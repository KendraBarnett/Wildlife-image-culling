"""Personal classifier — predicts the photographer's keep/cull decision
from a multi-signal feature vector + their own corrections.

Why this exists: the local triage LLM hit a hard ceiling on nuance
(eye-in-shadow vs eye-sharp, moody-backlight vs clean light). Instead
of asking a 7B-11B model to verbalize judgment about every image, we
predict the photographer's verdict directly from features we already
compute.

What's in the feature vector (~552 dims total):

  1. CLIP embedding                            512 dims
     The visual fingerprint of the image. Backlit silhouettes cluster
     together; clean portraits cluster together. CLIP was trained on
     400M image-text pairs — it already knows what "in focus" and
     "good light" mean at the feature level.

  2. focus_score (Laplacian variance)          1 dim
     Classical sharpness measure; cheap and useful.

  3. AI triage outputs (one-hot)               30 dims
     ai_keep (3), ai_eye_focus (4), ai_motion (4), ai_composition (3),
     ai_lighting (7), ai_is_silhouette (1), technical_issues (multi-hot, 11).
     The LLM's structured verdict — even when it's wrong, the signal
     is useful (the classifier learns "when AI says X but photographer
     says Y, here's the pattern").

  4. EXIF features                             4 dims
     log-normalized ISO, aperture, shutter, focal length. Captures
     "shot in poor light" / "wide-aperture portrait" / "telephoto crop"
     contexts the classifier can correlate with your taste.

  5. Claude scores                             3 dims
     technical, aesthetic, has_claude_score flag. When Phase 2 has run,
     these are the strongest single signals. When it hasn't, the flag
     tells the classifier to ignore the zero values.

Why logistic regression on this: 552 features × ~100 training examples
is well within LR's range. L2 regularization (sklearn default C=1.0)
handles correlated features gracefully. Trains in milliseconds, predicts
across a 10k-image library in microseconds. The CLIP embedding does the
heavy visual lifting; the extra 40 dims contextualize it.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import db
from .config import DATA_DIR


CLASSIFIER_PATH = Path(DATA_DIR) / "personal_classifier.pkl"
MIN_TRAIN_EXAMPLES = 10
MIN_PER_CLASS = 3
FEATURE_VERSION = 2   # bump on schema changes so old pickles refuse to load


# Controlled vocabularies — must match AI_VOCAB in ai.py.
# Kept here too so the classifier doesn't have to import ai.py
# (ai.py pulls in httpx and the model dependencies).
_KEEP_VALUES = ["yes", "maybe", "no"]
_EYE_FOCUS_VALUES = ["sharp", "soft", "not_visible", "n/a"]
_MOTION_VALUES = ["still", "subtle", "in_motion", "blurred"]
_COMPOSITION_VALUES = ["strong", "standard", "weak"]
_LIGHTING_VALUES = ["harsh", "soft", "golden", "low_light",
                    "backlit", "overcast", "mixed"]
_ISSUE_VALUES = ["out_of_focus", "soft_edges", "camera_shake",
                 "clipped_subject", "blown_highlights", "underexposed",
                 "harsh_backlight", "heavy_noise", "obstructed",
                 "cluttered_background", "subject_too_small"]


def _onehot(value, vocab):
    """One-hot encode a single value against vocab. Missing → all-zeros."""
    out = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        out[vocab.index(value)] = 1.0
    return out


def _multihot(values, vocab):
    """Multi-hot encode a list of values. Used for technical_issues
    where multiple flags can fire at once."""
    out = np.zeros(len(vocab), dtype=np.float32)
    if not values:
        return out
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            return out
    if not isinstance(values, list):
        return out
    for v in values:
        if v in vocab:
            out[vocab.index(v)] = 1.0
    return out


def _parse_shutter(s):
    """exif_shutter is stored as '1/2000' or '0.5' — convert to a
    float (seconds). Returns None if unparseable."""
    if s is None or s == "":
        return None
    try:
        s = str(s).strip()
        if "/" in s:
            num, _, den = s.partition("/")
            return float(num) / float(den)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _log_normalize(x, default=0.0):
    """log-normalize a positive number. Returns default if x is None
    or non-positive. log space squashes the wide dynamic range of ISO
    (100-25600) and shutter (1/8000 - 1) into something LR can use."""
    if x is None:
        return default
    try:
        x = float(x)
        if x <= 0:
            return default
        return float(np.log10(x))
    except (ValueError, TypeError):
        return default


def _normalize(x, scale=1.0, default=0.0):
    """Linear normalize, with a default for missing values."""
    if x is None:
        return default
    try:
        return float(x) / scale
    except (ValueError, TypeError):
        return default


def build_features(row) -> Optional[np.ndarray]:
    """Build the full feature vector for one row. Returns None if the
    row has no embedding (the visual fingerprint is required). All
    other features handle missing values with sensible defaults so a
    half-analyzed image still produces a vector."""
    emb_bytes = row["embedding"]
    if not emb_bytes:
        return None
    try:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
    except Exception:
        return None
    if emb.size == 0:
        return None

    # Each section below contributes to the feature vector in a fixed
    # order; predict-time and train-time use the same order because
    # they both call this function.

    # 1. CLIP visual fingerprint
    parts = [emb]

    # 2. Classical sharpness — log-normalize so a focus_score of 200
    # doesn't dominate the LR weights.
    focus_score = row["focus_score"] if "focus_score" in row.keys() else None
    parts.append(np.array([_log_normalize(focus_score, default=0.0)], dtype=np.float32))

    # 3. AI triage one-hot block
    parts.append(_onehot(row["ai_keep"] if "ai_keep" in row.keys() else None, _KEEP_VALUES))
    parts.append(_onehot(row["ai_eye_focus"] if "ai_eye_focus" in row.keys() else None, _EYE_FOCUS_VALUES))
    parts.append(_onehot(row["ai_motion"] if "ai_motion" in row.keys() else None, _MOTION_VALUES))
    parts.append(_onehot(row["ai_composition"] if "ai_composition" in row.keys() else None, _COMPOSITION_VALUES))
    parts.append(_onehot(row["ai_lighting"] if "ai_lighting" in row.keys() else None, _LIGHTING_VALUES))
    sil = row["ai_is_silhouette"] if "ai_is_silhouette" in row.keys() else None
    parts.append(np.array([1.0 if sil else 0.0], dtype=np.float32))
    parts.append(_multihot(row["ai_technical_issues"] if "ai_technical_issues" in row.keys() else None, _ISSUE_VALUES))

    # 4. EXIF — wide dynamic ranges, all log-normalized
    parts.append(np.array([
        _log_normalize(row["exif_iso"] if "exif_iso" in row.keys() else None),
        _log_normalize(row["exif_aperture"] if "exif_aperture" in row.keys() else None),
        _log_normalize(_parse_shutter(row["exif_shutter"] if "exif_shutter" in row.keys() else None)),
        _log_normalize(row["exif_focal_length"] if "exif_focal_length" in row.keys() else None),
    ], dtype=np.float32))

    # 5. Claude scores — present=score, absent=0 + flag=0. The flag
    # tells the classifier "ignore the zeros" so unscored images don't
    # get pushed toward the cull side just because they have 0.0 / 0.0.
    tech = row["claude_technical_score"] if "claude_technical_score" in row.keys() else None
    aest = row["claude_aesthetic_score"] if "claude_aesthetic_score" in row.keys() else None
    has_claude = 1.0 if (tech is not None and aest is not None) else 0.0
    parts.append(np.array([
        _normalize(tech, scale=10.0, default=0.0),
        _normalize(aest, scale=10.0, default=0.0),
        has_claude,
    ], dtype=np.float32))

    return np.concatenate(parts)


# Columns the classifier needs in addition to embedding/labels. All
# nullable — build_features handles missing values gracefully.
_FEATURE_COLUMNS = (
    "embedding, user_tags, user_rating, ai_feedback_json, "
    "focus_score, "
    "ai_keep, ai_eye_focus, ai_motion, ai_composition, ai_lighting, "
    "ai_is_silhouette, ai_technical_issues, "
    "exif_iso, exif_aperture, exif_shutter, exif_focal_length, "
    "claude_technical_score, claude_aesthetic_score"
)


def _infer_label(row) -> Optional[int]:
    """Translate a row's signals into a binary keep/no label, or None
    if there's no strong signal. Priority order (strongest wins):
      1. user tags 'Portfolio' / 'Keep' → 1
      2. user tag 'Reject' → 0
      3. explicit correction in ai_feedback_json.keep → use that
      4. user_rating >= 4 → 1
      5. user_rating <= 2 → 0
      6. otherwise: None (skip at train time)
    """
    try:
        tags = json.loads(row["user_tags"] or "[]")
    except Exception:
        tags = []

    if "Portfolio" in tags or "Keep" in tags:
        return 1
    if "Reject" in tags:
        return 0

    fb = {}
    if row["ai_feedback_json"]:
        try:
            fb = json.loads(row["ai_feedback_json"]) or {}
        except Exception:
            fb = {}

    if fb.get("keep") == "yes":
        return 1
    if fb.get("keep") == "no":
        return 0

    rating = row["user_rating"]
    if rating is not None:
        if rating >= 4:
            return 1
        if rating <= 2:
            return 0

    return None


def _gather_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Pull every image that has BOTH a feature vector AND a strong
    keep/cull label. Returns (X, y) for sklearn."""
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT {_FEATURE_COLUMNS} FROM images "
            "WHERE embedding IS NOT NULL "
            "AND (user_tags IS NOT NULL OR user_rating IS NOT NULL "
            "     OR ai_feedback_json IS NOT NULL)"
        ).fetchall()

    X, y = [], []
    for r in rows:
        label = _infer_label(r)
        if label is None:
            continue
        feats = build_features(r)
        if feats is None:
            continue
        X.append(feats)
        y.append(label)

    if not X:
        return np.array([]), np.array([])
    return np.vstack(X), np.array(y, dtype=np.int64)


def status() -> dict:
    """Summary used by the UI. Cheap to call."""
    X, y = _gather_training_data()
    n_total = int(len(y))
    n_keep = int(y.sum()) if n_total else 0
    n_cull = n_total - n_keep
    trained = CLASSIFIER_PATH.exists()
    trained_at = None
    feature_dim = 0
    if trained:
        try:
            trained_at = CLASSIFIER_PATH.stat().st_mtime
        except OSError:
            trained_at = None
        bundle = _load()
        if isinstance(bundle, dict):
            feature_dim = bundle.get("n_features", 0)
    ready = trained and n_total >= MIN_TRAIN_EXAMPLES
    return {
        "trained": trained,
        "trained_at": trained_at,
        "ready": ready,
        "labeled_count": n_total,
        "labeled_keep": n_keep,
        "labeled_cull": n_cull,
        "min_required": MIN_TRAIN_EXAMPLES,
        "feature_dim": feature_dim,
        "feature_version": FEATURE_VERSION,
    }


def train() -> dict:
    """Fit the classifier on current labels and persist to disk."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {
            "ok": False,
            "error": "scikit-learn not installed. Run: pip install -e .",
        }

    X, y = _gather_training_data()
    n_total = int(len(y))
    n_keep = int(y.sum()) if n_total else 0
    n_cull = n_total - n_keep

    if n_total < MIN_TRAIN_EXAMPLES:
        return {
            "ok": False,
            "error": (
                f"Need at least {MIN_TRAIN_EXAMPLES} labeled images to train. "
                f"You have {n_total}. Rate, tag (Keep/Portfolio/Reject), or "
                f"correct AI verdicts on more images."
            ),
            "labeled_count": n_total,
        }
    if n_keep < MIN_PER_CLASS or n_cull < MIN_PER_CLASS:
        return {
            "ok": False,
            "error": (
                f"Need at least {MIN_PER_CLASS} keep AND {MIN_PER_CLASS} cull "
                f"examples. You have {n_keep} keep, {n_cull} cull."
            ),
            "labeled_keep": n_keep,
            "labeled_cull": n_cull,
        }

    started = time.time()
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    train_acc = float(model.score(X, y))
    n_features = int(X.shape[1])

    bundle = {
        "model": model,
        "n_features": n_features,
        "feature_version": FEATURE_VERSION,
        "trained_at": time.time(),
    }
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return {
        "ok": True,
        "labeled_count": n_total,
        "labeled_keep": n_keep,
        "labeled_cull": n_cull,
        "train_accuracy": round(train_acc, 3),
        "feature_dim": n_features,
        "elapsed_secs": round(time.time() - started, 3),
    }


def _load() -> Optional[object]:
    if not CLASSIFIER_PATH.exists():
        return None
    try:
        with open(CLASSIFIER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _model_from_bundle(bundle):
    """Accepts either the new dict-bundled format or the old raw-model
    pickle from before FEATURE_VERSION=2. Returns (model, n_features)
    or (None, 0) if stale."""
    if bundle is None:
        return None, 0
    if isinstance(bundle, dict):
        if bundle.get("feature_version") != FEATURE_VERSION:
            return None, 0
        return bundle.get("model"), bundle.get("n_features", 0)
    # Old format — just the bare model, expected to take 512-dim CLIP only.
    return None, 0  # force retrain; old format is incompatible with new features


def predict_all() -> dict:
    """Run the trained classifier across every embedded image and write
    learned_keep + learned_keep_confidence into each row."""
    bundle = _load()
    model, n_features = _model_from_bundle(bundle)
    if model is None:
        return {
            "ok": False,
            "error": "No trained model (or stale format). Click Train first.",
        }

    started = time.time()
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT id, {_FEATURE_COLUMNS} FROM images "
            "WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return {"ok": True, "predicted": 0, "elapsed_secs": 0.0}

        ids: list[int] = []
        feats_list: list[np.ndarray] = []
        for r in rows:
            feats = build_features(r)
            if feats is None:
                continue
            if feats.shape[0] != n_features:
                # Stale or mismatched dim — skip rather than crash.
                continue
            ids.append(r["id"])
            feats_list.append(feats)

        if not feats_list:
            return {"ok": True, "predicted": 0, "elapsed_secs": 0.0}

        X = np.vstack(feats_list)
        probs = model.predict_proba(X)[:, 1]
        for image_id, prob in zip(ids, probs):
            verdict = "yes" if prob >= 0.5 else "no"
            conn.execute(
                "UPDATE images SET learned_keep=?, learned_keep_confidence=? "
                "WHERE id=?",
                (verdict, float(prob), image_id),
            )

    return {
        "ok": True,
        "predicted": len(ids),
        "feature_dim": n_features,
        "elapsed_secs": round(time.time() - started, 3),
    }
