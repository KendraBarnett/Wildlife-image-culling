"""Personal classifier — predicts the photographer's keep/cull decision
from CLIP embeddings + their own corrections.

The local triage LLM hit a hard ceiling on nuance (eye in shadow vs eye
sharp, moody-backlight vs clean light, etc.). Instead of asking a 7B-11B
model to verbalize about every image, we sidestep prompting entirely:

  1. Every image already has a 512-dim CLIP embedding (computed once at
     ingest, stored in images.embedding). These vectors place visually
     similar images near each other in embedding space.
  2. The photographer has been tagging / rating / correcting images.
     Those are ground-truth labels.
  3. A small logistic-regression classifier trained on (embedding, label)
     pairs predicts the photographer's keep decision on every other image.

Why logistic regression: CLIP embeddings are already a useful feature
space — there's no need for a deep net. LR works well on 50-2000 labeled
examples without overfitting, trains in milliseconds, and predicts in
microseconds. Fancier classifiers don't help here.

Why this works when prompting doesn't: the classifier doesn't have to
articulate WHY an image is a keeper. It just has to recognize the
neighborhood in embedding space. 'Backlit silhouette with prey in beak'
and 'sunlit perched bird, eye visible' end up far apart in CLIP space
even when the LLM can't write a coherent rule distinguishing them.
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


def _infer_label(row) -> Optional[int]:
    """Translate a row's signals into a binary keep/no label, or None
    if there's no strong signal.

    Priority (strongest signal wins):
      1. user tags 'Portfolio' / 'Keep' → 1 (keep)
      2. user tag 'Reject' → 0 (cull)
      3. explicit correction in ai_feedback_json.keep → use that
      4. user_rating >= 4 → 1
      5. user_rating <= 2 → 0
      6. otherwise: None (unlabeled — used at predict time but skipped
         at train time)
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
    # fb.get("keep") == "maybe" → ambiguous, no label

    rating = row["user_rating"]
    if rating is not None:
        if rating >= 4:
            return 1
        if rating <= 2:
            return 0

    return None


def _gather_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Pull every image that has BOTH a CLIP embedding AND a strong
    keep/cull signal from the user. Returns (X, y) for sklearn."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT embedding, user_tags, user_rating, ai_feedback_json "
            "FROM images "
            "WHERE embedding IS NOT NULL "
            "AND (user_tags IS NOT NULL OR user_rating IS NOT NULL "
            "     OR ai_feedback_json IS NOT NULL)"
        ).fetchall()

    X, y = [], []
    for r in rows:
        label = _infer_label(r)
        if label is None:
            continue
        try:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
        except Exception:
            continue
        if vec.size == 0:
            continue
        X.append(vec)
        y.append(label)

    if not X:
        return np.array([]), np.array([])
    return np.vstack(X), np.array(y, dtype=np.int64)


def status() -> dict:
    """Summary used by the UI: how many labeled examples, is a model
    trained, when. Cheap to call — used in the header / health poll."""
    X, y = _gather_training_data()
    n_total = int(len(y))
    n_keep = int(y.sum()) if n_total else 0
    n_cull = n_total - n_keep
    trained = CLASSIFIER_PATH.exists()
    trained_at = None
    if trained:
        try:
            trained_at = CLASSIFIER_PATH.stat().st_mtime
        except OSError:
            trained_at = None
    ready = trained and n_total >= MIN_TRAIN_EXAMPLES
    return {
        "trained": trained,
        "trained_at": trained_at,
        "ready": ready,
        "labeled_count": n_total,
        "labeled_keep": n_keep,
        "labeled_cull": n_cull,
        "min_required": MIN_TRAIN_EXAMPLES,
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
    # class_weight='balanced' compensates if you've labeled mostly one
    # side; C=1.0 is sklearn's default L2 strength and works fine on
    # 512-dim CLIP features without further tuning.
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    train_acc = float(model.score(X, y))

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(model, f)

    return {
        "ok": True,
        "labeled_count": n_total,
        "labeled_keep": n_keep,
        "labeled_cull": n_cull,
        "train_accuracy": round(train_acc, 3),
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


def predict_all() -> dict:
    """Run the trained classifier across every embedded image and write
    learned_keep + learned_keep_confidence into the row. Vectorized
    matrix multiply — fast even on 10k+ image libraries."""
    model = _load()
    if model is None:
        return {"ok": False, "error": "No trained model. Click Train first."}

    started = time.time()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM images WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return {"ok": True, "predicted": 0, "elapsed_secs": 0.0}

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            try:
                vec = np.frombuffer(r["embedding"], dtype=np.float32)
            except Exception:
                continue
            if vec.size == 0:
                continue
            ids.append(r["id"])
            vecs.append(vec)

        if not vecs:
            return {"ok": True, "predicted": 0, "elapsed_secs": 0.0}

        X = np.vstack(vecs)
        # predict_proba returns (n, 2); column 1 is P(keep=1).
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
        "elapsed_secs": round(time.time() - started, 3),
    }


def predict_one(embedding_bytes: bytes) -> Optional[tuple[str, float]]:
    """Single-image prediction for use right after a new embedding is
    computed (worker callback). Returns (verdict, confidence) or None."""
    model = _load()
    if model is None:
        return None
    try:
        vec = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
    except Exception:
        return None
    prob = float(model.predict_proba(vec)[0, 1])
    return ("yes" if prob >= 0.5 else "no", prob)
