"""Burst detection and best-of-burst selection.

A 'burst' is a contiguous run of frames taken close together in time AND
visually similar (same subject, same scene). On a typical wildlife shoot
this is what the photographer wanted in the moment — the next 10 frames
of the same flying egret, the same gorilla yawning. The user's job is to
pick ONE, maybe two, from each burst and discard the rest.

Algorithm:
- Sort all images by capture time (mtime is a fine proxy when EXIF time
  isn't available).
- Walk forward. Group adjacent images if BOTH:
    1. mtime delta < BURST_TIME_GAP_SEC, AND
    2. CLIP cosine similarity > BURST_SIM_THRESHOLD.
  Otherwise close the current burst and start a new one.
- Single-image groups are NOT bursts (no need to mark them).

For each burst we pick a 'best':
- Highest sharpness wins. Ties broken by Claude technical_score when present.
- Other frames in the burst get role='alt'; the picked one gets 'best'.
- Standalone images get burst_id=NULL, burst_role=NULL.

Bursts are recomputed for a folder on demand via /api/bursts/recompute.
We don't recompute automatically because (a) it requires CLIP embeddings
which aren't ready until the embedder has run, and (b) it churns when
the user is still ingesting more frames into the same shoot.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import db


# Defaults: 'same scene, same subject' thresholds, not 'rapid shutter
# burst'. A frame joins a group when it's still visually similar to the
# group's ANCHOR (first frame), not just the previous one — this keeps
# the burst together when the subject turns its head or you swap a lens
# mid-shoot. Time gap is to the anchor too, generous (2 minutes) so a
# brief pause doesn't split the same scene.
BURST_TIME_GAP_SEC = 120.0
BURST_SIM_THRESHOLD = 0.85


def _vec_from_bytes(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


# Field-by-field bonus/penalty weights for the burst composite score.
# Tuned so the AI fields dominate (which the user explicitly asked for —
# ranking should reflect the AI's judgment, not just file sharpness).
# Higher = better, sorting descending picks the keeper.
_KEEP_W = {"yes": 10.0, "no": -8.0}
_IN_FOCUS_W = {"yes": 5.0, "no": -10.0}
_EYE_FOCUS_W = {"sharp": 3.0, "soft": -2.0, "not_visible": -1.0, "n/a": 0.0}
_COMPOSITION_W = {"strong": 2.0, "standard": 0.0, "weak": -2.0}


def _composite_score(r) -> float:
    """Weighted score combining every AI signal we have for the image.
    Pure sharpness contributes a little (focus_score, 0-100ish range) as
    a tie-breaker; the AI's keep/in_focus/eye_focus calls dominate; and
    when Phase 2 has scored the image, the Claude technical + aesthetic
    scores layer on top. Two images with the same triage verdict get
    differentiated by Claude scores when available, by focus_score when
    not."""
    s = 0.0
    s += _KEEP_W.get(r["ai_keep"], 0.0)
    s += _IN_FOCUS_W.get(r["ai_in_focus"], 0.0)
    s += _EYE_FOCUS_W.get(r["ai_eye_focus"], 0.0)
    s += _COMPOSITION_W.get(r["ai_composition"], 0.0)
    if r["focus_score"] is not None:
        # Scale: focus_score is roughly 0-200; normalize so even a great
        # value (150+) only contributes ~1.5 — less than ai_in_focus.
        s += min(r["focus_score"], 200.0) / 100.0
    if r["claude_technical_score"] is not None:
        # /10 score; weight at 0.6 so it can flip ties but not override
        # the triage verdict.
        s += r["claude_technical_score"] * 0.6
    if r["claude_aesthetic_score"] is not None:
        s += r["claude_aesthetic_score"] * 0.6
    return s


def recompute_bursts_for_folder(
    folder: str,
    time_gap_sec: Optional[float] = None,
    sim_threshold: Optional[float] = None,
) -> dict:
    """Group adjacent same-scene frames into bursts. Returns rich
    diagnostics so the UI can explain a zero-burst result honestly:
    no embeddings yet? no capture times? scattered timestamps? The
    user shouldn't have to guess."""
    started = time.time()
    time_gap = float(time_gap_sec if time_gap_sec is not None else BURST_TIME_GAP_SEC)
    sim_min = float(sim_threshold if sim_threshold is not None else BURST_SIM_THRESHOLD)
    with db.connect() as conn:
        total_in_folder = conn.execute(
            "SELECT COUNT(*) AS n FROM images WHERE folder=?", (folder,)
        ).fetchone()["n"]
        with_embedding = conn.execute(
            "SELECT COUNT(*) AS n FROM images "
            "WHERE folder=? AND embedding IS NOT NULL",
            (folder,),
        ).fetchone()["n"]
        pending_analysis = conn.execute(
            "SELECT COUNT(*) AS n FROM images "
            "WHERE folder=? AND ai_status NOT IN ('done', 'error', 'cancelled')",
            (folder,),
        ).fetchone()["n"]

        # Bursts only group images the AI has reviewed. Ranking depends
        # on ai_keep / ai_in_focus / ai_eye_focus / ai_composition — if
        # those aren't filled yet the "best" pick would just be sharpness,
        # which is what the user explicitly didn't want. Click Find
        # bursts again once analysis completes.
        rows = conn.execute(
            "SELECT id, mtime, capture_time, embedding, focus_score, "
            "ai_status, ai_keep, ai_in_focus, ai_eye_focus, ai_composition, "
            "claude_technical_score, claude_aesthetic_score "
            "FROM images WHERE folder=? "
            "AND embedding IS NOT NULL "
            "AND ai_status='done' "
            "ORDER BY COALESCE(capture_time, mtime), id",
            (folder,),
        ).fetchall()

        with_capture_time = sum(1 for r in rows if r["capture_time"] is not None)

        # Reset existing burst membership for this folder before
        # recomputing — even if no bursts get found, old assignments
        # shouldn't linger.
        conn.execute(
            "UPDATE images SET burst_id=NULL, burst_role=NULL WHERE folder=?",
            (folder,),
        )

        if not rows:
            if total_in_folder == 0:
                reason = "Folder has no images."
            elif with_embedding == 0:
                reason = (
                    "No images in this folder have CLIP embeddings yet. "
                    "The embedder runs in the background after ingest — "
                    "wait a minute and try again."
                )
            else:
                reason = (
                    f"None of the {with_embedding} embedded images in this "
                    f"folder have finished AI analysis yet ({pending_analysis} "
                    f"still pending). Burst ranking uses the AI judgments to "
                    f"pick the best frame — click Find bursts again once "
                    f"analysis completes."
                )
            return {
                "folder": folder,
                "bursts": 0,
                "images_in_bursts": 0,
                "total_in_folder": total_in_folder,
                "with_embedding": with_embedding,
                "with_capture_time": 0,
                "pending_analysis": pending_analysis,
                "elapsed_sec": round(time.time() - started, 2),
                "reason": reason,
            }

        groups: list[list[dict]] = []
        current: list[dict] = []
        anchor_vec: np.ndarray | None = None
        anchor_t: float | None = None
        # Diagnostic counts — why an image didn't join the current group.
        rejected_time = 0
        rejected_sim = 0

        for r in rows:
            vec_bytes = r["embedding"]
            if not vec_bytes or len(vec_bytes) < 4:
                continue
            vec = _vec_from_bytes(vec_bytes)
            t = r["capture_time"] if r["capture_time"] is not None else (r["mtime"] or 0.0)
            entry = {
                "id": r["id"],
                "t": t,
                "vec": vec,
                "focus": r["focus_score"] if r["focus_score"] is not None else 0.0,
                "score": _composite_score(r),
            }
            if not current:
                current = [entry]
                anchor_vec = vec
                anchor_t = t
                continue

            # Anchor-based comparison: same SUBJECT (similar to first
            # frame) within a generous time window. Survives the bird
            # turning its head, a brief lens swap, etc.
            dt = abs(t - (anchor_t or t))
            sim = float(np.dot(vec, anchor_vec)) if anchor_vec is not None else 0.0
            if dt <= time_gap and sim >= sim_min:
                current.append(entry)
            else:
                if dt > time_gap:
                    rejected_time += 1
                elif sim < sim_min:
                    rejected_sim += 1
                if len(current) > 1:
                    groups.append(current)
                current = [entry]
                anchor_vec = vec
                anchor_t = t

        if len(current) > 1:
            groups.append(current)

        next_id = (conn.execute("SELECT COALESCE(MAX(burst_id), 0) AS m FROM images").fetchone()["m"]) or 0
        for g in groups:
            next_id += 1
            # Rank every frame in the group — not just one 'best'. Highest
            # composite score = rank 1 ('best'); the rest are ranked 2..N
            # so the photographer can see the runner-up next to the winner.
            ranked = sorted(g, key=lambda e: (e["score"], e["focus"]), reverse=True)
            for rank, entry in enumerate(ranked, start=1):
                role = "best" if rank == 1 else "alt"
                conn.execute(
                    "UPDATE images SET burst_id=?, burst_role=?, burst_rank=? "
                    "WHERE id=?",
                    (next_id, role, rank, entry["id"]),
                )

    reason = None
    if not groups:
        if with_capture_time == 0 and rejected_time > rejected_sim:
            reason = (
                f"No EXIF capture times on these images and file mtimes "
                f"span more than {int(time_gap)}s. Run 'Backfill capture "
                f"times' if EXIF data exists in the originals — or widen "
                f"the time-gap slider."
            )
        elif rejected_sim > 0 and rejected_time == 0:
            reason = (
                f"Frames are close in time but visually too different "
                f"(CLIP similarity to first-of-group < {sim_min:.2f}). "
                f"Try lowering the similarity slider."
            )
        else:
            reason = (
                f"Walked {len(rows)} embedded images; no run of 2+ "
                f"qualified (rejected on time: {rejected_time}, on "
                f"similarity: {rejected_sim})."
            )

    return {
        "folder": folder,
        "bursts": len(groups),
        "images_in_bursts": sum(len(g) for g in groups),
        "total_in_folder": total_in_folder,
        "with_embedding": with_embedding,
        "with_capture_time": with_capture_time,
        "rejected_time": rejected_time,
        "rejected_sim": rejected_sim,
        "time_gap_sec": time_gap,
        "sim_threshold": sim_min,
        "elapsed_sec": round(time.time() - started, 2),
        "reason": reason,
    }


def backfill_capture_times() -> dict:
    """Pull every EXIF field we track (DateTimeOriginal, camera, lens,
    focal length, ISO, aperture, shutter, exposure comp) for any image
    that's missing some of them. One exiftool call per image, same as
    ingest. Run once after upgrading; subsequent ingests fill these
    automatically."""
    from . import ingest
    started = time.time()
    updated = 0
    skipped = 0
    failed = 0
    errors: list[str] = []
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, path FROM images "
            "WHERE path IS NOT NULL "
            "AND (capture_time IS NULL OR exif_camera IS NULL)"
        ).fetchall()
        for r in rows:
            try:
                exif = ingest.extract_exif(Path(r["path"]))
                if exif:
                    conn.execute(
                        "UPDATE images SET "
                        "capture_time=COALESCE(?, capture_time), "
                        "exif_camera=COALESCE(?, exif_camera), "
                        "exif_lens=COALESCE(?, exif_lens), "
                        "exif_focal_length=COALESCE(?, exif_focal_length), "
                        "exif_iso=COALESCE(?, exif_iso), "
                        "exif_aperture=COALESCE(?, exif_aperture), "
                        "exif_shutter=COALESCE(?, exif_shutter), "
                        "exif_exposure_comp=COALESCE(?, exif_exposure_comp) "
                        "WHERE id=?",
                        (
                            exif.get("capture_time"),
                            exif.get("camera"),
                            exif.get("lens"),
                            exif.get("focal_length"),
                            exif.get("iso"),
                            exif.get("aperture"),
                            exif.get("shutter"),
                            exif.get("exposure_comp"),
                            r["id"],
                        ),
                    )
                    updated += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                if len(errors) < 10:
                    errors.append(f"{r['path']}: {type(exc).__name__}: {exc}")
    return {
        "scanned": updated + skipped + failed,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "elapsed_sec": round(time.time() - started, 2),
    }


def list_bursts(folder: str) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT burst_id, COUNT(*) AS n, "
            "SUM(CASE WHEN burst_role='best' THEN 1 ELSE 0 END) AS bests "
            "FROM images WHERE folder=? AND burst_id IS NOT NULL "
            "GROUP BY burst_id ORDER BY burst_id",
            (folder,),
        ).fetchall()
    return [dict(r) for r in rows]
