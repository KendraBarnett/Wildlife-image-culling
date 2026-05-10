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
from typing import Iterable

import numpy as np

from . import db


BURST_TIME_GAP_SEC = 15.0   # was 8 — too tight for typical wildlife bursts
BURST_SIM_THRESHOLD = 0.90  # was 0.92 — slightly more permissive


def _vec_from_bytes(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


def recompute_bursts_for_folder(folder: str) -> dict:
    """Group adjacent same-scene frames into bursts. Returns rich
    diagnostics so the UI can explain a zero-burst result honestly:
    no embeddings yet? no capture times? scattered timestamps? The
    user shouldn't have to guess."""
    started = time.time()
    with db.connect() as conn:
        total_in_folder = conn.execute(
            "SELECT COUNT(*) AS n FROM images WHERE folder=?", (folder,)
        ).fetchone()["n"]
        with_embedding = conn.execute(
            "SELECT COUNT(*) AS n FROM images "
            "WHERE folder=? AND embedding IS NOT NULL",
            (folder,),
        ).fetchone()["n"]

        rows = conn.execute(
            "SELECT id, mtime, capture_time, embedding, focus_score, "
            "claude_technical_score "
            "FROM images WHERE folder=? AND embedding IS NOT NULL "
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
            return {
                "folder": folder,
                "bursts": 0,
                "images_in_bursts": 0,
                "total_in_folder": total_in_folder,
                "with_embedding": 0,
                "with_capture_time": 0,
                "elapsed_sec": round(time.time() - started, 2),
                "reason": (
                    "No images in this folder have CLIP embeddings yet. "
                    "The embedder runs in the background after ingest — "
                    "wait a minute and try again."
                ) if total_in_folder > 0 else "Folder has no images.",
            }

        groups: list[list[dict]] = []
        current: list[dict] = []
        prev_vec: np.ndarray | None = None
        prev_t: float | None = None
        # Diagnostic counts — why an adjacent pair did NOT join.
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
                "art": r["claude_technical_score"] if r["claude_technical_score"] is not None else 0.0,
            }
            if not current:
                current = [entry]
                prev_vec = vec
                prev_t = t
                continue

            time_gap = abs(t - (prev_t or t))
            sim = float(np.dot(vec, prev_vec)) if prev_vec is not None else 0.0
            if time_gap <= BURST_TIME_GAP_SEC and sim >= BURST_SIM_THRESHOLD:
                current.append(entry)
            else:
                if time_gap > BURST_TIME_GAP_SEC:
                    rejected_time += 1
                elif sim < BURST_SIM_THRESHOLD:
                    rejected_sim += 1
                if len(current) > 1:
                    groups.append(current)
                current = [entry]
            prev_vec = vec
            prev_t = t

        if len(current) > 1:
            groups.append(current)

        next_id = (conn.execute("SELECT COALESCE(MAX(burst_id), 0) AS m FROM images").fetchone()["m"]) or 0
        for g in groups:
            next_id += 1
            best = max(g, key=lambda e: (e["focus"], e["art"]))
            for entry in g:
                role = "best" if entry["id"] == best["id"] else "alt"
                conn.execute(
                    "UPDATE images SET burst_id=?, burst_role=? WHERE id=?",
                    (next_id, role, entry["id"]),
                )

    reason = None
    if not groups:
        if with_capture_time == 0 and rejected_time > rejected_sim:
            reason = (
                "No EXIF capture times on these images and file mtimes are "
                "scattered across more than 15s — no two adjacent frames "
                "qualify as a burst. Run 'Backfill capture times' if EXIF "
                "data exists in the originals."
            )
        elif rejected_sim > 0 and rejected_time == 0:
            reason = (
                "Frames are close in time but visually too different to "
                "group as bursts (CLIP similarity < 0.90)."
            )
        else:
            reason = (
                f"Walked {len(rows)} embedded images; no contiguous run "
                f"qualified as a burst (rejected on time: {rejected_time}, "
                f"on similarity: {rejected_sim})."
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
        "elapsed_sec": round(time.time() - started, 2),
        "reason": reason,
    }


def backfill_capture_times() -> dict:
    """Pull EXIF DateTimeOriginal from disk for every image missing
    capture_time. Run once after upgrading; subsequent ingests fill it
    automatically. Slow on first run for large libraries (~one exiftool
    call per image) but only has to happen once."""
    from . import ingest
    started = time.time()
    updated = 0
    skipped = 0
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, path FROM images WHERE capture_time IS NULL "
            "AND path IS NOT NULL"
        ).fetchall()
        for r in rows:
            t = ingest.extract_capture_time(Path(r["path"]))
            if t is not None:
                conn.execute(
                    "UPDATE images SET capture_time=? WHERE id=?",
                    (t, r["id"]),
                )
                updated += 1
            else:
                skipped += 1
    return {
        "scanned": updated + skipped,
        "updated": updated,
        "skipped": skipped,
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
