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


BURST_TIME_GAP_SEC = 8.0
BURST_SIM_THRESHOLD = 0.92


def _vec_from_bytes(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


def recompute_bursts_for_folder(folder: str) -> dict:
    started = time.time()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, mtime, embedding, focus_score, claude_technical_score "
            "FROM images WHERE folder=? AND embedding IS NOT NULL "
            "ORDER BY mtime, id",
            (folder,),
        ).fetchall()
        if not rows:
            return {"folder": folder, "bursts": 0, "images_in_bursts": 0, "elapsed_sec": 0.0}

        # Reset existing burst membership for this folder.
        conn.execute(
            "UPDATE images SET burst_id=NULL, burst_role=NULL WHERE folder=?",
            (folder,),
        )

        groups: list[list[dict]] = []
        current: list[dict] = []
        prev_vec: np.ndarray | None = None
        prev_mtime: float | None = None

        for r in rows:
            vec_bytes = r["embedding"]
            if not vec_bytes or len(vec_bytes) < 4:
                continue
            vec = _vec_from_bytes(vec_bytes)
            mtime = r["mtime"] or 0.0
            entry = {
                "id": r["id"],
                "mtime": mtime,
                "vec": vec,
                "focus": r["focus_score"] if r["focus_score"] is not None else 0.0,
                "art": r["claude_technical_score"] if r["claude_technical_score"] is not None else 0.0,
            }
            if not current:
                current = [entry]
                prev_vec = vec
                prev_mtime = mtime
                continue

            time_gap = abs(mtime - (prev_mtime or mtime))
            sim = float(np.dot(vec, prev_vec)) if prev_vec is not None else 0.0
            if time_gap <= BURST_TIME_GAP_SEC and sim >= BURST_SIM_THRESHOLD:
                current.append(entry)
            else:
                if len(current) > 1:
                    groups.append(current)
                current = [entry]
            prev_vec = vec
            prev_mtime = mtime

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

    return {
        "folder": folder,
        "bursts": len(groups),
        "images_in_bursts": sum(len(g) for g in groups),
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
