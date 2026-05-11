import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai, bursts, classifier, db, ingest, xmp
from . import claude as claude_mod
from .config import SERVER_HOST, SERVER_PORT, VISION_MODEL
from .worker import BackgroundWorker

worker = BackgroundWorker()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Probe every known folder for filesystem reachability on startup.
    # External drives that aren't plugged in will get last_online=0;
    # the worker skips them and the UI shows an (offline) badge.
    with db.connect() as conn:
        for r in db.list_folders(conn):
            db.update_folder_online(
                conn, r["folder"], Path(r["folder"]).is_dir()
            )
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="Wildlife Image Culling", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/health")
def health() -> dict:
    issue = ai.ollama_health()
    with db.connect() as conn:
        fb_count = conn.execute(
            "SELECT COUNT(*) AS n FROM images WHERE ai_feedback_json IS NOT NULL"
        ).fetchone()["n"]
    return {
        "ok": issue is None,
        "ollama_issue": issue,
        "vision_model": VISION_MODEL,
        "judges": [{"name": j["name"], "label": j["label"], "model": j["model"]} for j in ai.JUDGES],
        "current_judge": worker.current_judge,
        "worker_last_error": worker.last_error,
        "worker_paused": worker.paused,
        "worker_last_timing": worker.last_timing,
        "worker_in_flight": worker.in_flight,
        "feedback_corrections": fb_count,
        "feedback_active": min(fb_count, 5),
        "classifier": classifier.status(),
        "claude": {**claude_mod.usage_summary(), "bulk": claude_mod.bulk_scorer.status()},
    }


@app.get("/api/claude/usage")
def api_claude_usage() -> dict:
    return {**claude_mod.usage_summary(), "bulk": claude_mod.bulk_scorer.status()}


@app.get("/api/admin/preview-stats")
def api_preview_stats() -> dict:
    """How much disk the preview cache is using right now."""
    from .config import PREVIEW_DIR
    total_bytes = 0
    count = 0
    for p in Path(PREVIEW_DIR).glob("*"):
        if p.is_file():
            try:
                total_bytes += p.stat().st_size
                count += 1
            except OSError:
                pass
    return {
        "file_count": count,
        "bytes": total_bytes,
        "mb": round(total_bytes / (1024 * 1024), 1),
        "gb": round(total_bytes / (1024 * 1024 * 1024), 2),
    }


@app.get("/api/classifier/status")
def api_classifier_status() -> dict:
    """How many labeled examples, is a model trained, when. Used by
    the UI to decide whether to enable the 'Train personal model'
    button and to show the current state in the header."""
    return classifier.status()


@app.post("/api/classifier/train")
def api_classifier_train() -> dict:
    """Train the personal classifier on current labels, then run
    predict_all so every embedded image gets a fresh learned_keep
    verdict. Synchronous — training is milliseconds, prediction across
    a 10k library is a couple seconds at most."""
    result = classifier.train()
    if not result.get("ok"):
        return result
    pred = classifier.predict_all()
    return {**result, "predicted": pred.get("predicted", 0)}


@app.post("/api/admin/refresh-sidecars")
def api_refresh_sidecars() -> dict:
    """Rewrite every image's XMP sidecar from its current DB state.

    Use this once after a schema change or after the AI-tag feature
    ships: existing images won't have the new WC:* keywords in their
    sidecars yet, and this is the one-click backfill. For a typical
    library this is fast (a few ms per file, no model inference)."""
    written = 0
    errors: list[str] = []
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, path FROM images WHERE path IS NOT NULL"
        ).fetchall()
        for row in rows:
            try:
                result = xmp.refresh_sidecar(conn, row["id"])
                if result is not None:
                    written += 1
            except Exception as exc:
                errors.append(f"{row['path']}: {type(exc).__name__}: {exc}")
                if len(errors) >= 20:
                    break
    return {
        "ok": True,
        "scanned": len(rows),
        "written": written,
        "errors": errors,
    }


class CompareReq(BaseModel):
    image_ids: list[int]


@app.post("/api/compare")
def api_compare(req: CompareReq) -> dict:
    """Compare two images side-by-side with the local vision model.
    Returns which is better and a short reason. Phase 1 cost (free,
    local) — no Anthropic API call. Synchronous; usually 5–15 sec."""
    if not req.image_ids or len(req.image_ids) != 2:
        raise HTTPException(status_code=400, detail="Need exactly 2 image_ids")
    if req.image_ids[0] == req.image_ids[1]:
        raise HTTPException(status_code=400, detail="Pick two different images")
    with db.connect() as conn:
        rows = {
            r["id"]: dict(r) for r in conn.execute(
                "SELECT id, filename, preview_path FROM images WHERE id IN (?, ?)",
                (req.image_ids[0], req.image_ids[1]),
            ).fetchall()
        }
    if len(rows) != 2:
        raise HTTPException(status_code=404, detail="One or both images not found")
    img1 = rows[req.image_ids[0]]
    img2 = rows[req.image_ids[1]]
    if not img1.get("preview_path") or not img2.get("preview_path"):
        raise HTTPException(status_code=409, detail="One or both images have no preview yet")
    try:
        parsed, raw, timing = ai.compare_two_images(
            Path(img1["preview_path"]),
            Path(img2["preview_path"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    winner_index = parsed["winner"]
    winner_id = req.image_ids[winner_index - 1]
    loser_id = req.image_ids[2 - winner_index]
    return {
        "ok": True,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "winner_filename": rows[winner_id]["filename"],
        "loser_filename": rows[loser_id]["filename"],
        "reasoning": parsed.get("reasoning", ""),
        "margin": parsed.get("margin", "close"),
        "elapsed_secs": timing["total_secs"],
    }


class FolderPathReq(BaseModel):
    folder: str


@app.post("/api/folders/drop-previews")
def api_drop_previews_for_folder(req: FolderPathReq) -> dict:
    """Delete cached previews and thumbnails for every image in this
    folder. The AI data, embeddings, ratings, tags, and XMP sidecars
    stay intact — only the local JPEG cache goes. Previews re-extract
    on demand the next time you open an image in this folder. Use this
    on shoots you're done with to reclaim disk."""
    removed = 0
    bytes_freed = 0
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, preview_path, thumb_path FROM images WHERE folder=?",
            (req.folder,),
        ).fetchall()
        for r in rows:
            for p in (r["preview_path"], r["thumb_path"]):
                if not p:
                    continue
                try:
                    path = Path(p)
                    if path.exists():
                        bytes_freed += path.stat().st_size
                        path.unlink()
                        removed += 1
                except OSError:
                    pass
        conn.execute(
            "UPDATE images SET preview_path=NULL, thumb_path=NULL WHERE folder=?",
            (req.folder,),
        )
    return {
        "ok": True,
        "folder": req.folder,
        "files_removed": removed,
        "mb_freed": round(bytes_freed / (1024 * 1024), 1),
    }


@app.post("/api/admin/clear-previews")
def api_clear_previews() -> dict:
    """Delete every cached preview/thumb file and null out preview_path in
    the DB. The next time you open the app, previews will be re-extracted
    from the original RAW/JPEG on demand (slow for RAW — minutes for a
    folder of hundreds — but reclaims gigs of disk)."""
    from .config import PREVIEW_DIR
    removed = 0
    bytes_freed = 0
    for p in Path(PREVIEW_DIR).glob("*"):
        if p.is_file():
            try:
                bytes_freed += p.stat().st_size
                p.unlink()
                removed += 1
            except OSError:
                pass
    with db.connect() as conn:
        conn.execute("UPDATE images SET preview_path=NULL, thumb_path=NULL")
    return {
        "ok": True,
        "files_removed": removed,
        "mb_freed": round(bytes_freed / (1024 * 1024), 1),
    }


@app.post("/api/claude/score-keepers")
def api_score_keepers() -> dict:
    try:
        return claude_mod.bulk_scorer.start()
    except claude_mod.NotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/claude/score-keepers/cancel")
def api_score_keepers_cancel() -> dict:
    claude_mod.bulk_scorer.cancel()
    return {"ok": True, "status": claude_mod.bulk_scorer.status()}


@app.get("/api/claude/score-keepers/status")
def api_score_keepers_status() -> dict:
    return claude_mod.bulk_scorer.status()


@app.post("/api/image/{image_id}/score-with-claude")
def api_score_with_claude(image_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT id, preview_path FROM images WHERE id=?", (image_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="image not found")
    if not row["preview_path"]:
        raise HTTPException(status_code=409, detail="image has no preview yet")
    try:
        result = claude_mod.score_image(image_id, Path(row["preview_path"]))
    except claude_mod.NotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except claude_mod.BudgetExceeded as exc:
        raise HTTPException(status_code=402, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    claude_mod.persist_score(image_id, result)
    return {
        "ok": True,
        "image_id": image_id,
        "score": result,
        "usage": claude_mod.usage_summary(),
    }


@app.get("/api/worker/status")
def worker_status() -> dict:
    return {"paused": worker.paused, "last_error": worker.last_error}


@app.post("/api/worker/pause")
def worker_pause() -> dict:
    worker.pause()
    return {"paused": worker.paused}


@app.post("/api/worker/resume")
def worker_resume() -> dict:
    worker.resume()
    return {"paused": worker.paused}


@app.post("/api/worker/cancel")
def worker_cancel() -> dict:
    cancelled = worker.cancel_pending()
    return {"cancelled": cancelled, "paused": worker.paused}


class IngestReq(BaseModel):
    folder: str
    recursive: bool = True


@app.post("/api/ingest")
def api_ingest(req: IngestReq) -> dict:
    try:
        return ingest.ingest_folder(req.folder, recursive=req.recursive)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/browse")
def api_browse(path: Optional[str] = None) -> dict:
    if not path:
        home = Path.home()
        entries = [{"name": f"Home  ({home.name})", "path": str(home)}]
        volumes = Path("/Volumes")
        if volumes.exists():
            for v in sorted(volumes.iterdir(), key=lambda x: x.name.lower()):
                if v.is_dir() and not v.name.startswith("."):
                    entries.append({"name": f"Volume:  {v.name}", "path": str(v)})
        users = Path("/Users")
        if users.exists():
            entries.append({"name": "/Users", "path": "/Users"})
        return {"path": "", "parent_path": None, "dirs": entries, "is_root": True}

    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Cannot resolve path: {path}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {p}")

    dirs = []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if entry.is_dir() and not entry.name.startswith("."):
                    dirs.append({"name": entry.name, "path": str(entry)})
            except (PermissionError, OSError):
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {p}")

    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent_path": parent, "dirs": dirs, "is_root": False}


@app.get("/api/folders")
def api_folders() -> list[dict]:
    with db.connect() as conn:
        rows = db.list_folders(conn)
    return [dict(r) for r in rows]


@app.get("/api/folder")
def api_folder(folder: str) -> dict:
    with db.connect() as conn:
        summary = db.folder_summary(conn, folder)
    return {"folder": folder, "summary": summary}


@app.get("/api/ai-vocab")
def api_ai_vocab() -> dict:
    return ai.AI_VOCAB


class BurstReq(BaseModel):
    folder: str
    time_gap_sec: Optional[float] = None
    sim_threshold: Optional[float] = None


@app.post("/api/bursts/recompute")
def api_bursts_recompute(req: BurstReq) -> dict:
    return bursts.recompute_bursts_for_folder(
        req.folder,
        time_gap_sec=req.time_gap_sec,
        sim_threshold=req.sim_threshold,
    )


@app.post("/api/admin/backfill-capture-times")
def api_backfill_capture_times() -> dict:
    """Pull EXIF DateTimeOriginal for every image that doesn't have a
    capture_time yet. One-shot — burst detection works much better with
    real capture times than with file mtimes."""
    return bursts.backfill_capture_times()


@app.get("/api/bursts")
def api_bursts(folder: str) -> dict:
    return {"bursts": bursts.list_bursts(folder)}


@app.get("/api/judges")
def api_judges() -> list[dict]:
    return [
        {"name": j["name"], "label": j["label"], "model": j["model"], "weight": j["weight"], "primary": bool(j.get("primary"))}
        for j in ai.JUDGES
    ]


@app.get("/api/images")
def api_images(
    folder: Optional[str] = None,
    rating_min: Optional[int] = None,
    rating_max: Optional[int] = None,
    only_rated: bool = False,
    ai_status: Optional[str] = None,
    eye_focus: Optional[str] = None,
    motion: Optional[str] = None,
    composition: Optional[str] = None,
    lighting: Optional[str] = None,
    silhouette: Optional[str] = None,
    animal_type: Optional[str] = None,
    species_contains: Optional[str] = None,
    subject_contains: Optional[str] = None,
    has_issue: Optional[str] = None,
    has_feedback: Optional[str] = None,
    focus: Optional[str] = None,
    burst_only: Optional[str] = None,
    keep: Optional[str] = None,
    in_focus: Optional[str] = None,
    scored: Optional[str] = None,
    learned: Optional[str] = None,
    min_technical: Optional[float] = None,
    min_aesthetic: Optional[float] = None,
    limit: int = Query(default=500, le=2000),
    offset: int = 0,
) -> dict:
    where = []
    params: list = []
    if folder:
        where.append("folder=?")
        params.append(folder)
    if rating_min is not None:
        where.append("user_rating >= ?")
        params.append(rating_min)
    if rating_max is not None:
        where.append("user_rating <= ?")
        params.append(rating_max)
    if only_rated:
        where.append("user_rating IS NOT NULL")
    if ai_status:
        where.append("ai_status=?")
        params.append(ai_status)
    if eye_focus:
        where.append("ai_eye_focus=?")
        params.append(eye_focus)
    if motion:
        where.append("ai_motion=?")
        params.append(motion)
    if composition:
        where.append("ai_composition=?")
        params.append(composition)
    if lighting:
        where.append("ai_lighting=?")
        params.append(lighting)
    if silhouette in ("yes", "no"):
        where.append("ai_is_silhouette=?")
        params.append(1 if silhouette == "yes" else 0)
    if animal_type:
        where.append("ai_animal_type=?")
        params.append(animal_type)
    if species_contains:
        where.append("ai_species LIKE ?")
        params.append(f"%{species_contains}%")
    if subject_contains:
        where.append("ai_subject LIKE ?")
        params.append(f"%{subject_contains}%")
    if has_issue:
        where.append("ai_technical_issues LIKE ?")
        params.append(f"%\"{has_issue}\"%")
    if has_feedback == "yes":
        where.append("ai_feedback_json IS NOT NULL")
    elif has_feedback == "no":
        where.append("ai_feedback_json IS NULL")
    if focus:
        where.append("focus_label=?")
        params.append(focus)
    if burst_only == "best":
        where.append("burst_role='best'")
    elif burst_only == "alts":
        where.append("burst_role='alt'")
    elif burst_only == "in_burst":
        where.append("burst_id IS NOT NULL")
    elif burst_only == "singletons":
        where.append("burst_id IS NULL")
    if keep in ("yes", "no"):
        where.append("ai_keep=?")
        params.append(keep)
    if in_focus in ("yes", "no"):
        where.append("ai_in_focus=?")
        params.append(in_focus)
    if scored == "yes":
        where.append("claude_scored_at IS NOT NULL")
    elif scored == "no":
        where.append("claude_scored_at IS NULL")
    if learned == "yes":
        where.append("learned_keep='yes'")
    elif learned == "no":
        where.append("learned_keep='no'")
    elif learned == "disagree":
        # AI vs personal model say different things — high-value review images.
        where.append(
            "learned_keep IS NOT NULL AND ai_keep IS NOT NULL "
            "AND learned_keep != ai_keep AND ai_keep != 'maybe'"
        )
    if min_technical is not None:
        where.append("claude_technical_score >= ?")
        params.append(min_technical)
    if min_aesthetic is not None:
        where.append("claude_aesthetic_score >= ?")
        params.append(min_aesthetic)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT id, filename, folder, is_raw, width, height, "
        "ai_status, ai_keep, ai_notes, "
        "ai_in_focus, ai_eye_focus, ai_motion, ai_composition, ai_lighting, "
        "ai_is_silhouette, ai_subject, ai_animal_type, ai_species, "
        "ai_technical_issues, ai_judges_json, ai_feedback_json, "
        "focus_score, focus_label, burst_id, burst_role, burst_rank, "
        "claude_technical_score, claude_aesthetic_score, claude_reasoning, "
        "claude_scored_at, claude_cost_usd, "
        "capture_time, exif_camera, exif_lens, exif_focal_length, "
        "exif_iso, exif_aperture, exif_shutter, exif_exposure_comp, "
        "learned_keep, learned_keep_confidence, "
        "user_rating, user_tags, user_notes "
        f"FROM images {where_sql} ORDER BY filename LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    with db.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        if d.get("user_tags"):
            try:
                d["user_tags"] = json.loads(d["user_tags"])
            except Exception:
                d["user_tags"] = []
        else:
            d["user_tags"] = []
        if d.get("ai_technical_issues"):
            try:
                d["ai_technical_issues"] = json.loads(d["ai_technical_issues"])
            except Exception:
                d["ai_technical_issues"] = []
        else:
            d["ai_technical_issues"] = []
        judges = {}
        if d.get("ai_judges_json"):
            try:
                judges = json.loads(d["ai_judges_json"])
            except Exception:
                judges = {}
        d["ai_judges"] = judges
        d["judges_done"] = sum(1 for v in judges.values() if isinstance(v, dict) and v.get("status") == "done")
        d["judges_total"] = len(ai.JUDGES)
        d.pop("ai_judges_json", None)
        feedback = None
        if d.get("ai_feedback_json"):
            try:
                feedback = json.loads(d["ai_feedback_json"])
            except Exception:
                feedback = None
        d["ai_feedback"] = feedback
        d.pop("ai_feedback_json", None)
        items.append(d)
    return {"images": items, "count": len(items)}


class ReanalyzeReq(BaseModel):
    ids: Optional[list[int]] = None
    status: Optional[str] = None
    folder: Optional[str] = None


@app.post("/api/image/{image_id}/reanalyze")
def api_reanalyze_one(image_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT id FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")
        n = db.reset_ai_for_reanalysis(conn, [image_id])
    return {"reset": n}


@app.post("/api/reanalyze")
def api_reanalyze_bulk(req: ReanalyzeReq) -> dict:
    with db.connect() as conn:
        ids = list(req.ids or [])
        if req.status:
            ids.extend(db.select_image_ids_by_status(conn, req.status, req.folder))
        ids = list(dict.fromkeys(ids))  # dedupe, preserve order
        if not ids:
            return {"reset": 0}
        n = db.reset_ai_for_reanalysis(conn, ids)
    return {"reset": n}


@app.get("/api/image/{image_id}")
def api_image(image_id: int) -> dict:
    with db.connect() as conn:
        r = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="not found")
    d = dict(r)
    d.pop("embedding", None)
    if d.get("user_tags"):
        try:
            d["user_tags"] = json.loads(d["user_tags"])
        except Exception:
            d["user_tags"] = []
    else:
        d["user_tags"] = []
    if d.get("ai_json"):
        try:
            d["ai"] = json.loads(d["ai_json"])
        except Exception:
            d["ai"] = {"_unparsed": d["ai_json"]}
    if d.get("ai_judges_json"):
        try:
            d["ai_judges"] = json.loads(d["ai_judges_json"])
        except Exception:
            d["ai_judges"] = {}
    else:
        d["ai_judges"] = {}
    if d.get("ai_feedback_json"):
        try:
            d["ai_feedback"] = json.loads(d["ai_feedback_json"])
        except Exception:
            d["ai_feedback"] = {}
    else:
        d["ai_feedback"] = None
    if d.get("ai_technical_issues"):
        try:
            d["ai_technical_issues"] = json.loads(d["ai_technical_issues"])
        except Exception:
            d["ai_technical_issues"] = []
    else:
        d["ai_technical_issues"] = []
    return d


class FeedbackReq(BaseModel):
    marked_wrong: bool = False
    keep: Optional[str] = None
    in_focus: Optional[str] = None
    eye_focus: Optional[str] = None
    motion: Optional[str] = None
    composition: Optional[str] = None
    lighting: Optional[str] = None
    animal_type: Optional[str] = None
    species: Optional[str] = None
    subject: Optional[str] = None
    is_silhouette: Optional[bool] = None
    technical_issues: Optional[list[str]] = None
    note: Optional[str] = None


@app.post("/api/image/{image_id}/feedback")
def api_feedback(image_id: int, req: FeedbackReq) -> dict:
    payload = req.model_dump(exclude_none=True)
    with db.connect() as conn:
        existing = conn.execute("SELECT id FROM images WHERE id=?", (image_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="not found")
        db.set_feedback(conn, image_id, payload)
        # A correction can flip the effective AI tag set (e.g. keep no→yes),
        # so refresh the sidecar so Lightroom sees the updated truth.
        try:
            xmp.refresh_sidecar(conn, image_id)
        except Exception:
            pass
    return {"ok": True, "feedback": payload}


@app.delete("/api/image/{image_id}/feedback")
def api_clear_feedback(image_id: int) -> dict:
    with db.connect() as conn:
        existing = conn.execute("SELECT id FROM images WHERE id=?", (image_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="not found")
        db.clear_feedback(conn, image_id)
        try:
            xmp.refresh_sidecar(conn, image_id)
        except Exception:
            pass
    return {"ok": True}


class RateReq(BaseModel):
    rating: Optional[int] = Field(None, ge=0, le=5)
    tags: Optional[list[str]] = None
    notes: Optional[str] = None


class BulkTagsReq(BaseModel):
    image_ids: list[int]
    tags: list[str]
    mode: str = "add"   # "add" merges into existing tags; "replace" overwrites


@app.post("/api/bulk/tags")
def api_bulk_tags(req: BulkTagsReq) -> dict:
    """Apply tags to many images at once. Writes to SQLite and to each
    image's XMP sidecar so Lightroom picks them up."""
    if not req.image_ids:
        raise HTTPException(status_code=400, detail="no images given")
    if req.mode not in ("add", "replace"):
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode}")
    new_tags = [t.strip() for t in req.tags if t and t.strip()]
    if not new_tags and req.mode == "add":
        raise HTTPException(status_code=400, detail="no tags to add")

    updated = 0
    sidecar_errors: list[str] = []
    ts = time.time()
    with db.connect() as conn:
        placeholders = ",".join("?" for _ in req.image_ids)
        rows = conn.execute(
            f"SELECT id, path, user_rating, user_tags, user_notes "
            f"FROM images WHERE id IN ({placeholders})",
            req.image_ids,
        ).fetchall()
        for row in rows:
            try:
                existing = json.loads(row["user_tags"]) if row["user_tags"] else []
            except Exception:
                existing = []
            if req.mode == "add":
                seen = set(existing)
                merged = list(existing) + [t for t in new_tags if t not in seen]
            else:
                merged = list(new_tags)
            db.set_user_rating(
                conn, row["id"], row["user_rating"], merged, row["user_notes"], ts,
            )
            try:
                xmp.refresh_sidecar(conn, row["id"])
            except Exception as exc:
                sidecar_errors.append(f"{row['path']}: {exc}")
            updated += 1
    return {
        "ok": True,
        "updated": updated,
        "sidecar_errors": sidecar_errors[:10],
    }


@app.post("/api/image/{image_id}/rate")
def api_rate(image_id: int, req: RateReq) -> dict:
    """Trust the request shape. The UI always sends all three fields on
    every save (see app.js doSave), so `rating=None` means 'clear', not
    'no change'. The earlier 'fall back to existing if None' logic broke
    the clear-rating button — null came in, we treated it as 'unchanged',
    and the existing rating stuck."""
    sent = req.model_fields_set
    with db.connect() as conn:
        row = conn.execute(
            "SELECT path, user_rating, user_tags, user_notes FROM images WHERE id=?",
            (image_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")

        rating = req.rating if "rating" in sent else row["user_rating"]
        if "tags" in sent:
            tags = req.tags or []
        else:
            try:
                tags = json.loads(row["user_tags"]) if row["user_tags"] else []
            except Exception:
                tags = []
        notes = req.notes if "notes" in sent else row["user_notes"]

        ts = time.time()
        db.set_user_rating(conn, image_id, rating, tags, notes, ts)

    try:
        with db.connect() as conn:
            xmp.refresh_sidecar(conn, image_id)
        sidecar_ok = True
        sidecar_err = None
    except Exception as exc:
        sidecar_ok = False
        sidecar_err = str(exc)

    return {"ok": True, "sidecar_written": sidecar_ok, "sidecar_error": sidecar_err}


@app.get("/api/preview/{image_id}")
def api_preview(image_id: int):
    with db.connect() as conn:
        r = conn.execute(
            "SELECT preview_path FROM images WHERE id=?", (image_id,)
        ).fetchone()
    if not r or not r["preview_path"]:
        raise HTTPException(status_code=404, detail="no preview")
    return FileResponse(r["preview_path"], media_type="image/jpeg")


@app.get("/api/thumb/{image_id}")
def api_thumb(image_id: int):
    with db.connect() as conn:
        r = conn.execute(
            "SELECT thumb_path FROM images WHERE id=?", (image_id,)
        ).fetchone()
    if not r or not r["thumb_path"]:
        raise HTTPException(status_code=404, detail="no thumb")
    return FileResponse(r["thumb_path"], media_type="image/jpeg")


@app.get("/api/stats")
def api_stats(folder: Optional[str] = None) -> dict:
    """When folder is set, scope counts to that folder. When omitted, the
    counts roll up across every folder (the 'All' tab)."""
    where = "WHERE folder=?" if folder else ""
    params = (folder,) if folder else ()
    with db.connect() as conn:
        row = conn.execute(
            f"""SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN ai_status='done' THEN 1 ELSE 0 END) AS analyzed,
                SUM(CASE WHEN ai_status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN ai_status='error' THEN 1 ELSE 0 END) AS errored,
                SUM(CASE WHEN ai_status='unreachable' THEN 1 ELSE 0 END) AS unreachable,
                SUM(CASE WHEN ai_status='pending' AND ai_judges_json IS NOT NULL AND ai_judges_json != '' AND ai_judges_json != '{{}}' THEN 1 ELSE 0 END) AS partial,
                SUM(CASE WHEN user_rating IS NOT NULL THEN 1 ELSE 0 END) AS rated,
                SUM(CASE WHEN ai_feedback_json IS NOT NULL THEN 1 ELSE 0 END) AS feedback_count,
                SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
            FROM images {where}""",
            params,
        ).fetchone()
    return dict(row) if row else {}


class FolderPauseReq(BaseModel):
    folder: str
    paused: bool


@app.post("/api/folders/pause")
def api_folder_pause(req: FolderPauseReq) -> dict:
    """Toggle pause for one folder. Worker picks this up on its next
    tick — within ~2 sec the folder stops accepting new analysis."""
    with db.connect() as conn:
        db.set_folder_paused(conn, req.folder, req.paused)
    return {"ok": True, "folder": req.folder, "paused": req.paused}


class FolderFocusReq(BaseModel):
    folder: Optional[str] = None


@app.post("/api/folders/focus")
def api_folder_focus(req: FolderFocusReq) -> dict:
    """The UI calls this when the user switches tabs. Worker uses the
    focused folder as priority when picking the next image. Passing
    None clears focus (use for the 'All' tab)."""
    worker.focused_folder = req.folder or None
    return {"ok": True, "focused_folder": worker.focused_folder}


@app.post("/api/folders/recheck")
def api_folders_recheck() -> dict:
    """Probe every folder's filesystem path. Updates last_online so the
    UI can drop the (offline) badge from folders whose drive is now
    plugged back in (and add it to ones that just got unplugged).
    Also resets ai_status='unreachable' rows in folders that came back
    online so the worker tries them again."""
    transitioned_online: list[str] = []
    transitioned_offline: list[str] = []
    with db.connect() as conn:
        folders = [r["folder"] for r in db.list_folders(conn)]
        for folder in folders:
            online_before = not (folder in db.list_offline_folders(conn))
            online_now = Path(folder).is_dir()
            db.update_folder_online(conn, folder, online_now)
            if online_now and not online_before:
                transitioned_online.append(folder)
                conn.execute(
                    "UPDATE images SET ai_status='pending' "
                    "WHERE folder=? AND ai_status='unreachable'",
                    (folder,),
                )
            elif not online_now and online_before:
                transitioned_offline.append(folder)
    return {
        "ok": True,
        "online_again": transitioned_online,
        "now_offline": transitioned_offline,
    }


def main() -> None:
    import uvicorn
    print(f"Wildlife Image Culling — open http://{SERVER_HOST}:{SERVER_PORT} in your browser")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
