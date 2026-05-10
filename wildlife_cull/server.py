import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai, db, ingest, xmp
from .config import SERVER_HOST, SERVER_PORT, VISION_MODEL
from .worker import BackgroundWorker

worker = BackgroundWorker()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
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
    return {
        "ok": issue is None,
        "ollama_issue": issue,
        "vision_model": VISION_MODEL,
        "worker_last_error": worker.last_error,
        "worker_paused": worker.paused,
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
    subject_contains: Optional[str] = None,
    has_issue: Optional[str] = None,
    min_artistic: Optional[int] = None,
    min_portfolio: Optional[int] = None,
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
    if subject_contains:
        where.append("ai_subject LIKE ?")
        params.append(f"%{subject_contains}%")
    if has_issue:
        where.append("ai_technical_issues LIKE ?")
        params.append(f"%\"{has_issue}\"%")
    if min_artistic is not None:
        where.append("ai_artistic_score >= ?")
        params.append(min_artistic)
    if min_portfolio is not None:
        where.append("ai_portfolio_score >= ?")
        params.append(min_portfolio)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT id, filename, folder, is_raw, width, height, "
        "ai_status, ai_artistic_score, ai_portfolio_score, "
        "ai_eye_focus, ai_motion, ai_composition, ai_lighting, "
        "ai_is_silhouette, ai_subject, ai_technical_issues, "
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
    return d


class RateReq(BaseModel):
    rating: Optional[int] = Field(None, ge=0, le=5)
    tags: Optional[list[str]] = None
    notes: Optional[str] = None


@app.post("/api/image/{image_id}/rate")
def api_rate(image_id: int, req: RateReq) -> dict:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT path, user_rating, user_tags, user_notes FROM images WHERE id=?",
            (image_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="not found")

        rating = req.rating if req.rating is not None else row["user_rating"]
        if req.tags is not None:
            tags = req.tags
        else:
            try:
                tags = json.loads(row["user_tags"]) if row["user_tags"] else []
            except Exception:
                tags = []
        notes = req.notes if req.notes is not None else row["user_notes"]

        ts = time.time()
        db.set_user_rating(conn, image_id, rating, tags, notes, ts)

    try:
        xmp.write_sidecar(row["path"], rating, tags or [], notes)
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
def api_stats() -> dict:
    with db.connect() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN ai_status='done' THEN 1 ELSE 0 END) AS analyzed,
                SUM(CASE WHEN ai_status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN ai_status='error' THEN 1 ELSE 0 END) AS errored,
                SUM(CASE WHEN user_rating IS NOT NULL THEN 1 ELSE 0 END) AS rated,
                SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
            FROM images"""
        ).fetchone()
    return dict(row) if row else {}


def main() -> None:
    import uvicorn
    print(f"Wildlife Image Culling — open http://{SERVER_HOST}:{SERVER_PORT} in your browser")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
