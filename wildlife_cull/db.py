import sqlite3
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    folder TEXT NOT NULL,
    file_size INTEGER,
    mtime REAL,
    is_raw INTEGER NOT NULL DEFAULT 0,
    preview_path TEXT,
    thumb_path TEXT,
    width INTEGER,
    height INTEGER,
    ingested_at REAL NOT NULL,

    ai_status TEXT NOT NULL DEFAULT 'pending',
    ai_json TEXT,
    ai_keep TEXT,
    ai_in_focus TEXT,
    ai_eye_focus TEXT,
    ai_motion TEXT,
    ai_composition TEXT,
    ai_lighting TEXT,
    ai_is_silhouette INTEGER,
    ai_subject TEXT,
    ai_animal_type TEXT,
    ai_species TEXT,
    ai_technical_issues TEXT,
    ai_notes TEXT,
    ai_analyzed_at REAL,
    ai_judges_json TEXT,
    ai_feedback_json TEXT,
    focus_score REAL,
    focus_label TEXT,
    burst_id INTEGER,
    burst_role TEXT,

    claude_technical_score REAL,
    claude_aesthetic_score REAL,
    claude_reasoning TEXT,
    claude_scored_at REAL,
    claude_input_tokens INTEGER,
    claude_output_tokens INTEGER,
    claude_cache_read_tokens INTEGER,
    claude_cost_usd REAL,

    embedding BLOB,
    embedding_model TEXT,

    user_rating INTEGER,
    user_tags TEXT,
    user_notes TEXT,
    user_updated_at REAL
);

CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder);
CREATE INDEX IF NOT EXISTS idx_images_ai_status ON images(ai_status);
CREATE INDEX IF NOT EXISTS idx_images_user_rating ON images(user_rating);

CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    ingested_at REAL NOT NULL,
    image_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claude_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_claude_usage_ts ON claude_usage(ts);
"""


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
    for name, ddl in [
        ("ai_composition", "ALTER TABLE images ADD COLUMN ai_composition TEXT"),
        ("ai_lighting", "ALTER TABLE images ADD COLUMN ai_lighting TEXT"),
        ("ai_technical_issues", "ALTER TABLE images ADD COLUMN ai_technical_issues TEXT"),
        ("ai_animal_type", "ALTER TABLE images ADD COLUMN ai_animal_type TEXT"),
        ("ai_species", "ALTER TABLE images ADD COLUMN ai_species TEXT"),
        ("ai_judges_json", "ALTER TABLE images ADD COLUMN ai_judges_json TEXT"),
        ("ai_feedback_json", "ALTER TABLE images ADD COLUMN ai_feedback_json TEXT"),
        ("focus_score", "ALTER TABLE images ADD COLUMN focus_score REAL"),
        ("focus_label", "ALTER TABLE images ADD COLUMN focus_label TEXT"),
        ("burst_id", "ALTER TABLE images ADD COLUMN burst_id INTEGER"),
        ("burst_role", "ALTER TABLE images ADD COLUMN burst_role TEXT"),
        ("ai_in_focus", "ALTER TABLE images ADD COLUMN ai_in_focus TEXT"),
        ("ai_keep", "ALTER TABLE images ADD COLUMN ai_keep TEXT"),
        ("ai_notes", "ALTER TABLE images ADD COLUMN ai_notes TEXT"),
        ("claude_technical_score", "ALTER TABLE images ADD COLUMN claude_technical_score REAL"),
        ("claude_aesthetic_score", "ALTER TABLE images ADD COLUMN claude_aesthetic_score REAL"),
        ("claude_reasoning", "ALTER TABLE images ADD COLUMN claude_reasoning TEXT"),
        ("claude_scored_at", "ALTER TABLE images ADD COLUMN claude_scored_at REAL"),
        ("claude_input_tokens", "ALTER TABLE images ADD COLUMN claude_input_tokens INTEGER"),
        ("claude_output_tokens", "ALTER TABLE images ADD COLUMN claude_output_tokens INTEGER"),
        ("claude_cache_read_tokens", "ALTER TABLE images ADD COLUMN claude_cache_read_tokens INTEGER"),
        ("claude_cost_usd", "ALTER TABLE images ADD COLUMN claude_cost_usd REAL"),
    ]:
        if name not in cols:
            conn.execute(ddl)

    # Wipe the legacy artistic/portfolio score columns. SQLite 3.35+ supports
    # DROP COLUMN; older versions silently fail and we keep going (the columns
    # will just sit there orphaned, which is harmless — nothing reads them).
    for legacy in ("ai_artistic_score", "ai_portfolio_score"):
        if legacy in cols:
            try:
                conn.execute(f"ALTER TABLE images DROP COLUMN {legacy}")
            except sqlite3.OperationalError:
                pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_burst ON images(burst_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_keep ON images(ai_keep)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_claude_usage_ts ON claude_usage(ts)")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_image(conn: sqlite3.Connection, row: dict) -> int:
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    col_list = ",".join(cols)
    update_clause = ",".join(f"{c}=excluded.{c}" for c in cols if c != "path")
    sql = (
        f"INSERT INTO images ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {update_clause}"
    )
    cur = conn.execute(sql, [row[c] for c in cols])
    if cur.lastrowid:
        return cur.lastrowid
    existing = conn.execute("SELECT id FROM images WHERE path=?", (row["path"],)).fetchone()
    return existing["id"]


def _read_judges_json(conn: sqlite3.Connection, image_id: int) -> dict:
    row = conn.execute("SELECT ai_judges_json FROM images WHERE id=?", (image_id,)).fetchone()
    if not row or not row["ai_judges_json"]:
        return {}
    try:
        data = json.loads(row["ai_judges_json"])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_judge_result(
    conn: sqlite3.Connection,
    image_id: int,
    judge_name: str,
    status: str,
    model: str,
    payload: Optional[dict],
    raw: Optional[str],
    error: Optional[str],
    ts: float,
) -> dict:
    """Update a single judge's slot in ai_judges_json. Returns the merged dict."""
    judges = _read_judges_json(conn, image_id)
    judges[judge_name] = {
        "status": status,
        "model": model,
        "result": payload,
        "raw": raw,
        "error": error,
        "analyzed_at": ts,
    }
    conn.execute(
        "UPDATE images SET ai_judges_json=? WHERE id=?",
        (json.dumps(judges), image_id),
    )
    return judges


def finalize_image_if_complete(
    conn: sqlite3.Connection,
    image_id: int,
    judge_names: list[str],
) -> bool:
    """If every requested judge has terminal status, recompute combined fields
    and flip ai_status to 'done' (or 'error' if every judge errored).
    Returns True if the image was finalized this call."""
    judges = _read_judges_json(conn, image_id)
    terminal = {"done", "error"}
    if not all(judges.get(n, {}).get("status") in terminal for n in judge_names):
        return False

    done = [judges[n] for n in judge_names if judges[n]["status"] == "done" and judges[n].get("result")]
    if not done:
        first_err = next((judges[n].get("error") or "unknown" for n in judge_names if judges[n]["status"] == "error"), "no judges succeeded")
        conn.execute(
            "UPDATE images SET ai_status='error' WHERE id=?",
            (image_id,),
        )
        # Keep last error message in ai_json for the UI's error display.
        conn.execute(
            "UPDATE images SET ai_json=? WHERE id=?",
            (json.dumps({"error": first_err}), image_id),
        )
        return True

    primary = next((judges[n]["result"] for n in judge_names if judges[n]["status"] == "done" and n == judge_names[0]), None)
    if primary is None:
        primary = done[0]["result"]
    issues = primary.get("technical_issues") or []
    if not isinstance(issues, list):
        issues = []

    conn.execute(
        """UPDATE images SET
            ai_status='done',
            ai_keep=?,
            ai_in_focus=?,
            ai_eye_focus=?,
            ai_motion=?,
            ai_composition=?,
            ai_lighting=?,
            ai_is_silhouette=?,
            ai_subject=?,
            ai_animal_type=?,
            ai_species=?,
            ai_technical_issues=?,
            ai_notes=?,
            ai_analyzed_at=?
        WHERE id=?""",
        (
            primary.get("keep"),
            primary.get("in_focus"),
            primary.get("eye_focus"),
            primary.get("motion"),
            primary.get("composition"),
            primary.get("lighting"),
            1 if primary.get("is_silhouette") else 0,
            primary.get("subject"),
            primary.get("animal_type"),
            primary.get("species"),
            json.dumps(issues),
            primary.get("notes"),
            time.time(),
            image_id,
        ),
    )
    return True


def list_pending_for_judge(
    conn: sqlite3.Connection,
    judge_name: str,
    limit: int = 1,
) -> list[sqlite3.Row]:
    """Find images whose ai_judges_json has no terminal entry for this judge.
    Skips images that aren't in a fresh 'pending' or 'done'-but-incomplete
    state and skips cancelled images."""
    rows = conn.execute(
        "SELECT * FROM images WHERE ai_status NOT IN ('cancelled') ORDER BY filename, id"
    ).fetchall()
    out = []
    for r in rows:
        data = {}
        if r["ai_judges_json"]:
            try:
                data = json.loads(r["ai_judges_json"])
            except Exception:
                data = {}
        existing = data.get(judge_name, {}).get("status")
        if existing not in ("done", "error"):
            out.append(r)
            if len(out) >= limit:
                break
    return out


def next_pending_image_and_judge(
    conn: sqlite3.Connection,
    judge_names: list[str],
) -> Optional[tuple[str, sqlite3.Row]]:
    """Per-image mode: walk images in filename order (matches the UI's
    default grid sort, so analysis fills in left-to-right, top-to-bottom)
    and return the first (judge_name, image) pair where that judge has not
    yet reached a terminal status (done/error)."""
    rows = conn.execute(
        "SELECT * FROM images WHERE ai_status NOT IN ('cancelled') "
        "ORDER BY filename, id"
    ).fetchall()
    for r in rows:
        data = {}
        if r["ai_judges_json"]:
            try:
                data = json.loads(r["ai_judges_json"])
            except Exception:
                data = {}
        for name in judge_names:
            existing = data.get(name, {}).get("status")
            if existing not in ("done", "error"):
                return name, r
    return None


def set_ai_result(conn: sqlite3.Connection, image_id: int, ai: dict, raw_json: str, ts: float) -> None:
    issues = ai.get("technical_issues") or []
    if not isinstance(issues, list):
        issues = []
    conn.execute(
        """UPDATE images SET
            ai_status='done',
            ai_json=?,
            ai_keep=?,
            ai_in_focus=?,
            ai_eye_focus=?,
            ai_motion=?,
            ai_composition=?,
            ai_lighting=?,
            ai_is_silhouette=?,
            ai_subject=?,
            ai_animal_type=?,
            ai_species=?,
            ai_technical_issues=?,
            ai_notes=?,
            ai_analyzed_at=?
        WHERE id=?""",
        (
            raw_json,
            ai.get("keep"),
            ai.get("in_focus"),
            ai.get("eye_focus"),
            ai.get("motion"),
            ai.get("composition"),
            ai.get("lighting"),
            1 if ai.get("is_silhouette") else 0,
            ai.get("subject"),
            ai.get("animal_type"),
            ai.get("species"),
            json.dumps(issues),
            ai.get("notes"),
            ts,
            image_id,
        ),
    )


def reset_ai_for_reanalysis(conn: sqlite3.Connection, image_ids: list[int]) -> int:
    if not image_ids:
        return 0
    placeholders = ",".join("?" for _ in image_ids)
    cur = conn.execute(
        f"""UPDATE images SET
            ai_status='pending',
            ai_json=NULL,
            ai_keep=NULL,
            ai_in_focus=NULL,
            ai_eye_focus=NULL,
            ai_motion=NULL,
            ai_composition=NULL,
            ai_lighting=NULL,
            ai_is_silhouette=NULL,
            ai_subject=NULL,
            ai_animal_type=NULL,
            ai_species=NULL,
            ai_technical_issues=NULL,
            ai_notes=NULL,
            ai_analyzed_at=NULL,
            ai_judges_json=NULL
        WHERE id IN ({placeholders})""",
        image_ids,
    )
    return cur.rowcount


def select_image_ids_by_status(
    conn: sqlite3.Connection,
    status: str,
    folder: Optional[str] = None,
) -> list[int]:
    sql = "SELECT id FROM images WHERE ai_status=?"
    params: list = [status]
    if folder:
        sql += " AND folder=?"
        params.append(folder)
    return [r["id"] for r in conn.execute(sql, params).fetchall()]


def set_ai_error(conn: sqlite3.Connection, image_id: int, message: str) -> None:
    conn.execute(
        "UPDATE images SET ai_status='error', ai_json=? WHERE id=?",
        (json.dumps({"error": message}), image_id),
    )


def set_feedback(conn: sqlite3.Connection, image_id: int, feedback: dict) -> None:
    feedback = dict(feedback or {})
    feedback["ts"] = time.time()
    conn.execute(
        "UPDATE images SET ai_feedback_json=? WHERE id=?",
        (json.dumps(feedback), image_id),
    )


def clear_feedback(conn: sqlite3.Connection, image_id: int) -> None:
    conn.execute(
        "UPDATE images SET ai_feedback_json=NULL WHERE id=?",
        (image_id,),
    )


def list_feedback_examples(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Return user-corrected images suitable for few-shot prompt injection."""
    rows = conn.execute(
        "SELECT id, filename, ai_subject, ai_animal_type, ai_species, "
        "ai_judges_json, ai_feedback_json, user_rating, user_notes "
        "FROM images WHERE ai_feedback_json IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["ai_feedback"] = json.loads(d.pop("ai_feedback_json") or "{}")
        except Exception:
            d["ai_feedback"] = {}
        try:
            d["ai_judges"] = json.loads(d.pop("ai_judges_json") or "{}")
        except Exception:
            d["ai_judges"] = {}
        out.append(d)
    return out


def set_embedding(conn: sqlite3.Connection, image_id: int, vec_bytes: bytes, model: str) -> None:
    conn.execute(
        "UPDATE images SET embedding=?, embedding_model=? WHERE id=?",
        (vec_bytes, model, image_id),
    )


def set_user_rating(
    conn: sqlite3.Connection,
    image_id: int,
    rating: Optional[int],
    tags: Optional[list],
    notes: Optional[str],
    ts: float,
) -> None:
    conn.execute(
        "UPDATE images SET user_rating=?, user_tags=?, user_notes=?, user_updated_at=? WHERE id=?",
        (rating, json.dumps(tags or []), notes, ts, image_id),
    )


def list_pending_for_ai(conn: sqlite3.Connection, limit: int = 1) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM images WHERE ai_status='pending' ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()


def list_pending_for_embedding(conn: sqlite3.Connection, limit: int = 8) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM images WHERE embedding IS NULL ORDER BY filename, id LIMIT ?",
        (limit,),
    ).fetchall()


def folder_summary(conn: sqlite3.Connection, folder: str) -> dict:
    row = conn.execute(
        """SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN ai_status='done' THEN 1 ELSE 0 END) AS analyzed,
            SUM(CASE WHEN ai_status='pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN ai_status='error' THEN 1 ELSE 0 END) AS errored,
            SUM(CASE WHEN user_rating IS NOT NULL THEN 1 ELSE 0 END) AS rated
        FROM images WHERE folder=?""",
        (folder,),
    ).fetchone()
    return dict(row) if row else {}


def list_folders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT folder, COUNT(*) AS n, MAX(ingested_at) AS last "
        "FROM images GROUP BY folder ORDER BY last DESC"
    ).fetchall()
