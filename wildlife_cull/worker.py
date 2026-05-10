import sys
import threading
import time
import traceback
from pathlib import Path

from . import db, ai


def _log(message: str) -> None:
    print(f"[wc-worker] {message}", file=sys.stderr, flush=True)


class BackgroundWorker:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        # Sticky judge index. We process all pending images for the current
        # judge before advancing, so each Ollama model loads at most once
        # per pass instead of three times per image.
        self._current_judge_idx = 0
        self._warmed_judge: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wc-worker", daemon=True)
        self._thread.start()
        _log("background worker started")

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        if not self._pause.is_set():
            self._pause.set()
            _log("paused")

    def resume(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
            _log("resumed")

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def current_judge(self) -> str:
        return ai.JUDGES[self._current_judge_idx % len(ai.JUDGES)]["name"]

    def cancel_pending(self) -> int:
        with db.connect() as conn:
            cur = conn.execute(
                "UPDATE images SET ai_status='cancelled' WHERE ai_status='pending'"
            )
            n = cur.rowcount
        _log(f"cancelled {n} pending image(s)")
        return n

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.5)
                continue
            did_work = self._tick()
            if not did_work:
                time.sleep(2.0)

    def _ensure_warm(self, judge: dict) -> None:
        if self._warmed_judge == judge["name"]:
            return
        _log(f"loading judge model: {judge['label']} ({judge['model']}). First call may take a minute.")
        err = ai.warmup_judge(judge)
        if err:
            _log(f"warmup for {judge['name']}: {err}")
        else:
            _log(f"judge ready: {judge['label']} ({judge['model']})")
        self._warmed_judge = judge["name"]

    def _next_judge_with_work(self) -> tuple[dict, list] | None:
        """Look for pending work starting at the current judge, falling
        through to others if the current judge has nothing to do."""
        n = len(ai.JUDGES)
        for offset in range(n):
            idx = (self._current_judge_idx + offset) % n
            judge = ai.JUDGES[idx]
            with db.connect() as conn:
                rows = db.list_pending_for_judge(conn, judge["name"], limit=1)
            if rows:
                if idx != self._current_judge_idx:
                    _log(f"switching judge: {judge['label']} ({judge['model']})")
                    self._current_judge_idx = idx
                return judge, rows
        return None

    def _tick(self) -> bool:
        # Embeddings first — they're independent of the judges.
        with db.connect() as conn:
            embed_rows = db.list_pending_for_embedding(conn, limit=1)
        if embed_rows:
            row = embed_rows[0]
            try:
                vec = ai.embed_image(Path(row["preview_path"]))
                with db.connect() as conn:
                    db.set_embedding(conn, row["id"], ai.vec_to_bytes(vec), ai.CLIP_MODEL)
            except Exception as exc:
                msg = f"embed FAILED for {row['filename']}: {exc!r}"
                self.last_error = msg
                _log("=" * 60)
                _log(msg)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                _log("=" * 60)
                with db.connect() as conn:
                    conn.execute(
                        "UPDATE images SET embedding=zeroblob(1), embedding_model='ERROR' WHERE id=?",
                        (row["id"],),
                    )
            return True

        result = self._next_judge_with_work()
        if not result:
            return False
        judge, rows = result
        row = rows[0]
        self._ensure_warm(judge)

        ts = time.time()
        try:
            parsed, raw = ai.analyze_with_judge(judge, Path(row["preview_path"]))
            with db.connect() as conn:
                db.set_judge_result(
                    conn, row["id"], judge["name"],
                    status="done", model=judge["model"],
                    payload=parsed, raw=raw, error=None, ts=ts,
                )
                db.finalize_image_if_complete(conn, row["id"], [j["name"] for j in ai.JUDGES])
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            msg = f"{judge['label']} FAILED for {row['filename']}: {detail}"
            self.last_error = msg
            _log("=" * 60)
            _log(msg)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            _log("=" * 60)
            with db.connect() as conn:
                db.set_judge_result(
                    conn, row["id"], judge["name"],
                    status="error", model=judge["model"],
                    payload=None, raw=None, error=detail, ts=ts,
                )
                db.finalize_image_if_complete(conn, row["id"], [j["name"] for j in ai.JUDGES])
        return True
