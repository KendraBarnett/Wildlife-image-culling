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

    def cancel_pending(self) -> int:
        from . import db as _db
        with _db.connect() as conn:
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

    def _tick(self) -> bool:
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

        with db.connect() as conn:
            ai_rows = db.list_pending_for_ai(conn, limit=1)
        if ai_rows:
            row = ai_rows[0]
            try:
                parsed, raw = ai.analyze_image(Path(row["preview_path"]))
                with db.connect() as conn:
                    db.set_ai_result(conn, row["id"], parsed, raw, time.time())
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                msg = f"analyze FAILED for {row['filename']}: {detail}"
                self.last_error = msg
                _log("=" * 60)
                _log(msg)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                _log("=" * 60)
                with db.connect() as conn:
                    db.set_ai_error(conn, row["id"], detail)
            return True

        return False
