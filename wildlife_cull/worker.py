import threading
import time
import traceback
from pathlib import Path

from . import db, ai


class BackgroundWorker:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wc-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
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
                self.last_error = f"embed {row['filename']}: {exc}"
                traceback.print_exc()
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
                self.last_error = f"analyze {row['filename']}: {exc}"
                traceback.print_exc()
                with db.connect() as conn:
                    db.set_ai_error(conn, row["id"], str(exc))
            return True

        return False
