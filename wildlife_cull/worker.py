import sys
import threading
import time
import traceback
from pathlib import Path

from . import db, ai
from .config import DATA_DIR


def _log(message: str) -> None:
    print(f"[wc-worker] {message}", file=sys.stderr, flush=True)


# A tiny sentinel file that survives restarts. If it exists when the
# worker starts, we boot in paused state. Pause writes it, resume
# removes it. No schema migration, no DB row needed.
PAUSE_FLAG = Path(DATA_DIR) / ".worker_paused"


class BackgroundWorker:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        # Per-image mode: finish one image fully (every judge in turn)
        # before moving to the next.
        self._warmed_model: str | None = None
        self.last_timing: dict | None = None
        self.in_flight: dict | None = None

        # Restore paused state from the sentinel file on construction so
        # restarting the app respects the last pause you set.
        if PAUSE_FLAG.exists():
            self._pause.set()
            _log("starting in paused state (.worker_paused flag found)")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wc-worker", daemon=True)
        self._thread.start()
        if self._pause.is_set():
            _log("background worker started (paused)")
        else:
            _log("background worker started")

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        if not self._pause.is_set():
            self._pause.set()
            try:
                PAUSE_FLAG.touch()
            except OSError as exc:
                _log(f"could not write pause sentinel: {exc}")
            _log("paused (persisted)")

    def resume(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
            try:
                PAUSE_FLAG.unlink(missing_ok=True)
            except OSError as exc:
                _log(f"could not remove pause sentinel: {exc}")
            _log("resumed")

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def current_judge(self) -> str:
        if self.in_flight:
            return self.in_flight.get("judge", "")
        return ""

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
        """Warm the judge's model. Tracks the warmed model (not judge) so
        switching between two judges that share a model is a no-op."""
        if self._warmed_model == judge["model"]:
            return
        _log(f"loading model: {judge['model']} (for {judge['label']}). First call may take a minute.")
        err = ai.warmup_judge(judge)
        if err:
            _log(f"warmup for {judge['model']}: {err}")
        else:
            _log(f"model ready: {judge['model']}")
        self._warmed_model = judge["model"]

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

        judge_names = [j["name"] for j in ai.JUDGES]
        with db.connect() as conn:
            result = db.next_pending_image_and_judge(conn, judge_names)
        if not result:
            return False
        judge_name, row = result
        judge = ai.judge_by_name(judge_name)
        if judge is None:
            return False
        self._ensure_warm(judge)

        ts = time.time()
        with db.connect() as conn:
            feedback_rows = db.list_feedback_examples(conn, limit=10)
        examples_block = ai.format_feedback_block(feedback_rows, max_examples=5)
        self.in_flight = {
            "filename": row["filename"],
            "judge": judge["name"],
            "started_at": ts,
        }
        try:
            parsed, raw, timing = ai.analyze_with_judge(
                judge,
                Path(row["preview_path"]),
                examples_block=examples_block,
            )
            self.last_timing = {
                "filename": row["filename"],
                "judge": judge["name"],
                **timing,
            }
            _log(
                f"{judge['name']} done in {timing['total_secs']}s for {row['filename']} "
                f"(encode={timing['image_encode_secs']}s, "
                f"prompt_eval={timing.get('prompt_eval_secs')}s/{timing.get('prompt_eval_count')}tok, "
                f"gen={timing.get('eval_secs')}s/{timing.get('eval_count')}tok, "
                f"prompt_chars={timing['prompt_chars']} feedback_chars={timing['examples_chars']})"
            )
            with db.connect() as conn:
                db.set_judge_result(
                    conn, row["id"], judge["name"],
                    status="done", model=judge["model"],
                    payload=parsed, raw=raw, error=None, ts=ts,
                )
                finalized = db.finalize_image_if_complete(
                    conn, row["id"], [j["name"] for j in ai.JUDGES]
                )
            # Outside the DB write transaction — refresh the sidecar so
            # Lightroom can see the fresh AI tags.
            if finalized:
                from . import xmp as _xmp
                try:
                    with db.connect() as conn:
                        _xmp.refresh_sidecar(conn, row["id"])
                except Exception as exc:
                    _log(f"sidecar refresh failed for {row['filename']}: {exc}")
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
        finally:
            self.in_flight = None
        return True
