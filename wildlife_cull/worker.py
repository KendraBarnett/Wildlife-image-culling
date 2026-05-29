import sys
import threading
import time
import traceback
from pathlib import Path

from . import db, ai
from .config import DATA_DIR


def _log(message: str) -> None:
    print(f"[wc-worker] {message}", file=sys.stderr, flush=True)


# Sharpness thresholds — Laplacian variance scale (same as focus.py).
# IMPORTANT: wildlife photos with shallow depth-of-field have low global
# variance because the bokeh background contributes near-zero gradient
# energy. A tack-sharp portrait of a bird with a creamy out-of-focus
# background can score in the 50-120 range. Above the OOF floor, there
# is sharp content somewhere in the frame.
_FOCUS_OOF_HARD_FLOOR = 18.0   # below this: deterministically OOF
_FOCUS_SHARP_FLOOR = 40.0      # above this: sharp content exists, LLM's
                               # claims of OOF/not-visible/soft are wrong


def _enforce_sharpness_from_focus_score(parsed: dict, focus_score) -> dict:
    """Bidirectional sharpness override — math beats LLM hallucinations
    in both directions.

    The LLM lies in BOTH directions: it confidently says 'sharp eye' on
    blurry images AND 'not in focus' on perfectly sharp images. The
    Laplacian variance is a deterministic measurement that cannot
    hallucinate. Use it to correct the LLM wherever it disagrees with
    the math.

    Three bands:
      focus_score < 18:   hard cull — image lacks gradient energy
                          everywhere. Override LLM to OOF / keep=no.
      focus_score 18-40:  ambiguous band — could be very soft or very
                          shallow DOF. Trust the LLM here.
      focus_score >= 40:  sharp content exists in the frame. If LLM
                          said in_focus=no / eye_focus=soft/not_visible
                          / keep=no with OOF in issues, override to
                          sharp/visible/keep=yes. The LLM was wrong.

    The bidirectional override is critical because the LLM is just as
    likely to say 'OOF' on a sharp image (the failure mode the user just
    showed) as it is to say 'sharp' on a blurry one (the failure mode
    we fixed earlier)."""
    if not isinstance(parsed, dict):
        return parsed
    if focus_score is None:
        return parsed
    try:
        fs = float(focus_score)
    except (TypeError, ValueError):
        return parsed

    issues = parsed.get("technical_issues") or []
    if not isinstance(issues, list):
        issues = []
    notes = parsed.get("notes") or ""

    if fs < _FOCUS_OOF_HARD_FLOOR:
        # Image really is out of focus everywhere — no gradient energy.
        if "out_of_focus" not in issues:
            issues.append("out_of_focus")
        parsed["in_focus"] = "no"
        parsed["eye_focus"] = "not_visible"
        parsed["technical_issues"] = issues
        parsed["keep"] = "no"
        override_note = (
            f" [Auto: focus_score={fs:.1f} < {_FOCUS_OOF_HARD_FLOOR:.0f} — "
            f"image-wide variance too low for bokeh to explain, "
            f"deterministically out of focus.]"
        )
        if override_note not in notes:
            parsed["notes"] = (notes + override_note).strip()
        return parsed

    if fs >= _FOCUS_SHARP_FLOOR:
        # Sharp content exists in the frame. The LLM saying 'not in focus'
        # / 'eye not visible' / 'soft' with this much gradient energy is
        # almost always wrong (it's hallucinating softness on a thumbnail
        # of a real keeper). Override the sharpness fields and, if the
        # only reason the LLM culled was sharpness, override the keep too.
        sharpness_issue_set = {"out_of_focus", "soft_edges"}
        issues_filtered = [i for i in issues if i not in sharpness_issue_set]

        overrode_any = False
        if parsed.get("in_focus") == "no":
            parsed["in_focus"] = "yes"
            overrode_any = True
        if parsed.get("eye_focus") in ("soft", "not_visible"):
            parsed["eye_focus"] = "sharp"
            overrode_any = True
        if len(issues_filtered) != len(issues):
            parsed["technical_issues"] = issues_filtered
            issues = issues_filtered
            overrode_any = True

        # If the LLM culled and the cull was driven by the sharpness
        # claims we just overrode, promote to keep="yes". The user
        # explicitly said false culls are more expensive than false
        # keeps. Other cull reasons (composition=weak, clipped_subject,
        # camera_shake, etc.) would still hold; we only flip when the
        # remaining issues list is sharpness-only.
        non_sharpness_issues = [
            i for i in issues
            if i not in sharpness_issue_set
        ]
        if (parsed.get("keep") == "no" and overrode_any
                and not non_sharpness_issues
                and parsed.get("composition") != "weak"):
            parsed["keep"] = "yes"
            overrode_any = True

        if overrode_any:
            override_note = (
                f" [Auto: focus_score={fs:.1f} ≥ {_FOCUS_SHARP_FLOOR:.0f} — "
                f"image has sharp content; LLM softness / not-visible "
                f"claims overridden by gradient math.]"
            )
            if override_note not in notes:
                parsed["notes"] = (notes + override_note).strip()
        return parsed

    # 18-40: ambiguous band. Trust the LLM (could be very soft, could
    # be very shallow DOF — math doesn't distinguish reliably here).
    return parsed


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
        # The folder the UI is currently focused on. Worker prefers
        # pending work in this folder before falling back to others.
        self.focused_folder: str | None = None

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
            result = db.next_pending_image_and_judge(
                conn, judge_names, focused_folder=self.focused_folder,
            )
        if not result:
            return False
        judge_name, row = result
        judge = ai.judge_by_name(judge_name)
        if judge is None:
            return False

        # Reachability check: the preview should exist locally, and the
        # original file should still be on disk. If either is missing
        # (drive unplugged, file moved), mark the image as unreachable
        # and the folder as offline, then continue with the next image.
        preview_ok = bool(row["preview_path"]) and Path(row["preview_path"]).exists()
        original_ok = bool(row["path"]) and Path(row["path"]).exists()
        if not preview_ok or not original_ok:
            _log(
                f"unreachable: {row['filename']} "
                f"(preview={preview_ok}, original={original_ok}) "
                f"— marking folder {row['folder']} offline"
            )
            with db.connect() as conn:
                conn.execute(
                    "UPDATE images SET ai_status='unreachable' WHERE id=?",
                    (row["id"],),
                )
                db.update_folder_online(conn, row["folder"], online=False)
            return True

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
            # Math beats hallucination on sharpness. The LLM keeps
            # saying eye_focus="sharp" on obviously soft images; the
            # Laplacian variance we computed at ingest is a deterministic
            # ground-truth signal. Override the LLM with it.
            parsed = _enforce_sharpness_from_focus_score(parsed, row["focus_score"])
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
