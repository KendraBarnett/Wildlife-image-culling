"""Phase 2: on-demand scoring of keepers via the Claude API.

Phase 1 (the local triage judge) decides which images are worth scoring.
This module sends those keepers to Claude Sonnet 4.6 for two real scores —
technical (1-10) and aesthetic (1-10) — plus a one-sentence rationale.

Cost control: every call's tokens and dollars are written to the
claude_usage table. The budget cap (CLAUDE_BUDGET_USD) is enforced before
each call: at >=80% spent the UI shows a warning, at >=100% the call is
refused. The user lifts the cap by raising CLAUDE_BUDGET_USD in run.sh.

Cost optimization: the scoring rubric is identical on every call (only the
image varies), so it's marked with cache_control. After the first call
the rubric is served from cache at ~10% of the input price. Per-image cost
on Sonnet 4.6 with caching is roughly $0.005-$0.01."""

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_BUDGET_USD
from . import db


# Per-million-token pricing for the supported Sonnet model. Source: the
# pricing table in the Claude API skill (cached 2026-04-29). Update if a
# newer Sonnet variant is targeted.
MODEL_PRICING_USD_PER_MTOK = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,   # 1.25x input
        "cache_read": 0.30,    # 0.10x input
    },
    "claude-opus-4-7": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
}


# The scoring prompt — kept stable across calls so prompt caching kicks in.
# Two scores on separate axes: technical = how well executed; aesthetic =
# how distinctive / artistic / worth showing.
SCORING_RUBRIC = """You are a senior wildlife-photography editor. The photographer has already \
triaged this image as a "keeper" using a smaller model. Your job is to give it two real scores \
and a one-sentence rationale, so the photographer can rank their keepers.

Return ONLY a JSON object, no prose, no markdown fences:

{
  "technical_score": <number 1.0-10.0, ONE decimal>,
  "aesthetic_score": <number 1.0-10.0, ONE decimal>,
  "reasoning": "<one tight sentence: what works, what doesn't, why these scores>"
}

TECHNICAL SCORE (1-10) — how well-executed is this as a photograph?
- Eye sharpness on the actual eye (not just the face)
- Subject focus across the relevant body
- Exposure: highlights not blown on the subject, shadows have detail
- Noise control at the resolution shown
- Motion: intentional or unintentional, does it serve the image
- Framing: subject not amputated awkwardly, no distracting elements at the edges

Calibration:
- 3 = significant flaws (soft eye, blown highlights on subject, heavy noise)
- 5 = competent, no major flaws but nothing notable
- 7 = clean and well-executed throughout
- 9 = technically excellent — the kind of frame other working wildlife photographers would compliment
- 10 = nothing to improve technically

AESTHETIC SCORE (1-10) — how distinctive, artistic, or compelling is the image, separate from technical execution?
- Light: is it doing something interesting (golden, dramatic, atmospheric, mood)?
- Behavior: is something genuinely happening (interaction, hunting, courtship, expression)?
- Moment: is this an unusual instant for this species/setting?
- Composition: deliberate framing, leading lines, intentional negative space
- Story: does the image suggest something beyond the literal subject?
- Distinctiveness: would this stand out next to other photos of the same subject?

Calibration:
- 3 = generic. A clean record shot of a common subject in flat light.
- 5 = competent, has one weak distinguishing thing going for it
- 7 = strong. A real portfolio candidate — has light OR behavior OR moment that makes you pause
- 9 = exceptional. Combines multiple of light/behavior/moment/composition. Rare in any shoot.
- 10 = once a year. Almost never give this.

The two scores measure different things and SHOULD often differ. A clean, sharp portrait of a perched cardinal in flat light might be technical=8, aesthetic=4. A slightly soft frame of a fox carrying prey in golden hour might be technical=5, aesthetic=8.

Be HONEST and DISCRIMINATING. The default is filler — you must EARN points up. Most images will land in the 4-6 range on each axis. Reserve 8+ for images that genuinely deserve it."""


def _read_b64(path: Path, max_size: int = 1568) -> tuple[str, str]:
    """Resize + JPEG-encode + base64 the preview. Sonnet vision processes
    images at up to 1568px on the long edge before downscaling for
    inference, so larger inputs are wasted bandwidth."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_size:
            im.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _calc_cost(model: str, usage: dict) -> float:
    """usage = {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}"""
    p = MODEL_PRICING_USD_PER_MTOK.get(model) or MODEL_PRICING_USD_PER_MTOK["claude-sonnet-4-6"]
    cost = (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("output_tokens", 0) * p["output"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
    ) / 1_000_000
    return round(cost, 6)


def usage_summary() -> dict:
    """Total spend, calls, image count, budget cap, and cap state."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS calls, "
            "COALESCE(SUM(cost_usd), 0) AS spent, "
            "COUNT(DISTINCT image_id) AS images "
            "FROM claude_usage WHERE success=1"
        ).fetchone()
    spent = float(row["spent"] or 0)
    cap = CLAUDE_BUDGET_USD
    pct = (spent / cap) if cap > 0 else 0
    state = "ok"
    if pct >= 1.0:
        state = "blocked"
    elif pct >= 0.80:
        state = "warning"
    return {
        "calls": int(row["calls"] or 0),
        "images": int(row["images"] or 0),
        "spent_usd": round(spent, 4),
        "budget_usd": round(cap, 2),
        "remaining_usd": round(max(cap - spent, 0), 4),
        "pct": round(pct * 100, 1),
        "state": state,
        "model": CLAUDE_MODEL,
        "api_key_set": bool(ANTHROPIC_API_KEY),
    }


class BudgetExceeded(Exception):
    pass


class NotConfigured(Exception):
    pass


def _record_usage(
    image_id: int,
    model: str,
    usage_obj,
    cost_usd: float,
    success: bool,
    error: Optional[str],
) -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO claude_usage "
            "(image_id, ts, model, input_tokens, cache_creation_tokens, "
            "cache_read_tokens, output_tokens, cost_usd, success, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                time.time(),
                model,
                getattr(usage_obj, "input_tokens", 0) or 0,
                getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
                getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
                getattr(usage_obj, "output_tokens", 0) or 0,
                cost_usd,
                1 if success else 0,
                error,
            ),
        )


def score_image(image_id: int, preview_path: Path) -> dict:
    """Score one image with Claude. Enforces budget BEFORE the call.

    Returns: {
        "technical_score": float,
        "aesthetic_score": float,
        "reasoning": str,
        "input_tokens": int, "output_tokens": int, "cache_read_tokens": int,
        "cost_usd": float,
    }

    Raises NotConfigured if the API key isn't set, BudgetExceeded if the
    next call would push past CLAUDE_BUDGET_USD."""
    if not ANTHROPIC_API_KEY:
        raise NotConfigured(
            "ANTHROPIC_API_KEY is not set. Add it to scripts/run.sh "
            "(export ANTHROPIC_API_KEY=sk-ant-...) and restart."
        )

    summary = usage_summary()
    if summary["state"] == "blocked":
        raise BudgetExceeded(
            f"Budget cap reached: ${summary['spent_usd']:.4f} of "
            f"${summary['budget_usd']:.2f} spent. Raise CLAUDE_BUDGET_USD "
            "in scripts/run.sh and restart to continue."
        )

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    image_b64, media_type = _read_b64(preview_path)
    started = time.time()
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=[
                {
                    "type": "text",
                    "text": SCORING_RUBRIC,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Score this wildlife photograph using the rubric above.",
                        },
                    ],
                }
            ],
        )
    except anthropic.APIError as exc:
        _record_usage(image_id, CLAUDE_MODEL, type("U", (), {})(), 0.0, False, str(exc))
        raise

    usage = response.usage
    usage_dict = {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    cost = _calc_cost(CLAUDE_MODEL, usage_dict)

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    parsed = _parse_json(text)

    if "technical_score" not in parsed or "aesthetic_score" not in parsed:
        _record_usage(image_id, CLAUDE_MODEL, usage, cost, False, f"unparsable: {text[:200]}")
        raise RuntimeError(f"Claude response missing scores: {text[:200]}")

    _record_usage(image_id, CLAUDE_MODEL, usage, cost, True, None)

    elapsed = round(time.time() - started, 2)
    return {
        "technical_score": float(parsed["technical_score"]),
        "aesthetic_score": float(parsed["aesthetic_score"]),
        "reasoning": str(parsed.get("reasoning") or ""),
        "input_tokens": usage_dict["input_tokens"],
        "output_tokens": usage_dict["output_tokens"],
        "cache_read_tokens": usage_dict["cache_read_input_tokens"],
        "cost_usd": cost,
        "elapsed_secs": elapsed,
    }


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


class BulkScorer:
    """Background scorer that walks all keep=yes, not-yet-scored images and
    scores them one at a time, respecting the budget cap. Single-flight:
    only one bulk job runs at a time. The UI polls status to render
    progress. Stops gracefully when budget hits 100%, when cancelled, or
    when the queue empties."""

    def __init__(self) -> None:
        import threading
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.state = "idle"        # idle | running | done | cancelled | budget_blocked | error
        self.total = 0
        self.done = 0
        self.failed = 0
        self.spent_usd_at_start = 0.0
        self.last_filename = ""
        self.last_error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def status(self) -> dict:
        return {
            "state": self.state,
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "last_filename": self.last_filename,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "spent_this_run": round(usage_summary()["spent_usd"] - self.spent_usd_at_start, 4),
        }

    def cancel(self) -> None:
        self._cancel.set()

    def start(self) -> dict:
        import threading
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"already_running": True, "status": self.status()}
            if not ANTHROPIC_API_KEY:
                raise NotConfigured(
                    "ANTHROPIC_API_KEY is not set. Add it to scripts/run.sh "
                    "(export ANTHROPIC_API_KEY=sk-ant-...) and restart."
                )
            self._cancel.clear()
            self.state = "running"
            self.done = 0
            self.failed = 0
            self.last_filename = ""
            self.last_error = None
            self.started_at = time.time()
            self.finished_at = None
            self.spent_usd_at_start = usage_summary()["spent_usd"]

            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT id, filename, preview_path FROM images "
                    "WHERE ai_keep='yes' "
                    "AND claude_scored_at IS NULL "
                    "AND preview_path IS NOT NULL "
                    "AND ai_status NOT IN ('cancelled') "
                    "ORDER BY filename"
                ).fetchall()
            queue = [(r["id"], r["filename"], r["preview_path"]) for r in rows]
            self.total = len(queue)
            if not queue:
                self.state = "done"
                self.finished_at = time.time()
                return {"started": False, "status": self.status()}

            self._thread = threading.Thread(
                target=self._run, args=(queue,), name="claude-bulk", daemon=True
            )
            self._thread.start()
            return {"started": True, "status": self.status()}

    def _run(self, queue: list) -> None:
        for image_id, filename, preview_path in queue:
            if self._cancel.is_set():
                self.state = "cancelled"
                self.finished_at = time.time()
                return
            self.last_filename = filename
            try:
                result = score_image(image_id, Path(preview_path))
                persist_score(image_id, result)
                self.done += 1
                self.last_error = None
            except BudgetExceeded as exc:
                self.state = "budget_blocked"
                self.last_error = str(exc)
                self.finished_at = time.time()
                return
            except Exception as exc:
                self.failed += 1
                self.last_error = f"{filename}: {type(exc).__name__}: {exc}"
                # keep going — one bad image shouldn't kill the whole run
        self.state = "done"
        self.finished_at = time.time()


# Module-level singleton; the server reuses it across requests.
bulk_scorer = BulkScorer()


def persist_score(image_id: int, result: dict) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE images SET "
            "claude_technical_score=?, claude_aesthetic_score=?, "
            "claude_reasoning=?, claude_scored_at=?, "
            "claude_input_tokens=?, claude_output_tokens=?, "
            "claude_cache_read_tokens=?, claude_cost_usd=? "
            "WHERE id=?",
            (
                result["technical_score"],
                result["aesthetic_score"],
                result["reasoning"],
                time.time(),
                result["input_tokens"],
                result["output_tokens"],
                result["cache_read_tokens"],
                result["cost_usd"],
                image_id,
            ),
        )
