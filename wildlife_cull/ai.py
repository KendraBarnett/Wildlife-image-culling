import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from .config import OLLAMA_HOST, VISION_MODEL, CLIP_MODEL

# Bigger inputs make the vision encoder slower without meaningfully
# improving culling judgment. On a 16 GB M-series Mac mini, sending the
# full 1600 px preview can push time-to-first-token past the read timeout.
AI_IMAGE_MAX_SIZE = 1024


_TRIAGE_SCHEMA = """Reply with ONLY a JSON object, no prose, no markdown fences. Use this schema and \
pick values STRICTLY from the listed options for every enum field:

{
  "subject": "<short phrase, e.g. 'great horned owl perched on branch'>",
  "animal_type": <PICK EXACTLY ONE FROM THIS DROPDOWN — no other value is allowed: "Mammal" | "Bird" | "Reptile" | "Amphibian" | "Fish" | "Insect" | "Other" | "Unknown">,
  "species": "<best-guess common name, e.g. 'Mallard', 'Western Lowland Gorilla'. Use 'Unknown' if unsure>",
  "in_focus": <PICK EXACTLY ONE: "yes" | "no">,
  "eye_focus": <PICK EXACTLY ONE: "sharp" | "soft" | "not_visible" | "n/a">,
  "motion": <PICK EXACTLY ONE: "still" | "subtle" | "in_motion" | "blurred">,
  "composition": <PICK EXACTLY ONE: "strong" | "standard" | "weak">,
  "lighting": <PICK EXACTLY ONE: "harsh" | "soft" | "golden" | "low_light" | "backlit" | "overcast" | "mixed">,
  "is_silhouette": true | false,
  "technical_issues": [<zero or more, each ONE OF: "out_of_focus","camera_shake","clipped_subject","blown_highlights","heavy_noise","obstructed">],
  "keep": <PICK EXACTLY ONE: "yes" | "no">,
  "notes": "<2-3 sentences. First sentence: what the image actually shows (subject, light, composition, behavior). Second sentence: technical and compositional state — what works, what doesn't. Third sentence: WHY keep or don't keep. Be SPECIFIC and CONCRETE. NEVER write empty contradictions like 'good, but blurry' — name strengths and flaws separately and tie each to what you can see.>"
}

Schema rules:
- Every enum field above must be filled with one of the listed values. Treat them like a dropdown menu — invented values are forbidden.
- `animal_type` MUST be exactly one of: Mammal, Bird, Reptile, Amphibian, Fish, Insect, Other, Unknown. Capitalized exactly as shown. If you are not sure of the type, return "Unknown" — do not write the species name there or invent a new category.
- "low_light" goes in `lighting`, never in `technical_issues`.
- `technical_issues` is for image flaws, not artistic choices.

DO NOT INVENT FLAWS. The single biggest failure mode is hallucinating problems that aren't visible. Apply these rules:
- `technical_issues` must be CONSERVATIVE. Only flag a problem you can clearly see at this resolution. When in doubt, leave the array EMPTY. An empty array is the correct answer for most images.
- For every entry you put in `technical_issues`, your notes MUST describe the specific area of the frame where it appears. If you cannot describe where, do not list the issue.
- Do not flag "out_of_focus" because the background is blurred — background blur is normal bokeh, not a flaw. Out-of-focus is only a flaw if it affects the SUBJECT.
- Do not flag "blown_highlights" unless you can actually see white pixels with no detail in the subject area.

INTERNAL CONSISTENCY — these must agree:
- `in_focus` is the simple yes/no overall question: is the SUBJECT sharp enough to use? If `in_focus` is "no", `technical_issues` MUST include "out_of_focus", and `eye_focus` MUST be "soft" or "not_visible". If `in_focus` is "yes", `eye_focus` MUST be "sharp" and "out_of_focus" MUST NOT appear in technical_issues.
- If `technical_issues` contains "out_of_focus", then `eye_focus` MUST be "soft" or "not_visible" AND `in_focus` MUST be "no". Saying "sharp" in one place while flagging out-of-focus in another is a forbidden contradiction.
- If `motion` is "blurred" because of camera shake, `technical_issues` MUST include "camera_shake".
- Notes must never contradict the structured fields. If you write "tack sharp" in notes, eye_focus cannot be "soft" and out_of_focus cannot be in issues and in_focus must be "yes".

KEEP DECISION — this is the cull decision. Be honest and somewhat strict; the photographer wants to spend time only on images worth keeping.
- "no" if: in_focus="no", or any out_of_focus / camera_shake flag, or composition="weak" combined with no other redeeming qualities, or the subject is unidentifiable / clipped in a ruinous way.
- "yes" if: technically clean (in_focus="yes", eye_focus="sharp" or n/a) AND has at least one of: clear subject, decent composition, interesting behavior, good light, or something distinctive about the moment.
- When uncertain between yes and no, default to "no" — the photographer would rather review a missed maybe than waste time on a clearly bad image.
- "keep" answers a different question than artistic quality. A technically perfect but boring portrait CAN still be a "keep" (it's usable). A flawed but rare moment CAN still be a "keep" (the moment matters). The Phase 2 Claude scorer will rank quality among keepers later — your job here is just to filter out the clearly unusable."""


_SHIPPED_AS_IS = """CRITICAL CONTEXT: this photographer SHIPS PHOTOS AS-IS. They will NOT do heavy \
post-processing. Assume at most a basic edit: white balance, exposure adjustment of about \
±1 stop, modest crop, light noise reduction. NO highlight recovery beyond what's already \
in the file, NO sky replacement, NO compositing, NO heavy dodging-and-burning. \
If a flaw cannot be removed by a basic edit, score the image AS IF the flaw stays. \
Do not give credit for "could be saved with work." Score what will actually leave the camera bag."""


TRIAGE_PROMPT = f"""You are a wildlife-photo cull assistant. Your job is to look at one image and \
produce a structured triage report for the photographer: what's in it, basic technical state, and \
a yes/no keep decision. You are NOT scoring artistic merit — a separate, more capable scorer (Claude) \
runs later, only on the photographer's keepers. Your job is to filter out the clearly unusable and \
let the keepers through.

{_SHIPPED_AS_IS}

{_TRIAGE_SCHEMA}

Your only job: identify, classify, flag genuine technical problems, and recommend keep / don't keep. \
Be honest about what you can actually see. When uncertain, default to "no" on `keep` — the \
photographer would rather review a missed maybe than waste time on a clearly bad image."""


# Single source of truth for the controlled vocab. The server exposes this
# to the UI so dropdowns show exactly the values the model can return.
AI_VOCAB = {
    "animal_type": ["Mammal", "Bird", "Reptile", "Amphibian", "Fish", "Insect", "Other", "Unknown"],
    "in_focus": ["yes", "no"],
    "keep": ["yes", "no"],
    "eye_focus": ["sharp", "soft", "not_visible", "n/a"],
    "motion": ["still", "subtle", "in_motion", "blurred"],
    "composition": ["strong", "standard", "weak"],
    "lighting": ["harsh", "soft", "golden", "low_light", "backlit", "overcast", "mixed"],
    "technical_issues": [
        "out_of_focus", "camera_shake", "clipped_subject",
        "blown_highlights", "heavy_noise", "obstructed",
    ],
}


# Each judge: a persona-specific prompt run against a specific Ollama model.
# Defaults below are tuned for a 16 GB Apple Silicon Mac mini. The worker
# processes one judge across all pending images before switching, so the
# user pays one model load per judge per batch instead of three loads per
# image.
JUDGES = [
    {
        "name": "triage",
        "label": "Triage",
        "model": "qwen2.5vl:7b",
        "prompt": TRIAGE_PROMPT,
        "weight": 1.0,
        "primary": True,
    },
]


def judge_by_name(name: str) -> Optional[dict]:
    for j in JUDGES:
        if j["name"] == name:
            return j
    return None


def primary_judge() -> dict:
    for j in JUDGES:
        if j.get("primary"):
            return j
    return JUDGES[0]


def _read_b64(path: Path) -> str:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > AI_IMAGE_MAX_SIZE:
            im.thumbnail((AI_IMAGE_MAX_SIZE, AI_IMAGE_MAX_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def format_feedback_block(rows: list[dict], max_examples: int = 5) -> str:
    """Turn the user's recent corrections into a few-shot prompt prefix.

    Each correction shows the AI's prior verdict and the user's override on
    that same image, so the judge can see the pattern of where it tends to
    be wrong from this user's perspective. We cap at max_examples to keep
    the prompt small and prefer corrections that include a free-text note
    (the most informative ones)."""
    if not rows:
        return ""

    rows = sorted(rows, key=lambda r: 0 if (r.get("ai_feedback") or {}).get("note") else 1)
    items: list[str] = []
    for r in rows:
        fb = r.get("ai_feedback") or {}
        if not fb:
            continue
        judges = r.get("ai_judges") or {}
        ai_bits = []
        for name, slot in judges.items():
            if isinstance(slot, dict) and slot.get("status") == "done" and slot.get("result"):
                res = slot["result"]
                ai_bits.append(
                    f"keep={res.get('keep')}, "
                    f"in_focus={res.get('in_focus')}, "
                    f"eye_focus={res.get('eye_focus')}, "
                    f"composition={res.get('composition')}, "
                    f"lighting={res.get('lighting')}, "
                    f"issues={res.get('technical_issues')}"
                )
        ai_text = " | ".join(ai_bits) if ai_bits else "(no AI fields)"

        user_bits = []
        for k in ("keep", "in_focus", "eye_focus", "motion", "composition", "lighting",
                  "animal_type", "species", "subject"):
            if k in fb and fb[k]:
                user_bits.append(f"{k}={fb[k]}")
        if "is_silhouette" in fb:
            user_bits.append(f"is_silhouette={fb['is_silhouette']}")
        if "technical_issues" in fb:
            user_bits.append(f"technical_issues={fb['technical_issues']}")
        if fb.get("marked_wrong"):
            user_bits.append("(user flagged whole AI analysis as wrong)")
        user_text = ", ".join(user_bits) if user_bits else "(no field changes)"

        subj = r.get("ai_subject") or "(subject unknown)"
        note = (fb.get("note") or "").strip()
        line = (
            f"- Subject: {subj}. AI said: {ai_text}. "
            f"User corrected to: {user_text}."
        )
        if note:
            line += f' User said: "{note}"'
        items.append(line)

        if len(items) >= max_examples:
            break

    if not items:
        return ""

    return (
        "PHOTOGRAPHER FEEDBACK ON YOUR PAST WORK — these are corrections this user has "
        "made to prior triage analyses. They are GROUND TRUTH. Use them as calibration anchors: "
        "if you see a similar pattern on the new image, lean toward the user's verdict. Pay "
        "attention to specific recurring complaints in the user's notes (e.g. 'don't hallucinate "
        "out_of_focus when the eye is sharp', 'be stricter on keep — most images aren't worth "
        "scoring', 'a clean portrait of a common bird is still a keep if technically clean').\n\n"
        + "\n".join(items)
        + "\n\nNow triage the new image, applying that calibration:\n\n"
    )


def analyze_with_judge(
    judge: dict,
    preview_path: Path,
    examples_block: str = "",
    idle_timeout: float = 90.0,
    total_timeout: float = 420.0,
) -> tuple[dict, str, dict]:
    if not preview_path.exists():
        raise FileNotFoundError(f"Preview missing: {preview_path}")
    full_prompt = (examples_block + judge["prompt"]) if examples_block else judge["prompt"]
    encode_start = time.time()
    image_b64 = _read_b64(preview_path)
    encode_secs = time.time() - encode_start
    payload = {
        "model": judge["model"],
        "prompt": full_prompt,
        "images": [image_b64],
        "stream": True,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.2},
    }
    timeout = httpx.Timeout(connect=10.0, read=idle_timeout, write=30.0, pool=10.0)
    chunks: list[str] = []
    started = time.time()
    first_token_ts: Optional[float] = None
    last_error: Optional[str] = None
    final_evt: dict = {}
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"Ollama HTTP {r.status_code}: {body}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    if time.time() - started > total_timeout:
                        raise RuntimeError(
                            f"WATCHDOG: ollama exceeded total timeout of {total_timeout:.0f}s "
                            f"(model={judge['model']}). Skipping this image."
                        )
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("error"):
                        last_error = str(evt["error"])
                        break
                    chunk = evt.get("response", "")
                    if chunk:
                        if first_token_ts is None:
                            first_token_ts = time.time()
                        chunks.append(chunk)
                    if evt.get("done"):
                        final_evt = evt
                        break
    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            f"WATCHDOG: ollama stopped sending data for {idle_timeout:.0f}s "
            f"(model={judge['model']}). Likely stuck on this image — skipping."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}") from exc
    if last_error:
        raise RuntimeError(f"Ollama error: {last_error}")
    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError(
            f"Ollama returned empty response for judge '{judge['name']}' "
            f"(model {judge['model']} may have failed to load or run out of memory)"
        )
    parsed = _parse_json_loose(text)
    total_secs = time.time() - encode_start
    ns = 1_000_000_000.0
    timing = {
        "total_secs": round(total_secs, 2),
        "image_encode_secs": round(encode_secs, 2),
        "time_to_first_token_secs": round((first_token_ts - started), 2) if first_token_ts else None,
        "prompt_eval_count": final_evt.get("prompt_eval_count"),
        "prompt_eval_secs": round(final_evt["prompt_eval_duration"] / ns, 2) if final_evt.get("prompt_eval_duration") else None,
        "eval_count": final_evt.get("eval_count"),
        "eval_secs": round(final_evt["eval_duration"] / ns, 2) if final_evt.get("eval_duration") else None,
        "load_secs": round(final_evt["load_duration"] / ns, 2) if final_evt.get("load_duration") else None,
        "prompt_chars": len(full_prompt),
        "examples_chars": len(examples_block),
    }
    return parsed, text, timing


# Back-compat alias for any caller still using the single-judge name.
def analyze_image(preview_path: Path, **kwargs) -> tuple[dict, str, dict]:
    return analyze_with_judge(primary_judge(), preview_path, **kwargs)


def _parse_json_loose(text: str) -> dict:
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
    return {"_unparsed": text}


_clip_model = None


def _get_clip():
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        _clip_model = SentenceTransformer(CLIP_MODEL)
    return _clip_model


def embed_image(preview_path: Path) -> np.ndarray:
    from PIL import Image
    model = _get_clip()
    with Image.open(preview_path) as im:
        im = im.convert("RGB")
        vec = model.encode([im], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype(np.float32)


def vec_to_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def warmup_judge(judge: dict, timeout: float = 600.0) -> Optional[str]:
    """Load this judge's model into memory and pin it via keep_alive."""
    payload = {"model": judge["model"], "keep_alive": "30m"}
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            if r.status_code != 200:
                return f"warmup HTTP {r.status_code}: {r.text[:200]}"
        return None
    except Exception as exc:
        return f"warmup failed: {exc}"


def warmup_vision_model(timeout: float = 600.0) -> Optional[str]:
    return warmup_judge(primary_judge(), timeout=timeout)


def ollama_health() -> Optional[str]:
    """Return a human-readable problem string, or None if everything is OK."""
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = [m.get("name") for m in tags if m.get("name")]
            missing = []
            for j in JUDGES:
                wanted = j["model"]
                base = wanted.split(":")[0]
                if not any(n == wanted or n.startswith(base + ":") or n == base for n in names):
                    missing.append(wanted)
            if missing:
                cmds = "\n".join(f"  ollama pull {m}" for m in missing)
                return f"Ollama is reachable but these models are not pulled:\n{cmds}"
            return None
    except Exception as exc:
        return f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}"
