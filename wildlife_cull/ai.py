import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from .config import OLLAMA_HOST, VISION_MODEL, CLIP_MODEL, TRIAGE_MODEL, COMPARE_MODEL

# Bigger inputs make the vision encoder slower without meaningfully
# improving culling judgment. On a 16 GB M-series Mac mini, sending the
# full 1600 px preview can push time-to-first-token past the read timeout.
AI_IMAGE_MAX_SIZE = 1024


TRIAGE_PROMPT = """You are a wildlife-photo TRIAGE filter. Your job is to throw out OBVIOUSLY broken
images and let everything else through. A second, more capable model (Claude) does
real quality scoring later — only on what you pass through. You are the coarse
filter, not the judge.

YOUR BIAS: keep more than you cull. The photographer would much rather glance at a
mediocre photo for half a second than lose a good one forever. False culls are
EXPENSIVE. False keeps are CHEAP — Claude catches them downstream.

You are looking at a small thumbnail through a small vision model. You CANNOT
reliably judge: exact eye sharpness, fine feather detail, subtle motion blur,
composition strength, or species ID. Do NOT pretend you can. When something looks
ambiguous, that is a "maybe", not a "no".

============================================================
OUTPUT — JSON only, no prose, no markdown fences.
============================================================

{
  "subject": "<short phrase, e.g. 'inca tern with fish in beak'>",
  "animal_type": "Mammal" | "Bird" | "Reptile" | "Amphibian" | "Fish" | "Insect" | "Other" | "Unknown",
  "species": "<best guess common name, or 'Unknown' — it's fine to be wrong here, Claude re-IDs later>",
  "in_focus": "yes" | "no" | "unsure",
  "eye_focus": "sharp" | "soft" | "not_visible" | "n/a",
  "motion": "still" | "subtle" | "in_motion" | "blurred",
  "composition": "strong" | "standard" | "weak",
  "lighting": "harsh" | "soft" | "golden" | "low_light" | "backlit" | "overcast" | "mixed",
  "is_silhouette": true | false,
  "technical_issues": [ ...zero or more from the list below... ],
  "keep": "yes" | "maybe" | "no",
  "notes": "<2-3 sentences: (1) what's in the frame, (2) what looks fine and what (if anything) looks broken, (3) why you chose this keep value.>"
}

All enum fields must use one of the listed values exactly. No new values, no
capitalisation variants, no nulls.

============================================================
KEEP DECISION — this is the only decision that matters.
============================================================

Three values. Read these carefully.

"yes" — looks like a real photo of a real animal.
   The subject is identifiable. The frame is not obviously broken. You don't have
   to love it. You don't have to be sure the eye is tack-sharp. You don't have to
   call the composition strong. If a human flipping through a contact sheet would
   pause on it for even a second, it's a "yes". Common-animal portraits in
   ordinary light with a visible subject = "yes", every time.

"maybe" — a human should look at this.
   Use this when:
     - You genuinely can't tell if it's sharp or soft.
     - The subject is partly hidden, partly out of frame, or hard to read.
     - It might be an intentional silhouette / artistic shot — not your call.
     - You see SOME issue but you're not sure it's bad enough to throw out.
     - Behaviour shot (feeding, fighting, flying) where execution is borderline.
   "maybe" is the safe answer. When in doubt, pick "maybe", not "no".

"no" — only for images that are OBVIOUSLY broken.
   You must be able to point to a specific gross defect from this list:
     - The frame is mostly black, mostly white, or mostly empty sky/ground with
       no animal in it.
     - The ENTIRE image is a smeared blur (camera dropped, massive shake) — not
       just shallow depth of field, not just a soft eye, not just a blurred wing.
     - The subject is so tiny you can't tell what species or even what type of
       animal it is (a speck in the frame).
     - The animal's head/face is entirely cut off by the frame edge AND nothing
       else interesting is happening.
     - Heavy blown highlights or crushed shadows across MOST of the subject so
       you can't see the animal at all.
     - It's clearly not a wildlife photo (test shot of a wall, lens cap, etc).

If you cannot name which of those applies, it is NOT a "no". Promote to "maybe".

============================================================
THINGS THAT ARE *NOT* CULL REASONS — do not let these push you to "no".
============================================================

- Blurry background. That's shallow depth of field. It's the LOOK of wildlife
  photography. The background SHOULD be blurred.
- Dark subject on a dark background. Black birds, dark mammals in shadow,
  back-of-the-animal shots — these can be deliberate. Send to "maybe" if unsure.
- Eye looks a little soft. You're looking at a thumbnail through a 7B-11B model.
  You cannot reliably tell sharp from slightly-soft. Mark eye_focus honestly
  ("soft" or "sharp" as best you can), but DO NOT cull on this alone.
- Subject in motion / wings spread / mid-action. Motion is content, not a flaw.
- "Composition is weak." Composition is taste. Claude scores that later.
- Backlighting / silhouette-ish look. Could be intentional. → "maybe".
- Animal facing away, eyes closed, yawning, etc. Could be the shot. → "maybe".
- Cluttered background. Not a cull reason on its own.

The previous version of this prompt was over-culling images where the subject was
dark or the background was strongly blurred. Do not repeat that mistake. A dark
bird with a sharp-looking head in front of a creamy out-of-focus background is a
TEXTBOOK keeper, not a cull.

============================================================
technical_issues — observations, not verdicts.
============================================================

Pick zero or more, ONLY when clearly visible. Empty list is fine and common.
Allowed values:
  "out_of_focus"        — the whole subject (not just background) is mushy.
  "soft_edges"          — subject edges are noticeably fuzzy.
  "camera_shake"        — directional smear across the whole frame.
  "clipped_subject"     — important part (head, full body when needed) cut off.
  "blown_highlights"    — large pure-white areas with no detail.
  "underexposed"        — subject is so dark you cannot read the animal.
  "harsh_backlight"     — strong light from behind crushing the subject.
  "heavy_noise"         — coarse grain across the image.
  "obstructed"          — branch / fence / glass blocks key part of subject.
  "cluttered_background"— busy background competing with subject.
  "subject_too_small"   — animal occupies a tiny fraction of the frame.

Rules:
  - Listing an issue does NOT require a cull. Many keepers have one or two.
  - Only "out_of_focus", "camera_shake", "blown_highlights" across the subject,
    "underexposed" across the subject, or "subject_too_small" can, on their own,
    justify a "no" — and only when severe and obvious.
  - "low_light" is a LIGHTING value, never a technical_issue.
  - If you flag "out_of_focus", the subject itself must look mushy. Background
    blur is not "out_of_focus".

============================================================
eye_focus — be honest, don't overclaim.
============================================================

- "sharp"       — eye is clearly visible and looks crisp.
- "soft"        — eye is visible but does not look crisp, OR you can't tell.
- "not_visible" — eye is hidden, turned away, behind something, or in shadow.
- "n/a"         — no animal, or no face to have an eye on (back-of-body shot).

Default to "soft" over "sharp" when uncertain. "soft" alone is NOT a cull.

============================================================
INTERNAL CONSISTENCY (quick check before you return)
============================================================

- If in_focus = "no" → motion is "blurred" OR technical_issues includes
  "out_of_focus" or "camera_shake".
- If eye_focus = "not_visible" → don't ALSO claim it's sharp or soft elsewhere.
- If is_silhouette = true → keep = "maybe" (let the human decide if it was on
  purpose), unless it's literally a black blob with no readable shape (then "no").
- If keep = "no" → notes must name the specific gross defect.
- If keep = "yes" → fine to still list 1-2 technical_issues. Keepers can have
  flaws.

============================================================
CALIBRATION
============================================================

Mostly-black frame of a dark bird, only beak edge visible
   → keep = "maybe"   (could be intentional low-key — let the human see it)

Inca tern in profile, dark body, sharp-looking head, beak open with fish,
heavily blurred soft background
   → keep = "yes"     (this is a textbook wildlife shot — DO NOT cull)

Snow leopard mid-yawn, sharp head, mottled background
   → keep = "yes"

Red panda peeking from foliage, face visible
   → keep = "yes"

Peacock close-up where most of the frame is feathers and you can barely see
the bird's head
   → keep = "maybe"

Bird at the edge of the frame with head cropped off, nothing else happening
   → keep = "no" (clipped_subject, no behaviour to justify it)

Entire image is a directional smear, can't tell what the animal is
   → keep = "no" (camera_shake / out_of_focus)

Tiny speck of a bird in a huge empty sky
   → keep = "no" (subject_too_small)

Backlit perched tern, recognisable shape, no eye visible
   → keep = "maybe" (could be the intended silhouette shot)

REMEMBER: you are a filter, not a judge. Pass the borderline ones through. The
human and Claude will sort them out."""


# Legacy variable kept for back-compat with the prompt-building code that
# expected a separate schema block. The new prompt is self-contained.
_TRIAGE_SCHEMA = ""
_SHIPPED_AS_IS = ""
# Single source of truth for the controlled vocab. The server exposes this
# to the UI so dropdowns show exactly the values the model can return.
AI_VOCAB = {
    "animal_type": ["Mammal", "Bird", "Reptile", "Amphibian", "Fish", "Insect", "Other", "Unknown"],
    "in_focus": ["yes", "no", "unsure"],
    "keep": ["yes", "maybe", "no"],
    "eye_focus": ["sharp", "soft", "not_visible", "n/a"],
    "motion": ["still", "subtle", "in_motion", "blurred"],
    "composition": ["strong", "standard", "weak"],
    "lighting": ["harsh", "soft", "golden", "low_light", "backlit", "overcast", "mixed"],
    "technical_issues": [
        "out_of_focus", "soft_edges", "camera_shake", "clipped_subject",
        "blown_highlights", "underexposed", "harsh_backlight",
        "heavy_noise", "obstructed", "cluttered_background",
        "subject_too_small",
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
        "model": TRIAGE_MODEL,
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


COMPARE_PROMPT = """You are a senior wildlife photo editor. The photographer has \
selected TWO images for you to compare directly. The first image attached is "Image 1", \
the second is "Image 2". Your job is to choose the better photo for keeping and explain \
why in one or two sentences.

Compare on:
- Sharpness (especially the eye — that matters most)
- Composition and framing
- Behavior or moment captured
- Light quality
- Technical execution (exposure, noise, motion handling)

If one image clearly wins, say so. If they're nearly identical, pick the one with the \
sharper eye or stronger composition and acknowledge it was close.

Reply with ONLY this JSON object, no prose, no markdown fences:

{
  "winner": <1 or 2>,
  "reasoning": "<one or two sentences explaining the choice. Be specific about what you can see — name the eye state, the light, the moment, the flaw. Avoid vague phrases like 'better composition'.>",
  "margin": <"clear" | "close">
}
"""


def compare_two_images(
    preview1_path: Path,
    preview2_path: Path,
    judge: Optional[dict] = None,
    idle_timeout: float = 90.0,
    total_timeout: float = 420.0,
) -> tuple[dict, str, dict]:
    """Send two images to the local vision model and ask which is the
    better photograph. Returns (parsed, raw_text, timing) — same shape as
    analyze_with_judge so existing log/watchdog plumbing applies."""
    if not preview1_path.exists():
        raise FileNotFoundError(f"Preview missing: {preview1_path}")
    if not preview2_path.exists():
        raise FileNotFoundError(f"Preview missing: {preview2_path}")

    # The compare task is simpler than triage (pick the better of two);
    # use the faster COMPARE_MODEL by default to keep latency low.
    # An explicit judge arg still overrides it.
    model_to_use = (judge["model"] if judge else COMPARE_MODEL)
    encode_start = time.time()
    img1_b64 = _read_b64(preview1_path)
    img2_b64 = _read_b64(preview2_path)
    encode_secs = time.time() - encode_start

    payload = {
        "model": model_to_use,
        "prompt": COMPARE_PROMPT,
        # Order matters here — the prompt refers to "first" and "second"
        # by index.
        "images": [img1_b64, img2_b64],
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
                            f"on image comparison."
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
            f"on image comparison."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}") from exc

    if last_error:
        raise RuntimeError(f"Ollama error: {last_error}")
    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Ollama returned empty response for image comparison")
    parsed = _parse_json_loose(text)

    if parsed.get("winner") not in (1, 2):
        raise RuntimeError(f"Comparison response missing valid winner: {text[:200]}")

    total_secs = time.time() - encode_start
    ns = 1_000_000_000.0
    timing = {
        "total_secs": round(total_secs, 2),
        "image_encode_secs": round(encode_secs, 2),
        "time_to_first_token_secs": round((first_token_ts - started), 2) if first_token_ts else None,
        "eval_secs": round(final_evt["eval_duration"] / ns, 2) if final_evt.get("eval_duration") else None,
    }
    return parsed, text, timing


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
