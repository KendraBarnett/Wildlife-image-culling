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

INTERNAL CONSISTENCY — these must agree. CHECK ALL OF THESE BEFORE RETURNING:
- `in_focus` is the simple yes/no overall question: is the SUBJECT sharp enough to use? If `in_focus` is "no", `technical_issues` MUST include "out_of_focus", and `eye_focus` MUST be "soft" or "not_visible".

**THE EYE_FOCUS RULE — STRICT. THIS IS THE #1 HALLUCINATION FAILURE MODE.**
`eye_focus = "sharp"` REQUIRES POSITIVE VISUAL EVIDENCE OF SHARPNESS in the actual eye area:
- A visible catchlight (highlight reflection in the iris/pupil) showing crisp edges, OR
- Clear visible iris/pupil structure that's in focus.
If you can only see WHERE the eye is (a dark shape, an outline, a guess based on the bird's face) but cannot see actual sharpness DETAIL in the eye itself, the answer is "soft" or "not_visible" — never "sharp". Sharp on adjacent features (beak, fur) does NOT transfer to the eye. Each is graded independently on what is actually visible at THIS resolution.
- **eye_focus = "sharp" is forbidden when the eye is in shadow, backlit-into-darkness, or shows no catchlight/iris detail.** Even if the face is technically visible. "I can see where the eye is" ≠ "the eye is sharp".

- **SILHOUETTE / DEEP SHADOW RULE: If `is_silhouette` is true, OR if the subject's face is in deep shadow with no visible eye / facial detail, `eye_focus` MUST be "not_visible". Never "sharp".**

- **HEAVY-SHADOW SUBJECT RULE (the silhouette-adjacent case): Even when `is_silhouette` is false, if the subject's body is mostly crushed-shadow with little visible feather/fur detail, treat it the same way for `keep`. A backlit bird where the only well-lit area is the beak — body in shadow, eye barely visible, edges defined but interior dark — is NOT a keeper just because the composition is strong. See the heavy-backlight example in CALIBRATION below.**

- If `technical_issues` contains "out_of_focus", then `eye_focus` MUST be "soft" or "not_visible" AND `in_focus` MUST be "no".
- If `motion` is "blurred" because of camera shake, `technical_issues` MUST include "camera_shake".
- Notes must never contradict the structured fields. Writing "the bird is in focus with sharp details" while the eye is in shadow is a forbidden hallucination. Be specific about WHAT is sharp — if it's the beak but not the eye, say so.

KEEP DECISION — this is the cull decision. **The default is "no". You must EARN a yes.** Most images on a wildlife shoot are not keepers — that's normal, and the photographer wants the cull to reflect that. Better to send a maybe to cull than to flood the keeper pile with mediocre frames.

A keeper has BOTH:
  (A) Technically usable: in_focus="yes", eye_focus="sharp" (NOT "not_visible" — see silhouette rule below), no major technical_issues. AND
  (B) Something that earns its place: a clearly visible subject doing something readable (calling, eating, flying, looking at camera with engagement), good light, clear strong composition, or a distinctive moment.

Default to "no" when:
- in_focus="no" → no
- Any "out_of_focus", "camera_shake", or "clipped_subject" issue → no
- composition="weak" AND nothing else carrying it → no
- Subject unidentifiable / face hidden / heavily obstructed → no
- **eye_focus="not_visible" on a still wildlife subject** → no, unless the moment is genuinely exceptional (dramatic action, predation, courtship). A bird with its head turned away or backlit-into-silhouette with no behavior happening is NOT a keeper.
- **eye_focus="soft" with no exceptional moment** → no. A soft eye on a static wildlife portrait is a fail; the keeper would have a sharper neighbor in the same burst.
- **Heavy-backlight / crushed-shadow subject** → no. If the subject's body is mostly in shadow (you cannot read feather or fur texture on most of the body), this is the silhouette-adjacent case. The image only works AS a silhouette — but the AI flagged is_silhouette=false because some rim or edge is visible, which is exactly the case where "strong composition + sharp beak" tricks the model into keeping a marginal frame. Default to no. The photographer marks it keep themselves if they specifically wanted that moody-backlight look.

**SILHOUETTES (`is_silhouette`=true) DEFAULT TO `keep`="no".** Silhouettes are an artistic choice the photographer makes deliberately for specific images — the cull AI should NOT pre-select them as keepers. A silhouette earns "yes" only if:
- The subject's shape is clearly readable as the species (not just a generic blob), AND
- Something is genuinely happening (dramatic flight pose, prey transfer, sunset-against-dramatic-sky), AND
- The light itself is the photograph (golden hour, dramatic sky), not just "the sun was behind it and I lost detail".
A backlit silhouette of a perched bird with no behavior is "no". The photographer will mark it "keep" themselves if they specifically wanted that silhouette.

CALIBRATION EXAMPLES:
- **Side profile of an Inca tern with a fish in beak, photographed against soft blue background. The bird's body is mostly in deep shadow — the eye is technically there but in shadow with no visible catchlight, only the beak and the fish are well-lit. is_silhouette is debatable — some rim light defines the white throat line, but the body interior has no readable feather detail.** → `eye_focus="not_visible"` (no catchlight, no iris structure visible), `is_silhouette` can be true OR false but it doesn't change the verdict, **`keep="no"`**. The composition is fine and the moment (prey transfer) is fine, but the image only works as a silhouette. The photographer will mark it keep if they specifically wanted that moody-backlight look — the cull AI should NOT pre-select it.
- Backlit silhouette of a perched tern, even with prey in beak → keep="no". Same reasoning as above.
- Clean portrait of a common cardinal in soft front-light, eye sharp with visible catchlight → keep="yes". Usable stock-grade image.
- Soft-focus shot of a rare species doing something exceptional (mating display) → keep="yes". The moment matters more than technical perfection.
- A bird with its back turned, eye not visible, no behavior → keep="no".
- A frame from a burst where the eye is closed → keep="no" (better frames almost always exist).
- Overexposed sky with clipped highlights on the subject → keep="no".
- A genuinely strong silhouette: bird in flight pose against a dramatic sunset, shape clearly readable, the sky IS the photograph → keep="yes". Note the difference: the LIGHT is doing something here, vs the heavy-backlight case where the light is just losing detail.

You are NOT scoring artistic merit — a separate scorer (Claude) handles ranking among keepers. Your job is the filter. Be strict at the filter."""


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

    j = judge or primary_judge()
    encode_start = time.time()
    img1_b64 = _read_b64(preview1_path)
    img2_b64 = _read_b64(preview2_path)
    encode_secs = time.time() - encode_start

    payload = {
        "model": j["model"],
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
