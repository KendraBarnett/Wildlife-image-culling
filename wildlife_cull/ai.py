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
  "technical_issues": [<zero or more, each ONE OF: "out_of_focus","soft_edges","camera_shake","clipped_subject","blown_highlights","underexposed","harsh_backlight","heavy_noise","obstructed","cluttered_background","subject_too_small">],
  "keep": <PICK EXACTLY ONE: "yes" | "maybe" | "no">,
  "notes": "<2-3 sentences. First: what the image actually shows (subject, light, composition, behavior). Second: technical and compositional state — name BOTH what works AND every flaw you can see, in plain language tied to specific parts of the frame ('body in shadow from the chest down', 'soft on the eye but beak is sharp', 'twig crossing the wing'). Third: WHY keep or don't keep, referencing what you just named. NEVER be a cheerleader — a keeper can have flaws and those flaws still need to be named. Don't write 'captures a clear and engaging moment' on a backlit-shadow shot. Don't write 'sharp details' if the sharpness is on the beak but the eye is in shadow. If technical_issues is empty, the notes should justify that — 'no visible technical issues; clean across the board'.>"
}

Schema rules:
- Every enum field above must be filled with one of the listed values. Treat them like a dropdown menu — invented values are forbidden.
- `animal_type` MUST be exactly one of: Mammal, Bird, Reptile, Amphibian, Fish, Insect, Other, Unknown. Capitalized exactly as shown. If you are not sure of the type, return "Unknown" — do not write the species name there or invent a new category.
- "low_light" goes in `lighting`, never in `technical_issues`.
- `technical_issues` is for image flaws, not artistic choices.

TECHNICAL ISSUES — DEFINITIONS, AND FLAG EVERYTHING YOU CAN SEE.
A keeper can and SHOULD have issues listed. Issues are observations, not the cull decision. The photographer wants to know what's wrong with each frame even when they decide to keep it — that's how they choose between similar shots and how they decide whether the image needs editing. Be HONEST about flaws regardless of the keep verdict.

Definitions — what each value means:
- `out_of_focus` — the SUBJECT is clearly out of focus. Background blur (bokeh) is NOT this. Only count it if the subject itself is soft enough to render the image unusable.
- `soft_edges` — slightly soft but not full out_of_focus. Subject is recognizable but details are mushy (slight motion, slight focus miss, atmospheric haze). Common on long-lens wildlife.
- `camera_shake` — handheld at too low a shutter speed; whole frame has motion smear including the background.
- `clipped_subject` — the subject is cut off at the frame edge in a way that damages the image (wingtip, tail, foot amputated).
- `blown_highlights` — pure white pixels with no recoverable detail on the SUBJECT. Sky blown is not an issue; subject feathers blown is.
- `underexposed` — the subject's body is so dark you cannot read feather/fur detail across most of it. A moody-backlight shot where only edges and bright features show counts here.
- `harsh_backlight` — heavy backlighting where the light source is behind the subject and the front (eye, face, body) is in deep shadow as a result. Often goes WITH underexposed. The image only works as a silhouette.
- `heavy_noise` — visible grain that hurts the image at viewing size, beyond what light NR can fix.
- `obstructed` — branches, grass, fence wires, or other elements crossing the subject in a way that can't be cropped out.
- `cluttered_background` — distracting elements behind the subject (busy twigs, harsh sun spots, bright distractions) that pull the eye and can't be cropped away.
- `subject_too_small` — subject occupies so little of the frame that the image is more landscape than wildlife portrait, with no narrative reason for it.

RULES:
- An EMPTY array is correct only when the image actually has no visible flaws. Most images have at least one. Don't withhold issues just because you've decided to keep the image.
- For every issue you flag, the notes MUST describe specifically where you see it ("body in deep shadow from the chest down", "soft on the eye but beak is sharp", "telephone wire crossing the bird's neck"). If you can't be specific, don't flag it.
- DO NOT invent issues you can't see at this resolution. Hallucinating flaws is just as bad as missing them.

INTERNAL CONSISTENCY — these must agree. CHECK ALL OF THESE BEFORE RETURNING:
- `in_focus` is the simple yes/no overall question: is the SUBJECT sharp enough to use? If `in_focus` is "no", `technical_issues` MUST include "out_of_focus", and `eye_focus` MUST be "soft" or "not_visible".

**THE EYE_FOCUS RULE — STRICT. THIS IS THE #1 HALLUCINATION FAILURE MODE.**

`eye_focus` refers ONLY to the literal eye organ on the animal's head — the eyeball in its socket. NOTHING ELSE counts as "the eye":
- **DECORATIVE EYESPOTS ARE NOT EYES.** The "eye" patterns on a peacock's train feathers, the eyespots on butterfly wings, the false eye-markings on the back of an owl's head, the ocelli on a moth — these are decorative patterns that LOOK like eyes. They are NOT what `eye_focus` is rating. If you cannot see the animal's actual head and its actual eye, `eye_focus` is "not_visible" regardless of how many decorative "eyes" appear in the frame.
- **THE FACE MUST BE VISIBLE.** If the animal's head is turned away, tucked into its body, hidden by feathers/fur, in deep shadow, or otherwise not clearly showing the actual eye, `eye_focus` is "not_visible". A back-of-animal shot where you see only feathers / fur / wings is `eye_focus="not_visible"`, period.

`eye_focus = "sharp"` REQUIRES, in the actual eye on the head:
- A visible catchlight (highlight reflection in the iris/pupil) showing crisp edges, OR
- Clear visible iris/pupil structure that's in focus at this resolution.

If you can only see WHERE the eye is (a dark shape, an outline, a guess based on the bird's face), or only see a decorative eyespot, or only see the back/side of the animal where the head isn't visible — the answer is "soft" or "not_visible" — never "sharp". Sharp on adjacent features (beak, fur, wing feathers, train feathers) does NOT transfer to the eye. Each is graded independently on what is actually visible at THIS resolution.
- **eye_focus = "sharp" is forbidden when the eye is in shadow, backlit-into-darkness, hidden, tucked, turned away, or shows no catchlight/iris detail.** Even if the face is technically visible. "I can see where the eye is" ≠ "the eye is sharp". "I can see decorative eye-patterns on the feathers" ≠ "the eye is sharp".

**SUBJECT VISIBILITY RULE.** If you cannot see the SUBJECT'S head AND face AND a substantial portion of its body clearly (more than half of the animal hidden, tucked, or in deep shadow), this is a candidate for keep="no" unless something exceptional is happening (rare behavior, dramatic moment). "Can't see half the animal" is not a strong wildlife photo by default. Flag this case in notes: name which parts of the animal are not visible.

- **SILHOUETTE / DEEP SHADOW RULE: If `is_silhouette` is true, OR if the subject's face is in deep shadow with no visible eye / facial detail, `eye_focus` MUST be "not_visible". Never "sharp".**

- **HEAVY-SHADOW SUBJECT RULE (the silhouette-adjacent case): Even when `is_silhouette` is false, if the subject's body is mostly crushed-shadow with little visible feather/fur detail, treat it the same way for `keep`. A backlit bird where the only well-lit area is the beak — body in shadow, eye barely visible, edges defined but interior dark — is NOT a keeper just because the composition is strong. See the heavy-backlight example in CALIBRATION below.**

- If `technical_issues` contains "out_of_focus", then `eye_focus` MUST be "soft" or "not_visible" AND `in_focus` MUST be "no".
- If `technical_issues` contains "soft_edges" (slight softness), `eye_focus` can be either "soft" (if the softness reaches the eye) or "sharp" (if only non-eye details are soft), but `in_focus` can stay "yes" — soft_edges is the in-between case before full out_of_focus.
- If `technical_issues` contains "underexposed" or "harsh_backlight", and the eye is among the dark areas, `eye_focus` MUST be "not_visible" or "soft" — never "sharp".
- If `motion` is "blurred" because of camera shake, `technical_issues` MUST include "camera_shake".
- Notes must never contradict the structured fields. Writing "the bird is in focus with sharp details" while the eye is in shadow is a forbidden hallucination. Be specific about WHAT is sharp — if it's the beak but not the eye, say so.

**FILL EVERY FIELD ON EVERY IMAGE, regardless of keep verdict.** Even if you decide keep="no", the photographer needs to know what was in the frame and what was wrong with it. Type, Species, Subject, In focus, Eye focus, Motion, Composition, Lighting, Silhouette, Issues, Keep, Notes — ALL must be populated based on what you actually see. Leaving fields blank or skipping the structured analysis when you decide to cull is forbidden. The photographer reviews rejected images too — they need the data.

KEEP DECISION — three values: "yes", "maybe", "no". Be HONEST, not strict for strictness's sake. The goal is for the photographer to TRUST the verdict: if you say "yes", they expect a usable image; if you say "no", they expect a clear flaw they can see. Inventing flaws to justify "no" is just as bad as missing real flaws. Don't panic-cull a clearly competent image because some prior rule said "default to no" — those rules only apply when the conditions actually fire.

**"yes" — clear keeper.** A clean, well-executed image with a visible subject. Specifically:
  (A) Technically usable: in_focus="yes", eye_focus="sharp", no severe technical_issues actually present in the frame.
  (B) Subject is clearly identifiable: the animal's head/face is visible and you can read the subject as a wildlife photo, not just texture.
  (C) Something supports it: good light, clear composition, an engaging subject (looking at camera, calling, doing something), or just a clean technically sound portrait of an interesting animal.

A clean shot of a common animal in good light with a sharp visible eye IS a keeper. You don't need a "wow" moment for "yes" — a usable image is a keeper. Stock-quality is enough.

**"maybe" — technically excellent but the SUBJECT is incomplete or ambiguous.** Use this when:
  - The image is technically clean (sharp, well-exposed, well-composed, no major flaws) BUT the subject is partial / cropped / abstract — only feathers, only fur, only a body part. No full animal, no face. The image works as a STUDY of texture/color/pattern but not as a wildlife portrait.
  - The composition cuts the animal in a way that's not clearly intentional but not damaging either.
  - The moment is ambiguous — you can't tell if the photographer wanted this frame or it's an in-between burst frame.
  - You're genuinely uncertain whether the photographer wanted this. The image is a HUMAN judgment call, not a clear cull.

Use "maybe" SPARINGLY. It's the small subset where the technique is right but the subject choice needs a human eye. If the image has technical_issues that are real flaws (out_of_focus, camera_shake, harsh_backlight, etc.), it goes to "no" — those aren't maybes.

**"no" — clear cull.** This is for images with actual identifiable flaws you can point to in the image. The condition has to actually FIRE in this specific image, not just be a default. Cull when any of these is TRUE about THIS image:
- in_focus is genuinely "no" because you can see the subject is soft → no
- You actually see out_of_focus, camera_shake, or clipped_subject damage to the subject → no
- composition really is weak (subject dead-center with no light or behavior, distracting background you can't crop) AND nothing else carrying it → no
- The subject's face is genuinely hidden / turned away / obstructed AND no exceptional moment is happening → no
- eye_focus is genuinely "not_visible" on a static wildlife subject with no behavior → no
- eye_focus is genuinely "soft" on a static portrait → no, the keeper neighbor is sharper
- The body is genuinely in deep shadow with no readable detail (heavy-backlight, silhouette-adjacent) → no

If NONE of these is actually visible in the image, it's NOT a "no". A clean, sharp portrait of an animal that just isn't doing anything dramatic is still a "yes" — usable images are keepers, "yes" doesn't require a wow moment.

**CRITICAL — DO NOT INVENT ISSUES TO JUSTIFY A KEEP DECISION.** technical_issues is for ACTUAL VISIBLE FLAWS at this resolution. If you decided keep="no" because the subject is incomplete (only feathers visible, no face), DO NOT then add "out_of_focus" to technical_issues — the image is in focus, that's not the problem. The problem goes in the notes, plain language. Lying about a technical flaw to back up the cull verdict is the single worst failure mode: it tells the photographer the wrong thing about their craft.

**SILHOUETTES (`is_silhouette`=true) DEFAULT TO `keep`="no".** Silhouettes are an artistic choice the photographer makes deliberately for specific images — the cull AI should NOT pre-select them as keepers. A silhouette earns "yes" only if:
- The subject's shape is clearly readable as the species (not just a generic blob), AND
- Something is genuinely happening (dramatic flight pose, prey transfer, sunset-against-dramatic-sky), AND
- The light itself is the photograph (golden hour, dramatic sky), not just "the sun was behind it and I lost detail".
A backlit silhouette of a perched bird with no behavior is "no". The photographer will mark it "keep" themselves if they specifically wanted that silhouette.

CALIBRATION EXAMPLES:
- **Close-up of a peacock's train feathers showing many decorative eyespots, with the bird's actual head not visible in the frame. Image is TECHNICALLY EXCELLENT — sharp feathers, good color, strong light, clean composition.** → `eye_focus="not_visible"` (the eyespots on feathers are NOT the bird's eye), `in_focus="yes"` (the FEATHERS are in focus — don't lie about this), `technical_issues=[]` (NO out_of_focus, the image is sharp — adding fake issues to justify the cull is the worst failure mode), **`keep="maybe"`** — technically excellent but the SUBJECT is a partial detail study rather than a full wildlife portrait, the photographer needs to decide if they meant to shoot this as an abstract or wanted the full bird. Notes: 'Sharp, well-lit close-up of peacock train feathers; the bird's actual head and face are not in the frame. Decorative eyespots on the feathers are not the real eye. Image works as a texture/pattern study but not as a wildlife portrait — needs photographer's call.'
- **A dark-furred animal (black cat, black bear, etc.) curled up or with its head tucked, where most of the body is in deep shadow and the face is barely visible** → `eye_focus="not_visible"`, add `underexposed` to technical_issues, `keep="no"`. Notes should say which parts of the animal you can and cannot see. 'Head is tucked into body / face mostly in shadow / can see fur but not facial features clearly' — be specific.
- **Side profile of an Inca tern with a fish in beak, photographed against soft blue background. The bird's body is mostly in deep shadow — the eye is technically there but in shadow with no visible catchlight, only the beak and the fish are well-lit. is_silhouette is debatable — some rim light defines the white throat line, but the body interior has no readable feather detail.** → `eye_focus="not_visible"` (no catchlight, no iris structure visible), `is_silhouette` can be true OR false but it doesn't change the verdict, add `underexposed` and `harsh_backlight` to technical_issues, **`keep="no"`**. The composition is fine and the moment (prey transfer) is fine, but the image only works as a silhouette. The photographer will mark it keep if they specifically wanted that moody-backlight look — the cull AI should NOT pre-select it.
- Backlit silhouette of a perched tern, even with prey in beak → keep="no". Same reasoning as above.
- **Close-up portrait of a red panda's face — eyes clearly visible looking at camera or near-camera, fur detail rendered well, soft natural light, clean composition, no harsh shadows on the face, sharp eye area.** → `in_focus="yes"`, `eye_focus="sharp"` (the eyes ARE visible and ARE rendered with detail), `technical_issues=[]` (no actual visible flaw), **`keep="yes"`**. This is the prototypical clean wildlife portrait. Do NOT flag it as out_of_focus to justify a cull — it is in focus and it is a keeper. Notes: 'Sharp portrait of a red panda, eyes visible with clear iris/catchlight detail, soft natural front-light, no technical issues.'
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
