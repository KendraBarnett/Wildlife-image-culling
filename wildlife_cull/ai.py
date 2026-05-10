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


_SHARED_SCHEMA = """Reply with ONLY a JSON object, no prose, no markdown fences. Use this schema and \
pick values STRICTLY from the listed options for every enum field:

{
  "subject": "<short phrase, e.g. 'great horned owl perched on branch'>",
  "animal_type": <PICK EXACTLY ONE FROM THIS DROPDOWN — no other value is allowed: "Mammal" | "Bird" | "Reptile" | "Amphibian" | "Fish" | "Insect" | "Other" | "Unknown">,
  "species": "<best-guess common name, e.g. 'Mallard', 'Western Lowland Gorilla'. Use 'Unknown' if unsure>",
  "eye_focus": <PICK EXACTLY ONE: "sharp" | "soft" | "not_visible" | "n/a">,
  "motion": <PICK EXACTLY ONE: "still" | "subtle" | "in_motion" | "blurred">,
  "composition": <PICK EXACTLY ONE: "strong" | "standard" | "weak">,
  "lighting": <PICK EXACTLY ONE: "harsh" | "soft" | "golden" | "low_light" | "backlit" | "overcast" | "mixed">,
  "is_silhouette": true | false,
  "technical_issues": [<zero or more, each ONE OF: "out_of_focus","camera_shake","clipped_subject","blown_highlights","heavy_noise","obstructed">],
  "artistic_score": <number 1.0-10.0, ONE decimal place, e.g. 4.3 or 7.1>,
  "portfolio_potential": <number 1.0-10.0, ONE decimal place, e.g. 5.7 or 8.2>,
  "notes": "<2-3 sentences. First sentence: what the image actually shows and what works about it (subject, light, composition, behavior). Second sentence: what holds it back, with specifics. Optional third sentence: why this scored where it did. Be SPECIFIC and CONCRETE. NEVER write empty contradictions like 'good, but blurry' — name strengths and flaws separately and tie each to what you can see.>"
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
- If `technical_issues` contains "out_of_focus", then `eye_focus` MUST be "soft" or "not_visible". Saying "sharp" in one place while flagging out-of-focus in another is a forbidden contradiction.
- If `motion` is "blurred" because of camera shake, `technical_issues` MUST include "camera_shake".
- Notes must never contradict the structured fields. If you write "tack sharp" in notes, eye_focus cannot be "soft" and out_of_focus cannot be in issues.

SCORE CAPS WHEN A REAL ISSUE IS PRESENT (only apply if the issue is genuine, not invented):
- Any `out_of_focus` flag: cap BOTH artistic_score AND portfolio_potential at 3.0. Out-of-focus images cannot ship.
- Any `camera_shake` flag: cap both at 3.0.
- Any `clipped_subject` flag: cap both at 4.0.
- Any `blown_highlights` flag where the highlights are on the subject: cap both at 4.5.
- Any `obstructed` flag: cap both at 4.5."""


_SHIPPED_AS_IS = """CRITICAL CONTEXT: this photographer SHIPS PHOTOS AS-IS. They will NOT do heavy \
post-processing. Assume at most a basic edit: white balance, exposure adjustment of about \
±1 stop, modest crop, light noise reduction. NO highlight recovery beyond what's already \
in the file, NO sky replacement, NO compositing, NO heavy dodging-and-burning. \
If a flaw cannot be removed by a basic edit, score the image AS IF the flaw stays. \
Do not give credit for "could be saved with work." Score what will actually leave the camera bag."""


_BASELINE_RUBRIC = """SCORING APPROACH — be HARSH and DISCRIMINATING.

The default verdict for any wildlife photo is "filler — not interesting." \
Most working photographers' shoots are 90% filler; say so. You must EARN \
points up from a base of 3 for unremarkable images. Do not start in the \
middle and adjust slightly — start LOW and add points for what makes a \
photo actually work.

START EVERY IMAGE AT artistic_score = 3. Then:

ADD +1 each (max +7) for any of these that genuinely apply:
- Eye is decisively sharp on the actual eye (not just "the bird is in focus")
- Light is doing something interesting (golden, dramatic side-light, rim, atmosphere, mood)
- Genuine behavior is happening (calling, eating, flight, interaction, hunting, courtship)
- Composition is deliberate and works (intentional negative space, leading lines, frame-within-frame, off-center balance that's clearly chosen)
- Moment is unusual or rare for the species
- Image makes a viewer pause — a strong response of any kind
- Would hold up next to working pros' published work

SUBTRACT -1 each for any of these:
- Blown highlights touch the subject and lose feather/fur detail
- Eye is soft, missed, or not visible on a static animal
- Camera-aware "looking at the lens" pose with no other interest
- Cluttered or distracting background you can't crop out
- Visible captive cues (enclosure walls, glass, fences, name placards)
- Redundant within a burst (this is one of many similar frames)
- Common species in a common pose with no special light or story

Final artistic_score = 3 + bonuses − penalties, with finer-grained decimals \
where useful (e.g. 4.3 if "barely above filler with one weak strength", 6.7 \
if "good and creeping toward strong"). Use ONE decimal place. Clamp to 1.0–10.0.

portfolio_potential answers a DIFFERENT question: "Would this be chosen over \
similar work?" It MUST often differ from artistic_score. A technically strong \
but generic shot has high artistic_score but lower portfolio_potential (too \
much competition). A flawed but rare moment can have a lower artistic_score \
yet higher portfolio_potential. If you find yourself writing the same number \
for both, reconsider — what would make a buyer choose this over the next \
photographer's frame?

Calibration anchors:
- 3 = filler. The base. Most images.
- 4 = barely above filler. One small thing going for it.
- 5 = competent average. Stock-grade at best.
- 6 = good. Worth keeping.
- 7 = strong. Portfolio candidate.
- 8 = excellent. Rare in any shoot.
- 9 = exceptional. Headline image.
- 10 = once-a-year. Almost never give this."""


NATGEO_PROMPT = f"""You are a senior National Geographic photo editor evaluating a wildlife image \
for potential editorial use. Your priorities, in strict order: (1) behavioral or biological \
story, (2) sense of place and conservation context, (3) rarity of the moment, (4) authenticity \
(no obvious zoo or captive cues), (5) image craft. A technically perfect portrait of a common \
animal in mundane light is editorial filler — score it as such.

{_SHIPPED_AS_IS}

{_SHARED_SCHEMA}

{_BASELINE_RUBRIC}

NATGEO-SPECIFIC weights when scoring:
- Behavior, environment, and rarity matter MOST. A clean technical portrait without those is filler (artistic_score around 3-5).
- Authenticity matters. Visible captive cues (enclosure, glass, fence, name placard) cap artistic_score at 5 and portfolio_potential at 4.
- "Would this run in a NatGeo feature?" is the portfolio_potential question. Most images do not survive.
- A flawed but unique behavior moment (e.g. predation, courtship, rare species) can score higher than a clean studio-style portrait."""


STOCK_PROMPT = f"""You are a senior stock-photo buyer evaluating broad commercial licensing potential. \
Your priorities: clean isolated subject, broadly appealing pose, universal relatability, \
room for negative space and text overlay, marketability across many contexts (greeting \
cards, calendars, conservation websites, advertorials). Niche or unsettling images do not \
license well; score them down even if they are artistically strong.

{_SHIPPED_AS_IS}

{_SHARED_SCHEMA}

{_BASELINE_RUBRIC}

STOCK-SPECIFIC weights when scoring:
- Clean isolation and clear subject matter most. A clean shot of a common animal can score well here even if NatGeo would reject it.
- Visible zoo cues are NOT necessarily disqualifying for stock, but cap artistic_score at 7 since editorial outlets won't use it.
- Cluttered background / no room for text overlay: cap artistic_score at 6.
- Subjects that are visually challenging for broad audiences (gore, large spiders, snakes in striking poses): cap artistic_score at 6.
- portfolio_potential here = "broad licensing appeal" — not artistic value. A boring but clean cardinal portrait may have higher portfolio_potential than a stunning predation shot, because licensing buyers want safe, generic, repeatable subjects."""


# Single source of truth for the controlled vocab. The server exposes this
# to the UI so dropdowns show exactly the values the model can return.
AI_VOCAB = {
    "animal_type": ["Mammal", "Bird", "Reptile", "Amphibian", "Fish", "Insect", "Other", "Unknown"],
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
        "name": "natgeo",
        "label": "NatGeo",
        "model": "llama3.2-vision:11b",
        "prompt": NATGEO_PROMPT,
        "weight": 1.0,
        "primary": True,  # this judge's structured fields populate the searchable columns
    },
    {
        "name": "stock",
        "label": "Stock",
        "model": "minicpm-v:8b",
        "prompt": STOCK_PROMPT,
        "weight": 1.0,
        "primary": False,
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


def analyze_with_judge(
    judge: dict,
    preview_path: Path,
    idle_timeout: float = 300.0,
    total_timeout: float = 900.0,
) -> tuple[dict, str]:
    if not preview_path.exists():
        raise FileNotFoundError(f"Preview missing: {preview_path}")
    payload = {
        "model": judge["model"],
        "prompt": judge["prompt"],
        "images": [_read_b64(preview_path)],
        "stream": True,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.2},
    }
    timeout = httpx.Timeout(connect=10.0, read=idle_timeout, write=30.0, pool=10.0)
    chunks: list[str] = []
    started = time.time()
    last_error: Optional[str] = None
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
                        raise RuntimeError(f"Ollama exceeded total timeout of {total_timeout:.0f}s")
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("error"):
                        last_error = str(evt["error"])
                        break
                    chunk = evt.get("response", "")
                    if chunk:
                        chunks.append(chunk)
                    if evt.get("done"):
                        break
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
    return parsed, text


# Back-compat alias for any caller still using the single-judge name.
def analyze_image(preview_path: Path, **kwargs) -> tuple[dict, str]:
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
