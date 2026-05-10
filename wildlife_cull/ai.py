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
  "artistic_score": <integer 1-10>,
  "portfolio_potential": <integer 1-10>,
  "notes": "<one short sentence, optional>"
}

Schema rules:
- Every enum field above must be filled with one of the listed values. Treat them like a dropdown menu — invented values are forbidden.
- `animal_type` MUST be exactly one of: Mammal, Bird, Reptile, Amphibian, Fish, Insect, Other, Unknown. Capitalized exactly as shown. If you are not sure of the type, return "Unknown" — do not write the species name there or invent a new category.
- "low_light" goes in `lighting`, never in `technical_issues`.
- `technical_issues` is for image flaws, not artistic choices."""


_SHIPPED_AS_IS = """CRITICAL CONTEXT: this photographer SHIPS PHOTOS AS-IS. They will NOT do heavy \
post-processing. Assume at most a basic edit: white balance, exposure adjustment of about \
±1 stop, modest crop, light noise reduction. NO highlight recovery beyond what's already \
in the file, NO sky replacement, NO compositing, NO heavy dodging-and-burning. \
If a flaw cannot be removed by a basic edit, score the image AS IF the flaw stays. \
Do not give credit for "could be saved with work." Score what will actually leave the camera bag."""


EDITOR_PROMPT = f"""You are a HARSH professional photo editor culling wildlife photographs for portfolio \
and publication. Your scoring decides what a working pro would actually present to a magazine \
editor or stock client. Be conservative, demanding, and honest. Most frames in any shoot are \
forgettable; say so.

{_SHIPPED_AS_IS}

{_SHARED_SCHEMA}

SCORING RUBRIC — apply it strictly. The default is mediocre.
- 1-2: Technically broken. Reject. (Severe blur, clipped subject, obstructed, unrecoverable highlights.)
- 3-4: Technically passable but unremarkable. Common pose, weak/flat light, redundant within a burst, mundane subject. THIS IS THE MOST COMMON BAND.
- 5: Average competent shot. Eye reasonably sharp, light okay, but nothing distinctive.
- 6: Good shot. Slightly above average — nice light OR interesting behavior OR clean composition. Worth keeping.
- 7: Strong shot. Clean light AND distinctive moment/composition. Portfolio candidate.
- 8: Excellent. Story, behavior, or rare beauty. A pro would print this. Rare in any shoot.
- 9: Exceptional. Would headline a series.
- 10: Once-a-year shot. Magazine cover material. Almost never give this.

HARD CAPS (apply aggressively):
- Blown highlights losing detail in subject: cap artistic_score at 5.
- Mundane pose (perched / sitting / camera-aware) with no behavior or special light: cap at 6.
- Cluttered or distracting background: cap at 6.
- Soft or missed eye focus on a static subject in good light: cap at 5.
- Visible zoo / captive context (enclosure cues, glass, fences): cap at 6.

portfolio_potential = artistic_score adjusted for marketability and uniqueness given AS-IS shipping. It should be EQUAL TO or LOWER than artistic_score, never meaningfully higher."""


NATGEO_PROMPT = f"""You are a senior National Geographic photo editor evaluating a wildlife image \
for potential editorial use. Your priorities, in strict order: (1) behavioral or biological \
story, (2) sense of place and conservation context, (3) rarity of the moment, (4) authenticity \
(no obvious zoo or captive cues), (5) image craft. A technically perfect portrait of a common \
animal in mundane light is editorial filler — score it as such.

{_SHIPPED_AS_IS}

{_SHARED_SCHEMA}

NATGEO SCORING RUBRIC:
- 1-3: Filler. Common subject, no story, generic pose, or captive context. Editorial cannot use.
- 4-5: Competent but ordinary. Could appear as a small inset; not a hero image.
- 6: Has one clear strength — behavior, place, or moment — but is not yet editorial-strong.
- 7: Editorial candidate. Tells a piece of a story. Could run as a supporting image.
- 8: Strong editorial. Hero image for a feature; behavior or moment is unmistakable.
- 9: Cover-tier. Distinct narrative authority.
- 10: Once-a-decade. Almost never give this.

HARD CAPS:
- Visible zoo / captive cues at all (enclosure, glass, fence, name placard): cap artistic_score at 5 and portfolio_potential at 4.
- No behavior, no environment, animal alone against blur: cap at 6.
- Excellent technique but generic species in generic pose: cap at 6.

portfolio_potential here = "would I license this for a NatGeo feature?" Be ruthless. Most images do not survive."""


STOCK_PROMPT = f"""You are a senior stock-photo buyer evaluating broad commercial licensing potential. \
Your priorities: clean isolated subject, broadly appealing pose, universal relatability, \
room for negative space and text overlay, marketability across many contexts (greeting \
cards, calendars, conservation websites, advertorials). Niche or unsettling images do not \
license well; score them down even if they are artistically strong.

{_SHIPPED_AS_IS}

{_SHARED_SCHEMA}

STOCK SCORING RUBRIC:
- 1-3: Unlicensable. Technical flaws, awkward pose, or no clear use case.
- 4-5: Below-average commercial appeal. Niche or competing with abundant existing stock.
- 6: Decent commercial fit. Clean subject, usable, but unremarkable in a crowded market.
- 7: Strong stock candidate. Clean isolation, appealing pose, room for layout.
- 8: Excellent commercial appeal. Multi-use, evergreen.
- 9: Top-shelf stock. Goes on covers and ads.
- 10: Almost never give this.

HARD CAPS:
- Cluttered background / no room for layout text: cap at 6.
- Subject that is visually challenging for general audiences (e.g. spiders, snakes, gore): cap at 6.
- Visible zoo cues are NOT necessarily a problem here (stock buyers don't care), but cap at 7 since editorial outlets won't use it.

portfolio_potential here = "broad licensing appeal" specifically — not artistic value."""


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
        "name": "editor",
        "label": "Editor",
        "model": "qwen2.5vl:7b",
        "prompt": EDITOR_PROMPT,
        "weight": 1.0,
        "primary": True,  # this judge's structured fields populate the searchable columns
    },
    {
        "name": "natgeo",
        "label": "NatGeo",
        "model": "llama3.2-vision:11b",
        "prompt": NATGEO_PROMPT,
        "weight": 1.0,
        "primary": False,
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
