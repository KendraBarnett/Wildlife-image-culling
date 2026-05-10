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

VISION_PROMPT = """You are evaluating a single wildlife photograph for culling. \
Reply with ONLY a JSON object, no prose, no markdown fences. Use this schema \
and pick values STRICTLY from the listed options for every enum field:

{
  "subject": "<short phrase, e.g. 'great horned owl perched on branch'>",
  "eye_focus": "sharp" | "soft" | "not_visible" | "n/a",
  "motion": "still" | "subtle" | "in_motion" | "blurred",
  "composition": "strong" | "okay" | "weak",
  "lighting": "harsh" | "soft" | "golden" | "low_light" | "backlit" | "overcast" | "mixed",
  "is_silhouette": true | false,
  "technical_issues": [<zero or more of: "out_of_focus","camera_shake","clipped_subject","blown_highlights","heavy_noise","obstructed">],
  "artistic_score": <integer 1-10>,
  "portfolio_potential": <integer 1-10>,
  "notes": "<one short sentence, optional>"
}

Important rules:
- Pick exactly one value for each enum (eye_focus, motion, composition, lighting). Do not invent new values.
- "low_light" goes in `lighting`, never in `technical_issues`.
- `technical_issues` is for image flaws, not artistic choices. A deliberate silhouette is not an "obstructed" or "out_of_focus" image.
- Score conservatively. Most frames in a burst will be average (4-6). Reserve 9-10 for genuinely exceptional work."""


# Single source of truth for the controlled vocab. The server exposes this
# to the UI so dropdowns show exactly the values the model can return.
AI_VOCAB = {
    "eye_focus": ["sharp", "soft", "not_visible", "n/a"],
    "motion": ["still", "subtle", "in_motion", "blurred"],
    "composition": ["strong", "okay", "weak"],
    "lighting": ["harsh", "soft", "golden", "low_light", "backlit", "overcast", "mixed"],
    "technical_issues": [
        "out_of_focus", "camera_shake", "clipped_subject",
        "blown_highlights", "heavy_noise", "obstructed",
    ],
}


def _read_b64(path: Path) -> str:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > AI_IMAGE_MAX_SIZE:
            im.thumbnail((AI_IMAGE_MAX_SIZE, AI_IMAGE_MAX_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def analyze_image(preview_path: Path, idle_timeout: float = 300.0, total_timeout: float = 900.0) -> tuple[dict, str]:
    if not preview_path.exists():
        raise FileNotFoundError(f"Preview missing: {preview_path}")
    payload = {
        "model": VISION_MODEL,
        "prompt": VISION_PROMPT,
        "images": [_read_b64(preview_path)],
        "stream": True,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0.2},
    }
    timeout = httpx.Timeout(connect=10.0, read=idle_timeout, write=30.0, pool=10.0)
    chunks: list[str] = []
    started = time.time()
    last_error: str | None = None
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
        raise RuntimeError("Ollama returned empty response (model may have failed to load or run out of memory)")
    parsed = _parse_json_loose(text)
    return parsed, text


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


def warmup_vision_model(timeout: float = 600.0) -> Optional[str]:
    """Force Ollama to load the vision model so the first real call is fast."""
    payload = {"model": VISION_MODEL, "keep_alive": "30m"}
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            if r.status_code != 200:
                return f"warmup HTTP {r.status_code}: {r.text[:200]}"
        return None
    except Exception as exc:
        return f"warmup failed: {exc}"


def ollama_health() -> Optional[str]:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = [m.get("name") for m in tags]
            if not any(n and n.startswith(VISION_MODEL.split(":")[0]) for n in names):
                return f"Ollama is reachable but model '{VISION_MODEL}' is not pulled. Run: ollama pull {VISION_MODEL}"
            return None
    except Exception as exc:
        return f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}"
