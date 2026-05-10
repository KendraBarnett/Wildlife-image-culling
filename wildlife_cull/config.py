from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREVIEW_DIR = DATA_DIR / "previews"
DB_PATH = DATA_DIR / "cull.db"

DATA_DIR.mkdir(exist_ok=True)
PREVIEW_DIR.mkdir(exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Triage model — Phase 1, runs on every image. Default is
# llama3.2-vision:11b because the 11B model handles the 200-line
# structured prompt + 10+ consistency rules more reliably than 7B
# alternatives. Override with WC_TRIAGE_MODEL=qwen2.5vl:7b for faster
# but less careful triage. Memory: ~10 GB while loaded.
TRIAGE_MODEL = os.environ.get("WC_TRIAGE_MODEL", "llama3.2-vision:11b")

# Compare model — used by the "Compare 2 with AI" button. Simpler
# task (pick the better of two), so we prefer the faster qwen here.
# Override with WC_COMPARE_MODEL.
COMPARE_MODEL = os.environ.get("WC_COMPARE_MODEL", "qwen2.5vl:7b")

# Back-compat alias: VISION_MODEL is the triage model. Old code that
# references VISION_MODEL still works.
VISION_MODEL = TRIAGE_MODEL

CLIP_MODEL = os.environ.get("WC_CLIP_MODEL", "clip-ViT-B-32")

SERVER_HOST = os.environ.get("WC_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("WC_PORT", "8765"))

RAW_EXTS = {".cr3", ".cr2", ".nef", ".arw", ".raf", ".orf", ".rw2", ".dng", ".pef", ".srw"}
JPEG_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".tif", ".tiff"}
IMAGE_EXTS = RAW_EXTS | JPEG_EXTS

# Preview size — used for the modal image AND as input to the local AI.
# 1024px matches what we resize to before sending to qwen2.5vl anyway,
# so going bigger only wastes disk. 1024px JPEG q85 is roughly 30-50 KB
# vs 80-120 KB at 1600px — about half the disk for ~5k images.
# Override with WC_PREVIEW_MAX_SIZE if you want larger previews for
# modal pixel-peeping (at the cost of disk).
PREVIEW_MAX_SIZE = int(os.environ.get("WC_PREVIEW_MAX_SIZE", "1024"))
THUMB_MAX_SIZE = 400

# Phase 2 — Claude API scoring (on-demand only)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
try:
    CLAUDE_BUDGET_USD = float(os.environ.get("CLAUDE_BUDGET_USD", "5.00"))
except ValueError:
    CLAUDE_BUDGET_USD = 5.00
