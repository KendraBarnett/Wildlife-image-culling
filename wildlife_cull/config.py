from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREVIEW_DIR = DATA_DIR / "previews"
DB_PATH = DATA_DIR / "cull.db"

DATA_DIR.mkdir(exist_ok=True)
PREVIEW_DIR.mkdir(exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
VISION_MODEL = os.environ.get("WC_VISION_MODEL", "qwen2.5vl:7b")
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
