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

PREVIEW_MAX_SIZE = 1600
THUMB_MAX_SIZE = 400
