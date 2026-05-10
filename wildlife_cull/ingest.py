import hashlib
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image, ImageOps

from . import db, focus
from .config import IMAGE_EXTS, RAW_EXTS, PREVIEW_DIR, PREVIEW_MAX_SIZE, THUMB_MAX_SIZE


def extract_capture_time(src: Path) -> Optional[float]:
    """Pull EXIF DateTimeOriginal via exiftool and return it as a Unix
    timestamp. Falls back to None when exiftool isn't installed, the
    file lacks EXIF, or the date is malformed. Burst detection needs
    real capture time — file mtime is often wrong on copied archives."""
    try:
        result = subprocess.run(
            ["exiftool", "-s3", "-DateTimeOriginal", "-d", "%Y-%m-%d %H:%M:%S", str(src)],
            capture_output=True,
            check=False,
            timeout=5,
            text=True,
        )
        out = (result.stdout or "").strip()
        if not out:
            return None
        dt = datetime.strptime(out, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
        return None


def iter_images(folder: Path, recursive: bool = True) -> Iterator[Path]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    walker = folder.rglob("*") if recursive else folder.iterdir()
    for p in walker:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def _stable_id(path: Path) -> str:
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return h


def _extract_raw_preview(src: Path, dst: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-b",
                "-JpgFromRaw",
                "-PreviewImage",
                "-OtherImage",
                str(src),
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            dst.write_bytes(result.stdout)
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def _make_preview_and_thumb(src: Path, image_id: str) -> tuple[Optional[Path], Optional[Path], Optional[int], Optional[int]]:
    preview_path = PREVIEW_DIR / f"{image_id}_preview.jpg"
    thumb_path = PREVIEW_DIR / f"{image_id}_thumb.jpg"

    is_raw = src.suffix.lower() in RAW_EXTS
    source_img_path: Optional[Path] = None

    if is_raw:
        raw_extracted = PREVIEW_DIR / f"{image_id}_raw.jpg"
        if _extract_raw_preview(src, raw_extracted):
            source_img_path = raw_extracted
    else:
        source_img_path = src

    if source_img_path is None:
        return None, None, None, None

    try:
        with Image.open(source_img_path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            preview = im.copy()
            preview.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.LANCZOS)
            preview.convert("RGB").save(preview_path, "JPEG", quality=85, optimize=True)

            thumb = im.copy()
            thumb.thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE), Image.LANCZOS)
            thumb.convert("RGB").save(thumb_path, "JPEG", quality=82)
    except Exception:
        return None, None, None, None
    finally:
        if is_raw and source_img_path is not None and source_img_path != src:
            source_img_path.unlink(missing_ok=True)

    return preview_path, thumb_path, w, h


def _read_existing_xmp_rating(image_path: Path) -> Optional[int]:
    sidecar = Path(str(image_path) + ".xmp")
    if not sidecar.exists():
        return None
    try:
        text = sidecar.read_text(errors="ignore")
    except Exception:
        return None
    import re
    m = re.search(r"xmp:Rating[^0-9-]*([0-5])", text)
    if m:
        return int(m.group(1))
    return None


def ingest_folder(folder: str, recursive: bool = True) -> dict:
    root = Path(folder).expanduser().resolve()
    started = time.time()
    seen = 0
    new = 0
    skipped = 0

    with db.connect() as conn:
        for img_path in iter_images(root, recursive=recursive):
            seen += 1
            stat = img_path.stat()
            image_id = _stable_id(img_path)

            existing = conn.execute(
                "SELECT id, mtime, file_size FROM images WHERE path=?",
                (str(img_path),),
            ).fetchone()
            if existing and existing["mtime"] == stat.st_mtime and existing["file_size"] == stat.st_size:
                skipped += 1
                continue

            preview, thumb, w, h = _make_preview_and_thumb(img_path, image_id)
            if preview is None:
                skipped += 1
                continue

            existing_rating = _read_existing_xmp_rating(img_path)
            f_score = focus.focus_score(preview)
            f_label = focus.focus_label(f_score)
            cap = extract_capture_time(img_path)

            row = {
                "path": str(img_path),
                "filename": img_path.name,
                "folder": str(img_path.parent),
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "capture_time": cap,
                "is_raw": 1 if img_path.suffix.lower() in RAW_EXTS else 0,
                "preview_path": str(preview),
                "thumb_path": str(thumb),
                "focus_score": f_score,
                "focus_label": f_label,
                "width": w,
                "height": h,
                "ingested_at": time.time(),
                "ai_status": "pending",
            }
            if existing_rating is not None:
                row["user_rating"] = existing_rating
                row["user_updated_at"] = time.time()

            db.upsert_image(conn, row)
            new += 1

    return {
        "folder": str(root),
        "seen": seen,
        "new": new,
        "skipped": skipped,
        "elapsed_sec": round(time.time() - started, 2),
    }
