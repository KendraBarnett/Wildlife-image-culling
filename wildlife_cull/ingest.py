import hashlib
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image, ImageOps

from . import db, focus
from .config import IMAGE_EXTS, RAW_EXTS, PREVIEW_DIR, PREVIEW_MAX_SIZE, THUMB_MAX_SIZE


_EXIF_TAGS = [
    "DateTimeOriginal",
    "Model",                 # camera body
    "LensModel",             # lens (modern cameras)
    "Lens",                  # lens (older cameras)
    "FocalLength",           # "500.0 mm" or just "500"
    "ISO",
    "FNumber",               # aperture, "7.1" or "f/7.1"
    "ShutterSpeed",          # "1/2000" preferred
    "ExposureTime",          # fallback for ShutterSpeed
    "ExposureCompensation",  # "+0.3" or "-1"
]


def _parse_float(s: str) -> Optional[float]:
    """Parse '500.0 mm', 'f/7.1', '+0.3' etc. into a float. exiftool
    formats vary across cameras; this is permissive."""
    if not s:
        return None
    s = s.strip()
    for prefix in ("f/", "F/", "f"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in (" mm", "mm", " s", " sec"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    s = s.strip("+ ")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(s: str) -> Optional[int]:
    v = _parse_float(s)
    return int(v) if v is not None else None


def extract_exif(src: Path) -> dict:
    """Pull every EXIF field we care about in a single exiftool call.
    Returns a dict with keys: capture_time (float unix ts), camera,
    lens, focal_length (mm), iso, aperture (f-number), shutter (text
    fraction), exposure_comp (EV). Missing fields are simply absent.

    Single exiftool call per image keeps ingest fast — pulling 10 tags
    is the same cost as pulling 1."""
    args = ["exiftool", "-s3"]
    for t in _EXIF_TAGS:
        args.append(f"-{t}")
    args.extend(["-d", "%Y-%m-%d %H:%M:%S", str(src)])
    try:
        result = subprocess.run(
            args, capture_output=True, check=False, timeout=8, text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}
    out = (result.stdout or "").splitlines()
    # exiftool with -s3 returns one value per line, in the same order
    # as the tag flags. Missing tags become blank lines, which is why
    # we walk by index instead of trying to be clever.
    out = [line.strip() for line in out]
    values = {}
    for i, tag in enumerate(_EXIF_TAGS):
        values[tag] = out[i] if i < len(out) else ""

    parsed: dict = {}
    if values.get("DateTimeOriginal"):
        try:
            dt = datetime.strptime(values["DateTimeOriginal"], "%Y-%m-%d %H:%M:%S")
            parsed["capture_time"] = dt.timestamp()
        except ValueError:
            pass
    if values.get("Model"):
        parsed["camera"] = values["Model"]
    lens = values.get("LensModel") or values.get("Lens")
    if lens:
        parsed["lens"] = lens
    fl = _parse_float(values.get("FocalLength", ""))
    if fl is not None:
        parsed["focal_length"] = fl
    iso = _parse_int(values.get("ISO", ""))
    if iso is not None:
        parsed["iso"] = iso
    ap = _parse_float(values.get("FNumber", ""))
    if ap is not None:
        parsed["aperture"] = ap
    # ShutterSpeed is already a nice fraction ("1/2000"); ExposureTime is
    # the same value in decimal. Prefer the fraction for display.
    sh = values.get("ShutterSpeed") or values.get("ExposureTime")
    if sh:
        parsed["shutter"] = sh
    ec = _parse_float(values.get("ExposureCompensation", ""))
    if ec is not None:
        parsed["exposure_comp"] = ec
    return parsed


def extract_capture_time(src: Path) -> Optional[float]:
    """Back-compat helper: still useful for the existing backfill, but
    new code should call extract_exif() instead and use ['capture_time']."""
    return extract_exif(src).get("capture_time")


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
            exif = extract_exif(img_path)

            row = {
                "path": str(img_path),
                "filename": img_path.name,
                "folder": str(img_path.parent),
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "capture_time": exif.get("capture_time"),
                "exif_camera": exif.get("camera"),
                "exif_lens": exif.get("lens"),
                "exif_focal_length": exif.get("focal_length"),
                "exif_iso": exif.get("iso"),
                "exif_aperture": exif.get("aperture"),
                "exif_shutter": exif.get("shutter"),
                "exif_exposure_comp": exif.get("exposure_comp"),
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
