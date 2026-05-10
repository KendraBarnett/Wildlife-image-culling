"""Deterministic out-of-focus detection.

We use the variance of the Laplacian, the classic image-processing metric for
sharpness. It is fast, deterministic, and correlates very well with human
perception of focus — much more reliable than asking a vision LLM whether a
photo is in focus.

For a proper sharpness score we run it on a center crop of the preview at a
moderate size: full-res introduces sensor noise that inflates variance for
soft images, and a tiny thumbnail loses the high-frequency detail that
distinguishes sharp from blurry. 1024 px center crop hits the sweet spot.

Output: a single float (focus_score). Higher = sharper. Typical ranges on
modern wildlife photography:
    - very sharp / detailed feathers / fur:   500-3000+
    - normal sharp:                            150-500
    - mildly soft / motion-blurred subject:    40-150
    - clearly out-of-focus:                    < 40
We expose `focus_label(score)` to map this to a coarse tier the UI can show.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


CENTER_CROP_SIZE = 1024


def focus_score(preview_path: Path) -> Optional[float]:
    try:
        from PIL import Image
        with Image.open(preview_path) as im:
            im = im.convert("L")
            w, h = im.size
            side = min(w, h)
            target = min(side, CENTER_CROP_SIZE)
            left = (w - target) // 2
            top = (h - target) // 2
            crop = im.crop((left, top, left + target, top + target))
            arr = np.asarray(crop, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    # 4-neighbour Laplacian: lap[y,x] = N + S + E + W - 4*center.
    lap = (
        arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
        - 4.0 * arr[1:-1, 1:-1]
    )
    return float(lap.var())


def focus_label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score < 40:
        return "out_of_focus"
    if score < 150:
        return "soft"
    if score < 500:
        return "sharp"
    return "very_sharp"
