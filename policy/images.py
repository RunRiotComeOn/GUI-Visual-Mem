"""Image normalization for OpenAI-compatible chat content.

Online benches give us screenshots in different shapes:
- online-mind2web / android_world: numpy ndarray (HxWx3 or HxWx4 uint8)
- WebArenaLiteV2: bytes (raw PNG)
- offline / static: PIL.Image
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


def to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 4:
            background = np.ones_like(arr[:, :, :3], dtype=np.uint8) * 255
            alpha = arr[:, :, 3:] / 255.0
            arr = (arr[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


def image_size(image: Any) -> tuple[int, int]:
    pil = to_pil(image)
    return pil.width, pil.height


def image_to_base64_jpeg(image: Any, max_dim: int = 1024, quality: int = 75) -> str:
    pil = to_pil(image).convert("RGB")
    if max_dim is not None and max(pil.size) > max_dim:
        pil = pil.copy()
        pil.thumbnail((max_dim, max_dim))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_base64_png(image: Any) -> tuple[str, tuple[int, int]]:
    """Lossless PNG encode, returns (b64, (w, h)). Used by policy prompt where
    screenshot fidelity matters more than payload size."""
    pil = to_pil(image)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return b64, (pil.width, pil.height)


def image_to_chat_content(image: Any, *, lossy: bool = True) -> dict:
    if lossy:
        b64 = image_to_base64_jpeg(image)
        url = f"data:image/jpeg;base64,{b64}"
    else:
        b64, _ = image_to_base64_png(image)
        url = f"data:image/png;base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url}}
