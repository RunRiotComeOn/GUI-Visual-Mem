"""Screenshot normalization for OpenAI-compatible multimodal chat calls."""
from __future__ import annotations

import base64
import io
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for non-array callers.
    np = None

from PIL import Image


def to_pil(image: Any) -> Image.Image:
    """Convert common benchmark screenshot objects to ``PIL.Image``."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if np is not None and isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 4:
            background = np.ones_like(arr[:, :, :3], dtype=np.uint8) * 255
            alpha = arr[:, :, 3:] / 255.0
            arr = (arr[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"unsupported screenshot type: {type(image)!r}")


def image_size(image: Any) -> tuple[int, int]:
    pil = to_pil(image)
    return pil.width, pil.height


def image_to_base64_jpeg(image: Any, max_dim: int | None = 1024, quality: int = 75) -> str:
    pil = to_pil(image).convert("RGB")
    if max_dim is not None and max(pil.size) > max_dim:
        pil = pil.copy()
        pil.thumbnail((max_dim, max_dim))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_base64_png(image: Any) -> str:
    pil = to_pil(image)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_chat_content(image: Any, *, lossy: bool = True) -> dict:
    if lossy:
        url = f"data:image/jpeg;base64,{image_to_base64_jpeg(image)}"
    else:
        url = f"data:image/png;base64,{image_to_base64_png(image)}"
    return {"type": "image_url", "image_url": {"url": url}}
