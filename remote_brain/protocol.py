import base64
import json
from typing import Any, Dict

import cv2
import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def encode_frame_to_base64(frame_rgb: np.ndarray, image_format: str = "jpg", jpeg_quality: int = 85) -> str:
    fmt = image_format.lower().lstrip(".")
    if fmt not in ("jpg", "jpeg", "png"):
        raise ValueError(f"Unsupported image format: {image_format}")

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ext = ".jpg" if fmt == "jpeg" else f".{fmt}"
    params = []
    if fmt in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    ok, buffer = cv2.imencode(ext, frame_bgr, params)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buffer).decode("ascii")


def decode_base64_to_frame_rgb(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode image")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def json_dumps(payload: Dict[str, Any]) -> bytes:
    return json.dumps(to_jsonable(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def json_loads(data: bytes) -> Dict[str, Any]:
    return json.loads(data.decode("utf-8"))

