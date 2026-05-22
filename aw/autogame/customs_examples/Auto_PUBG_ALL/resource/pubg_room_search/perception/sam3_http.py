from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3 import (
    config as sam3_config,
)

logger = logging.getLogger(__name__)

DEFAULT_SAM3_HTTP_URL = "http://10.41.182.148:8001"
SAM3_HTTP_BASE_URL = os.environ.get(
    "AUTOGAME_PUBG_SAM3_HTTP_URL",
    DEFAULT_SAM3_HTTP_URL,
).rstrip("/")
SAM3_HTTP_TIMEOUT = float(os.environ.get("AUTOGAME_PUBG_SAM3_HTTP_TIMEOUT", "60"))
SAM3_MASK_FILL_STRATEGY = os.environ.get(
    "AUTOGAME_PUBG_SAM3_MASK_FILL_STRATEGY",
    "black",
)


@dataclass
class Sam3RemoteResult:
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask: np.ndarray
    segmented_bgr: Optional[np.ndarray] = None
    cropped_bgr: Optional[np.ndarray] = None
    cropped_mask: Optional[np.ndarray] = None
    sam3_inference_ms: Optional[float] = None

    @property
    def mask_area(self) -> int:
        return int((self.mask > 0).sum()) if self.mask is not None else 0

    def to_summary_dict(self) -> dict:
        return {
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "score": float(self.score),
            "mask_shape": list(self.mask.shape) if self.mask is not None else [],
            "mask_area": self.mask_area,
            "sam3_inference_ms": self.sam3_inference_ms,
        }

    def to_visualization_dict(self, label: str) -> dict:
        return {
            "type": "sam3_mask",
            "coord": "local",
            "label": label,
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "score": float(self.score),
            "mask_area": self.mask_area,
            "color_bgr": [0, 255, 255],
            "bbox_color_bgr": [0, 255, 0],
            "alpha": 0.45,
            "contours": _mask_to_contours(self.mask),
        }


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def result_to_dict(result: Sam3RemoteResult | None, *, label: str = "sam3") -> dict:
    if result is None:
        return {"ok": False, "result": None, "__visualizations__": []}
    return {
        "ok": True,
        "result": result.to_summary_dict(),
        "__visualizations__": [result.to_visualization_dict(label)],
    }


def segment_house(
    img: np.ndarray,
    min_mask_area_ratio: Optional[float] = None,
) -> Optional[Sam3RemoteResult]:
    results = segment_image(
        img,
        prompt=sam3_config.SAM3_HOUSE_PROMPT,
        max_masks=1,
        min_mask_area_ratio=0.03 if min_mask_area_ratio is None else min_mask_area_ratio,
        label="house",
    )
    return results[0] if results else None


def segment_door(
    img: np.ndarray,
    min_mask_area_ratio: Optional[float] = None,
) -> Optional[Sam3RemoteResult]:
    results = segment_image(
        img,
        prompt=sam3_config.SAM3_DOOR_PROMPT,
        max_masks=1,
        min_mask_area_ratio=0.03 if min_mask_area_ratio is None else min_mask_area_ratio,
        label="door",
    )
    return results[0] if results else None


def segment_door_all(
    img: np.ndarray,
    *,
    max_masks: int = 8,
    min_mask_area_ratio: float = 0.001,
) -> list[Sam3RemoteResult]:
    return segment_image(
        img,
        prompt=sam3_config.SAM3_DOOR_PROMPT,
        max_masks=max_masks,
        min_mask_area_ratio=min_mask_area_ratio,
        label="door_all",
    )


def segment_image(
    img: np.ndarray,
    *,
    prompt: str,
    max_masks: int,
    min_mask_area_ratio: float,
    label: str,
) -> list[Sam3RemoteResult]:
    image_bgr = to_bgr(img)
    if image_bgr is None:
        return []

    payload = _post_multipart_png(
        f"{SAM3_HTTP_BASE_URL}/segment",
        _encode_png(image_bgr),
        {
            "prompt": prompt,
            "max_masks": str(max_masks),
            "min_mask_area_ratio": str(min_mask_area_ratio),
            "mask_fill_strategy": SAM3_MASK_FILL_STRATEGY,
        },
    )
    if not payload.get("success", False):
        logger.warning("SAM3 HTTP response failed: %s", payload)
        return []

    return [_decode_result(item, image_bgr) for item in payload.get("results") or []]


def _encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode SAM3 request image as PNG.")
    return buffer.tobytes()


def _post_multipart_png(url: str, image_png: bytes, fields: dict[str, str]) -> dict:
    boundary = uuid.uuid4().hex
    body_parts: list[bytes] = []

    for name, value in fields.items():
        body_parts.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "ascii"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    body_parts.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                'Content-Disposition: form-data; name="image"; '
                'filename="frame.png"\r\n'
            ).encode("ascii"),
            b"Content-Type: image/png\r\n\r\n",
            image_png,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )

    request = urllib.request.Request(
        url,
        data=b"".join(body_parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=SAM3_HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _decode_result(item: dict, image_bgr: np.ndarray) -> Sam3RemoteResult:
    bbox = tuple(int(v) for v in item.get("bbox_xyxy", (0, 0, 0, 0)))
    if len(bbox) != 4:
        bbox = (0, 0, 0, 0)
    mask = _decode_png_base64(item.get("mask_png_base64"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    cropped_bgr = _decode_png_base64(item.get("cropped_bgr_png_base64"), cv2.IMREAD_COLOR)
    cropped_mask = _decode_png_base64(
        item.get("cropped_mask_png_base64"),
        cv2.IMREAD_GRAYSCALE,
    )
    if cropped_bgr is None:
        cropped_bgr = _crop_bbox(image_bgr, bbox)
    if cropped_mask is None:
        cropped_mask = _crop_bbox(mask, bbox)

    inference_ms = item.get("inference_ms")
    return Sam3RemoteResult(
        bbox_xyxy=bbox,
        score=float(item.get("score", 0.0)),
        mask=(mask > 0).astype(np.uint8),
        segmented_bgr=_decode_png_base64(
            item.get("segmented_bgr_png_base64"),
            cv2.IMREAD_COLOR,
        ),
        cropped_bgr=cropped_bgr,
        cropped_mask=(cropped_mask > 0).astype(np.uint8)
        if cropped_mask is not None
        else None,
        sam3_inference_ms=float(inference_ms) if inference_ms is not None else None,
    )


def _decode_png_base64(value: Optional[str], flags: int) -> Optional[np.ndarray]:
    if not value:
        return None
    raw = base64.b64decode(value)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


def _crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    if image is None or len(bbox) != 4:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def _mask_to_contours(mask: np.ndarray) -> list[list[list[int]]]:
    if mask is None:
        return []
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    encoded: list[list[list[int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        encoded.append([[int(x), int(y)] for x, y in approx])
    return encoded
