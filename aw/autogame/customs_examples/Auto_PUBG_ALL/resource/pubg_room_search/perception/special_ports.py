from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
import uuid

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3 import (
    config as sam3_config,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.yolo_embedded import (
    detect_as_forward_scene,
    get_yolo_perception,
)

logger = logging.getLogger(__name__)

SAM3_HTTP_BASE_URL = os.environ.get(
    "AUTOGAME_PUBG_SAM3_HTTP_URL",
    "http://10.41.182.148:8001",
).rstrip("/")
SAM3_HTTP_TIMEOUT = float(os.environ.get("AUTOGAME_PUBG_SAM3_HTTP_TIMEOUT", "60"))
SAM3_MASK_FILL_STRATEGY = os.environ.get(
    "AUTOGAME_PUBG_SAM3_MASK_FILL_STRATEGY",
    "black",
)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode SAM3 request image as PNG.")
    return buffer.tobytes()


def _decode_mask_area(item: dict) -> tuple[list[int], int]:
    mask_b64 = item.get("mask_png_base64")
    if not mask_b64:
        return [], 0

    raw = base64.b64decode(mask_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], 0
    return list(mask.shape), int((mask > 0).sum())


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


def _sam3_http_segment(
    img: np.ndarray,
    *,
    prompt: str,
    max_masks: int,
    min_mask_area_ratio: float,
) -> list[dict]:
    image_bgr = _to_bgr(img)
    if image_bgr is None:
        return []

    image_png = _encode_png(image_bgr)
    payload = _post_multipart_png(
        f"{SAM3_HTTP_BASE_URL}/segment",
        image_png,
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
    return payload.get("results") or []


def _sam3_result_to_dict(item: dict | None) -> dict:
    if not item:
        return {"ok": False, "result": None}

    mask_shape, mask_area = _decode_mask_area(item)
    return {
        "ok": True,
        "result": {
            "bbox_xyxy": [int(v) for v in item.get("bbox_xyxy", [])],
            "score": float(item.get("score", 0.0)),
            "mask_shape": mask_shape,
            "mask_area": mask_area,
            "sam3_inference_ms": item.get("inference_ms"),
        },
    }


def pubg_yolo_detect(img: np.ndarray):
    return detect_as_forward_scene(_to_bgr(img))


def pubg_yolo_detect_detail(img: np.ndarray):
    return get_yolo_perception().detect(_to_bgr(img)).to_dict()


def pubg_yolo_classify(img: np.ndarray):
    return get_yolo_perception().classify(_to_bgr(img)).to_dict()


def pubg_yolo_detect_and_classify(img: np.ndarray):
    return get_yolo_perception().classify_and_detect(_to_bgr(img)).to_dict()


def pubg_yolo_reset_tracker(_img: np.ndarray = None):
    get_yolo_perception().reset_tracker()
    return {"ok": True, "status": "reset_requested"}


def pubg_sam3_segment_house(img: np.ndarray):
    try:
        results = _sam3_http_segment(
            img,
            prompt=sam3_config.SAM3_HOUSE_PROMPT,
            max_masks=1,
            min_mask_area_ratio=0.03,
        )
    except Exception as exc:
        logger.warning("SAM3 HTTP house segmentation failed: %s", exc)
        return {"ok": False, "result": None, "error": str(exc)}
    return _sam3_result_to_dict(results[0] if results else None)


def pubg_sam3_segment_door(img: np.ndarray):
    try:
        results = _sam3_http_segment(
            img,
            prompt=sam3_config.SAM3_DOOR_PROMPT,
            max_masks=1,
            min_mask_area_ratio=0.03,
        )
    except Exception as exc:
        logger.warning("SAM3 HTTP door segmentation failed: %s", exc)
        return {"ok": False, "result": None, "error": str(exc)}
    return _sam3_result_to_dict(results[0] if results else None)


def pubg_sam3_segment_door_all(img: np.ndarray):
    try:
        results = _sam3_http_segment(
            img,
            prompt=sam3_config.SAM3_DOOR_PROMPT,
            max_masks=8,
            min_mask_area_ratio=0.001,
        )
    except Exception as exc:
        logger.warning("SAM3 HTTP door_all segmentation failed: %s", exc)
        return {"ok": False, "results": [], "error": str(exc)}
    return {
        "ok": True,
        "results": [_sam3_result_to_dict(result)["result"] for result in results],
    }
