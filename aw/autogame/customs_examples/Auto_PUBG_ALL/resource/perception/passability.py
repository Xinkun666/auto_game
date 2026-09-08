"""PIDNet-S passability segmentation for HOS RGB frames."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

from .pidnet_s import PIDNet

INPUT_SIZE = (704, 328)  # width, height; must match training
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "pidnet_s_passability.pt"


class PIDNetSPassability:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"PIDNet-S 权重不存在：{weights}")
        self.model = PIDNet(
            m=2, n=3, num_classes=2, planes=32, ppm_planes=96, head_planes=128, augment=True
        )
        checkpoint = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        self.model.eval()

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("passable 需要 HOS RGB 三通道画面")
        height, width = frame_rgb.shape[:2]
        image = cv2.resize(frame_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        tensor = torch.from_numpy(((image - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(tensor)
            logits = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
        labels = logits.argmax(1).squeeze(0).numpy().astype(np.uint8)
        return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


_MODEL: PIDNetSPassability | None = None
_MODEL_LOCK = Lock()


def _model() -> PIDNetSPassability:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = PIDNetSPassability()
        return _MODEL


def _contours(mask: np.ndarray) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(contour) < 25:
            continue
        epsilon = 0.003 * cv2.arcLength(contour, True)
        result.append(cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(int).tolist())
    return result


def passable(frame_rgb: np.ndarray) -> dict:
    """Return 0=passable / 1=obstacle labels and visual overlays for one HOS frame."""
    labels = _model().predict(frame_rgb)
    passable_mask = labels == 0
    obstacle_mask = labels == 1
    return {
        "labels": labels,
        "passable_mask": passable_mask,
        "obstacle_mask": obstacle_mask,
        "passable_ratio": round(float(passable_mask.mean()), 4),
        "obstacle_ratio": round(float(obstacle_mask.mean()), 4),
        "__visualizations__": [
            {
                "type": "mask",
                "label": "passable",
                "contours": _contours(passable_mask),
                "color_bgr": [75, 210, 40],
                "bbox_color_bgr": [75, 210, 40],
                "alpha": 0.30,
            },
            {
                "type": "mask",
                "label": "obstacle",
                "contours": _contours(obstacle_mask),
                "color_bgr": [55, 60, 245],
                "bbox_color_bgr": [55, 60, 245],
                "alpha": 0.30,
            },
        ],
    }
