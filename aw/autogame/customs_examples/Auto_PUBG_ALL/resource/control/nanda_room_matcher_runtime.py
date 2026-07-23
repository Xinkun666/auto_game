"""auto_game 内置的南大 DINOv3 + MLP 房型匹配运行时。

该模块消费 ``sam3_tiny`` 生成的房屋立面、building mask、door frame
和 window 多目标 mask，实现南大最新版完整房型匹配链路：

1. DINOv3 masked-patch pooling 提取彩色/灰度房屋特征；
2. 对房型模板做向量召回并展开候选房型的全部模板；
3. 使用门框和窗户 mask 构建与南大一致的门窗结构向量；
4. 使用 ``rgb_mlp_struct_v7.pkl`` 做 MLP 重排；
5. 使用南大阈值拒绝弱匹配，再返回房型绑定的回放 DSL。

当前游戏画面的三类分割都来自搜房阶段的 SAM3 special_area，不使用 HTTP。
模板门窗结构在首次入选候选时使用同一个进程内 SAM3 模型计算并落盘缓存。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import pickle
import re
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np


LOGGER = logging.getLogger("NandaIntegratedRoomMatcher")

EMBEDDING_DIM = 1024
MIN_MASKED_PATCHES = 4
ROOM_SHORTLIST_TEMPLATE_MULTIPLIER = 4
STRUCTURE_VECTOR_MAX_ROWS = 4
STRUCTURE_VECTOR_CHANNELS = 4
STRUCTURE_VECTOR_GLOBAL_DIM = 6
STRUCTURE_VECTOR_DIM = (
    STRUCTURE_VECTOR_GLOBAL_DIM
    + STRUCTURE_VECTOR_MAX_ROWS * 3 * STRUCTURE_VECTOR_CHANNELS
)
FACADE_STRUCTURE_CACHE_VERSION = 1
FACADE_OPENING_MAX_MASKS = 12
FACADE_OPENING_MIN_MASK_AREA_RATIO = 0.001
FACADE_WINDOW_MERGE_Y_OVERLAP = 0.55
FACADE_WINDOW_MERGE_MAX_GAP_RATIO = 12.0 / 1024.0
FACADE_WINDOW_MERGE_HEIGHT_RATIO = 1.6
FACADE_DOOR_UNCERTAIN_SCORE_BELOW = 0.6
FACADE_WINDOW_UNCERTAIN_SCORE_BELOW = 0.6
FACADE_UNCERTAIN_DOOR_WEIGHT = 0.50
FACADE_UNCERTAIN_WINDOW_WEIGHT = 0.25

# 与南大最新版 control_proxy/pubg_room_explore/sam3/config.py 保持一致。
FACADE_TOP_K = 16
FACADE_STRUCT_SCORE_ALPHA = 0.30
FACADE_LOW_MLP_CONFIDENCE_SCORE = 0.05
FACADE_NO_MATCH_MIN_SCORE = 0.25
FACADE_NO_MATCH_MIN_DINO_SCORE = 0.55
FACADE_NO_MATCH_STRICT_DINO_SCORE = 0.65
FACADE_NO_MATCH_STRONG_STRUCTURE_SCORE = 0.45
FACADE_NO_MATCH_MIN_MARGIN = 0.0

RUNTIME_CACHE_VERSION = 1
INPUT_PREPROCESS_VERSION = "dino_input_no_black_holes_v2"


def _resource_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NandaMatcherAssetPaths:
    """进程内房型匹配所需的三个独立资产入口。"""

    dino_model_dir: Path
    mlp_model_path: Path
    room_library_path: Path

    @classmethod
    def auto_game_defaults(cls) -> "NandaMatcherAssetPaths":
        resource_root = _resource_root()
        weight_root = resource_root / "weights" / "nanda_room_matcher"
        return cls(
            dino_model_dir=weight_root / "dinov3_vitl16",
            mlp_model_path=weight_root / "rgb_mlp_struct_v7.pkl",
            room_library_path=resource_root / "nanda_room_library",
        )

    @classmethod
    def from_nanda_project(cls, project_root: Path) -> "NandaMatcherAssetPaths":
        package_root = (
            project_root.expanduser().resolve()
            / "control_proxy"
            / "src"
            / "gametest_proxy"
            / "pubg_room_explore"
        )
        return cls(
            dino_model_dir=(package_root / "img_similarity" / "dinov3_vitl16"),
            mlp_model_path=package_root / "models" / "rgb_mlp_struct_v7.pkl",
            room_library_path=package_root / "room_library",
        )

    def resolved(self) -> "NandaMatcherAssetPaths":
        return NandaMatcherAssetPaths(
            dino_model_dir=self.dino_model_dir.expanduser().resolve(),
            mlp_model_path=self.mlp_model_path.expanduser().resolve(),
            room_library_path=self.room_library_path.expanduser().resolve(),
        )

    def validate(self) -> None:
        paths = self.resolved()
        required_dino_files = (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        )
        missing = [
            str(paths.dino_model_dir / name)
            for name in required_dino_files
            if not (paths.dino_model_dir / name).is_file()
        ]
        if not paths.mlp_model_path.is_file():
            missing.append(str(paths.mlp_model_path))
        rooms_dir = paths.room_library_path / "rooms"
        if not rooms_dir.is_dir():
            missing.append(str(rooms_dir))
        if missing:
            raise FileNotFoundError("南大房型匹配缺少资产: " + ", ".join(missing))
        _ensure_real_model_file(
            paths.dino_model_dir / "model.safetensors", "DINOv3 权重",
        )
        _ensure_real_model_file(paths.mlp_model_path, "南大 MLP 权重")

    def to_jsonable(self) -> Dict[str, str]:
        paths = self.resolved()
        return {
            "dino_model_dir": str(paths.dino_model_dir),
            "mlp_model_path": str(paths.mlp_model_path),
            "room_library_path": str(paths.room_library_path),
        }


def _ensure_real_model_file(path: Path, label: str) -> None:
    with path.open("rb") as model_file:
        header = model_file.read(256)
    if path.stat().st_size < 1024 or header.startswith(
        b"version https://git-lfs.github.com/spec/"
    ):
        raise RuntimeError(f"{label}不是实际模型文件（可能仍是 Git LFS 指针）: {path}")


def _natural_key(text: str) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _normalize_embedding(value: np.ndarray) -> np.ndarray:
    embedding = np.asarray(value, dtype="float32").reshape(-1)
    if embedding.size != EMBEDDING_DIM:
        raise ValueError(
            f"DINOv3 特征维度异常: got={embedding.size}, expected={EMBEDDING_DIM}"
        )
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("DINOv3 返回了无效的零特征")
    return np.ascontiguousarray(embedding / norm, dtype="float32")


def _image_md5(image: Optional[np.ndarray]) -> Optional[str]:
    if image is None:
        return None
    digest = hashlib.md5()
    digest.update(repr(image.shape).encode("utf-8"))
    digest.update(str(image.dtype).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _normalize_mask(
    mask: Optional[np.ndarray], shape_hw: Tuple[int, int]
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    normalized = np.asarray(mask)
    if normalized.ndim == 3:
        normalized = np.any(normalized != 0, axis=2).astype(np.uint8)
    if normalized.ndim != 2:
        raise ValueError(f"房屋 mask 维度异常: {normalized.shape}")
    if normalized.shape != shape_hw:
        normalized = cv2.resize(
            normalized.astype(np.uint8),
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (normalized > 0).astype(np.uint8) * 255


def _fill_mask_holes(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or not np.any(mask):
        return mask
    mask_bool = mask.astype(bool)
    inverse = (~mask_bool).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, connectivity=8)
    if count <= 1:
        return mask
    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    filled = mask_bool.copy()
    for label in range(1, count):
        if label not in border_labels:
            filled[labels == label] = True
    return filled.astype(np.uint8) * 255


def _build_masked_dino_input(
    base_image: np.ndarray, mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if base_image.ndim != 3 or base_image.shape[2] != 3:
        raise ValueError(f"DINO 输入必须是 HxWx3 BGR 图像: {base_image.shape}")
    aligned_mask = _fill_mask_holes(_normalize_mask(mask, base_image.shape[:2]))
    if aligned_mask is None:
        return base_image.copy(), None
    output = np.zeros_like(base_image)
    mask_bool = aligned_mask.astype(bool)
    output[mask_bool] = base_image[mask_bool]
    return output, aligned_mask


def _gray_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@dataclass(frozen=True)
class FacadeMaskObservation:
    """One SAM3 opening mask in the cropped building coordinate system."""

    mask: np.ndarray
    score: float


@dataclass(frozen=True)
class FacadeOpening:
    kind: str
    bbox_xyxy: Tuple[float, float, float, float]
    score: float = 1.0
    reason: Optional[str] = None

    @property
    def width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class FacadeStructureFeature:
    anchor_door: Optional[FacadeOpening]
    row_bounds: List[Tuple[float, float]]
    structure_matrix: np.ndarray
    doors: List[FacadeOpening]
    uncertain_doors: List[FacadeOpening]
    windows_raw: List[FacadeOpening]
    windows: List[FacadeOpening]
    uncertain_windows: List[FacadeOpening]

    @property
    def available(self) -> bool:
        return self.anchor_door is not None


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs) or not len(ys):
        raise ValueError("门窗 mask 为空")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _opening_intersection(first: FacadeOpening, second: FacadeOpening) -> float:
    ax1, ay1, ax2, ay2 = first.bbox_xyxy
    bx1, by1, bx2, by2 = second.bbox_xyxy
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _opening_iou(first: FacadeOpening, second: FacadeOpening) -> float:
    intersection = _opening_intersection(first, second)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def _dedupe_openings(openings: List[FacadeOpening]) -> List[FacadeOpening]:
    kept: List[FacadeOpening] = []
    for opening in sorted(openings, key=lambda item: (-item.score, -item.area)):
        duplicate = False
        for existing in kept:
            intersection = _opening_intersection(opening, existing)
            smaller = min(opening.area, existing.area)
            if _opening_iou(opening, existing) >= 0.55 or (
                smaller > 0 and intersection / smaller >= 0.85
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(opening)
    return sorted(kept, key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))


def _normalize_facade_mask(
    facade_mask: Optional[np.ndarray],
    shape_hw: Tuple[int, int],
) -> np.ndarray:
    height, width = shape_hw
    if facade_mask is None:
        return np.ones((height, width), dtype=bool)
    normalized = _normalize_mask(facade_mask, shape_hw)
    if normalized is None or not np.any(normalized):
        return np.ones((height, width), dtype=bool)
    return normalized.astype(bool)


def _opening_from_observation(
    observation: FacadeMaskObservation,
    *,
    kind: str,
    facade_mask: np.ndarray,
) -> Optional[FacadeOpening]:
    raw_mask = np.asarray(observation.mask)
    if raw_mask.ndim == 3:
        raw_mask = np.any(raw_mask != 0, axis=2)
    if raw_mask.shape != facade_mask.shape:
        raw_mask = cv2.resize(
            raw_mask.astype(np.uint8),
            (facade_mask.shape[1], facade_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    raw_mask = raw_mask.astype(bool)
    raw_area = int(raw_mask.sum())
    if raw_area <= 0:
        return None
    inside_mask = raw_mask & facade_mask
    if int(inside_mask.sum()) / float(raw_area) <= 0.5 or not inside_mask.any():
        return None
    x1, y1, x2, y2 = _bbox_from_mask(inside_mask)
    return FacadeOpening(
        kind=kind,
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        score=float(observation.score),
    )


def _normalize_openings(
    openings: List[FacadeOpening],
    facade_bbox: Tuple[int, int, int, int],
) -> List[FacadeOpening]:
    x1, y1, x2, y2 = facade_bbox
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    normalized: List[FacadeOpening] = []
    for opening in openings:
        ox1, oy1, ox2, oy2 = opening.bbox_xyxy
        clipped_x1 = max(float(x1), ox1)
        clipped_y1 = max(float(y1), oy1)
        clipped_x2 = min(float(x2), ox2)
        clipped_y2 = min(float(y2), oy2)
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue
        normalized.append(
            FacadeOpening(
                kind=opening.kind,
                bbox_xyxy=(
                    (clipped_x1 - x1) / width,
                    (clipped_y1 - y1) / height,
                    (clipped_x2 - x1) / width,
                    (clipped_y2 - y1) / height,
                ),
                score=opening.score,
                reason=opening.reason,
            )
        )
    return normalized


def _retag_opening(
    opening: FacadeOpening,
    *,
    kind: str,
    reason: str,
) -> FacadeOpening:
    return FacadeOpening(
        kind=kind,
        bbox_xyxy=opening.bbox_xyxy,
        score=opening.score,
        reason=reason,
    )


def _merge_windows(windows: List[FacadeOpening]) -> List[FacadeOpening]:
    windows = _dedupe_openings(windows)
    changed = True
    while changed:
        changed = False
        merged: List[FacadeOpening] = []
        used: set[int] = set()
        for index, window in enumerate(windows):
            if index in used:
                continue
            current = window
            used.add(index)
            for other_index in range(index + 1, len(windows)):
                if other_index in used:
                    continue
                other = windows[other_index]
                overlap = max(
                    0.0,
                    min(current.bbox_xyxy[3], other.bbox_xyxy[3])
                    - max(current.bbox_xyxy[1], other.bbox_xyxy[1]),
                )
                y_overlap_ratio = overlap / max(
                    1e-6, min(current.height, other.height)
                )
                shorter = max(1e-6, min(current.height, other.height))
                taller = max(current.height, other.height)
                horizontal_gap = max(
                    0.0,
                    max(current.bbox_xyxy[0], other.bbox_xyxy[0])
                    - min(current.bbox_xyxy[2], other.bbox_xyxy[2]),
                )
                if (
                    y_overlap_ratio < FACADE_WINDOW_MERGE_Y_OVERLAP
                    or taller / shorter > FACADE_WINDOW_MERGE_HEIGHT_RATIO
                    or horizontal_gap > FACADE_WINDOW_MERGE_MAX_GAP_RATIO
                ):
                    continue
                current = FacadeOpening(
                    kind="window",
                    bbox_xyxy=(
                        min(current.bbox_xyxy[0], other.bbox_xyxy[0]),
                        min(current.bbox_xyxy[1], other.bbox_xyxy[1]),
                        max(current.bbox_xyxy[2], other.bbox_xyxy[2]),
                        max(current.bbox_xyxy[3], other.bbox_xyxy[3]),
                    ),
                    score=max(current.score, other.score),
                )
                used.add(other_index)
                changed = True
            merged.append(current)
        windows = sorted(
            merged, key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0])
        )
    return windows


def _choose_anchor_door(
    doors: List[FacadeOpening],
    uncertain_doors: List[FacadeOpening],
) -> Optional[FacadeOpening]:
    if doors:
        return min(
            doors,
            key=lambda item: (
                abs(item.center[0] - 0.5),
                -item.area,
                -item.score,
            ),
        )
    if uncertain_doors:
        return min(
            uncertain_doors,
            key=lambda item: (
                -item.score,
                abs(item.center[0] - 0.5),
                -item.area,
            ),
        )
    return None


def _dynamic_facade_rows(
    openings: List[FacadeOpening],
) -> List[Tuple[float, float, List[FacadeOpening]]]:
    rows: List[List[FacadeOpening]] = []
    for opening in openings:
        matching_indexes = []
        for index, row in enumerate(rows):
            if any(
                max(opening.bbox_xyxy[1], existing.bbox_xyxy[1])
                <= min(opening.bbox_xyxy[3], existing.bbox_xyxy[3])
                for existing in row
            ):
                matching_indexes.append(index)
        if not matching_indexes:
            rows.append([opening])
            continue
        first_index = matching_indexes[0]
        rows[first_index].append(opening)
        for index in reversed(matching_indexes[1:]):
            rows[first_index].extend(rows.pop(index))
    row_data = [
        (
            min(opening.bbox_xyxy[1] for opening in row),
            max(opening.bbox_xyxy[3] for opening in row),
            row,
        )
        for row in rows
    ]
    return sorted(row_data, key=lambda item: (item[1], item[0]), reverse=True)


def _build_facade_feature(
    *,
    doors: List[FacadeOpening],
    uncertain_doors: List[FacadeOpening],
    windows_raw: List[FacadeOpening],
    windows: List[FacadeOpening],
    uncertain_windows: List[FacadeOpening],
) -> FacadeStructureFeature:
    anchor = _choose_anchor_door(doors, uncertain_doors)
    all_openings = doors + uncertain_doors + windows + uncertain_windows
    rows = _dynamic_facade_rows(all_openings) if anchor is not None else []
    matrix = np.zeros((len(rows), 3, 4), dtype="float32")
    if anchor is not None:
        for channel, channel_openings in enumerate(
            (doors, uncertain_doors, windows, uncertain_windows)
        ):
            for opening in channel_openings:
                row_index = next(
                    index
                    for index, (_top, _bottom, row) in enumerate(rows)
                    if any(opening is row_opening for row_opening in row)
                )
                if opening.bbox_xyxy[2] < anchor.bbox_xyxy[0]:
                    side = 0
                elif opening.bbox_xyxy[0] > anchor.bbox_xyxy[2]:
                    side = 2
                else:
                    side = 1
                matrix[row_index, side, channel] += 1.0
    return FacadeStructureFeature(
        anchor_door=anchor,
        row_bounds=[(top, bottom) for top, bottom, _ in rows],
        structure_matrix=matrix,
        doors=doors,
        uncertain_doors=uncertain_doors,
        windows_raw=windows_raw,
        windows=windows,
        uncertain_windows=uncertain_windows,
    )


def build_facade_structure(
    facade_mask: Optional[np.ndarray],
    door_observations: Iterable[FacadeMaskObservation],
    window_observations: Iterable[FacadeMaskObservation],
) -> FacadeStructureFeature:
    """Convert SAM3 building/door/window masks to Nanda's facade feature contract."""
    if facade_mask is None:
        raise ValueError("构建南大门窗结构缺少 building mask")
    shape_hw = tuple(int(value) for value in facade_mask.shape[:2])
    facade_bool = _normalize_facade_mask(facade_mask, shape_hw)
    facade_bbox = _bbox_from_mask(facade_bool)

    raw_doors = [
        opening
        for observation in list(door_observations)[:FACADE_OPENING_MAX_MASKS]
        if (
            opening := _opening_from_observation(
                observation,
                kind="door",
                facade_mask=facade_bool,
            )
        )
        is not None
    ]
    raw_windows = [
        opening
        for observation in list(window_observations)[:FACADE_OPENING_MAX_MASKS]
        if (
            opening := _opening_from_observation(
                observation,
                kind="window",
                facade_mask=facade_bool,
            )
        )
        is not None
    ]
    raw_doors = _dedupe_openings(raw_doors)
    raw_windows = _dedupe_openings(raw_windows)

    confident_doors: List[FacadeOpening] = []
    uncertain_doors: List[FacadeOpening] = []
    for door in raw_doors:
        aspect_ratio = door.height / max(1e-6, door.width)
        if aspect_ratio < 1.0:
            uncertain_doors.append(
                _retag_opening(
                    door,
                    kind="uncertain_door",
                    reason="partial_door_frame",
                )
            )
        elif aspect_ratio > 4.0:
            uncertain_doors.append(
                _retag_opening(
                    door,
                    kind="uncertain_door",
                    reason="slender_aspect_ratio",
                )
            )
        elif door.score < FACADE_DOOR_UNCERTAIN_SCORE_BELOW:
            uncertain_doors.append(
                _retag_opening(
                    door,
                    kind="uncertain_door",
                    reason="low_score",
                )
            )
        else:
            confident_doors.append(door)

    confident_windows: List[FacadeOpening] = []
    uncertain_windows: List[FacadeOpening] = []
    for window in raw_windows:
        if window.score < FACADE_WINDOW_UNCERTAIN_SCORE_BELOW:
            uncertain_windows.append(
                _retag_opening(
                    window,
                    kind="uncertain_window",
                    reason="low_score",
                )
            )
        else:
            confident_windows.append(window)

    normalized_doors = _normalize_openings(confident_doors, facade_bbox)
    normalized_uncertain_doors = _normalize_openings(uncertain_doors, facade_bbox)
    normalized_windows_raw = _normalize_openings(confident_windows, facade_bbox)
    normalized_uncertain_windows = _normalize_openings(
        uncertain_windows, facade_bbox
    )
    normalized_windows = _merge_windows(normalized_windows_raw)
    return _build_facade_feature(
        doors=normalized_doors,
        uncertain_doors=normalized_uncertain_doors,
        windows_raw=normalized_windows_raw,
        windows=normalized_windows,
        uncertain_windows=normalized_uncertain_windows,
    )


def structure_feature_vector(
    feature: Optional[FacadeStructureFeature],
) -> np.ndarray:
    if feature is None or not feature.available:
        return np.zeros((STRUCTURE_VECTOR_DIM,), dtype="float32")
    matrix = np.asarray(feature.structure_matrix, dtype="float32")
    row_count = int(matrix.shape[0]) if matrix.ndim >= 1 else 0
    padded = np.zeros(
        (STRUCTURE_VECTOR_MAX_ROWS, 3, STRUCTURE_VECTOR_CHANNELS),
        dtype="float32",
    )
    if matrix.size:
        rows_to_copy = min(STRUCTURE_VECTOR_MAX_ROWS, row_count)
        channels_to_copy = min(STRUCTURE_VECTOR_CHANNELS, matrix.shape[2])
        padded[:rows_to_copy, :, :channels_to_copy] = matrix[
            :rows_to_copy, :, :channels_to_copy
        ]
    counts = np.asarray(
        [
            len(feature.doors),
            len(feature.uncertain_doors),
            len(feature.windows),
            len(feature.uncertain_windows),
            min(row_count, STRUCTURE_VECTOR_MAX_ROWS),
            max(0, row_count - STRUCTURE_VECTOR_MAX_ROWS),
        ],
        dtype="float32",
    )
    return np.concatenate([counts, padded.reshape(-1)]).astype("float32")


def compare_structure_features(
    query: Optional[FacadeStructureFeature],
    template: Optional[FacadeStructureFeature],
) -> float:
    if (
        query is None
        or template is None
        or not query.available
        or not template.available
    ):
        return 0.0
    query_matrix = query.structure_matrix.astype("float32")
    template_matrix = template.structure_matrix.astype("float32")
    observed = query_matrix > 0
    row_count = query_matrix.shape[0]
    channel_count = max(query_matrix.shape[2], template_matrix.shape[2], 4)
    if template_matrix.shape[0] < row_count:
        template_matrix = np.pad(
            template_matrix,
            ((0, row_count - template_matrix.shape[0]), (0, 0), (0, 0)),
        )
    elif template_matrix.shape[0] > row_count:
        template_matrix = template_matrix[:row_count]
    if query_matrix.shape[2] < channel_count:
        padding = channel_count - query_matrix.shape[2]
        query_matrix = np.pad(query_matrix, ((0, 0), (0, 0), (0, padding)))
        observed = np.pad(observed, ((0, 0), (0, 0), (0, padding)))
    if template_matrix.shape[2] < channel_count:
        template_matrix = np.pad(
            template_matrix,
            ((0, 0), (0, 0), (0, channel_count - template_matrix.shape[2])),
        )
    weights = np.ones((channel_count,), dtype="float32")
    weights[1] = FACADE_UNCERTAIN_DOOR_WEIGHT
    weights[3] = FACADE_UNCERTAIN_WINDOW_WEIGHT
    query_matrix *= weights.reshape((1, 1, channel_count))
    template_matrix *= weights.reshape((1, 1, channel_count))
    observed_float = observed.astype("float32")
    difference = float(
        (np.abs(query_matrix - template_matrix) * observed_float).sum()
    )
    denominator = max(
        float((query_matrix * observed_float).sum()),
        1.0,
    )
    return max(0.0, min(1.0, 1.0 - difference / denominator))


def facade_structure_payload(
    feature: Optional[FacadeStructureFeature],
) -> Dict[str, Any]:
    if feature is None:
        return {"available": False, "counts": {}, "rows": []}
    return {
        "available": feature.available,
        "counts": {
            "door_count": len(feature.doors),
            "uncertain_door_count": len(feature.uncertain_doors),
            "window_raw_count": len(feature.windows_raw),
            "window_count": len(feature.windows),
            "uncertain_window_count": len(feature.uncertain_windows),
        },
        "rows": np.asarray(feature.structure_matrix, dtype="float32").tolist(),
        "openings": {
            "doors": [_opening_payload(value) for value in feature.doors],
            "uncertain_doors": [
                _opening_payload(value) for value in feature.uncertain_doors
            ],
            "windows_raw": [
                _opening_payload(value) for value in feature.windows_raw
            ],
            "windows": [_opening_payload(value) for value in feature.windows],
            "uncertain_windows": [
                _opening_payload(value) for value in feature.uncertain_windows
            ],
        },
    }


def _opening_payload(opening: FacadeOpening) -> Dict[str, Any]:
    return {
        "kind": opening.kind,
        "bbox_xyxy": list(opening.bbox_xyxy),
        "score": opening.score,
        "reason": opening.reason,
    }


def _facade_structure_from_payload(
    payload: Mapping[str, Any],
) -> Optional[FacadeStructureFeature]:
    if not payload.get("available"):
        return None
    raw_openings = payload.get("openings")
    if not isinstance(raw_openings, Mapping):
        raise ValueError("门窗结构缓存缺少 openings")

    def load_openings(name: str) -> List[FacadeOpening]:
        values = raw_openings.get(name) or []
        result = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            bbox = value.get("bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            result.append(
                FacadeOpening(
                    kind=str(value.get("kind") or name),
                    bbox_xyxy=tuple(float(item) for item in bbox),
                    score=float(value.get("score", 1.0)),
                    reason=value.get("reason"),
                )
            )
        return result

    return _build_facade_feature(
        doors=load_openings("doors"),
        uncertain_doors=load_openings("uncertain_doors"),
        windows_raw=load_openings("windows_raw"),
        windows=load_openings("windows"),
        uncertain_windows=load_openings("uncertain_windows"),
    )


class DinoV3FeatureExtractor:
    """从 auto_game 指定目录离线加载 DINOv3，不访问模型网络。"""

    def __init__(self, model_dir: Path, device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3 匹配环境缺少 torch/transformers；"
                "请安装 requirements_nanda_room_matcher.txt"
            ) from exc

        self.torch = torch
        self.model_dir = model_dir.expanduser().resolve()
        self.device = self._resolve_device(device)
        LOGGER.info("加载 DINOv3：path=%s device=%s", self.model_dir, self.device)
        transformers_logging.disable_progress_bar()
        self.processor = AutoImageProcessor.from_pretrained(
            str(self.model_dir), use_fast=True, local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_dir), local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested:
            return str(requested)
        if self.torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(self.torch, "backends", None), "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"

    @staticmethod
    def _patch_mask_from_image_mask(
        mask: np.ndarray, input_height: int, input_width: int, patch_size: int,
    ) -> np.ndarray:
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.ndim == 3:
            mask_bool = np.any(mask_bool, axis=2)
        grid_h = input_height // patch_size
        grid_w = input_width // patch_size
        token_h = grid_h * patch_size
        token_w = grid_w * patch_size
        resized = cv2.resize(
            mask_bool.astype(np.uint8),
            (token_w, token_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        return (
            resized.reshape(grid_h, patch_size, grid_w, patch_size)
            .any(axis=(1, 3))
            .reshape(-1)
        )

    def extract(
        self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0]
        if mask is not None:
            pixel_values = inputs["pixel_values"]
            patch_size = int(getattr(self.model.config, "patch_size", 16))
            patch_mask = self._patch_mask_from_image_mask(
                mask,
                input_height=int(pixel_values.shape[-2]),
                input_width=int(pixel_values.shape[-1]),
                patch_size=patch_size,
            )
            prefix_tokens = 1 + int(
                getattr(self.model.config, "num_register_tokens", 0)
            )
            patch_tokens = outputs.last_hidden_state[:, prefix_tokens:]
            if (
                int(patch_mask.sum()) >= MIN_MASKED_PATCHES
                and patch_tokens.shape[1] == patch_mask.shape[0]
            ):
                selected = self.torch.from_numpy(patch_mask).to(
                    device=patch_tokens.device, dtype=self.torch.bool,
                )
                embedding = patch_tokens[:, selected, :].mean(dim=1)
        return _normalize_embedding(embedding.detach().cpu().numpy())


@dataclass(frozen=True)
class RoomMatchFeatureSpec:
    feature_set: str = "rgb"
    feature_mode: str = "pair_12d"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "RoomMatchFeatureSpec":
        spec = cls(
            feature_set=str(metadata.get("feature_set", "rgb")),
            feature_mode=str(metadata.get("feature_mode", "pair_12d")),
        )
        if spec.feature_set not in {"rgb", "gray", "rgb_gray"}:
            raise ValueError(f"不支持的 MLP feature_set: {spec.feature_set}")
        if spec.feature_mode not in {
            "pair_12d",
            "embedding_concat",
            "embedding_concat_structure",
        }:
            raise ValueError(f"不支持的 MLP feature_mode: {spec.feature_mode}")
        return spec

    @property
    def uses_gray_embedding(self) -> bool:
        return self.feature_set in {"gray", "rgb_gray"}

    @property
    def uses_structure(self) -> bool:
        return self.feature_mode == "embedding_concat_structure"

    def expected_n_features(self) -> int:
        if self.feature_mode == "pair_12d":
            return 12 if self.feature_set == "rgb_gray" else 6
        embedding_branches = 4 if self.feature_set == "rgb_gray" else 2
        result = embedding_branches * EMBEDDING_DIM
        if self.uses_structure:
            result += STRUCTURE_VECTOR_DIM * 3
        return result


def _pair_geometry_features(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = _normalize_embedding(first)
    second = _normalize_embedding(second)
    diff = first - second
    product = first * second
    return np.asarray(
        [
            float(np.dot(first, second)),
            float(np.linalg.norm(diff)),
            float(np.mean(np.abs(diff))),
            float(np.max(np.abs(diff))),
            float(np.mean(product)),
            float(np.std(product)),
        ],
        dtype="float32",
    )


def _build_mlp_features(
    spec: RoomMatchFeatureSpec,
    query_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    query_gray: Optional[np.ndarray],
    candidate_gray: Optional[np.ndarray],
    query_structure: Optional[FacadeStructureFeature] = None,
    candidate_structure: Optional[FacadeStructureFeature] = None,
) -> np.ndarray:
    if spec.feature_set == "gray":
        if query_gray is None or candidate_gray is None:
            raise ValueError("MLP 需要灰度 DINO 特征，但灰度特征不可用")
        base_query, base_candidate = query_gray, candidate_gray
    else:
        base_query, base_candidate = query_rgb, candidate_rgb

    if spec.feature_mode == "pair_12d":
        features = [_pair_geometry_features(base_query, base_candidate)]
        if spec.feature_set == "rgb_gray":
            if query_gray is None or candidate_gray is None:
                raise ValueError("rgb_gray MLP 缺少灰度 DINO 特征")
            features.append(_pair_geometry_features(query_gray, candidate_gray))
        return np.concatenate(features).astype("float32")

    parts = [
        _normalize_embedding(base_query),
        _normalize_embedding(base_candidate),
    ]
    if spec.feature_set == "rgb_gray":
        if query_gray is None or candidate_gray is None:
            raise ValueError("rgb_gray MLP 缺少灰度 DINO 特征")
        parts.extend(
            [_normalize_embedding(query_gray), _normalize_embedding(candidate_gray)]
        )
    if spec.uses_structure:
        query_structure_vector = structure_feature_vector(query_structure)
        candidate_structure_vector = structure_feature_vector(candidate_structure)
        parts.extend(
            [
                query_structure_vector,
                candidate_structure_vector,
                np.abs(
                    query_structure_vector - candidate_structure_vector
                ).astype("float32"),
            ]
        )
    return np.concatenate(parts).astype("float32")


class MlpRoomReranker:
    def __init__(self, model_path: Path):
        self.model_path = model_path.expanduser().resolve()
        try:
            with self.model_path.open("rb") as model_file:
                payload = pickle.load(model_file)
        except ImportError as exc:
            raise RuntimeError(
                "MLP 权重依赖与当前环境不一致；请安装 " "requirements_nanda_room_matcher.txt"
            ) from exc
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            metadata = payload.get("metadata")
            self.metadata = metadata if isinstance(metadata, dict) else {}
            self.positive_class_index = int(payload.get("positive_class_index", 1))
        else:
            self.model = payload
            self.metadata = {}
            self.positive_class_index = 1
        self.feature_spec = RoomMatchFeatureSpec.from_metadata(self.metadata)
        expected = self.feature_spec.expected_n_features()
        actual = self._model_n_features(self.model)
        metadata_features = self.metadata.get("n_features_in")
        if actual is not None and actual != expected:
            raise ValueError(
                f"MLP 输入维度不匹配: model={actual}, runtime={expected}, "
                f"path={self.model_path}"
            )
        if metadata_features is not None and int(metadata_features) != expected:
            raise ValueError(
                f"MLP metadata 输入维度不匹配: metadata={metadata_features}, "
                f"runtime={expected}"
            )
        LOGGER.info(
            "加载南大 MLP：path=%s feature_set=%s feature_mode=%s n_features=%s",
            self.model_path,
            self.feature_spec.feature_set,
            self.feature_spec.feature_mode,
            expected,
        )

    @staticmethod
    def _model_n_features(model: Any) -> Optional[int]:
        value = getattr(model, "n_features_in_", None)
        if value is None and hasattr(model, "named_steps"):
            final = model.named_steps.get("clf")
            value = getattr(final, "n_features_in_", None)
        return None if value is None else int(value)

    def predict(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype="float32").reshape(1, -1)
        if not np.all(np.isfinite(features)):
            raise ValueError("MLP 输入包含非有限数值")
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features)).reshape(-1)
            index = max(0, min(self.positive_class_index, probabilities.size - 1))
            return float(probabilities[index])
        if hasattr(self.model, "decision_function"):
            raw = float(
                np.asarray(self.model.decision_function(features)).reshape(-1)[0]
            )
            return float(1.0 / (1.0 + math.exp(-raw)))
        if hasattr(self.model, "predict"):
            return float(np.asarray(self.model.predict(features)).reshape(-1)[0])
        raise TypeError(f"不支持的 MLP 模型类型: {type(self.model)!r}")


@dataclass
class RoomTemplate:
    room_id: str
    template_id: str
    room_dir: Path
    template_dir: Path
    template_path: Path
    mask_path: Optional[Path]
    action_path: Path
    metadata: Dict[str, Any]
    rgb_embedding: np.ndarray
    gray_embedding: Optional[np.ndarray]
    structure: Optional[FacadeStructureFeature] = None
    structure_resolved: bool = False


def _iter_room_dirs(room_library_path: Path) -> Iterable[Path]:
    rooms_dir = room_library_path / "rooms"
    for path in sorted(rooms_dir.iterdir(), key=lambda value: _natural_key(value.name)):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def _load_room_metadata(room_dir: Path) -> Dict[str, Any]:
    metadata_path = room_dir / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取房型 metadata: {metadata_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"房型 metadata 必须是对象: {metadata_path}")
    replay = payload.get("replay")
    if not isinstance(replay, dict) or not isinstance(
        replay.get("allow_actions"), bool
    ):
        raise ValueError(f"房型 metadata.replay.allow_actions 无效: {metadata_path}")
    return payload


def _find_capture(room_dir: Path, template_id: str) -> Optional[Path]:
    for suffix in (".jpeg", ".jpg", ".png"):
        candidate = room_dir / "captures" / f"{template_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _locate_source_crop(
    source: np.ndarray, segment: np.ndarray, mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if source.shape[:2] == segment.shape[:2]:
        return source.copy()
    if source.shape[0] < segment.shape[0] or source.shape[1] < segment.shape[1]:
        return None
    if mask is None or mask.shape != segment.shape[:2] or not np.any(mask):
        return None
    visible = mask.astype(bool) & np.any(segment != 0, axis=2)
    if not np.any(visible):
        visible = mask.astype(bool)
    try:
        scores = cv2.matchTemplate(
            cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY),
            cv2.TM_CCORR_NORMED,
            mask=visible.astype(np.uint8) * 255,
        )
    except cv2.error:
        return None
    if scores.size == 0:
        return None
    _, best_score, _, best_location = cv2.minMaxLoc(scores)
    if not np.isfinite(best_score) or best_score < 0.80:
        return None
    x1, y1 = best_location
    height, width = segment.shape[:2]
    return source[y1 : y1 + height, x1 : x1 + width].copy()


def _load_template_dino_input(
    room_dir: Path, template_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    segment_path = template_dir / "segment.png"
    segment = cv2.imread(str(segment_path), cv2.IMREAD_COLOR)
    if segment is None:
        raise FileNotFoundError(f"无法读取房型模板: {segment_path}")
    mask_path = template_dir / "mask.png"
    raw_mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.is_file()
        else None
    )
    mask = _normalize_mask(raw_mask, segment.shape[:2])

    derived_path = room_dir / "derived" / "dino_inputs" / f"{template_dir.name}.png"
    derived = (
        cv2.imread(str(derived_path), cv2.IMREAD_COLOR)
        if derived_path.is_file()
        else None
    )
    if derived is not None and derived.shape[:2] == segment.shape[:2]:
        return derived, _fill_mask_holes(mask)

    source_path = _find_capture(room_dir, template_dir.name)
    source = (
        cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_path is not None
        else None
    )
    crop = _locate_source_crop(source, segment, mask) if source is not None else None
    base = crop if crop is not None else segment
    return _build_masked_dino_input(base, mask)


def _load_template_structure_input(
    room_dir: Path,
    template_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read the unmasked facade crop used by Nanda's opening extractor."""
    segment_path = template_dir / "segment.png"
    segment = cv2.imread(str(segment_path), cv2.IMREAD_COLOR)
    if segment is None:
        raise FileNotFoundError(f"无法读取房型结构模板: {segment_path}")
    mask_path = template_dir / "mask.png"
    raw_mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.is_file()
        else None
    )
    mask = _normalize_mask(raw_mask, segment.shape[:2])
    source_path = _find_capture(room_dir, template_dir.name)
    source = (
        cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_path is not None
        else None
    )
    located = _locate_source_crop(source, segment, mask) if source is not None else None
    image = located if located is not None else segment
    if mask is not None and mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if mask is None or not np.any(mask):
        return image.copy(), mask
    ys, xs = np.where(mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return (
        np.ascontiguousarray(image[y1:y2, x1:x2]).copy(),
        np.ascontiguousarray(mask[y1:y2, x1:x2]).copy(),
    )


def _template_structure_cache_path(room_dir: Path, template_id: str) -> Path:
    return room_dir / "derived" / "autogame_structure" / f"{template_id}.json"


def _load_cached_facade_structure(
    cache_path: Path,
    *,
    image: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[bool, Optional[FacadeStructureFeature]]:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, Mapping):
        return False, None
    expected = {
        "cache_version": FACADE_STRUCTURE_CACHE_VERSION,
        "image_md5": _image_md5(image),
        "mask_md5": _image_md5(mask),
    }
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        return False, None
    feature_payload = payload.get("feature")
    if not isinstance(feature_payload, Mapping):
        return False, None
    try:
        return True, _facade_structure_from_payload(feature_payload)
    except (TypeError, ValueError):
        return False, None


def _write_cached_facade_structure(
    cache_path: Path,
    *,
    image: np.ndarray,
    mask: Optional[np.ndarray],
    feature: Optional[FacadeStructureFeature],
) -> None:
    payload = {
        "metadata": {
            "cache_version": FACADE_STRUCTURE_CACHE_VERSION,
            "image_md5": _image_md5(image),
            "mask_md5": _image_md5(mask),
            "door_prompt": "door frame",
            "window_prompt": "window",
            "max_masks": FACADE_OPENING_MAX_MASKS,
        },
        "feature": facade_structure_payload(feature),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)


def _candidate_observations(
    candidates: Iterable[Mapping[str, Any]],
) -> List[FacadeMaskObservation]:
    observations = []
    for candidate in candidates:
        mask = candidate.get("mask")
        if mask is None:
            continue
        observations.append(
            FacadeMaskObservation(
                mask=np.asarray(mask, dtype=np.uint8),
                score=float(candidate.get("score", 0.0)),
            )
        )
    return observations


class IntegratedNandaRoomMatcher:
    """持久加载 DINOv3、MLP 和房型索引的独立运行时。"""

    def __init__(
        self, asset_paths: NandaMatcherAssetPaths, *, device: Optional[str] = None,
    ):
        self.asset_paths = asset_paths.resolved()
        self.asset_paths.validate()
        self.extractor = DinoV3FeatureExtractor(
            self.asset_paths.dino_model_dir, device=device,
        )
        self.reranker = MlpRoomReranker(self.asset_paths.mlp_model_path)
        self.templates: List[RoomTemplate] = []
        self.templates_by_room: Dict[str, List[RoomTemplate]] = {}
        self.template_matrix: Optional[np.ndarray] = None
        self.faiss_index = None
        self._load_room_library()

    def _cache_identity(self, variant: str) -> Dict[str, Any]:
        weights = self.asset_paths.dino_model_dir / "model.safetensors"
        return {
            "runtime_cache_version": RUNTIME_CACHE_VERSION,
            "preprocess_version": INPUT_PREPROCESS_VERSION,
            "model_bytes": int(weights.stat().st_size),
            "variant": variant,
        }

    def _load_or_compute_embedding(
        self,
        room_dir: Path,
        template_id: str,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        *,
        variant: str,
    ) -> np.ndarray:
        cache_dir = room_dir / "derived" / "autogame_embeddings"
        cache_path = cache_dir / f"{template_id}.{variant}.npy"
        metadata_path = cache_dir / f"{template_id}.{variant}.json"
        expected = {
            **self._cache_identity(variant),
            "image_md5": _image_md5(image),
            "mask_md5": _image_md5(mask),
        }
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached_metadata == expected:
                return _normalize_embedding(
                    np.load(str(cache_path), allow_pickle=False)
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

        embedding = self.extractor.extract(image, mask)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npy")
            np.save(str(temporary), embedding)
            temporary.replace(cache_path)
            metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            metadata_tmp.write_text(
                json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            metadata_tmp.replace(metadata_path)
        except OSError as exc:
            LOGGER.warning("无法写入 DINO 模板缓存 %s: %s", cache_path, exc)
        return embedding

    def _load_room_library(self) -> None:
        started = perf_counter()
        room_count = 0
        for room_dir in _iter_room_dirs(self.asset_paths.room_library_path):
            metadata = _load_room_metadata(room_dir)
            if metadata.get("status") == "disabled":
                continue
            action_path = room_dir / "actions" / "action_step.json"
            templates_dir = room_dir / "templates"
            if not action_path.is_file() or not templates_dir.is_dir():
                continue
            room_templates: List[RoomTemplate] = []
            template_dirs = sorted(
                (
                    path
                    for path in templates_dir.iterdir()
                    if path.is_dir()
                    and not path.name.startswith(".")
                    and not path.name.endswith("_structure_overlay")
                    and (path / "segment.png").is_file()
                ),
                key=lambda value: _natural_key(value.name),
            )
            for template_dir in template_dirs:
                try:
                    image, mask = _load_template_dino_input(room_dir, template_dir)
                    rgb_embedding = self._load_or_compute_embedding(
                        room_dir, template_dir.name, image, mask, variant="rgb",
                    )
                    gray_embedding = None
                    if self.reranker.feature_spec.uses_gray_embedding:
                        gray_embedding = self._load_or_compute_embedding(
                            room_dir,
                            template_dir.name,
                            _gray_bgr(image),
                            mask,
                            variant="gray",
                        )
                    structure = None
                    structure_resolved = False
                    if self.reranker.feature_spec.uses_structure:
                        structure_image, structure_mask = (
                            _load_template_structure_input(
                                room_dir,
                                template_dir,
                            )
                        )
                        structure_resolved, structure = _load_cached_facade_structure(
                            _template_structure_cache_path(
                                room_dir,
                                template_dir.name,
                            ),
                            image=structure_image,
                            mask=structure_mask,
                        )
                    record = RoomTemplate(
                        room_id=room_dir.name,
                        template_id=template_dir.name,
                        room_dir=room_dir,
                        template_dir=template_dir,
                        template_path=template_dir / "segment.png",
                        mask_path=(
                            template_dir / "mask.png"
                            if (template_dir / "mask.png").is_file()
                            else None
                        ),
                        action_path=action_path,
                        metadata=metadata,
                        rgb_embedding=rgb_embedding,
                        gray_embedding=gray_embedding,
                        structure=structure,
                        structure_resolved=structure_resolved,
                    )
                    room_templates.append(record)
                    self.templates.append(record)
                except Exception as exc:
                    LOGGER.warning(
                        "跳过无法索引的房型模板 room=%s template=%s: %s",
                        room_dir.name,
                        template_dir.name,
                        exc,
                    )
            if room_templates:
                self.templates_by_room[room_dir.name] = room_templates
                room_count += 1

        if not self.templates:
            raise RuntimeError(f"房型库没有可索引模板: {self.asset_paths.room_library_path}")
        self.template_matrix = np.ascontiguousarray(
            np.stack([item.rgb_embedding for item in self.templates]), dtype="float32",
        )
        try:
            import faiss

            index = faiss.IndexFlatIP(self.template_matrix.shape[1])
            index.add(self.template_matrix)
            self.faiss_index = index
            index_backend = "faiss"
        except ImportError:
            self.faiss_index = None
            index_backend = "numpy"
        LOGGER.info(
            "房型库索引完成：rooms=%d templates=%d backend=%s elapsed=%.2fs",
            room_count,
            len(self.templates),
            index_backend,
            perf_counter() - started,
        )

    def _ensure_template_structure(
        self,
        template: RoomTemplate,
    ) -> Optional[FacadeStructureFeature]:
        if template.structure_resolved:
            return template.structure
        image, mask = _load_template_structure_input(
            template.room_dir,
            template.template_dir,
        )
        try:
            from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.sam3_tiny.local_segmenter import (
                get_sam3_segmenter,
            )

            segmenter = get_sam3_segmenter()
            door_candidates = segmenter.infer_all_masks(
                image,
                prompt="door frame",
                max_masks=FACADE_OPENING_MAX_MASKS,
                min_mask_area_ratio=FACADE_OPENING_MIN_MASK_AREA_RATIO,
            )
            window_candidates = segmenter.infer_all_masks(
                image,
                prompt="window",
                max_masks=FACADE_OPENING_MAX_MASKS,
                min_mask_area_ratio=FACADE_OPENING_MIN_MASK_AREA_RATIO,
            )
            feature = build_facade_structure(
                mask,
                _candidate_observations(door_candidates),
                _candidate_observations(window_candidates),
            )
        except Exception as exc:
            raise RuntimeError(
                "南大模板门窗结构提取失败: "
                f"room={template.room_id}, template={template.template_id}: {exc}"
            ) from exc

        template.structure = feature if feature.available else None
        template.structure_resolved = True
        cache_path = _template_structure_cache_path(
            template.room_dir,
            template.template_id,
        )
        try:
            _write_cached_facade_structure(
                cache_path,
                image=image,
                mask=mask,
                feature=template.structure,
            )
        except OSError as exc:
            raise RuntimeError(f"无法写入南大模板门窗结构缓存: {cache_path}") from exc
        LOGGER.info(
            "南大模板门窗结构已生成：room=%s template=%s available=%s "
            "doors=%d uncertain_doors=%d windows=%d uncertain_windows=%d",
            template.room_id,
            template.template_id,
            feature.available,
            len(feature.doors),
            len(feature.uncertain_doors),
            len(feature.windows),
            len(feature.uncertain_windows),
        )
        return template.structure

    def _ensure_candidate_structures(
        self,
        candidates: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not self.reranker.feature_spec.uses_structure:
            return {
                "template_count": 0,
                "cached_count": 0,
                "generated_count": 0,
                "elapsed_ms": 0.0,
            }
        seen: set[Tuple[str, str]] = set()
        generated = 0
        cached = 0
        started = perf_counter()
        for candidate in candidates:
            template = candidate["template"]
            key = (template.room_id, template.template_id)
            if key in seen:
                continue
            seen.add(key)
            was_resolved = template.structure_resolved
            self._ensure_template_structure(template)
            if was_resolved:
                cached += 1
            else:
                generated += 1
        LOGGER.info(
            "南大候选模板门窗结构就绪：templates=%d cached=%d generated=%d "
            "elapsed=%.2fs",
            len(seen),
            cached,
            generated,
            perf_counter() - started,
        )
        return {
            "template_count": len(seen),
            "cached_count": cached,
            "generated_count": generated,
            "elapsed_ms": (perf_counter() - started) * 1000.0,
        }

    def health_payload(self) -> Dict[str, Any]:
        return {
            "assets": self.asset_paths.to_jsonable(),
            "room_count": len(self.templates_by_room),
            "template_count": len(self.templates),
            "dino_device": self.extractor.device,
            "mlp_feature_set": self.reranker.feature_spec.feature_set,
            "mlp_feature_mode": self.reranker.feature_spec.feature_mode,
            "structure_mode": "sam3_door_window_full",
        }

    def _retrieve(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        room_limit = min(FACADE_TOP_K, len(self.templates_by_room))
        template_limit = min(
            max(room_limit * ROOM_SHORTLIST_TEMPLATE_MULTIPLIER, room_limit),
            len(self.templates),
        )
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(
                np.ascontiguousarray(query_embedding.reshape(1, -1), dtype="float32"),
                template_limit,
            )
            pairs = zip(indices[0], scores[0])
        else:
            all_scores = np.dot(self.template_matrix, query_embedding)
            order = np.argsort(-all_scores)[:template_limit]
            pairs = ((index, all_scores[index]) for index in order)

        retrieved = []
        for index, score in pairs:
            if int(index) < 0:
                continue
            template = self.templates[int(index)]
            retrieved.append(
                {
                    "room_id": template.room_id,
                    "template": template,
                    "dino_score": float(score),
                }
            )
        retrieved.sort(key=lambda value: value["dino_score"], reverse=True)
        shortlisted_rooms: List[str] = []
        for candidate in retrieved:
            if candidate["room_id"] not in shortlisted_rooms:
                shortlisted_rooms.append(candidate["room_id"])
            if len(shortlisted_rooms) >= room_limit:
                break

        expanded = []
        for room_id in shortlisted_rooms:
            for template in self.templates_by_room[room_id]:
                expanded.append(
                    {
                        "room_id": room_id,
                        "template": template,
                        "dino_score": float(
                            np.dot(query_embedding, template.rgb_embedding)
                        ),
                    }
                )
        expanded.sort(key=lambda value: value["dino_score"], reverse=True)
        return expanded

    @staticmethod
    def _candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[float, ...]:
        return (
            float(candidate["total_score"]),
            float(candidate.get("room_best_dino_score", candidate["dino_score"])),
            float(candidate["structure_score"]),
            float(candidate["mlp_score"]),
        )

    def match(
        self,
        segmented_bgr: np.ndarray,
        cropped_mask: np.ndarray,
        cropped_bgr: np.ndarray,
        *,
        door_observations: Iterable[FacadeMaskObservation] = (),
        window_observations: Iterable[FacadeMaskObservation] = (),
    ) -> Tuple[Optional[str], Optional[Path], Dict[str, Any]]:
        started = perf_counter()
        query_image, query_mask = _build_masked_dino_input(
            cropped_bgr
            if cropped_bgr.shape[:2] == segmented_bgr.shape[:2]
            else segmented_bgr,
            cropped_mask,
        )
        query_rgb = self.extractor.extract(query_image, query_mask)
        query_gray = None
        if self.reranker.feature_spec.uses_gray_embedding:
            query_gray = self.extractor.extract(_gray_bgr(query_image), query_mask)
        query_structure = None
        query_structure_feature = None
        if self.reranker.feature_spec.uses_structure:
            query_structure_feature = build_facade_structure(
                cropped_mask,
                door_observations,
                window_observations,
            )
            if query_structure_feature.available:
                query_structure = query_structure_feature

        candidates = self._retrieve(query_rgb)
        if query_structure is not None:
            template_structure_cache = self._ensure_candidate_structures(candidates)
        else:
            template_structure_cache = {
                "template_count": 0,
                "cached_count": 0,
                "generated_count": 0,
                "elapsed_ms": 0.0,
                "skipped_reason": "query_structure_unavailable",
            }
        ranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            template = candidate["template"]
            candidate_structure = (
                template.structure if query_structure is not None else None
            )
            features = _build_mlp_features(
                self.reranker.feature_spec,
                query_rgb,
                template.rgb_embedding,
                query_gray,
                template.gray_embedding,
                query_structure,
                candidate_structure,
            )
            mlp_score = self.reranker.predict(features)
            mlp_rank_score = max(
                0.0, float(mlp_score) - FACADE_LOW_MLP_CONFIDENCE_SCORE
            )
            structure_score = compare_structure_features(
                query_structure,
                candidate_structure,
            )
            total_score = mlp_rank_score + FACADE_STRUCT_SCORE_ALPHA * structure_score
            ranked.append(
                {
                    **candidate,
                    "mlp_score": float(mlp_score),
                    "mlp_rank_score": mlp_rank_score,
                    "structure_score": structure_score,
                    "total_score": total_score,
                }
            )

        best_by_room: Dict[str, Dict[str, Any]] = {}
        best_dino_by_room: Dict[str, float] = {}
        for candidate in ranked:
            room_id = candidate["room_id"]
            best_dino_by_room[room_id] = max(
                best_dino_by_room.get(room_id, -1.0), float(candidate["dino_score"]),
            )
            previous = best_by_room.get(room_id)
            if previous is None or self._candidate_sort_key(
                candidate
            ) > self._candidate_sort_key(previous):
                best_by_room[room_id] = candidate
        room_candidates = []
        for room_id, candidate in best_by_room.items():
            room_candidates.append(
                {**candidate, "room_best_dino_score": best_dino_by_room[room_id],}
            )
        room_candidates.sort(key=self._candidate_sort_key, reverse=True)

        thresholds = {
            "min_dino_score": FACADE_NO_MATCH_MIN_DINO_SCORE,
            "strict_dino_score": FACADE_NO_MATCH_STRICT_DINO_SCORE,
            "strong_structure_score": FACADE_NO_MATCH_STRONG_STRUCTURE_SCORE,
            "min_total_score": FACADE_NO_MATCH_MIN_SCORE,
            "min_top2_margin": FACADE_NO_MATCH_MIN_MARGIN,
            "low_mlp_confidence_score": FACADE_LOW_MLP_CONFIDENCE_SCORE,
        }
        debug: Dict[str, Any] = {
            "thresholds": thresholds,
            "structure_mode": "sam3_door_window_full",
            "query_structure": facade_structure_payload(query_structure_feature),
            "template_structure_cache": template_structure_cache,
            "candidate_count": len(ranked),
            "room_candidate_count": len(room_candidates),
            "elapsed_ms": (perf_counter() - started) * 1000.0,
            "decision": {"status": "no_match"},
        }
        if not room_candidates:
            debug["no_match_reason"] = "empty_room_candidate_pool"
            return None, None, debug

        chosen = room_candidates[0]
        margin = (
            float(chosen["total_score"] - room_candidates[1]["total_score"])
            if len(room_candidates) > 1
            else float("inf")
        )
        template = chosen["template"]
        decision = {
            "status": "matched",
            "room_id": chosen["room_id"],
            "template_path": str(template.template_path),
            "dino_score": chosen["dino_score"],
            "room_best_dino_score": chosen["room_best_dino_score"],
            "mlp_score": chosen["mlp_score"],
            "mlp_rank_score": chosen["mlp_rank_score"],
            "structure_score": chosen["structure_score"],
            "total_score": chosen["total_score"],
            "top2_margin": margin,
            "room_status": template.metadata.get("status"),
            "replay_allow_actions": bool(
                (template.metadata.get("replay") or {}).get("allow_actions", False)
            ),
            "replay_disabled_reason": (template.metadata.get("replay") or {}).get(
                "disabled_reason"
            ),
        }
        debug["decision"] = decision
        debug["top2_margin"] = margin
        debug["top_candidates"] = [
            {
                "room_id": value["room_id"],
                "dino_score": value["dino_score"],
                "room_best_dino_score": value["room_best_dino_score"],
                "mlp_score": value["mlp_score"],
                "structure_score": value["structure_score"],
                "total_score": value["total_score"],
                "template_structure": facade_structure_payload(
                    value["template"].structure
                ),
                "template_path": str(value["template"].template_path),
            }
            for value in room_candidates[:5]
        ]

        gate_dino_score = float(chosen["room_best_dino_score"])
        no_match_reason = None
        if gate_dino_score < FACADE_NO_MATCH_MIN_DINO_SCORE:
            no_match_reason = "dino_score_below_threshold"
        elif (
            gate_dino_score < FACADE_NO_MATCH_STRICT_DINO_SCORE
            and float(chosen["structure_score"])
            < FACADE_NO_MATCH_STRONG_STRUCTURE_SCORE
        ):
            no_match_reason = "dino_score_below_strict_threshold_without_structure"
        elif float(chosen["total_score"]) < FACADE_NO_MATCH_MIN_SCORE:
            no_match_reason = "total_score_below_threshold"
        elif FACADE_NO_MATCH_MIN_MARGIN > 0 and margin < FACADE_NO_MATCH_MIN_MARGIN:
            no_match_reason = "top2_margin_below_threshold"

        if no_match_reason:
            debug["no_match_reason"] = no_match_reason
            debug["decision"]["status"] = "no_match"
            LOGGER.info(
                "房型匹配被拒绝：reason=%s room=%s dino=%.4f mlp=%.4f total=%.4f",
                no_match_reason,
                chosen["room_id"],
                gate_dino_score,
                chosen["mlp_score"],
                chosen["total_score"],
            )
            return None, None, debug

        LOGGER.info(
            "房型匹配成功：room=%s dino=%.4f mlp=%.4f total=%.4f margin=%s",
            chosen["room_id"],
            gate_dino_score,
            chosen["mlp_score"],
            chosen["total_score"],
            "inf" if not np.isfinite(margin) else f"{margin:.4f}",
        )
        return chosen["room_id"], template.action_path, debug


__all__ = [
    "FacadeMaskObservation",
    "FacadeStructureFeature",
    "IntegratedNandaRoomMatcher",
    "NandaMatcherAssetPaths",
    "build_facade_structure",
    "compare_structure_features",
    "facade_structure_payload",
    "structure_feature_vector",
]
