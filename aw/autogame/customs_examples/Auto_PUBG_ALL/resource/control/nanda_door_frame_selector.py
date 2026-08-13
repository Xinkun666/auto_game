"""南大 SAM3 door frame 几何筛选与目标锁定的本地适配。

算法顺序与 young-cloud-creator/pubg_test@5fa3f849 中
``sam3/geometry.py`` 和 ``door_calibration/vision.py`` 保持一致；
这里只去掉了与校准 UI、preview overlay 和 YOLO 后端有关的包装。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np


SAM3_DOOR_MIN_MASK_AREA_RATIO = 0.001
SAM3_DOOR_MAX_MASKS = 12
SAM3_MAX_TARGET_BBOX_AREA = 0.9
SAM3_MIN_DOOR_HEIGHT_WIDTH_RATIO = 1.0
SAM3_SLENDER_DOOR_HEIGHT_WIDTH_RATIO = 4.0
DOOR_SELECTION_AREA_TIE_RATIO = 0.40
DOOR_SELECTION_CONFIDENCE_TIE_BAND = 0.20
DOOR_SELECTION_FRONT_FACING_TARGET_RATIO = 1.8
MAX_MISSING_TARGET_COUNT = 3
SMALL_BBOX_AREA_RATIO = 0.01


@dataclass(frozen=True)
class NandaDoorFrameSelection:
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    bbox_area_ratio: float
    door_top_ratio: float
    aspect_ratio: float
    candidate_count: int
    selection_reason: str
    quality: str
    quality_flags: tuple[str, ...]
    score: float


@dataclass
class _DoorFrameGeometry:
    mask: np.ndarray
    primary_mask: np.ndarray
    primary_bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask_area_ratio: float
    bbox_area_ratio: float
    bbox_norm_xyxy: tuple[float, float, float, float]
    quality: str
    quality_flags: tuple[str, ...]
    left_support: float
    right_support: float
    top_support: float
    angle_confidence: float
    pixel_aspect_ratio: float


@dataclass
class _DoorTargetCandidate:
    geometry: _DoorFrameGeometry
    bbox_area: float
    aspect_ratio: float
    bbox_norm_xyxy: tuple[float, float, float, float]
    confidence: float
    feature: Optional[np.ndarray]
    is_slender_fallback: bool

    @property
    def center_x(self) -> float:
        return (self.bbox_norm_xyxy[0] + self.bbox_norm_xyxy[2]) / 2.0


@dataclass
class _DoorTargetLockState:
    bbox_area: float
    aspect_ratio: float
    bbox_norm_xyxy: tuple[float, float, float, float]
    feature: Optional[np.ndarray]
    consecutive_missing_count: int = 0


class NandaDoorFrameSelector:
    """保留南大首次候选排序和跨帧目标锁定。"""

    def __init__(self) -> None:
        self._target_lock: Optional[_DoorTargetLockState] = None

    def reset(self) -> None:
        self._target_lock = None

    def select(
        self, frame: np.ndarray, observations: Sequence[Any],
    ) -> Optional[NandaDoorFrameSelection]:
        geometries = _extract_door_frame_geometries(
            observations[:SAM3_DOOR_MAX_MASKS], frame.shape[:2],
        )
        candidates: list[_DoorTargetCandidate] = []
        for geometry in geometries:
            if not _is_usable_door_frame(geometry):
                continue
            ratio = geometry.pixel_aspect_ratio
            if ratio <= SAM3_MIN_DOOR_HEIGHT_WIDTH_RATIO:
                continue
            if geometry.bbox_area_ratio > SAM3_MAX_TARGET_BBOX_AREA:
                continue
            candidates.append(
                _DoorTargetCandidate(
                    geometry=geometry,
                    bbox_area=geometry.bbox_area_ratio,
                    aspect_ratio=ratio,
                    bbox_norm_xyxy=geometry.bbox_norm_xyxy,
                    confidence=geometry.score,
                    feature=_extract_bbox_feature(frame, geometry.primary_bbox_xyxy,),
                    is_slender_fallback=(ratio > SAM3_SLENDER_DOOR_HEIGHT_WIDTH_RATIO),
                )
            )

        selected, reason = self._select_target_candidate(candidates)
        if selected is None:
            return None
        geometry = selected.geometry
        x1, y1, x2, y2 = geometry.primary_bbox_xyxy
        return NandaDoorFrameSelection(
            mask=geometry.primary_mask,
            bbox_xyxy=geometry.primary_bbox_xyxy,
            bbox_area_ratio=geometry.bbox_area_ratio,
            door_top_ratio=geometry.bbox_norm_xyxy[1],
            aspect_ratio=float(y2 - y1) / max(1.0, float(x2 - x1)),
            candidate_count=len(candidates),
            selection_reason=reason,
            quality=geometry.quality,
            quality_flags=geometry.quality_flags,
            score=geometry.score,
        )

    def _select_target_candidate(
        self, candidates: list[_DoorTargetCandidate],
    ) -> tuple[Optional[_DoorTargetCandidate], str]:
        if not candidates:
            return self._handle_missing_target()

        if self._target_lock is None:
            selected = self._select_bootstrap_anchor_candidate(candidates)
            self._update_target_lock(selected)
            reason = "bootstrap_area_confidence_front_facing_completeness_center"
            if selected.is_slender_fallback:
                reason += "_slender_fallback"
            return selected, reason

        matching = [
            candidate
            for candidate in candidates
            if self._is_plausibly_same_door(candidate, self._target_lock)
        ]
        if not matching:
            selected, reason = self._handle_missing_target(candidates)
            if selected is not None:
                self._update_target_lock(selected)
            return selected, reason

        ranked = sorted(
            _prefer_non_slender_candidates(matching),
            key=lambda candidate: self._locked_candidate_rank(
                candidate, self._target_lock,
            ),
        )
        selected = ranked[0]
        reason = "lock_continue_shape"
        if selected.is_slender_fallback:
            reason += "_slender_fallback"
        if len(ranked) > 1:
            best_feature = _feature_similarity(
                selected.feature, self._target_lock.feature
            )
            next_feature = _feature_similarity(
                ranked[1].feature, self._target_lock.feature
            )
            if best_feature > next_feature + 1e-6:
                reason = "lock_continue_feature_tiebreak"
        self._update_target_lock(selected)
        return selected, reason

    def _handle_missing_target(
        self, reacquire_candidates: Optional[list[_DoorTargetCandidate]] = None,
    ) -> tuple[Optional[_DoorTargetCandidate], str]:
        if self._target_lock is None:
            return None, ""
        self._target_lock.consecutive_missing_count += 1
        if self._target_lock.consecutive_missing_count < MAX_MISSING_TARGET_COUNT:
            return None, ""
        self._target_lock = None
        if not reacquire_candidates:
            return None, ""
        selected = self._select_bootstrap_anchor_candidate(reacquire_candidates)
        reason = (
            "lock_expired_reacquire_area_confidence_front_facing_completeness_center"
        )
        if selected.is_slender_fallback:
            reason += "_slender_fallback"
        return selected, reason

    @staticmethod
    def _select_bootstrap_anchor_candidate(
        candidates: list[_DoorTargetCandidate],
    ) -> _DoorTargetCandidate:
        ranked_pool = _prefer_non_slender_candidates(candidates)
        best_area = max(_candidate_anchor_area(item) for item in ranked_pool)
        area_tied = [
            item
            for item in ranked_pool
            if _candidate_anchor_area(item)
            >= best_area * (1.0 - DOOR_SELECTION_AREA_TIE_RATIO)
        ]
        best_confidence = max(item.confidence for item in area_tied)
        confidence_tied = [
            item
            for item in area_tied
            if item.confidence >= best_confidence - DOOR_SELECTION_CONFIDENCE_TIE_BAND
        ]
        return min(
            confidence_tied,
            key=lambda item: (
                -item.confidence,
                -_candidate_anchor_area(item),
                -_candidate_completeness_score(item),
                abs(
                    item.geometry.pixel_aspect_ratio
                    - DOOR_SELECTION_FRONT_FACING_TARGET_RATIO
                ),
                abs(item.center_x - 0.5),
            ),
        )

    def _update_target_lock(self, candidate: _DoorTargetCandidate) -> None:
        self._target_lock = _DoorTargetLockState(
            bbox_area=candidate.bbox_area,
            aspect_ratio=candidate.aspect_ratio,
            bbox_norm_xyxy=candidate.bbox_norm_xyxy,
            feature=candidate.feature,
        )

    @staticmethod
    def _is_plausibly_same_door(
        candidate: _DoorTargetCandidate, lock_state: _DoorTargetLockState,
    ) -> bool:
        area_ratio = candidate.bbox_area / max(lock_state.bbox_area, 1e-6)
        if area_ratio < 0.4 or area_ratio > 2.5:
            return False
        aspect_change_ratio = abs(
            candidate.aspect_ratio - lock_state.aspect_ratio
        ) / max(lock_state.aspect_ratio, 1e-6)
        if aspect_change_ratio > 0.4:
            return False
        return (
            _vertical_overlap_ratio(
                candidate.bbox_norm_xyxy, lock_state.bbox_norm_xyxy,
            )
            >= 0.3
        )

    @staticmethod
    def _locked_candidate_rank(
        candidate: _DoorTargetCandidate, lock_state: _DoorTargetLockState,
    ) -> tuple[float, float, float, float, float]:
        area_change = abs(candidate.bbox_area - lock_state.bbox_area) / max(
            lock_state.bbox_area, 1e-6,
        )
        aspect_change = abs(candidate.aspect_ratio - lock_state.aspect_ratio) / max(
            lock_state.aspect_ratio, 1e-6,
        )
        horizontal_drift = abs(
            candidate.center_x
            - (lock_state.bbox_norm_xyxy[0] + lock_state.bbox_norm_xyxy[2]) / 2.0
        )
        return (
            area_change,
            aspect_change,
            -_feature_similarity(candidate.feature, lock_state.feature),
            horizontal_drift,
            -candidate.confidence,
        )


def _extract_door_frame_geometries(
    observations: Sequence[Any], image_shape: tuple[int, int],
) -> list[_DoorFrameGeometry]:
    image_h, image_w = image_shape
    image_area = float(max(1, image_h * image_w))
    raw: list[tuple[np.ndarray, float]] = []
    for observation in observations:
        mask = (np.asarray(observation.mask) > 0).astype(np.uint8)
        if (
            not mask.any()
            or float(mask.sum()) / image_area < SAM3_DOOR_MIN_MASK_AREA_RATIO
        ):
            continue
        raw.append((mask, float(observation.score)))

    geometries = [
        _extract_door_frame_geometry(mask, score, image_shape)
        for mask, score in _merge_door_frame_masks(raw, image_shape)
    ]
    return sorted(
        geometries, key=lambda geometry: (-geometry.score, -geometry.mask_area_ratio),
    )


def _extract_door_frame_geometry(
    mask: np.ndarray, score: float, image_shape: tuple[int, int],
) -> _DoorFrameGeometry:
    image_h, image_w = image_shape
    image_area = float(max(1, image_h * image_w))
    repaired = _repair_door_frame_mask(mask)
    significant, component_count = _significant_component_union(repaired)
    if not significant.any():
        significant = repaired.copy()
    bbox = _bbox_from_mask(significant)
    x1, y1, x2, y2 = bbox
    left, right, top, _bottom = _frame_support(significant, bbox)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    density = float(significant.sum()) / float(bbox_area)
    aspect = float(y2 - y1) / max(1.0, float(x2 - x1))
    quality, flags = _door_frame_quality(
        bbox, image_shape, left, right, top, density, component_count,
    )
    angle_confidence = _door_frame_angle_confidence(significant, bbox, quality)
    return _DoorFrameGeometry(
        mask=mask,
        primary_mask=significant,
        primary_bbox_xyxy=bbox,
        score=score,
        mask_area_ratio=float(significant.sum()) / image_area,
        bbox_area_ratio=float((x2 - x1) * (y2 - y1)) / image_area,
        bbox_norm_xyxy=(
            x1 / max(1, image_w),
            y1 / max(1, image_h),
            x2 / max(1, image_w),
            y2 / max(1, image_h),
        ),
        quality=quality,
        quality_flags=flags,
        left_support=left,
        right_support=right,
        top_support=top,
        angle_confidence=angle_confidence,
        pixel_aspect_ratio=aspect,
    )


def _merge_door_frame_masks(
    items: list[tuple[np.ndarray, float]], image_shape: tuple[int, int],
) -> list[tuple[np.ndarray, float]]:
    if not items:
        return []
    masks = [item[0] for item in items]
    bboxes = [_bbox_from_mask(mask) for mask in masks]
    parents = list(range(len(masks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    image_h, image_w = image_shape
    gap = max(8, min(48, int(round(min(image_h, image_w) * 0.02))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap * 2 + 1, gap * 2 + 1),)
    dilated = [cv2.dilate(mask, kernel, iterations=1) for mask in masks]
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            if _door_frame_masks_should_merge(
                bboxes[left], bboxes[right], dilated[left], masks[right], gap,
            ):
                union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(len(masks)):
        clusters.setdefault(find(index), []).append(index)
    merged: list[tuple[np.ndarray, float]] = []
    for indexes in clusters.values():
        union_mask = np.zeros_like(masks[0], dtype=np.uint8)
        best_score = max(items[index][1] for index in indexes)
        for index in indexes:
            union_mask |= masks[index]
        merged.append((union_mask, best_score))
    return merged


def _door_frame_masks_should_merge(
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    left_dilated: np.ndarray,
    right_mask: np.ndarray,
    gap: int,
) -> bool:
    merged_bbox = (
        min(left_bbox[0], right_bbox[0]),
        min(left_bbox[1], right_bbox[1]),
        max(left_bbox[2], right_bbox[2]),
        max(left_bbox[3], right_bbox[3]),
    )
    left_aspect = _bbox_aspect(left_bbox)
    right_aspect = _bbox_aspect(right_bbox)
    merged_aspect = _bbox_aspect(merged_bbox)
    if (
        merged_aspect < 1.0
        and (left_aspect >= 1.0 or right_aspect >= 1.0)
        and (left_aspect < 1.0 or right_aspect < 1.0)
    ):
        return False
    if bool((left_dilated & right_mask).any()):
        return True
    dx = max(0, max(left_bbox[0], right_bbox[0]) - min(left_bbox[2], right_bbox[2]))
    dy = max(0, max(left_bbox[1], right_bbox[1]) - min(left_bbox[3], right_bbox[3]))
    if dx > gap or dy > gap:
        return False
    left_w, right_w = (
        max(1, left_bbox[2] - left_bbox[0]),
        max(1, right_bbox[2] - right_bbox[0]),
    )
    left_h, right_h = (
        max(1, left_bbox[3] - left_bbox[1]),
        max(1, right_bbox[3] - right_bbox[1]),
    )
    x_overlap = max(
        0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0])
    )
    y_overlap = max(
        0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1])
    )
    return (
        x_overlap / max(1.0, float(min(left_w, right_w))) >= 0.20
        or y_overlap / max(1.0, float(min(left_h, right_h))) >= 0.20
    )


def _significant_component_union(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask_bool = mask > 0
    if not mask_bool.any():
        return np.zeros(mask.shape, dtype=np.uint8), 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), 8,
    )
    if count <= 1:
        return mask_bool.astype(np.uint8), 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max()) if areas.size else 0
    min_area = max(64, int(round(max_area * 0.05)))
    significant = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]
    if not significant:
        significant = [1 + int(np.argmax(areas))]
    return np.isin(labels, significant).astype(np.uint8), len(significant)


def _frame_support(
    mask: np.ndarray, bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    roi = mask[y1:y2, x1:x2] > 0
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    height, width = roi.shape[:2]
    side_w = max(1, int(round(width * 0.25)))
    top_h = max(1, int(round(height * 0.25)))
    return (
        float(np.any(roi[:, :side_w], axis=1).sum()) / max(1, height),
        float(np.any(roi[:, width - side_w :], axis=1).sum()) / max(1, height),
        float(np.any(roi[:top_h, :], axis=0).sum()) / max(1, width),
        float(np.any(roi[height - top_h :, :], axis=0).sum()) / max(1, width),
    )


def _door_frame_quality(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    left_support: float,
    right_support: float,
    top_support: float,
    bbox_density: float,
    component_count: int,
) -> tuple[str, tuple[str, ...]]:
    del component_count
    image_h, image_w = image_shape
    x1, y1, x2, y2 = bbox
    width, height = max(0, x2 - x1), max(0, y2 - y1)
    if width <= 0 or height <= 0:
        return "fragment", ("empty_bbox",)
    flags: list[str] = []
    bbox_area_ratio = float(width * height) / max(1.0, float(image_h * image_w))
    aspect = height / max(1.0, float(width))
    if aspect < 1.0:
        flags.append("flat_bbox")
    if aspect > 4.0:
        flags.append("over_slender_bbox")
    if bbox_density < 0.015:
        flags.append("sparse_fragment")
    if bbox_density > 0.65:
        flags.append("filled_mask_not_frame")
    side_pair = left_support >= 0.35 and right_support >= 0.35
    top_ready = top_support >= 0.35
    critical = {"flat_bbox", "sparse_fragment", "filled_mask_not_frame"}
    if any(flag in critical for flag in flags):
        return "fragment", tuple(flags)
    if bbox_area_ratio < SMALL_BBOX_AREA_RATIO:
        flags.append("small_bbox_area")
        if side_pair and top_ready:
            return "partial", tuple(flags)
        flags.append("insufficient_frame_support")
        return "fragment", tuple(flags)
    if not flags and side_pair and top_ready:
        return "complete", ()
    if aspect >= 1.0 and (left_support >= 0.35 or right_support >= 0.35 or top_ready):
        return "partial", tuple(flags) if flags else ("missing_frame_support",)
    return "fragment", tuple(flags) if flags else ("insufficient_frame_support",)


def _door_frame_angle_confidence(
    mask: np.ndarray, bbox: tuple[int, int, int, int], quality: str,
) -> float:
    if quality == "fragment":
        return 0.0
    x1, y1, x2, y2 = bbox
    width, height = max(0, x2 - x1), max(0, y2 - y1)
    if width <= 0 or height <= 0:
        return 0.0
    trim = int(round(width * 0.08))
    scan_x1, scan_x2 = x1 + trim, x2 - trim
    if scan_x2 - scan_x1 < 12:
        return 0.0
    top_limit = y1 + max(1, int(round(height * 0.25)))
    points: list[tuple[float, float]] = []
    for x in range(scan_x1, scan_x2):
        ys = np.flatnonzero(mask[y1:top_limit, x] > 0)
        if ys.size:
            points.append((float(x), float(y1 + int(ys[0]))))
    coverage = len(points) / max(1, scan_x2 - scan_x1)
    if coverage < 0.45 or len(points) < 16:
        return 0.0
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(points_np, cv2.DIST_L1, 0, 0.01, 0.01)
    vx, vy = float(np.squeeze(vx)), float(np.squeeze(vy))
    x0, y0 = float(np.squeeze(x0)), float(np.squeeze(y0))
    if abs(vx) < 1e-6:
        return 0.0
    residuals = [abs(py - (y0 + (px - x0) * vy / vx)) for px, py in points]
    confidence = max(0.0, min(1.0, coverage))
    if float(np.median(np.asarray(residuals, dtype=np.float32))) > max(
        3.0, 0.02 * height,
    ):
        confidence *= 0.5
    return confidence


def _repair_door_frame_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if not mask_u8.any():
        return mask_u8
    x1, y1, x2, y2 = _bbox_from_mask(mask_u8)
    span = max(3, min(33, int(round(min(x2 - x1, y2 - y1) * 0.04))))
    if span % 2 == 0:
        span += 1
    repaired = cv2.morphologyEx(
        mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (span, 3)),
    )
    return cv2.morphologyEx(
        repaired, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, span)),
    ).astype(np.uint8)


def _candidate_anchor_area(candidate: _DoorTargetCandidate) -> float:
    mask = candidate.geometry.primary_mask
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None or len(points) < 3:
        return candidate.bbox_area
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    image_area = float(max(1, mask.shape[0] * mask.shape[1]))
    if hull_area > 0.0:
        return hull_area / image_area
    _, (width, height), _ = cv2.minAreaRect(points)
    rotated_area = float(width * height)
    return rotated_area / image_area if rotated_area > 0.0 else candidate.bbox_area


def _candidate_completeness_score(candidate: _DoorTargetCandidate) -> float:
    geometry = candidate.geometry
    quality_tier = 1.0 if geometry.quality == "complete" else 0.5
    support = (
        geometry.left_support + geometry.right_support + geometry.top_support
    ) / 3.0
    return 0.6 * quality_tier + 0.3 * support + 0.1 * geometry.angle_confidence


def _is_usable_door_frame(geometry: _DoorFrameGeometry) -> bool:
    if geometry.quality == "complete":
        return True
    if geometry.quality != "partial":
        return False
    disqualifying = {"flat_bbox", "sparse_fragment", "filled_mask_not_frame"}
    return not any(flag in disqualifying for flag in geometry.quality_flags)


def _prefer_non_slender_candidates(
    candidates: list[_DoorTargetCandidate],
) -> list[_DoorTargetCandidate]:
    preferred = [item for item in candidates if not item.is_slender_fallback]
    return preferred or candidates


def _extract_bbox_feature(
    frame: np.ndarray, bbox: tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    crop = frame[
        max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)
    ]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    vector = (
        cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        .ravel()
    )
    return vector / (np.linalg.norm(vector) + 1e-12)


def _feature_similarity(
    left: Optional[np.ndarray], right: Optional[np.ndarray],
) -> float:
    if left is None or right is None:
        return -1.0
    left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _vertical_overlap_ratio(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    min_height = max(1e-6, min(left[3] - left[1], right[3] - right[1]))
    return overlap / min_height


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_aspect(bbox: tuple[int, int, int, int]) -> float:
    return max(0, bbox[3] - bbox[1]) / max(1.0, float(bbox[2] - bbox[0]))
