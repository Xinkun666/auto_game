from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np

from gametest_proxy.pubg_room_explore.door_calibration.types import CalibrationLine
from gametest_proxy.pubg_room_explore.sam3.segmenter import SegmentationResult


@dataclass
class MaskGeometry:
    mask: np.ndarray
    primary_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    primary_bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask_area_ratio: float
    bbox_norm_xyxy: tuple[float, float, float, float]
    centroid_xy_norm: tuple[float, float]
    largest_contour_xy: Optional[np.ndarray]
    largest_contour_area_ratio: float
    all_contours_xy: tuple[np.ndarray, ...] = field(default_factory=tuple)
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    fallback_reason: str = ""


DoorFrameQuality = Literal["complete", "partial", "fragment"]
SMALL_BBOX_AREA_RATIO = 0.01


@dataclass
class DoorFrameComponent:
    bbox_xyxy: tuple[int, int, int, int]
    area: int


@dataclass
class DoorFrameGeometry:
    mask: np.ndarray
    primary_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    primary_bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask_area_ratio: float
    bbox_norm_xyxy: tuple[float, float, float, float]
    centroid_xy_norm: tuple[float, float]
    largest_contour_xy: Optional[np.ndarray]
    largest_contour_area_ratio: float
    quality: DoorFrameQuality
    components: tuple[DoorFrameComponent, ...]
    left_support: float
    right_support: float
    top_support: float
    bottom_support: float
    bbox_density: float
    pixel_aspect_ratio: float
    all_contours_xy: tuple[np.ndarray, ...] = field(default_factory=tuple)
    angle_line: Optional[CalibrationLine] = None
    angle_confidence: float = 0.0
    angle_reason: str = ""
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    fallback_reason: str = ""


def extract_mask_geometry(
    segmentation_result: SegmentationResult,
    image_shape: tuple[int, int],
) -> MaskGeometry:
    image_h, image_w = image_shape
    mask = (segmentation_result.mask > 0).astype(np.uint8)
    image_area = float(max(1, image_h * image_w))
    primary_mask, contour, contour_area_ratio = _extract_primary_mask(mask, image_area)
    contours = mask_contours_xy(primary_mask)
    primary_x1, primary_y1, primary_x2, primary_y2 = _bbox_from_mask(
        primary_mask,
        fallback=segmentation_result.bbox_xyxy,
    )
    return MaskGeometry(
        mask=mask,
        primary_mask=primary_mask,
        bbox_xyxy=segmentation_result.bbox_xyxy,
        primary_bbox_xyxy=(primary_x1, primary_y1, primary_x2, primary_y2),
        score=float(segmentation_result.score),
        mask_area_ratio=float(primary_mask.sum()) / image_area,
        bbox_norm_xyxy=(
            primary_x1 / max(1, image_w),
            primary_y1 / max(1, image_h),
            primary_x2 / max(1, image_w),
            primary_y2 / max(1, image_h),
        ),
        centroid_xy_norm=_centroid_xy_norm(primary_mask, image_w, image_h),
        largest_contour_xy=contour,
        largest_contour_area_ratio=contour_area_ratio,
        all_contours_xy=contours,
    )


def extract_door_frame_geometry(
    segmentation_result: SegmentationResult,
    image_shape: tuple[int, int],
) -> DoorFrameGeometry:
    image_h, image_w = image_shape
    mask = (segmentation_result.mask > 0).astype(np.uint8)
    image_area = float(max(1, image_h * image_w))
    repaired_mask = _repair_door_frame_mask(mask)
    significant_mask, components = _significant_component_union(repaired_mask)
    if not significant_mask.any():
        significant_mask = repaired_mask.copy()
    x1, y1, x2, y2 = _bbox_from_mask(
        significant_mask,
        fallback=segmentation_result.bbox_xyxy,
    )
    contour, contour_area_ratio = _largest_contour(significant_mask, image_area)
    contours = mask_contours_xy(significant_mask)
    left, right, top, bottom = _frame_support(significant_mask, (x1, y1, x2, y2))
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    bbox_density = float(significant_mask.sum()) / float(bbox_area)
    pixel_aspect_ratio = float(y2 - y1) / max(1.0, float(x2 - x1))
    quality, flags = _door_frame_quality(
        bbox_xyxy=(x1, y1, x2, y2),
        image_shape=image_shape,
        left_support=left,
        right_support=right,
        top_support=top,
        bbox_density=bbox_density,
        component_count=len(components),
    )
    angle_line, angle_confidence, angle_reason = _fit_door_frame_angle_line(
        significant_mask,
        (x1, y1, x2, y2),
        quality=quality,
    )
    return DoorFrameGeometry(
        mask=mask,
        primary_mask=significant_mask,
        bbox_xyxy=segmentation_result.bbox_xyxy,
        primary_bbox_xyxy=(x1, y1, x2, y2),
        score=float(segmentation_result.score),
        mask_area_ratio=float(significant_mask.sum()) / image_area,
        bbox_norm_xyxy=(
            x1 / max(1, image_w),
            y1 / max(1, image_h),
            x2 / max(1, image_w),
            y2 / max(1, image_h),
        ),
        centroid_xy_norm=_centroid_xy_norm(significant_mask, image_w, image_h),
        largest_contour_xy=contour,
        largest_contour_area_ratio=contour_area_ratio,
        all_contours_xy=contours,
        quality=quality,
        components=tuple(components),
        left_support=left,
        right_support=right,
        top_support=top,
        bottom_support=bottom,
        bbox_density=bbox_density,
        pixel_aspect_ratio=pixel_aspect_ratio,
        angle_line=angle_line,
        angle_confidence=angle_confidence,
        angle_reason=angle_reason,
        quality_flags=flags,
        fallback_reason=angle_reason if angle_line is None else "",
    )


def extract_door_frame_geometries(
    segmentation_results: list[SegmentationResult],
    image_shape: tuple[int, int],
) -> list[DoorFrameGeometry]:
    merged_results = _merge_door_frame_segmentation_results(
        segmentation_results,
        image_shape,
    )
    geometries = [
        extract_door_frame_geometry(result, image_shape=image_shape)
        for result in merged_results
    ]
    return sorted(
        geometries,
        key=lambda geometry: (-geometry.score, -geometry.mask_area_ratio),
    )


def fit_top_profile_line(
    geometry: MaskGeometry,
    *,
    x_trim_ratio: float = 0.1,
    top_band_ratio: float = 0.35,
    min_height_width_ratio: float = 1.2,
    min_column_coverage: float = 0.4,
    min_sample_count: int = 12,
    max_angle_deg: float = 25.0,
) -> Optional[CalibrationLine]:
    geometry.quality_flags = ()
    geometry.fallback_reason = ""

    x1, y1, x2, y2 = geometry.primary_bbox_xyxy
    bbox_w = max(0, x2 - x1)
    bbox_h = max(0, y2 - y1)
    if bbox_w <= 0 or bbox_h <= 0:
        _set_quality_failure(geometry, "empty_bbox")
        return None
    if bbox_h / max(1, bbox_w) < min_height_width_ratio:
        _set_quality_failure(geometry, "bad_aspect_ratio")
        return None

    trim_px = int(round(bbox_w * x_trim_ratio))
    scan_x1 = x1 + trim_px
    scan_x2 = x2 - trim_px
    if scan_x2 - scan_x1 < 2:
        _set_quality_failure(geometry, "empty_bbox")
        return None

    band_y_limit = y1 + max(1, int(round(bbox_h * top_band_ratio)))
    points: list[tuple[float, float]] = []
    sampled_columns = 0
    for x in range(scan_x1, scan_x2):
        ys = np.flatnonzero(geometry.primary_mask[:, x] > 0)
        if ys.size == 0:
            continue
        top_y = int(ys[0])
        if top_y > band_y_limit:
            continue
        sampled_columns += 1
        points.append((float(x), float(top_y)))

    coverage = sampled_columns / max(1, scan_x2 - scan_x1)
    if coverage < min_column_coverage:
        _set_quality_failure(geometry, "insufficient_column_coverage")
        return None
    if len(points) < min_sample_count:
        _set_quality_failure(geometry, "insufficient_top_samples")
        return None

    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(points_np, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(np.squeeze(vx))
    vy = float(np.squeeze(vy))
    x0 = float(np.squeeze(x0))
    y0 = float(np.squeeze(y0))
    if abs(vx) < 1e-6:
        _set_quality_failure(geometry, "angle_out_of_range")
        return None

    start_x = float(scan_x1)
    end_x = float(scan_x2 - 1)
    start_y = y0 + (start_x - x0) * vy / vx
    end_y = y0 + (end_x - x0) * vy / vx
    angle_deg = float(np.degrees(np.arctan2(end_y - start_y, end_x - start_x)))
    if abs(angle_deg) > max_angle_deg:
        _set_quality_failure(geometry, "angle_out_of_range")
        return None

    return CalibrationLine(
        angle_deg=angle_deg,
        score=float(len(points)),
        label="sam3_top_profile",
        start=(int(round(start_x)), int(round(start_y))),
        end=(int(round(end_x)), int(round(end_y))),
    )


def _centroid_xy_norm(
    mask: np.ndarray,
    image_w: int,
    image_h: int,
) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0.5, 0.5
    return float(xs.mean() / max(1, image_w)), float(ys.mean() / max(1, image_h))


def _largest_contour(
    mask: np.ndarray,
    image_area: float,
) -> tuple[Optional[np.ndarray], float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    return contour, float(cv2.contourArea(contour) / max(1.0, image_area))


def mask_contours_xy(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return tuple(sorted(contours, key=cv2.contourArea, reverse=True))


def _repair_door_frame_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if not mask_u8.any():
        return mask_u8
    x1, y1, x2, y2 = _bbox_from_mask(mask_u8, fallback=(0, 0, 0, 0))
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    kernel_span = max(3, min(33, int(round(min(width, height) * 0.04))))
    if kernel_span % 2 == 0:
        kernel_span += 1
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_span, 3),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, kernel_span),
    )
    repaired = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, horizontal_kernel)
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, vertical_kernel)
    return repaired.astype(np.uint8)


def _merge_door_frame_segmentation_results(
    segmentation_results: list[SegmentationResult],
    image_shape: tuple[int, int],
) -> list[SegmentationResult]:
    masks: list[np.ndarray] = []
    bboxes: list[tuple[int, int, int, int]] = []
    source_indexes: list[int] = []
    for index, result in enumerate(segmentation_results):
        mask = (result.mask > 0).astype(np.uint8)
        if not mask.any():
            continue
        masks.append(mask)
        bboxes.append(_bbox_from_mask(mask, fallback=result.bbox_xyxy))
        source_indexes.append(index)
    if not masks:
        return []

    parents = list(range(len(masks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    image_h, image_w = image_shape
    merge_gap_px = max(8, min(48, int(round(min(image_h, image_w) * 0.02))))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (merge_gap_px * 2 + 1, merge_gap_px * 2 + 1),
    )
    dilated_masks = [cv2.dilate(mask, dilation_kernel, iterations=1) for mask in masks]

    for left_index in range(len(masks)):
        for right_index in range(left_index + 1, len(masks)):
            if _door_frame_masks_should_merge(
                bboxes[left_index],
                bboxes[right_index],
                dilated_masks[left_index],
                masks[right_index],
                merge_gap_px,
            ):
                union(left_index, right_index)

    clusters: dict[int, list[int]] = {}
    for index in range(len(masks)):
        clusters.setdefault(find(index), []).append(index)

    merged_results: list[SegmentationResult] = []
    for cluster_indexes in clusters.values():
        union_mask = np.zeros_like(masks[0], dtype=np.uint8)
        best_result = segmentation_results[source_indexes[cluster_indexes[0]]]
        best_score = float(best_result.score)
        for cluster_index in cluster_indexes:
            result = segmentation_results[source_indexes[cluster_index]]
            union_mask |= masks[cluster_index]
            if float(result.score) > best_score:
                best_result = result
                best_score = float(result.score)
        merged_results.append(
            SegmentationResult(
                segmented_bgr=best_result.segmented_bgr,
                mask=(union_mask * 255).astype(np.uint8),
                bbox_xyxy=_bbox_from_mask(union_mask, fallback=best_result.bbox_xyxy),
                score=best_score,
                cropped_bgr=best_result.cropped_bgr,
                cropped_mask=best_result.cropped_mask,
            )
        )
    return merged_results


def _door_frame_masks_should_merge(
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    left_dilated_mask: np.ndarray,
    right_mask: np.ndarray,
    merge_gap_px: int,
) -> bool:
    if _merge_would_flatten_vertical_door_candidate(left_bbox, right_bbox):
        return False

    if bool((left_dilated_mask & right_mask).any()):
        return True

    dx = max(0, max(left_bbox[0], right_bbox[0]) - min(left_bbox[2], right_bbox[2]))
    dy = max(0, max(left_bbox[1], right_bbox[1]) - min(left_bbox[3], right_bbox[3]))
    if dx > merge_gap_px or dy > merge_gap_px:
        return False

    left_w = max(1, left_bbox[2] - left_bbox[0])
    right_w = max(1, right_bbox[2] - right_bbox[0])
    left_h = max(1, left_bbox[3] - left_bbox[1])
    right_h = max(1, right_bbox[3] - right_bbox[1])
    x_overlap = max(
        0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0])
    )
    y_overlap = max(
        0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1])
    )
    x_overlap_ratio = x_overlap / max(1.0, float(min(left_w, right_w)))
    y_overlap_ratio = y_overlap / max(1.0, float(min(left_h, right_h)))
    return x_overlap_ratio >= 0.20 or y_overlap_ratio >= 0.20


def _merge_would_flatten_vertical_door_candidate(
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
) -> bool:
    left_aspect = _bbox_height_width_ratio(left_bbox)
    right_aspect = _bbox_height_width_ratio(right_bbox)
    merged_aspect = _bbox_height_width_ratio(_union_bbox(left_bbox, right_bbox))
    return (
        merged_aspect < 1.0
        and (left_aspect >= 1.0 or right_aspect >= 1.0)
        and (left_aspect < 1.0 or right_aspect < 1.0)
    )


def _bbox_height_width_ratio(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, y2 - y1) / max(1.0, float(x2 - x1))


def _union_bbox(
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(left_bbox[0], right_bbox[0]),
        min(left_bbox[1], right_bbox[1]),
        max(left_bbox[2], right_bbox[2]),
        max(left_bbox[3], right_bbox[3]),
    )


def _significant_component_union(
    mask: np.ndarray,
) -> tuple[np.ndarray, list[DoorFrameComponent]]:
    mask_bool = mask > 0
    if not mask_bool.any():
        return np.zeros(mask.shape, dtype=np.uint8), []
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype("uint8"),
        8,
    )
    if component_count <= 1:
        bbox = _bbox_from_mask(mask_bool.astype("uint8"), fallback=(0, 0, 0, 0))
        return mask_bool.astype("uint8"), [
            DoorFrameComponent(bbox_xyxy=bbox, area=int(mask_bool.sum()))
        ]

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max()) if areas.size else 0
    min_area = max(64, int(round(max_area * 0.05)))
    significant_labels: list[int] = []
    components: list[DoorFrameComponent] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        significant_labels.append(label)
        components.append(
            DoorFrameComponent(
                bbox_xyxy=(x, y, x + w, y + h),
                area=area,
            )
        )
    if not significant_labels:
        primary_label = 1 + int(np.argmax(areas))
        significant_labels = [primary_label]
        x = int(stats[primary_label, cv2.CC_STAT_LEFT])
        y = int(stats[primary_label, cv2.CC_STAT_TOP])
        w = int(stats[primary_label, cv2.CC_STAT_WIDTH])
        h = int(stats[primary_label, cv2.CC_STAT_HEIGHT])
        components = [
            DoorFrameComponent(
                bbox_xyxy=(x, y, x + w, y + h),
                area=int(stats[primary_label, cv2.CC_STAT_AREA]),
            )
        ]
    union_mask = np.isin(labels, significant_labels).astype(np.uint8)
    components.sort(key=lambda item: item.area, reverse=True)
    return union_mask, components


def _frame_support(
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    roi = mask[y1:y2, x1:x2] > 0
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    height, width = roi.shape[:2]
    side_w = max(1, int(round(width * 0.25)))
    top_h = max(1, int(round(height * 0.25)))
    left = float(np.any(roi[:, :side_w], axis=1).sum()) / max(1, height)
    right = float(np.any(roi[:, width - side_w :], axis=1).sum()) / max(1, height)
    top = float(np.any(roi[:top_h, :], axis=0).sum()) / max(1, width)
    bottom = float(np.any(roi[height - top_h :, :], axis=0).sum()) / max(1, width)
    return left, right, top, bottom


def _door_frame_quality(
    *,
    bbox_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    left_support: float,
    right_support: float,
    top_support: float,
    bbox_density: float,
    component_count: int,
) -> tuple[DoorFrameQuality, tuple[str, ...]]:
    image_h, image_w = image_shape
    x1, y1, x2, y2 = bbox_xyxy
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    flags: list[str] = []
    if width <= 0 or height <= 0:
        return "fragment", ("empty_bbox",)
    image_area = max(1.0, float(image_h * image_w))
    bbox_area_ratio = float(width * height) / image_area
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
    critical_flags = {
        "flat_bbox",
        "sparse_fragment",
        "filled_mask_not_frame",
    }
    if any(flag in critical_flags for flag in flags):
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


def _fit_door_frame_angle_line(
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    quality: DoorFrameQuality,
) -> tuple[Optional[CalibrationLine], float, str]:
    if quality == "fragment":
        return None, 0.0, "door_frame_fragment"

    x1, y1, x2, y2 = bbox_xyxy
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 0 or height <= 0:
        return None, 0.0, "empty_bbox"

    trim_px = int(round(width * 0.08))
    scan_x1 = x1 + trim_px
    scan_x2 = x2 - trim_px
    if scan_x2 - scan_x1 < 12:
        return None, 0.0, "insufficient_top_span"

    top_limit = y1 + max(1, int(round(height * 0.25)))
    points: list[tuple[float, float]] = []
    for x in range(scan_x1, scan_x2):
        ys = np.flatnonzero(mask[y1:top_limit, x] > 0)
        if ys.size == 0:
            continue
        points.append((float(x), float(y1 + int(ys[0]))))

    coverage = len(points) / max(1, scan_x2 - scan_x1)
    if coverage < 0.45 or len(points) < 16:
        return None, 0.0, "insufficient_top_support"

    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(points_np, cv2.DIST_L1, 0, 0.01, 0.01)
    vx = float(np.squeeze(vx))
    vy = float(np.squeeze(vy))
    x0 = float(np.squeeze(x0))
    y0 = float(np.squeeze(y0))
    if abs(vx) < 1e-6:
        return None, 0.0, "angle_out_of_range"

    start_x = float(scan_x1)
    end_x = float(scan_x2 - 1)
    start_y = y0 + (start_x - x0) * vy / vx
    end_y = y0 + (end_x - x0) * vy / vx
    angle_deg = float(np.degrees(np.arctan2(end_y - start_y, end_x - start_x)))

    residuals = [
        abs(point_y - (y0 + (point_x - x0) * vy / vx)) for point_x, point_y in points
    ]
    median_residual = float(np.median(np.asarray(residuals, dtype=np.float32)))

    line = CalibrationLine(
        angle_deg=angle_deg,
        score=float(len(points)),
        label="sam3_door_frame_top",
        start=(int(round(start_x)), int(round(start_y))),
        end=(int(round(end_x)), int(round(end_y))),
    )
    confidence = max(0.0, min(1.0, coverage))
    reason = "top_profile"
    if median_residual > max(3.0, 0.02 * height):
        reason = "top_profile_noisy"
        confidence *= 0.5
    return line, confidence, reason


def _extract_primary_mask(
    mask: np.ndarray,
    image_area: float,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 1:
        contour, contour_area_ratio = _largest_contour(mask, image_area)
        return mask.copy(), contour, contour_area_ratio

    primary_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    primary_mask = (labels == primary_label).astype(np.uint8)
    contour, contour_area_ratio = _largest_contour(primary_mask, image_area)
    return primary_mask, contour, contour_area_ratio


def _bbox_from_mask(
    mask: np.ndarray,
    *,
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    del fallback
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _set_quality_failure(geometry: MaskGeometry, reason: str) -> None:
    geometry.quality_flags = (reason,)
    geometry.fallback_reason = reason
