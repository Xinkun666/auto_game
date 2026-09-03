import cv2
import numpy as np
from typing import Optional


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class LocatePoints:
    def __init__(self,
                 big_map_path: str = r"aw/autogame/customs_examples/Auto_PUBG_ALL/resource/map/hpjy.png",
                 is_circle: bool = False,
                 init_stable_frames: int = 5,
                 stability_thresh: int = 50,
                 correction_thresh: int = 80,
                 max_corrections: int = 4,
                 min_good_matches: int = 8,
                 min_inliers: int = 6,
                 min_inlier_ratio: float = 0.5,
                 max_median_reprojection_error: float = 5.0,
                 min_inlier_coverage_ratio: float = 0.01,
                 unstable_output_min_samples: int = 3,
                 max_local_interference_ratio: float = 0.30,
                 local_residual_grid_size: int = 8,
                 min_clean_region_similarity: float = 0.55,
                 enable_local_map_validation: bool = True):

        self.big_map = cv2.imread(big_map_path)
        if self.big_map is None:
            raise FileNotFoundError(f"无法读取大地图文件: {big_map_path}")

        # 1. 基础灰度转换
        self.big_map_gray = cv2.cvtColor(self.big_map, cv2.COLOR_BGR2GRAY)

        # 2. 【优化】使用 CLAHE 增强地图纹理，解决特征点空白区问题
        # clipLimit 越大对比度越强，tileGridSize 决定局部增强的网格大小
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.big_map_gray = self.clahe.apply(self.big_map_gray)
        self.big_map_support = np.full(self.big_map_gray.shape, 255, dtype=np.uint8)

        self.is_circle = is_circle

        # 3. 【优化】SIFT 初始化：降低 contrastThreshold 从 0.04 到 0.02
        # 增加 nfeatures 到 20000 以应对大地图多出的细微特征
        self.sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.02, edgeThreshold=10)

        # 提取特征
        self.kp_big, self.des_big = self.sift.detectAndCompute(self.big_map_gray, None)

        # 4. 【可视化】保存特征点提取结果，用于检查空白区是否改善
        img_vis = cv2.drawKeypoints(
            self.big_map,  # 在原图上画
            self.kp_big,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/map/map_features_debug.png', img_vis)
        print(f"特征点提取完成，共 {len(self.kp_big)} 个点。可视化已保存至 map_features_debug.png")

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # --- 卡尔曼滤波器初始化 (保持不变) ---
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.mode = "unstable"
        self.history_points = []
        self.consecutive_corrections = 0
        self.stability_thresh = stability_thresh
        self.init_stable_frames = init_stable_frames
        self.correction_thresh = correction_thresh
        self.max_corrections = max_corrections
        self.min_good_matches = max(4, int(min_good_matches))
        self.min_inliers = max(4, int(min_inliers))
        self.min_inlier_ratio = min(1.0, max(0.0, float(min_inlier_ratio)))
        self.max_median_reprojection_error = max(
            0.0,
            float(max_median_reprojection_error),
        )
        self.min_inlier_coverage_ratio = min(
            1.0,
            max(0.0, float(min_inlier_coverage_ratio)),
        )
        self.unstable_output_min_samples = max(1, int(unstable_output_min_samples))
        # 动态小地图提示（枪声、脚步等）只应占据局部区域。这里不按颜色判断，
        # 而是在候选单应矩阵成立后，检查大地图回投影与当前画面的局部结构一致性。
        self.max_local_interference_ratio = min(
            1.0,
            max(0.0, float(max_local_interference_ratio)),
        )
        self.local_residual_grid_size = max(2, int(local_residual_grid_size))
        self.min_clean_region_similarity = min(
            1.0,
            max(-1.0, float(min_clean_region_similarity)),
        )
        self.enable_local_map_validation = bool(enable_local_map_validation)
        self.last_match_quality = {}
        self.last_local_interference_mask = None

    def reset_tracking(self) -> str:
        """清除跨场景的定位历史，强制下一帧从全局 SIFT 匹配重新收敛。"""
        self.mode = "unstable"
        self.history_points = []
        self.consecutive_corrections = 0
        zero_state = np.zeros((4, 1), dtype=np.float32)
        self.kf.statePre = zero_state.copy()
        self.kf.statePost = zero_state.copy()
        self.last_match_quality = {}
        self.last_local_interference_mask = None
        return self.mode

    def _reject_global_match(self, reason: str, **metrics):
        self.last_match_quality = {
            "accepted": False,
            "reason": reason,
            **metrics,
        }
        return None

    @staticmethod
    def _point_coverage_ratio(points, image_width: int, image_height: int) -> float:
        if points is None or len(points) < 2 or image_width <= 0 or image_height <= 0:
            return 0.0
        flat = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        width = max(0.0, float(np.max(flat[:, 0]) - np.min(flat[:, 0])))
        height = max(0.0, float(np.max(flat[:, 1]) - np.min(flat[:, 1])))
        return (width * height) / float(image_width * image_height)

    def _single_match_fallback(self, matches, keypoints, image_shape):
        """给特征不足的离线帧提供仅供审计显示的单点候选。

        单一特征无法估计单应矩阵，故调用方不得将此结果作为正式定位。这里选择
        Lowe 比率最低的匹配，并假定它与小地图中心只有平移差，用于在地图上标红。
        """
        best = None
        for pair in matches:
            if len(pair) < 2:
                continue
            first, second = pair[0], pair[1]
            denominator = max(float(second.distance), 1e-12)
            ratio = float(first.distance) / denominator
            ranking = (ratio, float(first.distance))
            if best is None or ranking < best[0]:
                best = (ranking, first)
        if best is None:
            return None

        _, match = best
        query_x, query_y = keypoints[match.queryIdx].pt
        map_x, map_y = self.kp_big[match.trainIdx].pt
        height, width = image_shape[:2]
        candidate_x = float(map_x) + (width / 2.0 - float(query_x))
        candidate_y = float(map_y) + (height / 2.0 - float(query_y))
        map_height, map_width = self.big_map_gray.shape[:2]
        if not (0 <= candidate_x < map_width and 0 <= candidate_y < map_height):
            return None
        ratio, distance = best[0]
        return {
            "point": (int(round(candidate_x)), int(round(candidate_y))),
            "lowe_ratio": ratio,
            "distance": distance,
            "confidence": max(0.0, min(1.0, 1.0 - ratio)),
        }

    def _preprocess_query(self, gray_curr):
        clahe = getattr(self, "clahe", None)
        return clahe.apply(gray_curr) if clahe is not None else gray_curr

    @staticmethod
    def _local_gradient(image: np.ndarray) -> np.ndarray:
        """返回对整体色偏不敏感的局部地图结构。"""
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y)

    def _local_valid_mask(self, image_shape: tuple[int, int]) -> np.ndarray:
        """生成真正参与小地图一致性计算的区域，圆外永远不参与。"""
        height, width = image_shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        if getattr(self, "is_circle", False):
            mask.fill(0)
            cv2.circle(mask, (width // 2, height // 2), min(height, width) // 2 - 2, 255, -1)
        return mask

    @staticmethod
    def _masked_correlation(
        left: np.ndarray, right: np.ndarray, valid: np.ndarray,
    ) -> Optional[float]:
        left_values = left[valid]
        right_values = right[valid]
        if len(left_values) < 16:
            return None
        left_centered = left_values - float(np.mean(left_values))
        right_centered = right_values - float(np.mean(right_values))
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denominator <= 1e-6:
            return None
        return float(np.dot(left_centered, right_centered) / denominator)

    def _validate_local_map_agreement(
        self, gray_curr: np.ndarray, homography: np.ndarray,
    ) -> dict:
        """验证候选地图变换是否仅有不超过阈值的局部结构残差。

        红区会改变整张小地图的色调，因此这里比较 CLAHE 后的梯度结构，而不是
        RGB/HSV 颜色。返回的 mask 仅用于诊断和内点分布检查，绝不参与导航。
        """
        height, width = gray_curr.shape[:2]
        valid_mask = self._local_valid_mask((height, width))
        try:
            inverse_homography = np.linalg.inv(homography)
            reference = cv2.warpPerspective(
                self.big_map_gray,
                inverse_homography,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            source_support = getattr(self, "big_map_support", None)
            if source_support is None or source_support.shape != self.big_map_gray.shape:
                source_support = np.full(self.big_map_gray.shape, 255, dtype=np.uint8)
                self.big_map_support = source_support
            reference_support = cv2.warpPerspective(
                source_support,
                inverse_homography,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        except (cv2.error, np.linalg.LinAlgError):
            return {"evaluated": False, "reason": "local_reference_warp_failed"}

        valid = (valid_mask > 0) & (reference_support > 0)
        valid_area = int(np.count_nonzero(valid))
        if valid_area < 64:
            return {"evaluated": False, "reason": "local_valid_area_too_small"}

        query_gradient = self._local_gradient(gray_curr)
        reference_gradient = self._local_gradient(reference)
        grid_size = max(2, int(getattr(self, "local_residual_grid_size", 8)))
        y_edges = np.linspace(0, height, grid_size + 1, dtype=int)
        x_edges = np.linspace(0, width, grid_size + 1, dtype=int)
        low_similarity_cells = np.zeros((grid_size, grid_size), dtype=np.uint8)
        cell_areas = np.zeros((grid_size, grid_size), dtype=np.int32)
        similarities = []

        for row in range(grid_size):
            for column in range(grid_size):
                y1, y2 = y_edges[row], y_edges[row + 1]
                x1, x2 = x_edges[column], x_edges[column + 1]
                cell_valid = valid[y1:y2, x1:x2]
                area = int(np.count_nonzero(cell_valid))
                cell_areas[row, column] = area
                if area < 16:
                    continue
                similarity = self._masked_correlation(
                    query_gradient[y1:y2, x1:x2],
                    reference_gradient[y1:y2, x1:x2],
                    cell_valid,
                )
                if similarity is None:
                    continue
                similarities.append(similarity)
                if similarity < float(getattr(self, "min_clean_region_similarity", 0.55)):
                    low_similarity_cells[row, column] = 1

        # 纯色/无纹理小地图无法进行这种验证；保留原有的 SIFT 质量门槛处理它，
        # 而不是因辅助检查不可评估而拒绝一个正常定位。
        if len(similarities) < 4:
            return {
                "evaluated": False,
                "reason": "local_structure_not_evaluable",
                "valid_area": valid_area,
                "evaluated_cells": len(similarities),
            }

        interference_mask = np.zeros((height, width), dtype=np.uint8)
        for row, column in zip(*np.nonzero(low_similarity_cells)):
            interference_mask[
                y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]
            ] = valid_mask[
                y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]
            ]
        interference_mask[reference_support == 0] = 0
        interference_area = int(np.count_nonzero(interference_mask))
        interference_ratio = interference_area / float(valid_area)

        component_count, labels, _, _ = cv2.connectedComponentsWithStats(
            low_similarity_cells, connectivity=8,
        )
        largest_component_area = 0
        for label in range(1, component_count):
            largest_component_area = max(
                largest_component_area,
                int(np.sum(cell_areas[labels == label])),
            )
        largest_component_ratio = largest_component_area / float(valid_area)

        clean_similarities = []
        for row in range(grid_size):
            for column in range(grid_size):
                if cell_areas[row, column] < 16 or low_similarity_cells[row, column]:
                    continue
                y1, y2 = y_edges[row], y_edges[row + 1]
                x1, x2 = x_edges[column], x_edges[column + 1]
                similarity = self._masked_correlation(
                    query_gradient[y1:y2, x1:x2],
                    reference_gradient[y1:y2, x1:x2],
                    valid[y1:y2, x1:x2],
                )
                if similarity is not None:
                    clean_similarities.append(similarity)

        clean_similarity = float(np.median(clean_similarities)) if clean_similarities else -1.0
        max_ratio = float(getattr(self, "max_local_interference_ratio", 0.30))
        # ``interference_ratio`` 是所有低一致性格子的并集，固定 UI、压缩噪声或
        # 轻微色偏会让它分散在地图多处。用户定义的 30% 是单个局部污染块上限，
        # 因此只限制最大连通块；其余区域仍须满足结构一致性和既有 SIFT 质量门槛。
        accepted = (
            largest_component_ratio <= max_ratio
            and bool(clean_similarities)
            and clean_similarity >= float(getattr(self, "min_clean_region_similarity", 0.55))
        )
        if accepted:
            reason = "accepted"
        elif largest_component_ratio > max_ratio:
            reason = "local_interference_exceeds_limit"
        else:
            reason = "local_clean_structure_too_weak"
        return {
            "evaluated": True,
            "accepted": accepted,
            "reason": reason,
            "valid_area": valid_area,
            "evaluated_cells": len(similarities),
            "low_similarity_cells": int(np.count_nonzero(low_similarity_cells)),
            "interference_ratio": interference_ratio,
            "largest_interference_ratio": largest_component_ratio,
            "clean_similarity": clean_similarity,
            "interference_mask": interference_mask,
        }

    def _get_global_measured_point(self, gray_curr):
        gray_curr = self._preprocess_query(gray_curr)
        mask = None
        if self.is_circle:
            mask = self._local_valid_mask(gray_curr.shape[:2])

        kp_small, des_small = self.sift.detectAndCompute(gray_curr, mask)
        if des_small is None or len(kp_small) < 4:
            return self._reject_global_match(
                "insufficient_query_features",
                query_keypoints=len(kp_small or []),
            )

        try:
            matches = self.flann.knnMatch(des_small, self.des_big, k=2)
        except cv2.error:
            return self._reject_global_match("flann_match_failed")
        good = [
            pair[0]
            for pair in matches
            if len(pair) >= 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        good_count = len(good)
        if good_count < self.min_good_matches:
            single_match_fallback = self._single_match_fallback(
                matches,
                kp_small,
                gray_curr.shape,
            )
            return self._reject_global_match(
                "insufficient_good_matches",
                good_matches=good_count,
                single_match_fallback=single_match_fallback,
            )

        src_pts = np.float32([kp_small[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp_big[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except cv2.error:
            return self._reject_global_match(
                "homography_estimation_failed",
                good_matches=good_count,
            )
        if M is None or inlier_mask is None or not np.all(np.isfinite(M)):
            return self._reject_global_match(
                "invalid_homography",
                good_matches=good_count,
            )

        # 从此处开始，单应矩阵已经可把小地图中心投影回大地图。即使后续质量门槛
        # 拒绝该帧，也保留 candidate_point 供离线审计显示；调用方仍只会收到 None。
        h, w = gray_curr.shape
        center_pts = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
        try:
            dst_center = cv2.perspectiveTransform(center_pts, M).reshape(-1, 2)[0]
        except cv2.error:
            return self._reject_global_match("center_projection_failed")
        if not np.all(np.isfinite(dst_center)):
            return self._reject_global_match("invalid_projected_center")

        center_x, center_y = float(dst_center[0]), float(dst_center[1])
        map_h, map_w = self.big_map_gray.shape[:2]
        if not (0 <= center_x < map_w and 0 <= center_y < map_h):
            return self._reject_global_match(
                "projected_center_out_of_map",
                center=(center_x, center_y),
                map_size=(map_w, map_h),
            )
        candidate_point = (int(round(center_x)), int(round(center_y)))

        inlier_flags = np.asarray(inlier_mask).reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inlier_flags))
        inlier_ratio = inlier_count / float(good_count)
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return self._reject_global_match(
                "weak_ransac_consensus",
                good_matches=good_count,
                inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                candidate_point=candidate_point,
            )

        inlier_src = src_pts[inlier_flags]
        inlier_dst = dst_pts[inlier_flags]
        coverage_ratio = self._point_coverage_ratio(inlier_src, w, h)
        if coverage_ratio < self.min_inlier_coverage_ratio:
            return self._reject_global_match(
                "clustered_inliers",
                good_matches=good_count,
                inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                coverage_ratio=coverage_ratio,
                candidate_point=candidate_point,
            )

        try:
            projected_inliers = cv2.perspectiveTransform(inlier_src, M)
        except cv2.error:
            return self._reject_global_match(
                "inlier_projection_failed",
                good_matches=good_count,
                inliers=inlier_count,
                candidate_point=candidate_point,
            )
        reprojection_errors = np.linalg.norm(
            projected_inliers.reshape(-1, 2) - inlier_dst.reshape(-1, 2),
            axis=1,
        )
        median_reprojection_error = float(np.median(reprojection_errors))
        if (
            not np.isfinite(median_reprojection_error)
            or median_reprojection_error > self.max_median_reprojection_error
        ):
            return self._reject_global_match(
                "high_reprojection_error",
                good_matches=good_count,
                inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                coverage_ratio=coverage_ratio,
                median_reprojection_error=median_reprojection_error,
                candidate_point=candidate_point,
            )

        if getattr(self, "enable_local_map_validation", True):
            local_validation = self._validate_local_map_agreement(gray_curr, M)
            interference_mask = local_validation.pop("interference_mask", None)
        else:
            local_validation = {"evaluated": False, "reason": "disabled_for_this_locator"}
            interference_mask = None
        self.last_local_interference_mask = interference_mask
        if local_validation.get("evaluated") and not local_validation.get("accepted"):
            return self._reject_global_match(
                local_validation.get("reason", "local_map_agreement_failed"),
                good_matches=good_count,
                inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                coverage_ratio=coverage_ratio,
                median_reprojection_error=median_reprojection_error,
                local_validation=local_validation,
                candidate_point=candidate_point,
            )

        if interference_mask is not None and int(np.count_nonzero(interference_mask)):
            inlier_coordinates = np.rint(inlier_src.reshape(-1, 2)).astype(int)
            height, width = interference_mask.shape
            inside_interference = [
                bool(interference_mask[y, x])
                for x, y in inlier_coordinates
                if 0 <= x < width and 0 <= y < height
            ]
            outside_inliers = inlier_count - int(np.count_nonzero(inside_interference))
            if outside_inliers < self.min_inliers:
                return self._reject_global_match(
                    "inliers_concentrated_in_local_interference",
                    good_matches=good_count,
                    inliers=inlier_count,
                    inliers_outside_interference=outside_inliers,
                    local_validation=local_validation,
                    candidate_point=candidate_point,
                )
            local_validation["inliers_outside_interference"] = outside_inliers
        measured_point = candidate_point
        self.last_match_quality = {
            "accepted": True,
            "reason": "accepted",
            "good_matches": good_count,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "coverage_ratio": coverage_ratio,
            "median_reprojection_error": median_reprojection_error,
            "local_validation": local_validation,
            "candidate_point": candidate_point,
            "point": measured_point,
        }
        return measured_point

    def get_global_location(self, img) -> tuple:
        """对单张、非连续画面直接做全局配准，不使用卡尔曼历史状态。

        ``get_location`` 面向连续视频流：稳定后会根据上一帧预测拒绝跳变较大的
        测量值。离线截图、抽帧数据或跨场景图片之间的位置通常不连续，应调用
        本方法，让每一张图片独立用 SIFT/单应性匹配到大地图。
        """
        if img is None or img.size == 0:
            return (None, None), "global"

        gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        measured_point = self._get_global_measured_point(gray_curr)
        if measured_point is None:
            return (None, None), "global"
        return measured_point, "global"

    def get_location(self, img) -> tuple:
        if img is None or img.size == 0:
            return (None, None), self.mode

        gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        measured_point = self._get_global_measured_point(gray_curr)

        if self.mode == "unstable":
            if measured_point is None:
                return (None, None), self.mode

            if self.history_points:
                movement = get_distance(measured_point, self.history_points[-1])
                if movement > self.stability_thresh:
                    self.history_points = [measured_point]
                    return (None, None), self.mode

            self.history_points.append(measured_point)

            if len(self.history_points) >= self.init_stable_frames:
                movements = [get_distance(self.history_points[i], self.history_points[i - 1])
                             for i in range(1, len(self.history_points))]

                if all(0 <= m <= self.stability_thresh for m in movements):
                    self.mode = "stable"
                    self.consecutive_corrections = 0
                    curr_x, curr_y = measured_point
                    self.kf.statePost = np.array([[curr_x], [curr_y], [0], [0]], np.float32)

                self.history_points.pop(0)
            if len(self.history_points) < self.unstable_output_min_samples:
                return (None, None), self.mode
            recent_points = np.asarray(
                self.history_points[-self.unstable_output_min_samples:],
                dtype=np.float32,
            )
            median_point = tuple(
                int(round(value))
                for value in np.median(recent_points, axis=0)
            )
            return median_point, self.mode

        elif self.mode == "stable":
            prediction = self.kf.predict()
            predicted_point = (int(prediction[0]), int(prediction[1]))

            needs_correction = measured_point is None or \
                               get_distance(measured_point, predicted_point) > self.correction_thresh

            if needs_correction:
                final_point = predicted_point
                self.consecutive_corrections += 1
                self.kf.statePost = prediction
            else:
                meas = np.array([[np.float32(measured_point[0])],
                                 [np.float32(measured_point[1])]], np.float32)
                self.kf.correct(meas)
                final_point = measured_point
                self.consecutive_corrections = 0

            if self.consecutive_corrections >= self.max_corrections:
                self.mode = "unstable"
                self.consecutive_corrections = 0
                self.history_points = []
            elif final_point is not None:
                self.history_points.append(final_point)
                if len(self.history_points) > 30:
                    self.history_points.pop(0)

            return final_point, self.mode
        return (None, None), self.mode
