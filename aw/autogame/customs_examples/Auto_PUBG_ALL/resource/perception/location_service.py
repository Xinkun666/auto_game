import cv2
import numpy as np


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
                 unstable_output_min_samples: int = 3):

        self.big_map = cv2.imread(big_map_path)
        if self.big_map is None:
            raise FileNotFoundError(f"无法读取大地图文件: {big_map_path}")

        # 1. 基础灰度转换
        self.big_map_gray = cv2.cvtColor(self.big_map, cv2.COLOR_BGR2GRAY)

        # 2. 【优化】使用 CLAHE 增强地图纹理，解决特征点空白区问题
        # clipLimit 越大对比度越强，tileGridSize 决定局部增强的网格大小
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.big_map_gray = self.clahe.apply(self.big_map_gray)

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
        self.last_match_quality = {}

    def reset_tracking(self) -> str:
        """清除跨场景的定位历史，强制下一帧从全局 SIFT 匹配重新收敛。"""
        self.mode = "unstable"
        self.history_points = []
        self.consecutive_corrections = 0
        zero_state = np.zeros((4, 1), dtype=np.float32)
        self.kf.statePre = zero_state.copy()
        self.kf.statePost = zero_state.copy()
        self.last_match_quality = {}
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

    def _preprocess_query(self, gray_curr):
        clahe = getattr(self, "clahe", None)
        return clahe.apply(gray_curr) if clahe is not None else gray_curr

    def _get_global_measured_point(self, gray_curr):
        gray_curr = self._preprocess_query(gray_curr)
        mask = None
        if self.is_circle:
            mask = np.zeros(gray_curr.shape, dtype=np.uint8)
            h, w = gray_curr.shape
            cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 2, 255, -1)

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
            return self._reject_global_match(
                "insufficient_good_matches",
                good_matches=good_count,
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

        inlier_flags = np.asarray(inlier_mask).reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inlier_flags))
        inlier_ratio = inlier_count / float(good_count)
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return self._reject_global_match(
                "weak_ransac_consensus",
                good_matches=good_count,
                inliers=inlier_count,
                inlier_ratio=inlier_ratio,
            )

        inlier_src = src_pts[inlier_flags]
        inlier_dst = dst_pts[inlier_flags]
        h, w = gray_curr.shape
        coverage_ratio = self._point_coverage_ratio(inlier_src, w, h)
        if coverage_ratio < self.min_inlier_coverage_ratio:
            return self._reject_global_match(
                "clustered_inliers",
                good_matches=good_count,
                inliers=inlier_count,
                inlier_ratio=inlier_ratio,
                coverage_ratio=coverage_ratio,
            )

        try:
            projected_inliers = cv2.perspectiveTransform(inlier_src, M)
        except cv2.error:
            return self._reject_global_match(
                "inlier_projection_failed",
                good_matches=good_count,
                inliers=inlier_count,
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
            )

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

        measured_point = (int(round(center_x)), int(round(center_y)))
        self.last_match_quality = {
            "accepted": True,
            "reason": "accepted",
            "good_matches": good_count,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "coverage_ratio": coverage_ratio,
            "median_reprojection_error": median_reprojection_error,
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
