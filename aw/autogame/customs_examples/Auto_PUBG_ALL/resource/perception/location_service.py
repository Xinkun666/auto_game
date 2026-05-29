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
                 max_corrections: int = 4):

        self.big_map = cv2.imread(big_map_path)
        if self.big_map is None:
            raise FileNotFoundError(f"无法读取大地图文件: {big_map_path}")

        # 1. 基础灰度转换
        self.big_map_gray = cv2.cvtColor(self.big_map, cv2.COLOR_BGR2GRAY)

        # 2. 【优化】使用 CLAHE 增强地图纹理，解决特征点空白区问题
        # clipLimit 越大对比度越强，tileGridSize 决定局部增强的网格大小
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.big_map_gray = clahe.apply(self.big_map_gray)

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

    def _get_global_measured_point(self, gray_curr):
        mask = None
        if self.is_circle:
            mask = np.zeros(gray_curr.shape, dtype=np.uint8)
            h, w = gray_curr.shape
            cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 2, 255, -1)

        kp_small, des_small = self.sift.detectAndCompute(gray_curr, mask)
        if des_small is None or len(kp_small) < 4:
            return None

        matches = self.flann.knnMatch(des_small, self.des_big, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) >= 4:
            src_pts = np.float32([kp_small[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_big[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = gray_curr.shape
                center_pts = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
                dst_center = cv2.perspectiveTransform(center_pts, M)
                return (int(dst_center[0][0][0]), int(dst_center[0][0][1]))
        return None

    def get_location(self, img) -> tuple:
        if img is None or img.size == 0:
            return (None, None), self.mode

        gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 建议：实时画面也可以考虑做一次 CLAHE，但如果游戏画面本来就很清晰则不需要
        measured_point = self._get_global_measured_point(gray_curr)

        if self.mode == "unstable":
            if measured_point is None:
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
            return measured_point, self.mode

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