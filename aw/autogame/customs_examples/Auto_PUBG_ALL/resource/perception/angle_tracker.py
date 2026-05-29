from collections import deque
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


class AngleTracker:
    def __init__(
        self,
        window_size=30,
        white_thresh=180,
        eps=3,
        min_samples=1,
        eps_angle=3,
        angle_thresh=3,
        stable_min_count=20,
    ):
        self.window_size = window_size
        self.white_thresh = white_thresh
        self.eps = eps
        self.min_samples = min_samples
        self.eps_angle = eps_angle
        self.angle_thresh = angle_thresh
        self.stable_min_count = stable_min_count

        # 维护一个固定长度窗口
        self.angle_window = deque(maxlen=window_size)

    def round_to_nearest_5(self, angle):
        return int(round(angle / 5.0) * 5)

    def detect_angle(self, image):
        b, g, r = cv2.split(image)
        white_mask = ((b > self.white_thresh) &
                      (g > self.white_thresh) &
                      (r > self.white_thresh)).astype(np.uint8)

        ys, xs = np.where(white_mask == 1)
        points = np.stack([xs, ys], axis=1)

        if len(points) == 0:
            return None

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_

        centers = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_pts = points[labels == label]
            cx = int(np.mean(cluster_pts[:, 0]))
            cy = int(np.mean(cluster_pts[:, 1]))
            centers.append((cx, cy))

        h, w = image.shape[:2]
        cx_img, cy_img = w / 2, h / 2

        angles = []
        for (px, py) in centers:
            dx = px - cx_img
            dy = cy_img - py
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 360
            angles.append(angle)

        angles = np.array(angles)

        if len(angles) >= self.angle_thresh:
            angle_clustering = DBSCAN(
                eps=self.eps_angle,
                min_samples=1
            ).fit(angles.reshape(-1, 1))

            angle_labels = angle_clustering.labels_
            unique_labels, counts = np.unique(angle_labels, return_counts=True)

            best_label = unique_labels[np.argmax(counts)]
            best_count = counts[np.argmax(counts)]

            if best_count < self.angle_thresh:
                return None
            else:
                final_angle = float(np.mean(angles[angle_labels == best_label]))
                final_angle = (450 - final_angle) % 360
                rect_angle = self.round_to_nearest_5(final_angle)
                if rect_angle == 0:
                    rect_angle = 360
                return rect_angle

        return None

    def stable_angle(self, angle_list):
        valid_angles = [a for a in angle_list if a is not None]

        if len(valid_angles) < 3:
            return None

        angles = np.array(valid_angles).reshape(-1, 1)

        clustering = DBSCAN(eps=self.eps_angle, min_samples=1).fit(angles)
        labels = clustering.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)

        best_label = unique_labels[np.argmax(counts)]
        best_count = counts[np.argmax(counts)]

        if best_count >= self.stable_min_count:
            cluster_angles = angles[labels == best_label].flatten()
            return float(np.mean(cluster_angles))

        return None

    def get_angle(self, img):
        """
        输入一张图：
        1. 先做单帧检测
        2. 放入30帧窗口
        3. 返回稳定角度
        """
        current_angle = self.detect_angle(img)
        self.angle_window.append(current_angle)

        final_angle = self.stable_angle(list(self.angle_window))
        return final_angle