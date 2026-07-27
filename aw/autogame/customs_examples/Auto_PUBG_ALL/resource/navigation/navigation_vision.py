"""Image-based navigation calibration and direction detection helpers."""

import json

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    round_to_nearest_5,
)


def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[index:index + 2], 16) for index in (0, 2, 4))


def extract_color_centers(
    image_path,
    target_hex="#00a2e8",
    tolerance=60,
    visualize=False,
):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_rgb = np.array(hex_to_rgb(target_hex), dtype=np.float32)
    distance = np.sqrt(np.sum((image_rgb - target_rgb) ** 2, axis=2))
    mask = (distance <= tolerance).astype(np.uint8)
    count, _, _, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    centers = [
        [int(centroids[index][0]), int(centroids[index][1])]
        for index in range(1, count)
    ]

    if visualize:
        visualization = image_bgr.copy()
        for x, y in centers:
            cv2.circle(visualization, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow(
            "Detected Centers",
            cv2.resize(visualization, (1500, 1500)),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Detected {len(centers)} color regions near {target_hex}")
    return centers


def find_boundaries(rgb_img):
    height, width = rgb_img.shape[:2]
    center_y, center_x = height // 2, width // 2

    upper_y = center_y
    for y in range(center_y, -1, -1):
        if not np.allclose(rgb_img[y, center_x], [0, 0, 0], atol=5):
            upper_y = y
            break

    left_x = 0
    for x in range(center_x, -1, -1):
        if np.allclose(rgb_img[center_y, x], [34, 154, 251], atol=10):
            left_x = x

    right_x = width - 1
    for x in range(center_x, width):
        if np.allclose(rgb_img[center_y, x], [255, 255, 255], atol=10):
            right_x = x

    return upper_y, left_x, right_x


def correct_speed_roi(img):
    config_path = r"config\config.json"
    template_path = r"resource\correct\zero.jpg"
    roi = (0.140, 0.888, 0.192, 0.946)
    x_ratio, y_ratio = 0.143, 0.473
    template = cv2.imread(template_path)
    height, width = img.shape[:2]
    x1, y1, x2, y2 = (
        int(roi[0] * width),
        int(roi[1] * height),
        int(roi[2] * width),
        int(roi[3] * height),
    )
    cropped = img[y1:y2, x1:x2]
    crop_height, crop_width = cropped.shape[:2]
    template_height, template_width = template.shape[:2]
    patch_height = max(4, int(crop_height * y_ratio))
    patch_width = max(4, int(crop_width * x_ratio))
    best_score = -1
    best_position = (0, 0)
    for y in range(0, crop_height - patch_height + 1):
        for x in range(0, crop_width - patch_width + 1):
            patch = cropped[y:y + patch_height, x:x + patch_width]
            resized = cv2.resize(patch, (template_width, template_height))
            score = cv2.matchTemplate(
                resized,
                template,
                cv2.TM_CCOEFF_NORMED,
            )[0][0]
            if score > best_score:
                best_score = score
                best_position = (x, y)

    normalized_x1 = (best_position[0] + x1) / width
    normalized_y1 = (best_position[1] + y1) / height
    result = (
        f"({normalized_x1:.4f}, {normalized_y1:.4f}, "
        f"{normalized_x1 + 0.0183:.4f}, {normalized_y1 + 0.0266:.4f})"
    )
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["roi"]["speed"] = result
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def correct_mini_map_roi(img):
    config_path = r"config\config.json"
    roi = (0.845, 0.228, 0.968, 0.267)
    height, width = img.shape[:2]
    x1, y1, x2, y2 = (
        int(roi[0] * width),
        int(roi[1] * height),
        int(roi[2] * width),
        int(roi[3] * height),
    )
    cropped = img[y1:y2, x1:x2]
    upper_y, left_x, right_x = find_boundaries(
        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
    )
    result = (
        f"({(x1 + left_x) / width:.4f}, 0.0000, "
        f"{(x1 + right_x) / width:.4f}, {(y1 + upper_y) / height:.4f})"
    )
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config["roi"]["location"] = result
    config["roi"]["white_angle"] = result
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def get_fast_running_status(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 80, 150]),
        np.array([35, 255, 255]),
    )
    yellow_ratio = np.sum(yellow_mask > 0) / (
        image.shape[0] * image.shape[1]
    )
    return int(0.05 < yellow_ratio < 0.75)


def detect_angel(
    image,
    white_thresh=180,
    eps=3,
    min_samples=1,
    eps_angle=3,
    angle_thresh=3,
):
    blue, green, red = cv2.split(image)
    white_mask = (
        (blue > white_thresh)
        & (green > white_thresh)
        & (red > white_thresh)
    ).astype(np.uint8)
    ys, xs = np.where(white_mask == 1)
    if len(xs) == 0:
        return None

    points = np.stack([xs, ys], axis=1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    centers = []
    for label in set(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        centers.append((int(np.mean(cluster[:, 0])), int(np.mean(cluster[:, 1]))))

    height, width = image.shape[:2]
    image_x, image_y = width / 2, height / 2
    angles = []
    for point_x, point_y in centers:
        angle = np.degrees(np.arctan2(image_y - point_y, point_x - image_x))
        angles.append(angle + 360 if angle < 0 else angle)

    if len(angles) < angle_thresh:
        return None
    angle_array = np.array(angles)
    angle_labels = DBSCAN(
        eps=eps_angle,
        min_samples=1,
    ).fit(angle_array.reshape(-1, 1)).labels_
    unique_labels, counts = np.unique(angle_labels, return_counts=True)
    best_label = unique_labels[np.argmax(counts)]
    if counts[np.argmax(counts)] < angle_thresh:
        return None

    final_angle = float(np.mean(angle_array[angle_labels == best_label]))
    rounded = round_to_nearest_5((450 - final_angle) % 360)
    return 360 if rounded == 0 else rounded
