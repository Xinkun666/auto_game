import re
import cv2
import ast
import math
import json
import subprocess
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Optional, Any, Tuple
from aw.autogame.tools.ProcessUtils import hidden_subprocess_kwargs

def hex_to_rgb(hex_str: str):
    """'#00a2e8' → (0, 162, 232)"""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def extract_color_centers(image_path, target_hex="#00a2e8", tolerance=60, visualize=False):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 转 RGB 并转换为 float32 便于距离计算
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    target_rgb = np.array(hex_to_rgb(target_hex), dtype=np.float32)

    # 计算颜色距离
    diff = img_rgb - target_rgb
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # 阈值筛选
    mask = (dist <= tolerance).astype(np.uint8)  # 1 = 近似目标颜色

    # 寻找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    for i in range(1, num_labels):  # 跳过背景 (label=0)
        x, y = centroids[i]  # 注意 centroids 是 (x, y)
        centers.append([int(x), int(y)])

    if visualize:
        vis = img_bgr.copy()
        for (x, y) in centers:
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        vis_resized = cv2.resize(vis, (1500, 1500))
        cv2.imshow("Detected Centers", vis_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Detected {len(centers)} color regions near {target_hex}")
    return centers

def find_nearest_point(points, point):
    if not points:
        raise ValueError("points 列表为空")

    pts = np.array(points)
    px, py = point
    # 计算所有点到 point 的欧氏距离平方
    dist_sq = (pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2
    idx = np.argmin(dist_sq)
    return tuple(pts[idx])


def get_relative_sector(base_dir: float, dir_angle: float, delta:int = 5) -> int:
    """
    判断目标角度相对于基准朝向的方位（扇区）。

    区间定义：
    1: [ 5,  30)  右前 (小)
    2: [ 30, 90)  右前 (大)
    3: [ 90, 180] 右后
    4: (-30, -5 ] 左前 (小)
    5: (-90, -30] 左前 (大)
    6: [-180,-90] 左后
    None: (-5, 5) 正前方死区

    :param base_dir: 当前朝向 (0~360)
    :param dir_angle: 目标朝向 (0~360)
    :return: 扇区编号 1-6 或 None
    """

    # 1. 计算相对角度差
    diff = dir_angle - base_dir

    # 2. 核心步骤：将角度归一化到 [-180, 180] 区间
    # 这样处理后：
    #   正右是 90，正左是 -90，正后是 -180 (或180)，正前是 0
    diff = (diff + 180) % 360 - 180

    # 3. 严格的分段判定 (使用半开半闭区间消除歧义)

    # --- 右侧 (Diff > 0) ---
    if delta <= diff < 30:  # 包含5，不包含30
        return 1
    elif 30 <= diff < 90:  # 包含30，不包含90
        return 2
    elif 90 <= diff <= 180:  # 包含90，包含180
        return 3

    # --- 左侧 (Diff < 0) ---
    elif -30 < diff <= -delta:  # 不包含-30，包含-5
        return 4
    elif -90 < diff <= -30:  # 不包含-90，包含-30
        return 5
    elif -180 <= diff <= -90:  # 包含-180，包含-90
        return 6

    # --- 死区 ( -5 < diff < 5 ) ---
    return None

def get_dms_rotation_mode():
    try:
        result = subprocess.run(
            ["hdc", "shell", "hidumper", "-s", "DisplayManagerService", "-a", "-a"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,  # 防止命令卡住
            **hidden_subprocess_kwargs(),
        )

        # 匹配第一个 Rotation: 后的数字，排除 ScreenRotation
        match = re.search(r"^\s*Rotation:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
        if match:
            return int(match.group(1))
        else:
            return None
    except subprocess.SubprocessError as e:
        print("Error running hdc:", e)
        return None

def parse_tuple_str(s):
    if not isinstance(s, str):
        return s  # 如果本身不是字符串，直接返回
    s = s.strip().strip('"').strip("'")
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"字符串格式错误：{s}")
    s = s[1:-1].strip()
    if not s:
        return ()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return tuple(float(x) for x in parts)
    except ValueError as e:
        raise ValueError(f"无法解析为数字：{s}") from e

def run_shell(cmd: str, r = False):
    try:
        if r:
            result = subprocess.run(
                cmd,
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                **hidden_subprocess_kwargs(),
            )
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            if result.returncode != 0 and not output:
                print(f"命令执行失败: {cmd}\nreturncode={result.returncode}")
                return None
            return output or None
        subprocess.run(cmd, shell=True, check=True, **hidden_subprocess_kwargs())
    except Exception as e:
        print(f"命令执行失败: {cmd}\n{e}")
        if r:
            return None


def _parse_screen_resolution(screen_info: str):
    if not screen_info:
        return None

    patterns = (
        r'activeMode:\s*(\d+)\s*x\s*(\d+)',
        r'render\s+resolution\s*=\s*(\d+)\s*x\s*(\d+)',
        r'physical\s+resolution\s*=\s*(\d+)\s*x\s*(\d+)',
        r'supportedMode\[\d+\]:\s*(\d+)\s*x\s*(\d+)',
    )
    for pattern in patterns:
        match = re.search(pattern, screen_info, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def get_resolution(r = True):
    resolution_mode = run_shell('hdc shell hidumper -s RenderService -a screen', r)
    resolution = _parse_screen_resolution(resolution_mode)
    if resolution:
        width, height = resolution
        return max(width, height), min(width, height)

    print('未能获取分辨率信息!')
    if resolution_mode:
        print(f"[Resolution] RenderService 输出片段: {resolution_mode[:500]}")
    return None, None

def get_buttons():
    resolution = get_resolution()
    assert resolution[0] is not None, '分辨率获取失败'
    button_dict = {}
    with open(r'config\config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        buttons = config["button"]
        for k,v in buttons.items():
            v = parse_tuple_str(v)
            button_dict[k] = (int(v[0] * resolution[0]), int(v[1] * resolution[1]))
    return button_dict

def get_wh():
    resolution = get_resolution()
    assert resolution[0] is not None, '分辨率获取失败'
    if resolution[0] > resolution[1]:
        w_h = resolution[0] / resolution[1]
    else:
        w_h = resolution[1] / resolution[0]
    with open(r'config\config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        width = config["width"]
        height = int(width * w_h)
        return width, height

def get_rois():
    w, h = get_wh()
    if w > h:
        resize_w, resize_h = w, h
    else:
        resize_w, resize_h = h, w
    roi_dict = {}
    with open(r'config\config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        rois = config["roi"]

        for k, v in rois.items():
            v = parse_tuple_str(v)
            roi_dict[k] = (int(v[0] * resize_w), int(v[1] * resize_h), int(v[2] * resize_w), int(v[3] * resize_h))

        return roi_dict

def get_brightness():
    command = 'hdc shell hidumper -s 3308'
    text = run_shell(command, r=True)
    value = int(re.search(r'DeviceBrightness=(\d+)', text).group(1))
    return value

def get_auto_brightness():
    command = 'hdc shell hidumper -s 3308'
    text = run_shell(command, r=True)
    state = re.search(r'Auto Adjust Brightness:\s*(ON|OFF)', text).group(1) == "ON"
    return state

def find_boundaries(rgb_img):
    height, width = rgb_img.shape[:2]
    h_c, w_c = height // 2, width // 2
    y_up = h_c
    for y in range(h_c, -1, -1):
        pixel = rgb_img[y, w_c]
        if not np.allclose(pixel, [0, 0, 0], atol=5):
            y_up = y
            break

    # 左边界
    x_left = 0
    for x in range(w_c, -1, -1):
        pixel = rgb_img[h_c, x]
        if np.allclose(pixel, [34, 154, 251], atol=10):  # 允许一定误差
            x_left = x
        else:
            continue

    # 右边界
    x_right = width - 1
    for x in range(w_c, width):
        pixel = rgb_img[h_c, x]
        if np.allclose(pixel, [255, 255, 255], atol=10):
            x_right = x
        else:
            continue

    return y_up, x_left, x_right

def correct_speed_roi(img):
    config_path = r'config\config.json'
    template_path = r'resource\correct\zero.jpg'
    ROI = (0.140, 0.888, 0.192, 0.946)
    x_ratio, y_ratio = 0.143, 0.473
    template = cv2.imread(template_path)
    h, w = img.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = int(ROI[0] * w), int(ROI[1] * h), int(ROI[2] * w), int(ROI[3] * h)
    crop_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
    h_c, w_c = crop_img.shape[:2]
    temp_h, temp_w = template.shape[:2]
    h_r = int(h_c * y_ratio)
    w_r = int(w_c * x_ratio)
    h_r = max(4, h_r)
    w_r = max(4, w_r)
    best_score = -1
    best_pos = (0, 0)
    for y in range(0, h_c - h_r + 1):
        for x in range(0, w_c - w_r + 1):
            patch = crop_img[y:y + h_r, x:x + w_r]
            patch_resized = cv2.resize(patch, (temp_w, temp_h))
            score = cv2.matchTemplate(patch_resized, template, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score = score
                best_pos = (x, y)
    x1, y1 = (best_pos[0] + roi_x1) / w, (best_pos[1] + roi_y1) / h
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    w_s, h_s = 0.0183, 0.0266
    x2, y2 = x1 + w_s, y1 + h_s
    result = f'({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})'
    config['roi']['speed'] = result
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def correct_mini_map_roi(img):
    config_path = r'config\config.json'
    ROI = [0.845, 0.228, 0.968, 0.267]
    h, w = img.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = int(ROI[0] * w), int(ROI[1] * h), int(ROI[2] * w), int(ROI[3] * h)
    crop_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
    rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    a, b, c = find_boundaries(rgb_img)
    x1 = (roi_x1 + b) / w
    y1 = 0 / h
    x2 = (roi_x1 + c) / w
    y2 = (roi_y1 + a) / h
    result = f'({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['roi']['location'] = result
    config['roi']['white_angle'] = result
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def round_to_nearest_5(angle):
    if angle > 357.5:
        return 360
    base = round(angle / 5) * 5
    if base == 5 and angle % 5 < 2.5:
        return 0
    return base % 360

def calculate_angle(current_point, target_point):
    x1, y1 = current_point
    x2, y2 = target_point
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    dx = x2 - x1
    dy = y1 - y2
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360
    if angle_deg < 0:
        angle_deg += 360
    angle_deg = (450 - angle_deg) % 360
    rect_angle = round_to_nearest_5(angle_deg)
    if rect_angle == 0:
        rect_angle = 360
    return rect_angle

def calculate_move_count(current_angle, target_angle):
    if target_angle is None or current_angle is None:
        return None, None, None
    if target_angle == current_angle:
        return 'right', 0, 0
    pixel_angle = {5 : 38, 10 : 70, 15 : 105, 20 :145, 25 : 183, 30: 220, 35: 250, 40: 290, 45 : 330, 55 : 405, 65 : 485,  90 : 580, 105: 660,  120 : 699, 150 : 821}
    angle_list = list(pixel_angle.keys())
    current_angle = current_angle % 360
    target_angle = target_angle % 360
    clockwise_distance = (target_angle - current_angle) % 360
    counterclockwise_distance = (current_angle - target_angle) % 360
    i = -1
    if clockwise_distance <= counterclockwise_distance:
        for angle in angle_list:
            if clockwise_distance >= angle:
                i = i + 1
            else:
                break
        direction = 'right'
    else:
        for angle in angle_list:
            if counterclockwise_distance >= angle:
                i = i + 1
            else:
                break
        direction = 'left'
    if i == -1:
        pixel = 0
        angle = 0
    else:
        pixel = pixel_angle[angle_list[i]]
        angle = angle_list[i]
    return direction, pixel, angle

def get_distance(coord1, coord2):
    if coord1[0] is None or coord1[1] is None:
        return -1
    return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

def get_fast_running_status(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 150])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(mask_yellow > 0) / (img.shape[0] * img.shape[1])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(mask_white > 0) / (img.shape[0] * img.shape[1])

    if 0.05 < yellow_ratio < 0.75:  # 黄色比例超过5%
        return 1
    else:
        return 0

def analyze_distance(dists):
    if len(dists) > 1:
        if dists[-1] == -1:
            return 1
        if dists[-1] <= 3:
            return 0
        else:
            return 1
    return 1

def extract_keys(src_dict, key_list):
    return {k: src_dict[k] for k in key_list if k in src_dict}


CAR_POINT_CONFIG = {
    "m_city": {
        "destination": (1534, 1228),
        "road_points": [
            (1484, 1190),
            (1481, 1204),
            (1471, 1222),
            (1451, 1252),
            (1502, 1211),
            (1520, 1222),
            (1559, 1232),
            (1585, 1242),
            (1680, 1234),
            (1598, 1215),
            (1635, 1236),
        ],
    },
    "r_city": {
        "destination": (1131, 763),
        "road_points": [
            (1134, 766),
            (1134, 763),
            (1130, 770),
            (1121, 767),
            (1118, 748),
            (1147, 745),
            (1147, 769),
        ],
    },
}

def generate_shortest_path(start_point):
    with open('config\config.json', 'r') as f:
        config = json.load(f)
    car_points = ast.literal_eval(config['car_points'])
    if not car_points:
        return []
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    def nearest_neighbor_route(start, points):
        unvisited = points.copy()
        route = [start]
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda p: dist(current, p))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return route
    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    d1 = dist(route[i], route[i + 1]) + dist(route[j], route[j + 1])
                    d2 = dist(route[i], route[j]) + dist(route[i + 1], route[j + 1])
                    if d2 < d1:
                        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                        improved = True
        return route
    start = start_point if start_point is not None else car_points[0]
    route = nearest_neighbor_route(start, car_points)
    road_lists = two_opt(route)

    return road_lists[1:]

def find_path(start, city="r_city", tol=1):
    car_cfg = CAR_POINT_CONFIG.get(city)
    if car_cfg is None:
        raise ValueError(f"未知 car_points 配置: {city}")

    target = car_cfg["destination"]
    points = car_cfg["road_points"]

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    pts = set(points)
    cur = start
    path = [start]

    while True:
        if d(cur, target) <= tol:
            path.append(target)
            return path[1:]

        nxt = None
        for point in sorted(pts, key=lambda p: d(cur, p)):
            if d(point, target) < d(cur, target):
                nxt = point
                break

        if nxt is None:
            path.append(target)
            return path[1:]

        path.append(nxt)
        pts.remove(nxt)
        cur = nxt

def detect_angel(image,white_thresh=180,eps=3,min_samples=1,eps_angle=3,angle_thresh=3):
    b, g, r = cv2.split(image)
    white_mask = ((b > white_thresh) &
                  (g > white_thresh) &
                  (r > white_thresh)).astype(np.uint8)

    ys, xs = np.where(white_mask == 1)
    points = np.stack([xs, ys], axis=1)
    if len(points) == 0:
        return None
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
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
        dy = cy_img - py  # 图像 y 轴向下，需反向
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360
        angles.append(angle)
    angles = np.array(angles)
    if len(angles) >= angle_thresh:
        angle_clustering = DBSCAN(eps=eps_angle, min_samples=1).fit(angles.reshape(-1,1))
        angle_labels = angle_clustering.labels_
        unique_labels, counts = np.unique(angle_labels, return_counts=True)
        best_label = unique_labels[np.argmax(counts)]
        best_count = counts[np.argmax(counts)]
        if best_count < angle_thresh:
            return None
        else:
            final_angle = float(np.mean(angles[angle_labels == best_label]))
            final_angle = (450 - final_angle) % 360
            rect_angle = round_to_nearest_5(final_angle)
            if rect_angle == 0:
                rect_angle = 360
            return rect_angle
    return None

def stable_angle(angle_list, eps_angle=3):
    valid_angles = [a for a in angle_list if a is not None]

    if len(valid_angles) < 3:
        return None

    angles = np.array(valid_angles).reshape(-1, 1)

    clustering = DBSCAN(eps=eps_angle, min_samples=1).fit(angles)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)

    best_label = unique_labels[np.argmax(counts)]
    best_count = counts[np.argmax(counts)]

    if best_count > 20:
        cluster_angles = angles[labels == best_label].flatten()
        return float(np.mean(cluster_angles))
    return None


def is_location_stagnant(history_points):
    """
    判断历史轨迹是否在原地打转（基于几何中心聚类）。

    Args:
        history_points: list of tuples, e.g. [(10, 20), (None, None), (11, 21)...]

    Returns:
        bool: True (困死/原地打转), False (正常移动)
    """
    # 1. 数据清洗：过滤掉 (None, None) 或包含 None 的坐标
    # 这里的判断条件确保 x 和 y 都不为 None
    valid_points = [
        pt for pt in history_points
        if pt is not None and pt[0] is not None and pt[1] is not None
    ]

    # 如果有效数据太少（比如全是 None），无法计算，默认为 False
    if not valid_points:
        return False

    n = len(valid_points)

    # 2. 计算均值点 (Centroid)
    sum_x = sum(p[0] for p in valid_points)
    sum_y = sum(p[1] for p in valid_points)
    centroid_x = sum_x / n
    centroid_y = sum_y / n

    # 3. 距离检测
    # 阈值设置为 3，为了避免开根号 (sqrt) 带来的性能损耗，直接比较平方值
    # 距离 < 3  等价于  距离平方 < 9
    threshold_sq = 3 ** 2

    for x, y in valid_points:
        # 计算当前点到均值点的 欧几里得距离的平方
        dist_sq = (x - centroid_x) ** 2 + (y - centroid_y) ** 2

        # 只要有一个点距离均值点太远，说明没有被困死
        if dist_sq >= threshold_sq:
            return False

    # 所有有效点都在均值点半径 3 的范围内
    return True

def parse_route_to_dicts(route: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    终极合并版函数：
    1. 解析原始 route 嵌套结构。
    2. 自动提取 move_press 的持续时间。
    3. 自动合并连续的 view_slide 角度（支持左右抵消）。
    """
    final_actions = []

    # 视角角度累加器：左为正，右为负
    pending_angle = 0.0

    i = 0
    total_steps = len(route)

    while i < total_steps:
        step = route[i]
        item_type = step.get("item_type")

        # 只关心 ACTION，忽略孤立的 INTERVAL（move_press 后的除外）
        if item_type == "ACTION":
            action_body = step.get("action", {})
            method = action_body.get("method")
            args = action_body.get("args", {})

            # -------------------------------------------------
            # 情况 A: 视角滑动 (只累加，不生成)
            # -------------------------------------------------
            if method == "view_slide":
                angle = args.get("angle", 0.0)
                direction = args.get("direction")

                if direction == "LEFT":
                    pending_angle += angle
                elif direction == "RIGHT":
                    pending_angle -= angle

            # -------------------------------------------------
            # 情况 B: 移动按下 (先结算视角，再生成移动)
            # -------------------------------------------------
            elif method == "move_press":
                # 1. 【结算】检查是否有未保存的视角动作
                if abs(pending_angle) > 1e-5:
                    final_dir = "LEFT" if pending_angle > 0 else "RIGHT"
                    final_val = abs(pending_angle)
                    final_actions.append({
                        "action": "view_slide",
                        "direction": final_dir,
                        "angle": round(final_val, 2)
                    })
                    pending_angle = 0.0  # 结算完清零

                # 2. 【前瞻】获取移动持续时间
                interval_time = 0.0
                if i + 1 < total_steps:
                    next_step = route[i + 1]
                    if next_step.get("item_type") == "INTERVAL":
                        interval_time = next_step.get("interval", 0.0)

                # 3. 【生成】移动动作
                final_actions.append({
                    "action": "move_press",
                    "init_angle": args.get("init_angle"),
                    "interval": round(interval_time, 4)
                })

        # 继续循环
        i += 1

    # -------------------------------------------------
    # 收尾: 循环结束后，如果还有没结算的视角动作，补加上去
    # -------------------------------------------------
    if abs(pending_angle) > 1e-5:
        final_dir = "LEFT" if pending_angle > 0 else "RIGHT"
        final_val = abs(pending_angle)
        final_actions.append({
            "action": "view_slide",
            "direction": final_dir,
            "angle": round(final_val, 2)
        })

    return final_actions

if __name__ == '__main__':
    print(find_path((1623, 1220)))
