import os
import cv2
import math
import json
import numpy as np
from sklearn.cluster import DBSCAN

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROUTE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "map", "hpjy.png")
DEFAULT_ROUTE_OUTPUT_PATH = os.path.join("aw", "autogame", "temp", "road", "route.jpg")

def draw_points_with_arrows(
    road_list,
    image_path=DEFAULT_ROUTE_IMAGE_PATH,
    output_path=DEFAULT_ROUTE_OUTPUT_PATH,
):
    from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.map_navigation import (
        draw_points_with_arrows as _draw_points_with_arrows,
    )

    return _draw_points_with_arrows(road_list, image_path=image_path, output_path=output_path)

def detect_red_text(image_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return False, 0

    # 2. 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 定义红色的范围 (红色在HSV中分布在两端)
    # 范围1: 0-10
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    # 范围2: 170-180
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 4. 创建红色掩膜并合并
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    # 5. 形态学处理：去除微小噪声
    kernel = np.ones((2, 2), np.uint8)
    # 开运算：先腐蚀后膨胀
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 6. 统计红色像素占比
    red_pixel_count = cv2.countNonZero(red_mask)
    total_pixels = img.shape[0] * img.shape[1]
    red_ratio = (red_pixel_count / total_pixels) * 100

    # 7. 根据比例或像素数判断 (阈值可根据实际情况微调)
    # 对于点阵大字，通常像素数会比较多
    has_red_text = (red_pixel_count > 500 )

    return has_red_text, red_pixel_count

def get_distance(coord1, coord2):
    if coord1[0] is None or coord1[1] is None:
        return -1
    return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

def check_location(location):
    # 1. 首先排除最基本的 None（比如获取不到信息的情况）
    if location is None:
        return None

    # 2. 针对 (None, None) 或 [None, None] 的情况进行判断
    # 只要 X 或 Y 只要有一个是 None，我们就认为这个点是无效的
    try:
        x, y = location[0], location[1]
        if x is None or y is None:
            return None
    except (IndexError, TypeError):
        # 如果 location 格式不对（比如不是列表或元组），直接按无效处理
        return None

    # 3. 校验通过，返回有效的坐标
    return location

def load_json(json_file):
    if not os.path.exists(json_file):
        print(f"错误: 找不到指定的 JSON 文件 -> {json_file}")
        return None

    try:
        # 2. 以读取模式打开文件，并指定 utf-8 编码防止中文或特殊字符乱码
        with open(json_file, 'r', encoding='utf-8') as f:
            # 3. 将文件内容解析为 Python 字典
            data = json.load(f)

        print(f"成功加载 JSON 数据，共包含 {len(data)} 个房子的信息。")
        return data

    except Exception as e:
        print(f"读取 JSON 文件时发生未知错误: {e}")
        return None

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

def calculate_move_count_old(current_angle, target_angle):
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

def calculate_move_count(current_dir, target_angle):
    """
    计算从 current_dir 转到 target_angle 所需的:
    1. 转向方向 ('left' 或 'right')
    2. 滑动像素
       - 小角度(<20°): 使用经验查表，避免线性模型过冲
       - 大角度(>=20°): 使用线性回归模型: px = 5.31 * angle + 49
    3. 角度差值
    """
    if current_dir is None or target_angle is None:
        return None, None, None
    try:
        current_dir = float(current_dir)
        target_angle = float(target_angle)
    except (TypeError, ValueError):
        return None, None, None

    # 1. 计算最小角度差 (0-180)
    diff = (target_angle - current_dir) % 360
    if diff > 180:
        diff = 360 - diff
        turn_dir = 'left'
    else:
        turn_dir = 'right'

    # 2. 如果角度差极小，直接返回 0
    if diff < 1.0:
        return turn_dir, 0, diff

    # 3. 小角度使用查表，大角度使用线性拟合
    if diff < 20:
        pixel_angle = {
            5: 38,
            10: 70,
            15: 105,
            20: 145,
            25: 183,
            30: 220,
            35: 250,
            40: 290,
            45: 330,
            55: 405,
            65: 485,
            90: 580,
            105: 660,
            120: 699,
            150: 821,
        }
        angle_list = sorted(pixel_angle.keys())

        matched_angle = 0
        for angle in angle_list:
            if diff >= angle:
                matched_angle = angle
            else:
                break

        pixel = pixel_angle[matched_angle] if matched_angle > 0 else 0
    else:
        pixel = int(diff * 5.31 + 49)

    return turn_dir, pixel, diff


class AdaptiveTurnTable:
    MIN_ANGLE = 5
    MAX_ANGLE = 180
    ANGLE_STEP = 5
    UPDATE_ALPHA = 0.45
    MIN_OBSERVED_DEG = 1.0
    MIN_PX = 8
    MIN_DURA = 80
    SCALE_MIN = 0.45
    SCALE_MAX = 1.8

    def __init__(self):
        self.table = {"left": {}, "right": {}}

    def angle_bin(self, angle):
        try:
            value = float(angle)
        except (TypeError, ValueError):
            return None
        if value < self.MIN_OBSERVED_DEG:
            return None
        value = max(self.MIN_ANGLE, min(self.MAX_ANGLE, value))
        return int(round(value / self.ANGLE_STEP) * self.ANGLE_STEP)

    def get(self, turn_dir, diff, fallback_px, fallback_dura):
        angle_key = self.angle_bin(diff)
        if turn_dir not in self.table or angle_key is None:
            return int(fallback_px or 0), int(fallback_dura or 0), angle_key

        entry = self.table[turn_dir].get(angle_key)
        if entry:
            return int(entry["px"]), int(entry["dura"]), angle_key

        return int(fallback_px or 0), int(fallback_dura or 0), angle_key

    def observe(self, turn_dir, desired_diff, before_angle, after_angle, used_px, used_dura):
        angle_key = self.angle_bin(desired_diff)
        if turn_dir not in self.table or angle_key is None:
            return

        observed = self._observed_turn_degrees(turn_dir, before_angle, after_angle)
        if observed is None or observed < self.MIN_OBSERVED_DEG:
            return

        scale = angle_key / observed
        scale = max(self.SCALE_MIN, min(self.SCALE_MAX, scale))
        measured_px = max(self.MIN_PX, int(round(abs(float(used_px or 0)) * scale)))
        measured_dura = max(self.MIN_DURA, int(round(float(used_dura or 0) * scale)))

        entry = self.table[turn_dir].get(angle_key)
        if not entry:
            self.table[turn_dir][angle_key] = {
                "px": measured_px,
                "dura": measured_dura,
                "samples": 1,
            }
            return

        alpha = self.UPDATE_ALPHA
        entry["px"] = int(round(entry["px"] * (1.0 - alpha) + measured_px * alpha))
        entry["dura"] = int(round(entry["dura"] * (1.0 - alpha) + measured_dura * alpha))
        entry["samples"] = int(entry.get("samples", 0)) + 1

    @staticmethod
    def _observed_turn_degrees(turn_dir, before_angle, after_angle):
        try:
            before = float(before_angle) % 360.0
            after = float(after_angle) % 360.0
        except (TypeError, ValueError):
            return None

        if turn_dir == "right":
            observed = (after - before) % 360.0
        elif turn_dir == "left":
            observed = (before - after) % 360.0
        else:
            return None

        if observed > 180.0:
            return None
        return observed


adaptive_turn_table = AdaptiveTurnTable()


def get_adaptive_turn_motion(turn_dir, diff, fallback_px, fallback_dura):
    return adaptive_turn_table.get(turn_dir, diff, fallback_px, fallback_dura)


def update_adaptive_turn_motion(turn_dir, desired_diff, before_angle, after_angle, used_px, used_dura):
    adaptive_turn_table.observe(turn_dir, desired_diff, before_angle, after_angle, used_px, used_dura)


class AdaptiveForwardMoveTable:
    MIN_DISTANCE = 1
    MAX_DISTANCE = 60
    DISTANCE_STEP = 1
    UPDATE_ALPHA = 0.45
    MIN_OBSERVED_DISTANCE = 0.2
    MIN_Y_BIAS = 80
    MAX_Y_BIAS = 520
    MIN_DURA = 80
    MAX_DURA = 2600
    MIN_WAIT = 120
    MAX_WAIT = 7000
    MIN_WAIT_PAD = 120
    SCALE_MIN = 0.55
    SCALE_MAX = 1.85

    def __init__(self):
        self.table = {}

    def distance_bin(self, distance):
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return None
        if value < self.MIN_OBSERVED_DISTANCE:
            return None
        value = max(self.MIN_DISTANCE, min(self.MAX_DISTANCE, value))
        return int(round(value / self.DISTANCE_STEP) * self.DISTANCE_STEP)

    def get(self, mode, desired_distance, fallback_y_bias, fallback_dura, fallback_wait):
        distance_key = self.distance_bin(desired_distance)
        if distance_key is None:
            return int(fallback_y_bias or 0), int(fallback_dura or 0), int(fallback_wait or 0), distance_key

        mode_table = self.table.get(mode, {})
        entry = mode_table.get(distance_key)
        if entry:
            return int(entry["y_bias"]), int(entry["dura"]), int(entry["wait"]), distance_key

        return int(fallback_y_bias or 0), int(fallback_dura or 0), int(fallback_wait or 0), distance_key

    def observe(self, mode, desired_distance, before_distance, after_distance, used_y_bias, used_dura, used_wait):
        distance_key = self.distance_bin(desired_distance)
        if distance_key is None:
            return

        observed = self._observed_forward_distance(before_distance, after_distance)
        if observed is None or observed < self.MIN_OBSERVED_DISTANCE:
            return

        scale = distance_key / observed
        scale = max(self.SCALE_MIN, min(self.SCALE_MAX, scale))

        measured_y = self._scaled_forward_bias(used_y_bias, scale)
        measured_dura = max(self.MIN_DURA, min(self.MAX_DURA, int(round(float(used_dura or 0) * scale))))
        measured_wait = max(
            measured_dura + self.MIN_WAIT_PAD,
            max(self.MIN_WAIT, min(self.MAX_WAIT, int(round(float(used_wait or 0) * scale)))),
        )

        mode_table = self.table.setdefault(mode, {})
        entry = mode_table.get(distance_key)
        if not entry:
            mode_table[distance_key] = {
                "y_bias": measured_y,
                "dura": measured_dura,
                "wait": measured_wait,
                "samples": 1,
            }
            return

        alpha = self.UPDATE_ALPHA
        entry["y_bias"] = int(round(entry["y_bias"] * (1.0 - alpha) + measured_y * alpha))
        entry["dura"] = int(round(entry["dura"] * (1.0 - alpha) + measured_dura * alpha))
        entry["wait"] = int(round(entry["wait"] * (1.0 - alpha) + measured_wait * alpha))
        entry["samples"] = int(entry.get("samples", 0)) + 1

    def _scaled_forward_bias(self, y_bias, scale):
        try:
            value = float(y_bias)
        except (TypeError, ValueError):
            value = -self.MIN_Y_BIAS
        sign = -1 if value <= 0 else 1
        magnitude = max(self.MIN_Y_BIAS, min(self.MAX_Y_BIAS, int(round(abs(value) * scale))))
        return sign * magnitude

    @staticmethod
    def _observed_forward_distance(before_distance, after_distance):
        try:
            before = float(before_distance)
            after = float(after_distance)
        except (TypeError, ValueError):
            return None
        observed = before - after
        if observed <= 0:
            return None
        return observed


adaptive_forward_move_table = AdaptiveForwardMoveTable()


def get_adaptive_forward_motion(mode, desired_distance, fallback_y_bias, fallback_dura, fallback_wait):
    return adaptive_forward_move_table.get(
        mode,
        desired_distance,
        fallback_y_bias,
        fallback_dura,
        fallback_wait,
    )


def update_adaptive_forward_motion(
    mode,
    desired_distance,
    before_distance,
    after_distance,
    used_y_bias,
    used_dura,
    used_wait,
):
    adaptive_forward_move_table.observe(
        mode,
        desired_distance,
        before_distance,
        after_distance,
        used_y_bias,
        used_dura,
        used_wait,
    )


class AdaptiveSideMoveTable:
    MIN_DISTANCE = 1
    MAX_DISTANCE = 60
    DISTANCE_STEP = 1
    UPDATE_ALPHA = 0.45
    MIN_OBSERVED_DISTANCE = 0.2
    MIN_X_BIAS = 80
    MAX_X_BIAS = 520
    MIN_DURA = 80
    MAX_DURA = 1600
    MIN_WAIT = 120
    MAX_WAIT = 3600
    MIN_WAIT_PAD = 120
    SCALE_MIN = 0.55
    SCALE_MAX = 1.85

    def __init__(self):
        self.table = {"left": {}, "right": {}}

    def distance_bin(self, distance):
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return None
        if value < self.MIN_OBSERVED_DISTANCE:
            return None
        value = max(self.MIN_DISTANCE, min(self.MAX_DISTANCE, value))
        return int(round(value / self.DISTANCE_STEP) * self.DISTANCE_STEP)

    def get(self, side, desired_distance, fallback_x_bias, fallback_dura, fallback_wait):
        distance_key = self.distance_bin(desired_distance)
        if side not in self.table or distance_key is None:
            return int(fallback_x_bias or 0), int(fallback_dura or 0), int(fallback_wait or 0), distance_key

        entry = self.table[side].get(distance_key)
        if entry:
            x_bias = int(entry["x_bias"])
            return x_bias, int(entry["dura"]), int(entry["wait"]), distance_key

        return int(fallback_x_bias or 0), int(fallback_dura or 0), int(fallback_wait or 0), distance_key

    def observe(self, side, desired_distance, before_distance, after_distance, used_x_bias, used_dura, used_wait):
        distance_key = self.distance_bin(desired_distance)
        if side not in self.table or distance_key is None:
            return

        observed = self._observed_side_distance(before_distance, after_distance)
        if observed is None or observed < self.MIN_OBSERVED_DISTANCE:
            return

        scale = distance_key / observed
        scale = max(self.SCALE_MIN, min(self.SCALE_MAX, scale))

        measured_x = self._scaled_side_bias(used_x_bias, side, scale)
        measured_dura = max(self.MIN_DURA, min(self.MAX_DURA, int(round(float(used_dura or 0) * scale))))
        measured_wait = max(
            measured_dura + self.MIN_WAIT_PAD,
            max(self.MIN_WAIT, min(self.MAX_WAIT, int(round(float(used_wait or 0) * scale)))),
        )

        entry = self.table[side].get(distance_key)
        if not entry:
            self.table[side][distance_key] = {
                "x_bias": measured_x,
                "dura": measured_dura,
                "wait": measured_wait,
                "samples": 1,
            }
            return

        alpha = self.UPDATE_ALPHA
        entry["x_bias"] = int(round(entry["x_bias"] * (1.0 - alpha) + measured_x * alpha))
        entry["dura"] = int(round(entry["dura"] * (1.0 - alpha) + measured_dura * alpha))
        entry["wait"] = int(round(entry["wait"] * (1.0 - alpha) + measured_wait * alpha))
        entry["samples"] = int(entry.get("samples", 0)) + 1

    def _scaled_side_bias(self, x_bias, side, scale):
        try:
            value = float(x_bias)
        except (TypeError, ValueError):
            value = -self.MIN_X_BIAS if side == "left" else self.MIN_X_BIAS
        sign = -1 if side == "left" else 1
        magnitude = max(self.MIN_X_BIAS, min(self.MAX_X_BIAS, int(round(abs(value) * scale))))
        return sign * magnitude

    @staticmethod
    def _observed_side_distance(before_distance, after_distance):
        try:
            before = float(before_distance)
            after = float(after_distance)
        except (TypeError, ValueError):
            return None
        observed = before - after
        if observed <= 0:
            return None
        return observed


adaptive_side_move_table = AdaptiveSideMoveTable()


def get_adaptive_side_motion(side, desired_distance, fallback_x_bias, fallback_dura, fallback_wait):
    return adaptive_side_move_table.get(
        side,
        desired_distance,
        fallback_x_bias,
        fallback_dura,
        fallback_wait,
    )


def update_adaptive_side_motion(
    side,
    desired_distance,
    before_distance,
    after_distance,
    used_x_bias,
    used_dura,
    used_wait,
):
    adaptive_side_move_table.observe(
        side,
        desired_distance,
        before_distance,
        after_distance,
        used_x_bias,
        used_dura,
        used_wait,
    )

def get_time_from_distance(distance):
    """
    根据距离计算所需时间（毫秒）。
    基于线性回归模型: Time = 815.6 * Distance - 127.5
    """
    # 核心公式 (y = kx + b)
    # k (斜率) ≈ 815.6 ms/unit
    # b (截距) ≈ -127.5 ms
    slope = 815.6
    intercept = -127.5

    estimated_time = (slope * distance) + intercept

    # 逻辑保护：时间不能为负数
    # 实际上，移动距离小于 0.2 单位时，物理上可能还没开始动，这里设个最低阈值
    if estimated_time < 0:
        return 0

    return int(estimated_time)

def align_direction(w, tar_loc, threshold=5):
    cur_loc = w.get_info('location')[0]

    # --- [修改] 在这里也加入方向容错 ---
    cur_dir = w.get_info('direction')

    target_angle = calculate_angle(cur_loc, tar_loc)
    turn_dir, px, diff = calculate_move_count(cur_dir, target_angle)

    # [Print] 仅在偏差存在时打印，避免刷屏
    print(f"[Check] 当前: {cur_dir:.1f}° | 目标: {target_angle:.1f}° | 偏差: {diff:.1f}°")

    if abs(diff) > threshold:
        fallback_dura = 800
        used_px, used_dura, _ = get_adaptive_turn_motion(turn_dir, diff, px, fallback_dura)
        move_px = used_px if turn_dir == 'right' else -used_px

        print(f"[Align] 修正视角: 当前 {cur_dir:.1f}° -> 目标 {target_angle:.1f}° (偏差 {diff:.1f}°)")
        print(f"        执行: {'右转' if turn_dir == 'right' else '左转'} {int(move_px)} px")

        w.tap_single('视角', x_bias=int(move_px), dura=used_dura, wait=500)
        w.refresh_frame()
        update_adaptive_turn_motion(turn_dir, diff, cur_dir, w.get_info('direction'), used_px, used_dura)

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
