import os
import cv2
import math
import json
import numpy as np
from sklearn.cluster import DBSCAN

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROUTE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "map", "hpjy.png")
DEFAULT_ROUTE_OUTPUT_PATH = os.path.join("aw", "autogame", "temp", "road", "route.jpg")
ADAPTIVE_MOTION_TABLE_PATH = os.path.join(RESOURCE_DIR, "adaptive_motion_table.json")
_ADAPTIVE_MOTION_DATA = None


def _load_adaptive_motion_data():
    global _ADAPTIVE_MOTION_DATA
    if _ADAPTIVE_MOTION_DATA is not None:
        return _ADAPTIVE_MOTION_DATA

    if not os.path.exists(ADAPTIVE_MOTION_TABLE_PATH):
        _ADAPTIVE_MOTION_DATA = {}
        return _ADAPTIVE_MOTION_DATA

    try:
        with open(ADAPTIVE_MOTION_TABLE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[AdaptiveMotion] 加载持久化动作表失败: {e}")
        raw = {}

    _ADAPTIVE_MOTION_DATA = raw if isinstance(raw, dict) else {}
    return _ADAPTIVE_MOTION_DATA


def _get_adaptive_motion_section(section):
    raw = _load_adaptive_motion_data().get(section)
    return raw if isinstance(raw, dict) else {}


def _persist_adaptive_motion_section(section, table):
    data = _load_adaptive_motion_data()
    data[section] = table
    tmp_path = f"{ADAPTIVE_MOTION_TABLE_PATH}.tmp"
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_MOTION_TABLE_PATH), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, ADAPTIVE_MOTION_TABLE_PATH)
    except Exception as e:
        print(f"[AdaptiveMotion] 保存持久化动作表失败: {e}")


def load_adaptive_motion_section(section):
    return dict(_get_adaptive_motion_section(section))


def persist_adaptive_motion_section(section, table):
    _persist_adaptive_motion_section(section, table if isinstance(table, dict) else {})


def _motion_entry_as_ints(entry, required_fields):
    if not isinstance(entry, dict):
        return None

    cleaned = {}
    for field in required_fields:
        if field not in entry:
            return None
        try:
            cleaned[field] = int(round(float(entry[field])))
        except (TypeError, ValueError):
            return None

    try:
        cleaned["samples"] = int(entry.get("samples", 1))
    except (TypeError, ValueError):
        cleaned["samples"] = 1
    return cleaned

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
    MODEL_UPDATE_ALPHA = 0.28
    MIN_OBSERVED_DEG = 1.0
    MIN_OBSERVED_RATIO = 0.2
    MAX_OBSERVED_RATIO = 2.2
    STRONG_SAMPLE_MIN_RATIO = 0.65
    STRONG_SAMPLE_MAX_RATIO = 1.45
    MIN_SAMPLE_WEIGHT = 0.25
    MIN_PX = 8
    MAX_PX = 1200
    MIN_DURA = 80
    MAX_DURA = 1800
    MIN_SLOPE = 2.0
    MAX_SLOPE = 10.0
    MIN_INTERCEPT = 0.0
    MAX_INTERCEPT = 220.0
    MIN_DURA_SCALE = 0.9
    MAX_DURA_SCALE = 2.4
    SCALE_MIN = 0.45
    SCALE_MAX = 1.8
    MODEL_KEY = "models"
    DEFAULT_MODELS = {
        "left": {"slope": 5.31, "intercept": 49.0, "dura_scale": 1.5, "samples": 0},
        "right": {"slope": 5.31, "intercept": 49.0, "dura_scale": 1.5, "samples": 0},
    }

    def __init__(self):
        self.table = {"left": {}, "right": {}}
        self.models = {
            turn_dir: dict(model)
            for turn_dir, model in self.DEFAULT_MODELS.items()
        }
        self._load_persisted_table()

    def _load_persisted_table(self):
        raw = _get_adaptive_motion_section("turn")
        dirty = False
        model_entries = raw.get(self.MODEL_KEY)
        if isinstance(model_entries, dict):
            for turn_dir in ("left", "right"):
                entry = model_entries.get(turn_dir)
                if not isinstance(entry, dict):
                    continue
                model = dict(self.models[turn_dir])
                raw_slope = entry.get("slope", model["slope"])
                raw_intercept = entry.get("intercept", model["intercept"])
                raw_dura_scale = entry.get("dura_scale", model["dura_scale"])
                model["slope"] = self._clamp_slope(raw_slope)
                model["intercept"] = self._clamp_intercept(raw_intercept)
                model["dura_scale"] = self._clamp_dura_scale(raw_dura_scale)
                try:
                    model["samples"] = int(entry.get("samples", 0))
                except (TypeError, ValueError):
                    model["samples"] = 0
                dirty = dirty or (
                    model["slope"] != raw_slope
                    or model["intercept"] != raw_intercept
                    or model["dura_scale"] != raw_dura_scale
                )
                self.models[turn_dir] = model

        for turn_dir in ("left", "right"):
            entries = raw.get(turn_dir)
            if not isinstance(entries, dict):
                continue

            for angle, entry in entries.items():
                angle_key = self.angle_bin(angle)
                cleaned = _motion_entry_as_ints(entry, ("px", "dura"))
                if angle_key is not None and cleaned:
                    raw_px = cleaned["px"]
                    raw_dura = cleaned["dura"]
                    cleaned["px"] = self._clamp_px(cleaned["px"])
                    cleaned["dura"] = self._clamp_dura(cleaned["dura"])
                    dirty = dirty or cleaned["px"] != raw_px or cleaned["dura"] != raw_dura
                    self.table[turn_dir][angle_key] = cleaned
        if dirty:
            self._persist()

    def _persist(self):
        payload = {
            "left": self.table.get("left", {}),
            "right": self.table.get("right", {}),
            self.MODEL_KEY: self.models,
        }
        _persist_adaptive_motion_section("turn", payload)

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
            return self._clamp_px(fallback_px), self._clamp_dura(fallback_dura), angle_key

        entry = self.table[turn_dir].get(angle_key)
        if entry:
            return self._clamp_px(entry["px"]), self._clamp_dura(entry["dura"]), angle_key

        model_px, model_dura = self._predict_model_motion(turn_dir, angle_key, fallback_px, fallback_dura)
        return model_px, model_dura, angle_key

    def observe(self, turn_dir, desired_diff, before_angle, after_angle, used_px, used_dura):
        angle_key = self.angle_bin(desired_diff)
        if turn_dir not in self.table or angle_key is None:
            return

        observed = self._observed_turn_degrees(turn_dir, before_angle, after_angle)
        if observed is None or observed < self.MIN_OBSERVED_DEG:
            return
        if angle_key >= 20 and observed < angle_key * self.MIN_OBSERVED_RATIO:
            print(
                f"[AdaptiveMotion] 跳过异常转向样本: turn={turn_dir}, "
                f"desired={angle_key}, observed={observed:.1f}"
            )
            return
        observed_ratio = observed / float(max(angle_key, 1))
        if angle_key >= 20 and observed_ratio > self.MAX_OBSERVED_RATIO:
            print(
                f"[AdaptiveMotion] 跳过异常转向样本: turn={turn_dir}, "
                f"desired={angle_key}, observed={observed:.1f}, ratio={observed_ratio:.2f}"
            )
            return

        had_entry = angle_key in self.table[turn_dir]
        scale = angle_key / observed
        scale = max(self.SCALE_MIN, min(self.SCALE_MAX, scale))
        measured_px = self._clamp_px(abs(float(used_px or 0)) * scale)
        measured_dura = self._clamp_dura(float(used_dura or 0) * scale)
        sample_weight = self._sample_weight(observed_ratio)

        entry = self.table[turn_dir].get(angle_key)
        if not entry:
            self._update_model(turn_dir, angle_key, measured_px, measured_dura, sample_weight=sample_weight)
            self.table[turn_dir][angle_key] = {
                "px": measured_px,
                "dura": measured_dura,
                "samples": 1,
                "confidence": round(sample_weight, 3),
            }
            print(
                f"[AdaptiveMotion] 建立转向表: turn={turn_dir}, angle={angle_key}, "
                f"observed={observed:.1f}, weight={sample_weight:.2f}, "
                f"px={measured_px}, dura={measured_dura}"
            )
            self._persist()
            return

        self._update_model(turn_dir, angle_key, measured_px, measured_dura, sample_weight=sample_weight)
        alpha = self.UPDATE_ALPHA * sample_weight
        entry["px"] = self._clamp_px(entry["px"] * (1.0 - alpha) + measured_px * alpha)
        entry["dura"] = self._clamp_dura(entry["dura"] * (1.0 - alpha) + measured_dura * alpha)
        entry["samples"] = int(entry.get("samples", 0)) + 1
        entry["confidence"] = round(
            float(entry.get("confidence", sample_weight)) * 0.7 + sample_weight * 0.3,
            3,
        )
        if had_entry:
            print(
                f"[AdaptiveMotion] 更新转向表: turn={turn_dir}, angle={angle_key}, "
                f"observed={observed:.1f}, weight={sample_weight:.2f}, "
                f"px={entry['px']}, dura={entry['dura']}, samples={entry['samples']}"
            )
        self._persist()

    def _predict_model_motion(self, turn_dir, angle_key, fallback_px, fallback_dura):
        model = self.models.get(turn_dir, self.DEFAULT_MODELS.get(turn_dir, self.DEFAULT_MODELS["right"]))
        try:
            angle = float(angle_key)
        except (TypeError, ValueError):
            angle = 0.0
        if angle <= 0:
            return self._clamp_px(fallback_px), self._clamp_dura(fallback_dura)

        predicted_px = self._clamp_px(float(model["slope"]) * angle + float(model["intercept"]))
        predicted_dura = float(predicted_px) * float(model["dura_scale"])
        try:
            predicted_dura = max(predicted_dura, float(fallback_dura or 0))
        except (TypeError, ValueError):
            pass
        return predicted_px, self._clamp_dura(predicted_dura)

    def _update_model(self, turn_dir, angle_key, measured_px, measured_dura, sample_weight=1.0):
        model = self.models.get(turn_dir)
        if not model:
            return
        try:
            angle = max(float(angle_key), 1.0)
            px = float(measured_px)
            dura = float(measured_dura)
        except (TypeError, ValueError):
            return

        target_slope = self._clamp_slope((px - float(model["intercept"])) / angle)
        target_intercept = self._clamp_intercept(px - float(model["slope"]) * angle)
        target_dura_scale = self._clamp_dura_scale(dura / max(px, 1.0))
        alpha = self.MODEL_UPDATE_ALPHA * self._clamp_sample_weight(sample_weight)
        intercept_alpha = alpha * 0.35
        model["slope"] = self._clamp_slope(model["slope"] * (1.0 - alpha) + target_slope * alpha)
        model["intercept"] = self._clamp_intercept(
            model["intercept"] * (1.0 - intercept_alpha) + target_intercept * intercept_alpha
        )
        model["dura_scale"] = self._clamp_dura_scale(
            model["dura_scale"] * (1.0 - alpha) + target_dura_scale * alpha
        )
        model["samples"] = int(model.get("samples", 0)) + 1
        print(
            f"[AdaptiveMotion] 更新转向模型: turn={turn_dir}, angle={angle_key}, "
            f"slope={model['slope']:.3f}, intercept={model['intercept']:.1f}, "
            f"dura_scale={model['dura_scale']:.3f}, weight={sample_weight:.2f}, samples={model['samples']}"
        )

    def _sample_weight(self, observed_ratio):
        try:
            ratio = float(observed_ratio)
        except (TypeError, ValueError):
            return self.MIN_SAMPLE_WEIGHT
        if self.STRONG_SAMPLE_MIN_RATIO <= ratio <= self.STRONG_SAMPLE_MAX_RATIO:
            return 1.0
        if ratio < self.STRONG_SAMPLE_MIN_RATIO:
            span = max(0.000001, self.STRONG_SAMPLE_MIN_RATIO - self.MIN_OBSERVED_RATIO)
            progress = (ratio - self.MIN_OBSERVED_RATIO) / span
        else:
            span = max(0.000001, self.MAX_OBSERVED_RATIO - self.STRONG_SAMPLE_MAX_RATIO)
            progress = (self.MAX_OBSERVED_RATIO - ratio) / span
        return self._clamp_sample_weight(progress)

    def _clamp_sample_weight(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.MIN_SAMPLE_WEIGHT
        return max(self.MIN_SAMPLE_WEIGHT, min(1.0, value))

    def _clamp_px(self, value):
        try:
            value = int(round(float(value or 0)))
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            return 0
        return max(self.MIN_PX, min(self.MAX_PX, value))

    def _clamp_dura(self, value):
        try:
            value = int(round(float(value or 0)))
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            return 0
        return max(self.MIN_DURA, min(self.MAX_DURA, value))

    def _clamp_slope(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.DEFAULT_MODELS["right"]["slope"]
        return max(self.MIN_SLOPE, min(self.MAX_SLOPE, value))

    def _clamp_intercept(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.DEFAULT_MODELS["right"]["intercept"]
        return max(self.MIN_INTERCEPT, min(self.MAX_INTERCEPT, value))

    def _clamp_dura_scale(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.DEFAULT_MODELS["right"]["dura_scale"]
        return max(self.MIN_DURA_SCALE, min(self.MAX_DURA_SCALE, value))

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


def plan_view_turn_motion(
    current_angle,
    target_angle,
    fallback_dura=None,
    min_dura=None,
    max_dura=None,
    max_px=None,
):
    turn_dir, fallback_px, diff = calculate_move_count(current_angle, target_angle)
    if turn_dir is None or diff is None:
        return None

    if fallback_dura is None:
        fallback_dura = max(250, int((fallback_px or 0) * 1.5))

    used_px, used_dura, angle_key = get_adaptive_turn_motion(turn_dir, diff, fallback_px, fallback_dura)
    if max_px is not None:
        used_px = min(int(max_px), int(used_px or 0))
    if min_dura is not None:
        used_dura = max(int(min_dura), int(used_dura or 0))
    if max_dura is not None:
        used_dura = min(int(max_dura), int(used_dura or 0))

    x_bias = used_px if turn_dir == "right" else -used_px
    return {
        "turn_dir": turn_dir,
        "diff": diff,
        "angle_key": angle_key,
        "px": int(used_px or 0),
        "dura": int(used_dura or 0),
        "x_bias": int(x_bias or 0),
        "fallback_px": int(fallback_px or 0),
        "fallback_dura": int(fallback_dura or 0),
    }


def execute_view_turn(
    w,
    current_angle,
    target_angle,
    threshold=5,
    max_steps=1,
    wait=250,
    fallback_dura=None,
    min_dura=None,
    max_dura=None,
    max_px=None,
    log_prefix="[Turn]",
):
    for _ in range(max_steps):
        motion = plan_view_turn_motion(
            current_angle,
            target_angle,
            fallback_dura=fallback_dura,
            min_dura=min_dura,
            max_dura=max_dura,
            max_px=max_px,
        )
        if motion is None:
            return False
        if motion["diff"] <= threshold:
            return True
        if not motion["px"]:
            return True

        print(
            f"{log_prefix} current={current_angle}, target={target_angle}, "
            f"diff={motion['diff']:.1f}, bin={motion['angle_key']}, "
            f"x_bias={motion['x_bias']}, dura={motion['dura']}"
        )
        before_angle = current_angle
        w.tap_single("视角", x_bias=motion["x_bias"], dura=motion["dura"], wait=wait)
        w.refresh_frame()
        current_angle = w.get_info("direction")
        update_adaptive_turn_motion(
            motion["turn_dir"],
            motion["diff"],
            before_angle,
            current_angle,
            motion["px"],
            motion["dura"],
        )
        if current_angle is None:
            return False
    return False


class AdaptiveForwardMoveTable:
    MIN_DISTANCE = 0.2
    MAX_DISTANCE = 60
    UPDATE_ALPHA = 0.55
    MIN_OBSERVED_DISTANCE = 0.2
    MIN_OBSERVED_RATIO = 0.08
    MAX_OBSERVED_RATIO = 2.0
    STRONG_SAMPLE_MIN_RATIO = 0.65
    STRONG_SAMPLE_MAX_RATIO = 1.35
    MIN_SAMPLE_WEIGHT = 0.25
    ARRIVAL_DISTANCE = 1.0
    FIXED_DURA = 300
    MIN_WAIT = 180
    MAX_WAIT = 7000
    MIN_SLOPE = 12.0
    MAX_SLOPE = 260.0
    MIN_INTERCEPT = 120.0
    MAX_INTERCEPT = 1000.0
    SCALE_MIN = 0.65
    SCALE_MAX = 1.85
    DEFAULT_MODELS = {
        "slow": {"y_bias": -100, "slope": 60.0, "intercept": 300.0},
        "fast": {"y_bias": -300, "slope": 32.0, "intercept": 220.0},
    }

    def __init__(self):
        self.table = {
            mode: dict(model, samples=0)
            for mode, model in self.DEFAULT_MODELS.items()
        }
        self._load_persisted_table()

    def _load_persisted_table(self):
        raw = _get_adaptive_motion_section("forward")
        if not isinstance(raw, dict):
            return

        dirty = False
        for mode in self.DEFAULT_MODELS:
            entry = raw.get(mode)
            if not isinstance(entry, dict):
                continue

            if "slope" not in entry or "intercept" not in entry:
                continue

            model = dict(self.table[mode])
            raw_y = entry.get("y_bias", model["y_bias"])
            raw_slope = entry.get("slope", model["slope"])
            raw_intercept = entry.get("intercept", model["intercept"])
            model["y_bias"] = self.DEFAULT_MODELS[mode]["y_bias"]
            model["slope"] = self._clamp_slope(raw_slope)
            model["intercept"] = self._clamp_intercept(raw_intercept)
            try:
                model["samples"] = int(entry.get("samples", 0))
            except (TypeError, ValueError):
                model["samples"] = 0

            dirty = dirty or (
                model["y_bias"] != raw_y
                or model["slope"] != raw_slope
                or model["intercept"] != raw_intercept
            )
            self.table[mode] = model

        if dirty:
            self._persist()

    def _persist(self):
        _persist_adaptive_motion_section("forward", self.table)

    def _clamp_distance(self, distance):
        try:
            value = float(distance)
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, min(self.MAX_DISTANCE, value))

    def _normalize_mode(self, mode, fallback_y_bias=None):
        mode = str(mode or "").lower()
        if mode in self.DEFAULT_MODELS:
            return mode
        try:
            y_bias = abs(float(fallback_y_bias or 0))
        except (TypeError, ValueError):
            y_bias = 0
        return "slow" if y_bias <= 150 else "fast"

    def _get_model(self, mode, fallback_y_bias=None):
        mode = self._normalize_mode(mode, fallback_y_bias)
        return mode, self.table.setdefault(mode, dict(self.DEFAULT_MODELS[mode], samples=0))

    def get(self, mode, desired_distance, fallback_y_bias, fallback_dura, fallback_wait):
        mode, model = self._get_model(mode, fallback_y_bias)
        distance = self._clamp_distance(desired_distance)
        if distance < self.MIN_DISTANCE:
            return int(model["y_bias"]), self.FIXED_DURA, 0, 0

        wait = self._predict_wait(model, distance)
        return int(model["y_bias"]), self.FIXED_DURA, wait, round(distance, 2)

    def observe(self, mode, desired_distance, before_distance, after_distance, used_y_bias, used_dura, used_wait):
        mode, model = self._get_model(mode, used_y_bias)
        desired = self._clamp_distance(desired_distance)
        if desired < self.MIN_DISTANCE:
            return

        observed = self._observed_forward_distance(before_distance, after_distance)
        if after_distance is not None:
            try:
                if float(after_distance) <= self.ARRIVAL_DISTANCE:
                    observed = desired
            except (TypeError, ValueError):
                pass

        if observed is None or observed < self.MIN_OBSERVED_DISTANCE:
            return
        if observed < desired * self.MIN_OBSERVED_RATIO:
            print(
                f"[AdaptiveMotion] 跳过异常前推样本: mode={mode}, "
                f"desired={desired:.2f}, observed={observed:.2f}"
            )
            return
        observed_ratio = observed / float(max(desired, self.MIN_DISTANCE))
        if observed_ratio > self.MAX_OBSERVED_RATIO:
            print(
                f"[AdaptiveMotion] 跳过异常前推样本: mode={mode}, "
                f"desired={desired:.2f}, observed={observed:.2f}, ratio={observed_ratio:.2f}"
            )
            return

        scale = desired / observed
        scale = max(self.SCALE_MIN, min(self.SCALE_MAX, scale))
        measured_wait = self._clamp_wait(float(used_wait or 0) * scale)
        target_slope = (measured_wait - float(model["intercept"])) / max(desired, self.MIN_DISTANCE)
        target_slope = self._clamp_slope(target_slope)
        sample_weight = self._sample_weight(observed_ratio)
        alpha = self.UPDATE_ALPHA * sample_weight
        model["slope"] = self._clamp_slope(model["slope"] * (1.0 - alpha) + target_slope * alpha)
        model["samples"] = int(model.get("samples", 0)) + 1
        model["confidence"] = round(
            float(model.get("confidence", sample_weight)) * 0.7 + sample_weight * 0.3,
            3,
        )
        print(
            f"[AdaptiveMotion] 更新前推模型: mode={mode}, desired={desired:.2f}, "
            f"observed={observed:.2f}, weight={sample_weight:.2f}, wait={used_wait}->{measured_wait}, "
            f"slope={model['slope']:.2f}, samples={model['samples']}"
        )
        self._persist()

    def _predict_wait(self, model, distance):
        return self._clamp_wait(float(model["slope"]) * float(distance) + float(model["intercept"]))

    def _clamp_wait(self, value):
        try:
            value = int(round(float(value or 0)))
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            return 0
        return max(self.MIN_WAIT, min(self.MAX_WAIT, value))

    def _clamp_slope(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.MIN_SLOPE
        return max(self.MIN_SLOPE, min(self.MAX_SLOPE, value))

    def _clamp_intercept(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.MIN_INTERCEPT
        return max(self.MIN_INTERCEPT, min(self.MAX_INTERCEPT, value))

    def _sample_weight(self, observed_ratio):
        try:
            ratio = float(observed_ratio)
        except (TypeError, ValueError):
            return self.MIN_SAMPLE_WEIGHT
        if self.STRONG_SAMPLE_MIN_RATIO <= ratio <= self.STRONG_SAMPLE_MAX_RATIO:
            return 1.0
        if ratio < self.STRONG_SAMPLE_MIN_RATIO:
            span = max(0.000001, self.STRONG_SAMPLE_MIN_RATIO - self.MIN_OBSERVED_RATIO)
            progress = (ratio - self.MIN_OBSERVED_RATIO) / span
        else:
            span = max(0.000001, self.MAX_OBSERVED_RATIO - self.STRONG_SAMPLE_MAX_RATIO)
            progress = (self.MAX_OBSERVED_RATIO - ratio) / span
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            progress = self.MIN_SAMPLE_WEIGHT
        return max(self.MIN_SAMPLE_WEIGHT, min(1.0, progress))

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
        self._load_persisted_table()

    def _load_persisted_table(self):
        raw = _get_adaptive_motion_section("side")
        for side in ("left", "right"):
            entries = raw.get(side)
            if not isinstance(entries, dict):
                continue

            for distance, entry in entries.items():
                distance_key = self.distance_bin(distance)
                cleaned = _motion_entry_as_ints(entry, ("x_bias", "dura", "wait"))
                if distance_key is not None and cleaned:
                    self.table[side][distance_key] = cleaned

    def _persist(self):
        _persist_adaptive_motion_section("side", self.table)

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
            self._persist()
            return

        alpha = self.UPDATE_ALPHA
        entry["x_bias"] = int(round(entry["x_bias"] * (1.0 - alpha) + measured_x * alpha))
        entry["dura"] = int(round(entry["dura"] * (1.0 - alpha) + measured_dura * alpha))
        entry["wait"] = int(round(entry["wait"] * (1.0 - alpha) + measured_wait * alpha))
        entry["samples"] = int(entry.get("samples", 0)) + 1
        self._persist()

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
    if cur_dir is None or target_angle is None:
        return False
    return execute_view_turn(
        w,
        cur_dir,
        target_angle,
        threshold=threshold,
        max_steps=1,
        wait=500,
        fallback_dura=800,
        log_prefix="[Align]",
    )

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
