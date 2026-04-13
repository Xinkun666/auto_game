import re
import os
import cv2
import time
import math
import json
import shutil
import subprocess
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = ROOT_DIR / "aw" / "autogame" / "temp"
LOG_DIR = TEMP_DIR / "logs"
PROCESS_TEMP_LOGS_DIR = LOG_DIR / "process_temp_logs"


def _safe_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_top_level_log_files(dst_dir: Path) -> list[str]:
    copied = []
    if not LOG_DIR.exists():
        return copied

    for path in sorted(LOG_DIR.iterdir()):
        if not path.is_file():
            continue
        shutil.copy2(path, dst_dir / path.name)
        copied.append(path.name)
    return copied


def _copy_process_temp_logs(dst_dir: Path) -> list[str]:
    copied = []
    if not PROCESS_TEMP_LOGS_DIR.exists():
        return copied

    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(PROCESS_TEMP_LOGS_DIR.iterdir()):
        target = dst_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)
            copied.append(path.name)
        elif path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
            copied.append(path.name + "/")
    return copied


def _build_archive_dir(run_index: int) -> Path:
    timestamp = time.strftime("%Y%m%d%H%M%S")
    base_dir = TEMP_DIR / f"game_{timestamp}_第{run_index}次用例"
    archive_dir = base_dir
    suffix = 1
    while archive_dir.exists():
        archive_dir = TEMP_DIR / f"{base_dir.name}_{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def archive_run_artifacts(
    run_index: int,
    source: str,
    extra_text_files: Optional[dict[str, str]] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    archive_dir = _build_archive_dir(run_index)
    log_archive_dir = archive_dir / "logs"
    process_archive_dir = archive_dir / "process_temp_logs"

    copied_log_files = _copy_top_level_log_files(log_archive_dir)
    copied_process_files = _copy_process_temp_logs(process_archive_dir)

    if extra_text_files:
        for name, content in extra_text_files.items():
            if not name:
                continue
            _safe_write_text(log_archive_dir / name, content)

    metadata = {
        "source": source,
        "run_index": run_index,
        "archive_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archive_dir": str(archive_dir),
        "copied_log_files": copied_log_files,
        "copied_process_temp_logs": copied_process_files,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    _safe_write_text(
        archive_dir / "archive_info.json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    return archive_dir

def find_template_center_multiscale(target_img, template_input, threshold=0.7, match_mode="gray"):
    def _normalize_match_mode(mode):
        mode = str(mode or "gray").strip().lower()
        return mode if mode in ("gray", "rgb", "hsv") else "gray"

    def _prepare_match_image(img_data, mode):
        if isinstance(img_data, str):
            try:
                arr = np.fromfile(img_data, dtype=np.uint8)
                flag = cv2.IMREAD_GRAYSCALE if mode == "gray" else cv2.IMREAD_COLOR
                img = cv2.imdecode(arr, flag)
            except Exception:
                return None
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            return None

        if img is None:
            return None

        if mode == "gray":
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if mode == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mode == "hsv":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _match_template(search_img, tpl_img):
        if len(search_img.shape) == 2:
            return cv2.matchTemplate(search_img, tpl_img, cv2.TM_CCOEFF_NORMED)

        channel_scores = []
        for channel_idx in range(search_img.shape[2]):
            channel_scores.append(
                cv2.matchTemplate(
                    search_img[:, :, channel_idx],
                    tpl_img[:, :, channel_idx],
                    cv2.TM_CCOEFF_NORMED,
                )
            )
        return np.mean(channel_scores, axis=0)

    match_mode = _normalize_match_mode(match_mode)
    prepared_target = _prepare_match_image(target_img, match_mode)
    prepared_template = _prepare_match_image(template_input, match_mode)

    if prepared_target is None or prepared_template is None:
        return None

    tH, tW = prepared_template.shape[:2]
    best_match = None

    # 缩放范围：由于主缩放已在外部完成，这里仅作微调
    scales = np.linspace(0.8, 1.2, 10)

    for scale in scales:
        new_w = int(tW * scale)
        new_h = int(tH * scale)
        if new_w < 10 or new_h < 10 or new_h > prepared_target.shape[0] or new_w > prepared_target.shape[1]:
            continue

        resized_tpl = cv2.resize(prepared_template, (new_w, new_h))
        res = _match_template(prepared_target, resized_tpl)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best_match is None or max_val > best_match[0]:
            cX = max_loc[0] + new_w // 2
            cY = max_loc[1] + new_h // 2
            best_match = (max_val, (cX, cY))

    if best_match and best_match[0] >= threshold:
        return best_match[1]
    return None

def run_shell(cmd: str, r = False):
    try:
        if r:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout.strip()
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print(f"命令执行失败: {cmd}\n{e}")
        if r:
            return None

def get_resolution(r = True):
    resolution_mode = run_shell('hdc shell hidumper -s RenderService -a screen', r)
    match = re.search(r'activeMode:\s*(\d+)x(\d+)', resolution_mode)
    if match:
        h, w = int(match.group(1)), int(match.group(2))
        return w, h
    else:
        print('未能获取分辨率信息!')
        return None, None

def get_wh():
    resolution = get_resolution()
    assert resolution[0] is not None, '分辨率获取失败'
    if resolution[0] > resolution[1]:
        w_h = resolution[0] / resolution[1]
    else:
        w_h = resolution[1] / resolution[0]
    with open(r'aw\autogame\config\config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        width = config["width"]
        height = int(width * w_h)
        return width, height

def normalize_rotation(rotation):
    if rotation is None:
        return None
    try:
        value = int(rotation)
    except (TypeError, ValueError):
        return None
    if value in (0, 90, 180, 270):
        return value
    mapping = {1: 90, 2: 180, 3: 270}
    return mapping.get(value)

def is_landscape(width, height):
    return width >= height

def _clamp_point(x, y, width, height):
    x = int(round(x))
    y = int(round(y))
    x = min(max(x, 0), max(int(width) - 1, 0))
    y = min(max(y, 0), max(int(height) - 1, 0))
    return x, y

def scale_point(x, y, src_width, src_height, dst_width, dst_height):
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return int(round(x)), int(round(y))
    dst_x = float(x) * float(dst_width) / float(src_width)
    dst_y = float(y) * float(dst_height) / float(src_height)
    return _clamp_point(dst_x, dst_y, dst_width, dst_height)

def convert_display_point_by_rotation(x, y, screen_width, screen_height, current_rotation):
    """
    将“当前画面左上角坐标系”中的点，转换为设备物理屏幕左上角坐标系。

    约定:
    - x, y 是已经映射到 screen_size 下的画面坐标
    - current_rotation 表示设备当前屏幕旋转角
    - rotation=90/270 时需要交换宽高轴来做映射
    """
    if screen_width <= 0 or screen_height <= 0:
        return int(round(x)), int(round(y))

    current_rotation = normalize_rotation(current_rotation)
    nx = float(x) / float(screen_width)
    ny = float(y) / float(screen_height)

    if current_rotation == 90:
        dst_x = ny * screen_height
        dst_y = (1.0 - nx) * screen_width
        return _clamp_point(dst_x, dst_y, screen_width, screen_height)

    if current_rotation == 180:
        dst_x = (1.0 - nx) * screen_width
        dst_y = (1.0 - ny) * screen_height
        return _clamp_point(dst_x, dst_y, screen_width, screen_height)

    if current_rotation == 270:
        dst_x = (1.0 - ny) * screen_height
        dst_y = nx * screen_width
        return _clamp_point(dst_x, dst_y, screen_width, screen_height)

    return _clamp_point(x, y, screen_width, screen_height)

def convert_scene_point_by_current_rotation(x, y, scene_width, scene_height,
                                            current_width, current_height, current_rotation):
    if (
        scene_width <= 0 or scene_height <= 0
        or current_width <= 0 or current_height <= 0
    ):
        return int(round(x)), int(round(y))
    scaled_x, scaled_y = scale_point(
        x, y,
        scene_width, scene_height,
        current_width, current_height,
    )
    return convert_display_point_by_rotation(
        scaled_x, scaled_y,
        current_width, current_height,
        current_rotation,
    )

def extract_absolute_points(stage_info):
    """
    将游戏各阶段场景中的控点（Points）从百分比归一化坐标转换为屏幕绝对像素坐标。

    在自动化标注过程中，为了适配不同分辨率的屏幕，我们通常使用 0.0 到 1.0 之间的浮点数（归一化坐标）
    来表示按钮的位置。但在实际执行点击操作（如使用 hdc 或 uinput）时，系统需要具体的像素坐标。
    该函数会自动遍历整个配置表，读取每个场景的原始宽高，计算出每个按钮中心点的像素位置，
    并生成一个方便直接查询的扁平化字典。

    参数:
        stage_info (dict): 包含游戏阶段、场景、长宽信息及控点矩形区域的原始嵌套字典。
                           要求每个场景必须包含 'width' 和 'height' 键。

    返回:
        dict: 转换后的绝对坐标字典。
              键格式为: '阶段名_控点名' (str)
              值格式为:
              {
                  "pos": (x, y),
                  "scene_width": int,
                  "scene_height": int,
              }
    """
    absolute_points = {}

    for stage_name, stage_content in stage_info.items():
        scenes = stage_content.get('scenes', {})

        for scene_name, scene_content in scenes.items():
            # 获取当前场景的画布大小（标注时的原始分辨率）
            img_w = scene_content.get('width', 1)
            img_h = scene_content.get('height', 1)
            points = scene_content.get('points', {})

            for point_name, point_content in points.items():
                # 获取归一化矩形区域 [x_start, y_start, x_end, y_end]
                rect = point_content.get('rect', [0, 0, 0, 0])

                # 计算中心点的归一化位置
                norm_x = (rect[0] + rect[2]) / 2
                norm_y = (rect[1] + rect[3]) / 2

                # 核心转换：归一化比例 * 原始分辨率 = 绝对像素坐标
                abs_x = int(norm_x * img_w)
                abs_y = int(norm_y * img_h)

                # 扁平化存储：使用 阶段_控点 作为唯一索引，方便 Controller 直接调用
                key = f"{stage_name}_{point_name}"
                absolute_points[key] = {
                    "pos": (abs_x, abs_y),
                    "scene_width": int(img_w),
                    "scene_height": int(img_h),
                }

    return absolute_points

def get_formatted_time():
    # 1. 获取当前本地时间
    now = datetime.now()

    # 2. 使用 strftime 格式化，%f 代表微秒（6位）
    # 格式含义： %m月, %d日, %H时, %M分, %S秒, %f微秒
    time_str = now.strftime("%m-%d %H:%M:%S.%f")

    # 3. 因为你要的是 3 位毫秒，所以截取掉末尾的 3 位微秒
    # 原字符串：01-22 09:31:44.732000 -> 截取后：01-22 09:31:44.732
    final_time = time_str[:-3]

    return final_time

def analyze_fps_value(log_path, time_txt_path):
    """
    在 log.txt 中寻找每个时间区间内【第一个】出现的 'targetFps = 数值'。
    返回结果将阶段名中的 Time 替换为 Fps。
    """
    # 1. 解析 time.txt
    time_data = {}
    time_keys = []
    if not os.path.exists(time_txt_path):
        print(f" [错误] 找不到文件: {time_txt_path}")
        return

    with open(time_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                key, val = line.split(':', 1)
                time_data[key.strip()] = val.strip()
                time_keys.append(key.strip())

    # 2. 读取 log.txt
    if not os.path.exists(log_path):
        print(f" [错误] 找不到日志文件: {log_path}")
        return
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_lines = f.readlines()

    results = {}

    # 3. 按区间搜索第一个匹配项
    for idx, key in enumerate(time_keys):
        start_time_str = time_data[key]
        end_time_str = time_data[time_keys[idx+1]] if idx + 1 < len(time_keys) else "99-99 99:99:99.999"

        found_value = "No Matching"

        # 转换键名: Hall Time -> Hall Fps
        new_key_name = key.replace("Time", "Fps")

        for line in log_lines:
            log_time_str = line[:18]

            # 时间区间判断 (log_time 必须在当前起始点和下一个起始点之间)
            if start_time_str <= log_time_str < end_time_str:
                # 匹配 targetFps = 数字
                if "targetFps" in line:
                    match = re.search(r'targetFps\s*=\s*(\d+)', line)
                    if match:
                        found_value = match.group(1)
                        break # 只记录第一次出现，跳出当前日志循环

            # 性能优化
            if log_time_str >= end_time_str:
                break

        results[new_key_name] = found_value

    # 4. 写入结果到 results.txt
    output_path = os.path.join(os.path.dirname(time_txt_path), "results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 按照转换后的 Fps 键名写入
        for key in results:
            f.write(f"{key}: {results[key]}\n")

    print(f" [完成] 分析报告已生成: {output_path}")

def draw_chinese_text(img, text, position, font_path="msyh.ttc", font_size=25, color=(0, 255, 0)):
    """
    使用 PIL 在图像上绘制中文
    """
    # 1. 先切断与外部 numpy/PIL buffer 的共享关系，再转为 PIL
    img_bgr = np.ascontiguousarray(img).copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb, mode="RGB").copy()
    draw = ImageDraw.Draw(img_pil)

    # 2. 加载字体
    try:
        # Windows常用字体: "msyh.ttc" (微软雅黑), "simhei.ttf" (黑体)
        # 如果是Linux，通常路径为: "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 如果找不到指定字体，使用默认字体（可能还是不支持中文，建议放一个ttf在工程目录）
        font = ImageFont.load_default()

    # 3. 绘制文字 (PIL使用RGB，但我们传进来的是BGR转RGB，所以color(0,255,0)依然是绿色)
    draw.text(position, text, font=font, fill=color)

    # 4. PIL 转回 OpenCV (BGR)，显式拷贝避免持有 PIL 内部内存视图
    out_rgb = np.array(img_pil, copy=True)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def visualizer_process(queue, visual=True):
    """
    独立进程：处理图像旋转、JSON保存及目标检测框可视化
    """
    vis_mode = os.environ.get("AUTOGAME_VIS_MODE", "window").strip().lower()
    show_window = visual and vis_mode != "launcher"
    print(f"[Visualizer] 显示进程已启动, 可视化状态: {visual}, mode: {vis_mode}")
    window_name = "Frame Monitor"
    log_dir = "aw/autogame/temp/logs/process_temp_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 统一显示尺寸 (826 * 2) * (384 * 2) = 1652 * 768
    target_width = 826 * 2
    target_height = 384 * 2

    def is_detection_list(value):
        if not isinstance(value, list) or not value:
            return False

        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                return False
            try:
                float(item[0])
                float(item[1])
                float(item[2])
                float(item[3])
                float(item[4])
            except (TypeError, ValueError):
                return False
        return True

    def draw_detection_list(frame, detections, font_size_getter, font_size):
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_id} {conf:.2f}"
            frame = draw_chinese_text(
                frame,
                label,
                (x1, max(font_size_getter(25), y1 - font_size_getter(10))),
                "simhei.ttf",
                font_size,
                (0, 255, 0),
            )
        return frame

    while True:
        try:
            data = queue.get()
            if data == "STOP":
                break

            frame_rgb, stage, info, index = data
            if not isinstance(frame_rgb, np.ndarray):
                frame_rgb = np.array(frame_rgb, copy=True)
            else:
                frame_rgb = np.ascontiguousarray(frame_rgb).copy()

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_rotated = frame_bgr.copy()

            # --- 关键修改：计算字体缩放比 ---
            # 获取旋转后原图的尺寸
            orig_h, orig_w = frame_rotated.shape[:2]
            # 计算缩放比例 (基于宽度的倍数)
            scale = target_width / orig_w

            # 根据比例动态调整字号
            # 如果原图很小，scale > 1，字号变大；如果原图很大，scale < 1，字号减小，从而在 resize 后保持视觉一致
            def get_scaled_size(base_size):
                return int(base_size / scale)

            # 调整后的基础参数
            font_main = get_scaled_size(24)  # 标题字号
            font_info = get_scaled_size(18)  # 信息字号
            y_offset = int(20 / scale)
            line_height = int(30 / scale)
            # ---------------------------

            # 2. 可视化检测框
            detection_keys = set()
            sorted_info_items = []
            if isinstance(info, dict):
                sorted_info_items = sorted(info.items(), key=lambda item: str(item[0]).lower())

            if visual and isinstance(info, dict):
                for k, v in sorted_info_items:
                    if is_detection_list(v):
                        detection_keys.add(k)
                        frame_rotated = draw_detection_list(frame_rotated, v, get_scaled_size, font_info)

            safe_info = {}
            if visual:
                base_text = f"ID: {index} | Stage: {stage}"
                frame_rotated = draw_chinese_text(frame_rotated, base_text, (int(10 / scale), y_offset),
                                                  "simhei.ttf", font_main, (255, 255, 255))
                y_offset += line_height

            if isinstance(info, dict):
                for k, v in sorted_info_items:
                    raw_str = str(v)
                    safe_info[k] = raw_str

                    if visual:
                        if k in detection_keys:
                            continue
                        val_str = raw_str
                        if len(val_str) > 50:
                            val_str = val_str[:50] + "..."
                        info_line = f"{k}: {val_str}"
                        frame_rotated = draw_chinese_text(
                            frame_rotated, info_line, (int(10 / scale), y_offset),
                            "simhei.ttf", font_info, (0, 255, 255)
                        )
                        y_offset += line_height
            else:
                safe_info = str(info)

            # 4. 存储原始图
            base_filename = os.path.join(log_dir, f"frame_{index:05d}")
            cv2.imwrite(f"{base_filename}.jpg", frame_rotated)
            with open(f"{base_filename}.json", "w", encoding="utf-8") as f:
                json.dump({"index": index, "stage": stage, "info": safe_info}, f, ensure_ascii=False, indent=4)

            # 5. 缩放显示 (此时文字会因为前面的反向补偿，在显示窗口中看起来大小适中)
            if show_window:
                frame_display = cv2.resize(frame_rotated, (target_width, target_height))
                cv2.imshow(window_name, frame_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"\n[Visualizer Error] {e}")
            break

    if show_window:
        cv2.destroyAllWindows()

def get_screen_mode(config_path="aw/autogame/config/config.json"):
    """
    仅读取并返回 config.json 中的 screen_mode 字段
    """
    if not os.path.exists(config_path):
        return "0"  # 默认值

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("screen_mode", "0")
    except Exception:
        return "0"

def get_display_rotation():
    """
    获取屏幕真实旋转角度（0/90/180/270）
    来源：[SCREEN PROPERTY] -> Rotation
    """

    try:
        result = subprocess.run(
            ["hdc", "shell", "hidumper", "-s", "DisplayManagerService", "-a", "-a"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print("[Rotation] Command failed:", result.stderr)
            return None

        output = result.stdout

        # 1️⃣ 先截取 [SCREEN PROPERTY] 区块
        block_match = re.search(
            r"\[SCREEN PROPERTY\](.*?)(\n\[|\Z)",
            output,
            re.S
        )

        if not block_match:
            print("[Rotation] SCREEN PROPERTY block not found")
            return None

        block = block_match.group(1)

        # 2️⃣ 在这个 block 里找 Rotation
        rot_match = re.search(r"Rotation:\s*(\d+)", block)

        if rot_match:
            return int(rot_match.group(1))

        print("[Rotation] Rotation not found in SCREEN PROPERTY")
        return None

    except Exception as e:
        print("[Rotation] Error:", e)
        return None

def insert_logs(log_name, time_dura, *key_words):
    """
    在指定日志文件中追加一行记录，并休眠 time_dura 秒。

    Args:
        log_name: 任务名称/日志标识
        time_dura: 持续时间（单位：秒）
        *key_words: 额外的关键词列表
    """
    # 1. 获取环境变量中的 task_name
    task_name = os.environ.get("TARGET_GAME_CASE", "default_task")

    # 2. 准备文件路径
    log_dir = os.path.join(r"aw/autogame/temp/results", task_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = os.path.join(log_dir, "time.txt")

    # 3. 处理时间格式
    start_dt = datetime.now()
    # 格式化 start_time: 月-日 时:分:秒.毫秒
    start_time_str = start_dt.strftime("%m-%d %H:%M:%S") + f".{int(start_dt.microsecond / 1000):03d}"

    # 4. 计算 end_time: start_time + time_dura (秒)
    # 修改点：这里改为 seconds=time_dura
    end_dt = start_dt + timedelta(seconds=time_dura)
    end_time_str = end_dt.strftime("%m-%d %H:%M:%S") + f".{int(end_dt.microsecond / 1000):03d}"

    # 5. 组合关键词字符串
    keywords_str = " ".join(map(str, key_words))

    # 6. 构建整行数据
    log_entry = f"{log_name} {start_time_str} {end_time_str} {keywords_str}\n"

    # 7. 追加模式写入文件
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"写入日志失败: {e}")

    # 8. 执行休眠
    # 修改点：直接 sleep time_dura，不需要再乘 60
    print(f"[Log] 已记录 {log_name}，开始休眠 {time_dura} 秒...")
    time.sleep(time_dura)

def analyze_txt(log_path, frame_path, time_txt_path, result_path):
    """
    解析 time.txt，根据时间段分析帧数据并将最终汇总结果记录在 result_path。

    :param log_path: 供子函数使用的日志路径
    :param frame_path: 截图帧所在的目录
    :param time_txt_path: 输入的包含时间段信息的 time.txt 路径
    :param result_path: 最终结果汇总输出路径 (results.txt)
    """
    if not os.path.exists(time_txt_path):
        print(f"Error: {time_txt_path} not found.")
        return

    # 定义关键字与函数的映射关系
    func_map = {
        "帧率": analyze_fps,
        "插帧": analyze_insert_frame,
        "超分": analyze_super_resolution
    }

    results_to_write = []

    with open(time_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # 基础格式要求：log_name, start_date, start_time, end_date, end_time (共5个parts)
            if len(parts) < 5:
                continue

            # 1. 解析基础字段
            # time.txt 格式：log_name 02-24 14:24:07.574 02-24 14:24:12.574 帧率 插帧
            log_name = parts[0]
            start_str = f"{parts[1]} {parts[2]}" # 拼接日期和时间
            end_str = f"{parts[3]} {parts[4]}"
            keywords = parts[5:]

            # 2. 执行分析逻辑
            current_line_results = [log_name]

            for kw in keywords:
                if kw in func_map:
                    # 按照你更新后的逻辑，传入 log_path 供子函数参考
                    res = func_map[kw](log_path, frame_path, start_str, end_str)
                    current_line_results.append(f"{kw}:{res}")

            # 3. 组合成一行：log_name 关键字1:结果1 关键字2:结果2
            results_to_write.append(" ".join(current_line_results))

    # 4. 写入最终结果文件 (使用 result_path)
    # 先确保输出目录存在
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_to_write) + "\n")

    print(f"Analysis complete. Summary saved to {result_path}")

def analyze_fps(log_path, frame_path, start_time, end_time):
    """
    读取 log_path，提取 start_time 到 end_time 之间所有的 targetFps 值。

    :param log_path: 日志文件路径 (例如 hdc log 产生的文本)
    :param frame_path: 帧目录 (此函数暂不用)
    :param start_time: 开始时间字符串 "02-24 14:24:07.574"
    :param end_time: 结束时间字符串 "02-24 14:24:12.574"
    :return: 包含所有 targetFps 的字符串，例如 "60 90"
    """
    if not os.path.exists(log_path):
        return "LogNotFound"

    # 匹配 "targetFps":60 或 "targetFps": 60 或 targetFps:60 等格式
    # 使用正则表达式匹配 targetFps 后面跟着的数字
    fps_pattern = re.compile(r'targetFps["\']?\s*:\s*(\d+)')

    found_fps = []

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1. 提取行首时间 (前 18 位左右: 02-24 15:12:25.942)
                # 假设日志格式固定，前18位是时间
                line_time = line[:18]

                # 2. 时间区间过滤 (利用字符串直接比较大小)
                if start_time <= line_time <= end_time:
                    # 3. 搜索该行是否存在 targetFps
                    matches = fps_pattern.findall(line)
                    for val in matches:
                        if val not in found_fps:  # 去重，只记录出现过的不同帧率
                            found_fps.append(val)

                # 如果当前行时间已经超过了 end_time，可以提前结束读取提高效率
                elif line_time > end_time:
                    # 注意：如果日志不是严格按时间顺序写的，请删掉这两行
                    break

    except Exception as e:
        print(f"Error reading log: {e}")
        return "Error"

    # 返回结果，如果没有找到则返回空字符串或特定的默认值
    return " ".join(found_fps) if found_fps else "None"

def analyze_insert_frame(log_path, frame_path, start_time, end_time):
    """
    分析指定时间段内的截图，只要有一张图片红色像素占比 > 0.1 则立即返回“生效”。
    """
    if not os.path.exists(frame_path):
        return "目录不存在"

    # 1. 筛选时间段内的图片
    all_files = os.listdir(frame_path)
    target_frames = []

    # 将标准时间格式中的 : 替换为 - 以便匹配文件名 (02-24 13-56-42.982.jpg)
    s_time_cmp = start_time.replace(":", "-")
    e_time_cmp = end_time.replace(":", "-")

    for f in all_files:
        if f.endswith((".jpg", ".png")):
            file_name_no_ext = os.path.splitext(f)[0]
            if s_time_cmp <= file_name_no_ext <= e_time_cmp:
                target_frames.append(f)

    if not target_frames:
        return "未找到帧"

    # 2. 逐张分析，一旦发现满足条件的图片立即返回
    for f in target_frames:
        file_path = os.path.join(frame_path, f)
        img = cv2.imread(file_path)
        if img is None:
            continue

        # 获取总像素数
        height, width = img.shape[:2]
        total_pixels = height * width

        # --- 核心红色提取逻辑 ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义红色范围 (S/V 下限 200/220)
        lower_red1 = np.array([0, 200, 220])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 200, 220])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.add(mask1, mask2)

        # 形态学处理：3x3 核腐蚀再膨胀
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        # 统计处理后的红色像素点
        red_pixel_count = cv2.countNonZero(red_mask)

        # 计算比例
        if red_pixel_count > 3000:
            # --- 关键改进：找到第一个满足条件的就停止后续所有计算 ---
            return "生效"

    # 如果遍历完所有图片都没有满足条件的
    return "失效"

def analyze_super_resolution(log_path, frame_path, start_time, end_time):
    return "Done"

if __name__ == '__main__':
    print(get_resolution())
