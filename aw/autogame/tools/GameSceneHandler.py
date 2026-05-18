import os
import importlib
import concurrent.futures
from aw.autogame.tools.Utils import *
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
import numpy as np

project_case = os.environ.get("TARGET_PROJECT_CASE")
if not project_case:
    raise ValueError("TARGET_PROJECT_CASE 未设置，无法定位资源路径")

handler_path = f"aw.autogame.customs_examples.{project_case}.resource.SpecialSceneHandler"
try:
    special_handler_module = importlib.import_module(handler_path)
    SpecialHandler = special_handler_module
    print(f"成功从项目 [{project_case}] 加载 SpecialSceneHandler")
except ImportError as e:
    print(f"路径错误: 无法在 {handler_path} 找到 SpecialSceneHandler 模块")
    raise e

class GameImageProcessor:
    def __init__(self, project_name):
        self.project_root = os.path.join(r"aw/autogame/customs_examples", project_name)
        self.template_cache = self._load_templates()
        self.special_handler = SpecialHandler
        self.task_config = None
        self.screen_w, self.screen_h = self._resolve_screen_resolution()

    def _resolve_screen_resolution(self):
        env_w = os.environ.get("AUTOGAME_SCREEN_WIDTH")
        env_h = os.environ.get("AUTOGAME_SCREEN_HEIGHT")
        if env_w and env_h:
            try:
                return int(env_w), int(env_h)
            except ValueError:
                pass

        screen_w, screen_h = get_resolution()
        if screen_w and screen_h:
            return int(screen_w), int(screen_h)
        return None, None

    def _load_templates(self):
        cache = {}
        if not os.path.exists(self.project_root):
            print(f"Warning: Project directory not found: {self.project_root}")
            return cache
        valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(valid_exts):
                    abs_path = os.path.join(root, file)
                    img_arr = np.fromfile(abs_path, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    cache[os.path.normpath(abs_path)] = img
        return cache

    def process(self, raw_frame, tasks_config, buffer_ratio=0.1):
        self.task_config = tasks_config
        curr_h, curr_w = raw_frame.shape[:2]
        results = {}

        def _execute_task(task_id, config):
            try:
                task_type = config.get('type')
                origin_w = config.get('origin_width')
                origin_h = config.get('origin_height')

                if not origin_w or not origin_h:
                    return task_id, "Error: Missing origin resolution info"

                global_scale = curr_w / origin_w

                # Case 1: 特殊区域
                if task_type == 'special':
                    area_config = config.get('area_config') or config
                    if 'anchor' in area_config or 'rect' in area_config:
                        x1, y1, x2, y2 = resolve_area_rect_for_frame(
                            curr_w,
                            curr_h,
                            area_config,
                            self.screen_w,
                            self.screen_h,
                            origin_w,
                            origin_h,
                        )
                        x1 = max(0, min(curr_w, x1))
                        y1 = max(0, min(curr_h, y1))
                        x2 = max(0, min(curr_w, x2))
                        y2 = max(0, min(curr_h, y2))
                        target_img = np.ascontiguousarray(raw_frame[y1:y2, x1:x2]).copy()
                    else:
                        target_img = np.ascontiguousarray(raw_frame).copy()

                    handler_name = config.get('handler_name', task_id)
                    method = getattr(self.special_handler, handler_name, None)
                    return task_id, method(target_img) if method else f"Error: {handler_name} not found"

                # Case 2: 模板匹配
                elif task_type == 'template':
                    scope = config.get('scope')
                    tpl_relative_path = config.get('template_path')
                    match_mode = config.get('match_mode', 'gray')

                    if scope:
                        px_min_x = int(scope[0] * curr_w)
                        px_min_y = int(scope[1] * curr_h)
                        px_max_x = int(scope[2] * curr_w)
                        px_max_y = int(scope[3] * curr_h)

                        w_rect = px_max_x - px_min_x
                        h_rect = px_max_y - px_min_y
                        buf_w = int(w_rect * buffer_ratio)
                        buf_h = int(h_rect * buffer_ratio)

                        crop_x1 = max(0, px_min_x - buf_w)
                        crop_y1 = max(0, px_min_y - buf_h)
                        crop_x2 = min(curr_w, px_max_x + buf_w)
                        crop_y2 = min(curr_h, px_max_y + buf_h)

                        search_img = raw_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        offset = (crop_x1, crop_y1)
                    else:
                        search_img = raw_frame
                        offset = (0, 0)

                    full_tpl_path = os.path.normpath(os.path.join(self.project_root, tpl_relative_path))
                    tpl_img_raw = self.template_cache.get(full_tpl_path)

                    if tpl_img_raw is None and os.path.exists(full_tpl_path):
                        arr = np.fromfile(full_tpl_path, dtype=np.uint8)
                        tpl_img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if tpl_img_raw is None:
                        return task_id, False

                    tpl_h, tpl_w = tpl_img_raw.shape[:2]
                    new_w = int(tpl_w * global_scale)
                    new_h = int(tpl_h * global_scale)

                    if new_w > 0 and new_h > 0:
                        tpl_img_resized = cv2.resize(tpl_img_raw, (new_w, new_h))
                    else:
                        tpl_img_resized = tpl_img_raw

                    match_res = find_template_center_multiscale(
                        search_img,
                        tpl_img_resized,
                        match_mode=match_mode,
                    )

                    if match_res:
                        local_x, local_y = match_res
                        final_pixel_x = local_x + offset[0]
                        final_pixel_y = local_y + offset[1]
                        norm_x = final_pixel_x / curr_w
                        norm_y = final_pixel_y / curr_h
                        return task_id, (norm_x, norm_y)
                    else:
                        return task_id, False

            except Exception as e:
                return task_id, f"Err: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_execute_task, k, v) for k, v in tasks_config.items()]
            for future in concurrent.futures.as_completed(futures):
                tid, res = future.result()
                results[tid] = res
        return results

class StageLogicController:
    def __init__(self):
        """
        初始化：动态加载环境变量指定的项目配置
        """
        # 从环境变量获取项目 case 名
        project_case = os.environ.get("TARGET_PROJECT_CASE")
        if not project_case:
            raise ValueError("Environment variable 'TARGET_PROJECT_CASE' is not set!")

        # 动态导包
        module_path = f"aw.autogame.customs_examples.{project_case}.info"
        info_mod = importlib.import_module(module_path)

        self.project_name = getattr(info_mod, "PROJECT_NAME")
        self.stage_info = getattr(info_mod, "STAGE_INFO")

        self.processor = GameImageProcessor(project_case)
        print(f"[{self.project_name}] 逻辑控制器已就绪。")

    def process_frame(self, frame_img, current_stage_name):
        """
        处理单帧逻辑。

        Args:
            frame_img: 当前视频帧
            current_stage_name (str): 由 Framework 传入的当前阶段名称 (如 '关闭弹窗')

        Returns:
            dict: 检测结果
        """
        # 1. 如果 Framework 没传阶段名，或者传了 None，直接返回
        if not current_stage_name:
            return {}

        # 2. 构建任务配置 (根据传入的 stage_name 查找配置)
        # 这里不再读取全局 STAGE_DICT，而是完全依赖传入的参数
        stage_data = self.stage_info.get(current_stage_name, {})
        scenes = stage_data.get('scenes', {})
        tasks_config = {}

        # 遍历该阶段下的所有场景和区域
        for scene_name, scene_info in scenes.items():
            origin_w = scene_info.get('width')
            origin_h = scene_info.get('height')

            # 普通 Areas (模板匹配)
            areas = scene_info.get('areas', {})
            for area_name, area_data in areas.items():
                task_key = f"{scene_name}__{area_name}"
                scope = area_data.get('search_scope', area_data.get('rect'))
                tasks_config[task_key] = {
                    'type': 'template',
                    'scope': scope,
                    'template_path': area_data.get('template'),
                    'match_mode': area_data.get('match_mode', 'gray'),
                    'origin_width': origin_w,
                    'origin_height': origin_h
                }

            # Special Areas (特殊函数)
            special_areas = scene_info.get('special_areas', {})
            for sa_name, sa_data in special_areas.items():
                task_key = f"{scene_name}__{sa_name}"
                tasks_config[task_key] = {
                    'type': 'special',
                    'rect': sa_data.get('rect'),
                    'area_config': sa_data,
                    'handler_name': sa_name,
                    'origin_width': origin_w,
                    'origin_height': origin_h
                }

        # 3. 调用处理器执行具体计算
        if not tasks_config:
            return {}

        final_results = self.processor.process(frame_img, tasks_config, buffer_ratio=0.1)

        return final_results

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 初始化 (仅一次)
    logic_controller = StageLogicController()

    # 模拟数据
    mock_frame = np.zeros((384, 826, 3), dtype=np.uint8)

    # 2. 循环调用 (只需传 frame)
    # 假设在其他地方 STAGE_DICT['开始游戏'] 已经被置为 True
    results = logic_controller.process_frame(mock_frame)

    print("\n运行结果:")
    for k, v in results.items():
        print(f"{k}: {v}")
