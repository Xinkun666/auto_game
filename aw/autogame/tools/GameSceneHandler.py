import os
import importlib
import concurrent.futures
import copy
from aw.autogame.tools.Utils import *
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
import numpy as np

DEFAULT_GROUP_NAME = "默认"
GROUPABLE_ITEM_TYPES = ("area", "special_area")


def load_stage_info(project_case, info_mod):
    """加载阶段配置，并合并项目内可选的运行时分组覆盖。"""
    stage_info = copy.deepcopy(getattr(info_mod, "STAGE_INFO"))
    override_module_path = (
        f"aw.autogame.customs_examples.{project_case}.resource.stage_group_config"
    )
    try:
        override_mod = importlib.import_module(override_module_path)
    except ModuleNotFoundError as exc:
        missing_name = str(exc.name or "")
        if not (
            missing_name == override_module_path
            or override_module_path.startswith(f"{missing_name}.")
        ):
            raise
        return stage_info

    overrides = getattr(override_mod, "STAGE_GROUP_OVERRIDES", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"{override_module_path}.STAGE_GROUP_OVERRIDES 必须是字典")
    for stage_name, override in overrides.items():
        if not isinstance(override, dict):
            continue
        stage_data = stage_info.get(stage_name)
        if not isinstance(stage_data, dict):
            raise ValueError(f"分组覆盖引用了不存在的阶段: {stage_name}")
        initial_group = override.get("initial_group")
        if initial_group:
            stage_data["initial_group"] = str(initial_group)
        override_groups = override.get("groups", {})
        if isinstance(override_groups, dict):
            stage_data.setdefault("groups", {}).update(copy.deepcopy(override_groups))
    return stage_info

def load_special_handler(project_case):
    if not project_case:
        raise ValueError("TARGET_PROJECT_CASE 未设置，无法定位资源路径")

    handler_path = f"aw.autogame.customs_examples.{project_case}.resource.SpecialSceneHandler"
    try:
        special_handler_module = importlib.import_module(handler_path)
        print(f"成功从项目 [{project_case}] 加载 SpecialSceneHandler")
        return special_handler_module
    except ImportError as e:
        print(f"路径错误: 无法在 {handler_path} 找到 SpecialSceneHandler 模块")
        raise e

class GameImageProcessor:
    def __init__(self, project_name, special_handler=None):
        self.project_root = os.path.join(r"aw/autogame/customs_examples", project_name)
        self.template_cache = self._load_templates()
        self.special_handler = special_handler or load_special_handler(project_name)
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

    @staticmethod
    def _offset_bbox(bbox, offset_x, offset_y):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return bbox
        return [
            int(bbox[0]) + offset_x,
            int(bbox[1]) + offset_y,
            int(bbox[2]) + offset_x,
            int(bbox[3]) + offset_y,
        ]

    @staticmethod
    def _offset_contours(contours, offset_x, offset_y):
        if not isinstance(contours, list):
            return contours
        shifted = []
        for contour in contours:
            if not isinstance(contour, list):
                continue
            points = []
            for point in contour:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                points.append([int(point[0]) + offset_x, int(point[1]) + offset_y])
            if len(points) >= 2:
                shifted.append(points)
        return shifted

    def _map_special_visualizations(self, result, crop_xyxy):
        if not isinstance(result, dict):
            return result

        visuals = result.get("__visualizations__")
        if not isinstance(visuals, list):
            return result

        x1, y1, x2, y2 = crop_xyxy
        mapped_visuals = []
        for visual in visuals:
            if not isinstance(visual, dict):
                continue
            item = dict(visual)
            if item.get("coord", "local") == "local":
                item["bbox_xyxy"] = self._offset_bbox(item.get("bbox_xyxy"), x1, y1)
                item["contours"] = self._offset_contours(item.get("contours"), x1, y1)
                item["coord"] = "frame"
                item["source_crop_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
            mapped_visuals.append(item)

        mapped_result = dict(result)
        mapped_result["__visualizations__"] = mapped_visuals
        return mapped_result

    @staticmethod
    def _split_special_timing_result(method, result):
        if getattr(method, "__special_timing_enabled__", False):
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
        return result, None

    @staticmethod
    def _format_special_result_for_info(result, timing_ms):
        if timing_ms is None:
            return result
        return [result, [timing_ms]]

    def process(self, raw_frame, tasks_config, buffer_ratio=0.3):
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
                        x1, y1, x2, y2 = 0, 0, curr_w, curr_h
                        target_img = np.ascontiguousarray(raw_frame).copy()

                    handler_name = config.get('handler_name', task_id)
                    method = getattr(self.special_handler, handler_name, None)
                    if not method:
                        return task_id, f"Error: {handler_name} not found"
                    raw_special_result = method(target_img)
                    special_result, timing_ms = self._split_special_timing_result(method, raw_special_result)
                    mapped_result = self._map_special_visualizations(
                        special_result,
                        (x1, y1, x2, y2),
                    )
                    return task_id, self._format_special_result_for_info(mapped_result, timing_ms)

                # Case 2: 模板匹配
                elif task_type == 'template':
                    scope = config.get('scope')
                    scope_config = config.get('scope_config')
                    tpl_relative_path = config.get('template_path')
                    match_mode = config.get('match_mode', 'gray')

                    if scope_config or scope:
                        px_min_x, px_min_y, px_max_x, px_max_y = resolve_area_rect_for_frame(
                            curr_w,
                            curr_h,
                            scope_config or {"rect": scope},
                            self.screen_w,
                            self.screen_h,
                            origin_w,
                            origin_h,
                        )
                        px_min_x = max(0, min(curr_w, px_min_x))
                        px_min_y = max(0, min(curr_h, px_min_y))
                        px_max_x = max(0, min(curr_w, px_max_x))
                        px_max_y = max(0, min(curr_h, px_max_y))

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
        self.processor = GameImageProcessor(project_case)
        raw_stage_info = load_stage_info(project_case, info_mod)
        self.stage_info = lock_stage_info_scene_resolutions(
            raw_stage_info,
            self.processor.screen_w,
            self.processor.screen_h,
        )
        print(f"[{self.project_name}] 场景分辨率已锁定: {self.processor.screen_w}x{self.processor.screen_h}")
        print(f"[{self.project_name}] 逻辑控制器已就绪。")

    def get_stage_groups(self, stage_name):
        stage_data = self.stage_info.get(stage_name, {})
        groups = stage_data.get('groups', {}) if isinstance(stage_data, dict) else {}
        if not isinstance(groups, dict) or not groups:
            return [DEFAULT_GROUP_NAME]
        names = [DEFAULT_GROUP_NAME]
        for name in groups.keys():
            if name != DEFAULT_GROUP_NAME:
                names.append(name)
        return names

    def has_group(self, stage_name, group_name):
        if not group_name:
            group_name = DEFAULT_GROUP_NAME
        return group_name in self.get_stage_groups(stage_name)

    def get_initial_group(self, stage_name):
        """返回进入阶段时应启用的运行分组。

        未配置 initial_group 的老工程继续使用内置的“默认”全量组；
        配置值无效时也安全回退，避免阶段切换后处于不存在的分组。
        """
        stage_data = self.stage_info.get(stage_name, {})
        configured = (
            stage_data.get('initial_group', DEFAULT_GROUP_NAME)
            if isinstance(stage_data, dict)
            else DEFAULT_GROUP_NAME
        )
        group_name = str(configured or DEFAULT_GROUP_NAME).strip()
        if self.has_group(stage_name, group_name):
            return group_name
        print(
            f"[WARN] 阶段 '{stage_name}' 的 initial_group '{group_name}' 不存在，"
            f"回退到 '{DEFAULT_GROUP_NAME}'。"
        )
        return DEFAULT_GROUP_NAME

    def _resolve_group_filter(self, stage_data, group_name):
        if not group_name or group_name == DEFAULT_GROUP_NAME:
            return None
        groups = stage_data.get('groups', {}) if isinstance(stage_data, dict) else {}
        if not isinstance(groups, dict):
            return None
        group_data = groups.get(group_name)
        if group_data is None:
            return set()
        if isinstance(group_data, dict) and group_data.get('all'):
            return None

        def parse_item_refs(raw_items):
            refs = set()
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                scene_name = str(item.get('scene', '')).strip()
                item_type = str(item.get('type', '')).strip()
                item_name = str(item.get('name', '')).strip()
                if scene_name and item_name and item_type in GROUPABLE_ITEM_TYPES:
                    refs.add((scene_name, item_type, item_name))
            return refs

        raw_items = group_data.get('items', []) if isinstance(group_data, dict) else []
        allowed = parse_item_refs(raw_items)
        excluded_items = (
            group_data.get('exclude_items')
            if isinstance(group_data, dict)
            else None
        )
        if not isinstance(excluded_items, list):
            return allowed

        allowed = set()
        scenes = stage_data.get('scenes', {}) if isinstance(stage_data, dict) else {}
        for scene_name, scene_info in scenes.items():
            if not isinstance(scene_info, dict):
                continue
            for area_name in scene_info.get('areas', {}):
                allowed.add((scene_name, 'area', area_name))
            for area_name in scene_info.get('special_areas', {}):
                allowed.add((scene_name, 'special_area', area_name))
        return allowed - parse_item_refs(excluded_items)

    def process_frame(self, frame_img, current_stage_name, group_name=DEFAULT_GROUP_NAME):
        """
        处理单帧逻辑。

        Args:
            frame_img: 当前视频帧
            current_stage_name (str): 由 Framework 传入的当前阶段名称 (如 '关闭弹窗')
            group_name (str): 当前阶段内要识别的分组名，默认分组识别全部区域和特殊区域

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
        group_filter = self._resolve_group_filter(stage_data, group_name)
        tasks_config = {}

        # 遍历该阶段下的所有场景和区域
        for scene_name, scene_info in scenes.items():
            origin_w = scene_info.get('width')
            origin_h = scene_info.get('height')

            # 普通 Areas (模板匹配)
            areas = scene_info.get('areas', {})
            for area_name, area_data in areas.items():
                if group_filter is not None and (scene_name, 'area', area_name) not in group_filter:
                    continue
                task_key = f"{scene_name}__{area_name}"
                scope = area_data.get('search_scope', area_data.get('rect'))
                if area_data.get('search_scope'):
                    scope_config = {'rect': area_data.get('search_scope')}
                elif 'anchor' in area_data:
                    scope_config = area_data
                elif area_data.get('rect'):
                    scope_config = {'rect': area_data.get('rect')}
                else:
                    scope_config = None
                tasks_config[task_key] = {
                    'type': 'template',
                    'scope': scope,
                    'scope_config': scope_config,
                    'template_path': area_data.get('template'),
                    'match_mode': area_data.get('match_mode', 'gray'),
                    'origin_width': origin_w,
                    'origin_height': origin_h
                }

            # Special Areas (特殊函数)
            special_areas = scene_info.get('special_areas', {})
            for sa_name, sa_data in special_areas.items():
                if group_filter is not None and (scene_name, 'special_area', sa_name) not in group_filter:
                    continue
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

        final_results = self.processor.process(frame_img, tasks_config, buffer_ratio=0.3)

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
