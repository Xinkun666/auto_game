import cv2
import math
import random
import heapq
import time
from itertools import count
from collections import deque
import os
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import log_step
from aw.autogame.tools.Utils import resolve_process_temp_logs_dir, write_image_unicode

NAVIGATION_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.dirname(NAVIGATION_DIR)
DEFAULT_ROUTE_IMAGE_PATH = os.path.join(RESOURCE_DIR, "map", "hpjy.png")
DEFAULT_ROUTE_OUTPUT_PATH = os.path.join("aw", "autogame", "temp", "road", "route.jpg")
_ROUTE_IMAGE_COUNTER = count(1)


def _route_point(value):
    try:
        return (int(round(float(value[0]))), int(round(float(value[1]))))
    except (TypeError, ValueError, IndexError):
        return None


def _route_image_filename(start, end):
    start_loc = _route_point(start) or ("x", "x")
    end_loc = _route_point(end) or ("x", "x")
    timestamp = time.strftime("%Y%m%d%H%M%S")
    seq = next(_ROUTE_IMAGE_COUNTER)
    return (
        f"route_{timestamp}_{seq:04d}_"
        f"{start_loc[0]}_{start_loc[1]}_to_{end_loc[0]}_{end_loc[1]}.png"
    )


def save_route_image_for_log(
    road_list,
    start_pos=None,
    end_pos=None,
    image_path=None,
):
    if not road_list:
        return None, "路径点为空"

    image_path = image_path or DEFAULT_ROUTE_IMAGE_PATH
    if not os.path.exists(image_path):
        return None, f"底图不存在: {image_path}"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, f"底图读取失败: {image_path}"

    points = [_route_point(point) for point in road_list]
    points = [point for point in points if point is not None]
    if not points:
        return None, "路径点无法解析"

    route_dir = resolve_process_temp_logs_dir() / "route"
    route_dir.mkdir(parents=True, exist_ok=True)
    start = start_pos if start_pos is not None else points[0]
    end = end_pos if end_pos is not None else points[-1]
    filename = _route_image_filename(start, end)
    output_path = route_dir / filename

    line_color = (0, 220, 0)
    point_color = (0, 0, 255)
    start_color = (255, 0, 0)
    end_color = (0, 165, 255)
    for index, point in enumerate(points):
        if index > 0:
            cv2.line(image, points[index - 1], point, line_color, 2, lineType=cv2.LINE_AA)
        cv2.circle(image, point, 3, point_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, points[0], 4, start_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, points[-1], 4, end_color, -1, lineType=cv2.LINE_AA)

    if not write_image_unicode(output_path, image):
        return None, f"路径图写入失败: {output_path}"
    return f"route/{filename}", None


class MapNavigator:
    def __init__(self, map_mask_path = r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/map/hpjy_mask.tif'):
        """
        初始化导航器
        :param map_mask_path: 地图掩码图片的路径 (黑白图)
        """
        # 以灰度模式读取图片
        self.map_img = cv2.imread(map_mask_path, cv2.IMREAD_GRAYSCALE)

        if self.map_img is None:
            raise ValueError(f"无法读取图片路径: {map_mask_path}")

        # 二值化处理：确保只有0(黑)和255(白)
        # 假设：255(白) = 可通行, 0(黑) = 障碍物
        _, self.binary_map = cv2.threshold(self.map_img, 127, 255, cv2.THRESH_BINARY)


        # 获取地图尺寸 (高度=rows, 宽度=cols)
        self.height, self.width = self.binary_map.shape
        self._forbidden_region_labels = None

    def _is_walkable(self, x, y):
        """内部辅助函数：检查某个坐标是否可通行"""
        # 1. 检查边界
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        # 2. 检查颜色 (注意：opencv中 img[y, x])
        return self.binary_map[int(y), int(x)] == 255

    def is_walkable(self, pos):
        if pos is None or len(pos) < 2:
            return False
        return self._is_walkable(int(pos[0]), int(pos[1]))

    def _forbidden_labels(self):
        if self._forbidden_region_labels is None:
            forbidden_mask = (self.binary_map != 255).astype("uint8")
            _, labels, _, _ = cv2.connectedComponentsWithStats(forbidden_mask, connectivity=8)
            self._forbidden_region_labels = labels
        return self._forbidden_region_labels

    def same_forbidden_region(self, start_pos, end_pos):
        """判断两个点是否落在同一个连续不可通行区域。"""
        if start_pos is None or end_pos is None:
            return False
        try:
            sx, sy = int(start_pos[0]), int(start_pos[1])
            ex, ey = int(end_pos[0]), int(end_pos[1])
        except (TypeError, ValueError, IndexError):
            return False

        if (
            sx < 0 or sx >= self.width or sy < 0 or sy >= self.height
            or ex < 0 or ex >= self.width or ey < 0 or ey >= self.height
        ):
            return False

        if self._is_walkable(sx, sy) or self._is_walkable(ex, ey):
            return False

        labels = self._forbidden_labels()
        start_label = int(labels[sy, sx])
        end_label = int(labels[ey, ex])
        return start_label != 0 and start_label == end_label

    def nearest_walkable_within_radius(self, pos, radius):
        """在当前位置映射到 mask 后，只扫描局部半径内的可通行像素。"""
        if pos is None or len(pos) < 2:
            return None, float("inf")

        try:
            center_x = int(round(float(pos[0])))
            center_y = int(round(float(pos[1])))
        except (TypeError, ValueError):
            return None, float("inf")

        radius = max(0, int(math.ceil(float(radius))))
        best_point = None
        best_dist_sq = None

        min_x = max(0, center_x - radius)
        max_x = min(self.width - 1, center_x + radius)
        min_y = max(0, center_y - radius)
        max_y = min(self.height - 1, center_y + radius)
        radius_sq = radius * radius

        for y in range(min_y, max_y + 1):
            dy = y - center_y
            for x in range(min_x, max_x + 1):
                dx = x - center_x
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius_sq:
                    continue
                if self.binary_map[y, x] != 255:
                    continue
                if best_dist_sq is None or dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_point = (x, y)
                    if dist_sq == 0:
                        return best_point, 0.0

        if best_point is None:
            return None, float("inf")
        return best_point, math.sqrt(best_dist_sq)

    def check_safety_ahead(self, current_pos, angle, distance=10, step_size=1.0):
        """
        功能1（优化版）：射线检测。
        :param angle: 游戏角度 (上=0, 右=90, 下=180, 左=270)
        """
        curr_x, curr_y = current_pos

        # 【核心修改】：将游戏角度(上0)转换为数学坐标角度(右0)
        # 游戏0度(上) -> 数学-90度 (sin为-1，即向上)
        # 游戏90度(右) -> 数学0度 (cos为1，即向右)
        rad = math.radians(angle - 90)

        dir_x = math.cos(rad)
        dir_y = math.sin(rad)

        # ... (后续代码保持不变) ...
        # 2. 循环采样检测
        current_dist = step_size
        while current_dist <= distance:
            check_x = curr_x + dir_x * current_dist
            check_y = curr_y + dir_y * current_dist

            if not self._is_walkable(check_x, check_y):
                return "Pause"
            current_dist += step_size

        # 3. 检查终点
        target_x = curr_x + dir_x * distance
        target_y = curr_y + dir_y * distance
        if not self._is_walkable(target_x, target_y):
            return "Pause"

        return "Safe"

    def plan_path(self, start_pos, end_pos):
        """
        功能2：规划路径，尽量走直线并避障 (A*算法 + 路径平滑)
        :param start_pos: tuple (x, y)
        :param end_pos: tuple (x, y)
        :return: list of tuples [(x, y), ...] 路径点列表
        """
        start = (int(start_pos[0]), int(start_pos[1]))
        end = (int(end_pos[0]), int(end_pos[1]))
        direct_distance = math.hypot(end[0] - start[0], end[1] - start[1])
        log_step(
            f"当前要规划地图路径：current_loc={start}, target_loc={end}, "
            f"straight_distance={direct_distance:.2f}",
            target="地图路径规划",
            action="检查终点是否可通行，然后用 A* 生成路径",
            method="MapNavigator.plan_path(start_pos, end_pos)",
            result="准备开始 A* 搜索",
        )

        if not self._is_walkable(*end):
            print("警告：目的地不可达（位于黑色区域）")
            log_step(
                f"地图路径规划失败：current_loc={start}, target_loc={end}，目标点在不可通行黑区",
                target="地图路径规划",
                action="放弃本次路径",
                method="_is_walkable(target_loc)",
                result="返回空路径，调用方需要选择安全点或回退策略",
            )
            return []

        # --- 第一步：A* 算法寻找基础路径 ---
        # 优先队列: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}

        # g_score: 从起点到当前点的代价
        g_score = {start: 0}

        # f_score: g_score + 启发式估算(到终点的直线距离)
        f_score = {start: self._heuristic(start, end)}

        path_found = False

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path_found = True
                break

            # 遍历8个邻居节点 (上下左右 + 对角线)
            neighbors = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                # 移动代价：直线是1，对角线是sqrt(2) ≈ 1.414
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + move_cost

                if self._is_walkable(*neighbor):
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        if not path_found:
            log_step(
                f"地图路径规划失败：current_loc={start}, target_loc={end}，A* 未找到可达路线",
                target="地图路径规划",
                action="放弃本次路径",
                method="A* open_set 搜索结束",
                result="返回空路径，调用方需要重新规划或回退自由巡航",
            )
            return []  # 无法到达

        # 重建路径 (从终点回溯到起点)
        path = []
        curr = end
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.append(start)
        path.reverse()

        # --- 第二步：路径平滑 (Floyd's Algorithm / Line of Sight) ---
        # A* 生成的路径是基于格子的锯齿状，这里我们将其简化为直线段
        smoothed_path = self._smooth_path(path)
        route_image_name, route_image_error = save_route_image_for_log(smoothed_path, start, end)
        if route_image_name:
            route_result = (
                f"已经规划好路径，图片名称是 {route_image_name}；"
                f"下一步使用首个路径点 {smoothed_path[0] if smoothed_path else None} 导航"
            )
        else:
            route_result = (
                f"已经规划好路径，但路径图未生成：{route_image_error}；"
                f"下一步使用首个路径点 {smoothed_path[0] if smoothed_path else None} 导航"
            )
        log_step(
            f"路径规划成功：current_loc={start}, target_loc={end}, "
            f"raw_points={len(path)}, smooth_points={len(smoothed_path)}",
            target="地图路径规划",
            action="把 A* 原始路径压缩成更少的直线路径点",
            method="_smooth_path(path)",
            result=route_result,
        )
        return smoothed_path

    def _heuristic(self, a, b):
        """欧几里得距离启发函数，鼓励走直线"""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _smooth_path(self, path):
        """
        简化路径：如果点A和点C之间可以直接连线（无障碍），则去掉点B
        """
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # 尝试找最远的可视点
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx, -1):
                if self._check_line_of_sight(path[current_idx], path[i]):
                    next_idx = i
                    break

            smoothed_path.append(path[next_idx])
            current_idx = next_idx

        return smoothed_path

    def _check_line_of_sight(self, p1, p2):
        """使用Bresenham算法或采样法检查两点连线是否有障碍"""
        x1, y1 = p1
        x2, y2 = p2

        # 简单采样检测：沿着线段每隔几个像素检查一次
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0: return True

        steps = int(dist)  # 每像素检查一次
        for i in range(steps):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            if not self._is_walkable(x, y):
                return False
        return True

    def get_target_point(self, current_pos, angle, distance):
        """
        根据当前位置、游戏角度和距离，计算目标点。
        如果计算出的目标点是障碍物，则自动搜索并返回离该目标点最近的可通行点。

        :param current_pos: tuple (x, y) 当前坐标
        :param angle: float 游戏角度 (上=0, 右=90, 下=180, 左=270, 顺时针)
        :param distance: float 距离
        :return: tuple (x, y) 目标点坐标 (一定可通行)
        """
        curr_x, curr_y = current_pos

        # 1. 角度转换 & 理论坐标计算
        # 游戏角度转数学弧度：radians(angle - 90)
        rad = math.radians(angle - 90)

        raw_target_x = curr_x + distance * math.cos(rad)
        raw_target_y = curr_y + distance * math.sin(rad)

        # 2. 坐标取整与边界钳制 (Clamp)
        # 即使算到了地图外面，也先拉回地图边缘，作为搜索的起点
        target_x = int(max(0, min(raw_target_x, self.width - 1)))
        target_y = int(max(0, min(raw_target_y, self.height - 1)))

        # 3. 如果该点本身是可通行的，直接返回
        if self._is_walkable(target_x, target_y):
            return (target_x, target_y)

        # 4. 如果不可通行，搜索最近的可通行点 (BFS)
        # 这种场景下，我们以“理论目标点”为圆心向外找
        queue = deque([(target_x, target_y)])
        visited = set()
        visited.add((target_x, target_y))

        # 搜索范围限制 (防止全图无路死循环)
        max_search_steps = 5000
        steps = 0

        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]

        while queue:
            cx, cy = queue.popleft()
            steps += 1
            if steps > max_search_steps:
                break

            # 找到第一个白点，这就是离目标最近的有效点
            if self._is_walkable(cx, cy):
                return (cx, cy)

            # 继续向外扩散
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                # 检查边界
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        # 极少数情况：如果周围全是黑的找不到路，返回当前位置或者 None
        # 这里选择返回当前位置作为保底，意味着“没法移动”
        return current_pos

    def _find_nearest_safe_point(self, start_pos, max_search_dist=100):
        """
        内部辅助函数：使用 BFS 寻找离当前点最近的可通行点 (白色区域)。
        用于当人物卡在墙里时快速脱困。
        """
        start_x, start_y = int(start_pos[0]), int(start_pos[1])

        # 如果起点本身就是安全的，直接返回
        if self._is_walkable(start_x, start_y):
            return (start_x, start_y)

        # BFS 初始化
        queue = deque([(start_x, start_y)])
        visited = set()
        visited.add((start_x, start_y))

        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]

        # 防止无限搜索
        iterations = 0
        limit = max_search_dist * max_search_dist * 4  # 估算的一个上限

        while queue:
            cx, cy = queue.popleft()
            iterations += 1
            if iterations > limit:
                break

            # 找到最近的白点！
            if self._is_walkable(cx, cy):
                return (cx, cy)

            # 继续向外扩散
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return None  # 如果在范围内没找到安全点

    def get_nearest_safe_point(self, start_pos, max_search_dist=100):
        return self._find_nearest_safe_point(start_pos, max_search_dist=max_search_dist)

    def get_random_visible_points(self, current_pos, num_points=10, min_dist=30, max_dist=100):
        """
        功能升级版：
        1. 如果当前位置在障碍物内 -> 先找到最近的安全点作为路径起点，再以该安全点为中心生成随机点。
        2. 如果当前位置安全 -> 正常生成。
        所有生成的随机点与“基准中心”的连线都保证不穿墙。

        :return: list of tuples [(x, y), ...]
                 如果发生脱困，列表第一个点是安全出口点。
        """
        curr_x, curr_y = current_pos
        path_points = []

        # --- 步骤 1: 确定生成基准点 (Generation Center) ---
        generation_center = current_pos

        # 检测当前是否卡在墙里
        if not self._is_walkable(curr_x, curr_y):
            # print("警告：当前位于不可通行区域，正在计算最近脱困点...")
            safe_pos = self._find_nearest_safe_point(current_pos, max_search_dist=max_dist)

            if safe_pos:
                # 找到了安全点：
                # 1. 将安全点加入路径的第一个位置，让角色先走出来
                path_points.append(safe_pos)
                # 2. 更新生成中心，后续的随机点都基于这个安全点来找
                generation_center = safe_pos
            else:
                print("错误：周围全是障碍物，无法找到安全点！")
                return []

        # --- 步骤 2: 基于基准点生成随机巡逻点 ---
        center_x, center_y = generation_center
        random_points = []

        attempts = 0
        max_attempts = num_points * 20

        while len(random_points) < num_points and attempts < max_attempts:
            attempts += 1

            # A. 随机生成角度和距离
            rand_angle = random.uniform(0, 2 * math.pi)
            rand_dist = random.uniform(min_dist, max_dist)

            # B. 计算目标坐标 (基于 generation_center)
            target_x = int(center_x + rand_dist * math.cos(rand_angle))
            target_y = int(center_y + rand_dist * math.sin(rand_angle))

            # C. 越界检查
            if target_x < 0 or target_x >= self.width or target_y < 0 or target_y >= self.height:
                continue

            # D. 核心检查
            # 1. 目标点必须是白色的
            if self._is_walkable(target_x, target_y):
                # 2. 【关键】从“基准中心”到“目标点”的连线必须是通的
                # 注意：这里传入的是 generation_center，而不是 current_pos
                # 如果之前卡墙了，这里保证的是从脱困点看过去是通的
                if self._check_line_of_sight(generation_center, (target_x, target_y)):
                    random_points.append((target_x, target_y))

        if len(random_points) < num_points:
            pass  # 可以选择打印日志

        # --- 步骤 3: 合并结果 ---
        # 结果可能是 [安全点, 随机点1, 随机点2...]
        # 也可能是 [随机点1, 随机点2...] (如果本来就没卡墙)
        return path_points + random_points

    def get_avoidance_action(self, current_pos, current_angle, check_distance=15):
        """
        当检测到前方有障碍时，计算最佳的避障动作。
        新增逻辑：如果当前位置已经在障碍物内，计算最快脱离方向（前/后）。

        :return: 1-6 (常规避障), 7 (向前脱困), 8 (向后脱困)
        """
        curr_x, curr_y = current_pos

        # =========================================================
        # 【新增功能】：检测当前是否已经陷在障碍物里 (脱困模式)
        # =========================================================
        if not self._is_walkable(curr_x, curr_y):
            # 已经在黑区了，计算向前还是向后能更快出去

            # 计算单位向量
            rad = math.radians(current_angle - 90)
            dx = math.cos(rad)
            dy = math.sin(rad)

            # 最大搜索距离 (防止地图全黑死循环)
            max_search_dist = 50

            dist_forward = float('inf')
            dist_backward = float('inf')

            # 1. 向前探测
            for i in range(1, max_search_dist):
                nx = curr_x + dx * i
                ny = curr_y + dy * i
                # 只要找到一个白点，就记录距离并停止
                if self._is_walkable(nx, ny):
                    dist_forward = i
                    break

            # 2. 向后探测 (注意这里的减号)
            for i in range(1, max_search_dist):
                nx = curr_x - dx * i
                ny = curr_y - dy * i
                if self._is_walkable(nx, ny):
                    dist_backward = i
                    break

            # 3. 比较并决策
            # 如果向前更近或者一样近，返回7；向后更近，返回8
            if dist_forward <= dist_backward:
                return 7  # Forward to escape
            else:
                return 8  # Backward to escape

        # =========================================================
        # 下面是原有的常规避障逻辑 (当前位置是安全的，但前方有障碍)
        # =========================================================

        # 1. 探测前方极近距离 (用于判断是否需要倒车)
        critical_distance = 5
        is_critical = False

        if self.check_safety_ahead(current_pos, current_angle, critical_distance) == "Pause":
            is_critical = True

        # 2. 探测左右空旷程度
        def get_ray_distance(offset_angle):
            target_ang_rad = math.radians(current_angle + offset_angle - 90)
            dx = math.cos(target_ang_rad)
            dy = math.sin(target_ang_rad)
            curr_dist = 1
            cx, cy = current_pos
            while curr_dist < check_distance:
                nx = cx + dx * curr_dist
                ny = cy + dy * curr_dist
                if not self._is_walkable(nx, ny):
                    break
                curr_dist += 2
            return curr_dist

        left_score = get_ray_distance(-30) + get_ray_distance(-60)
        right_score = get_ray_distance(30) + get_ray_distance(60)

        # 3. 决策逻辑
        # === 场景 A: 右边更空旷 ===
        if right_score > left_score:
            if is_critical:
                # 距离太近，需要倒车变向。
                # 这里返回的编号语义是“车辆实际后退偏向哪一侧”，不是“方向键按哪边”。
                # 因此右侧更空旷时返回 3，交给上层映射为 backward_turn_right；
                # 底层实现可以是 down + left，但它的实际车辆效果应是“后退并向右脱离”。
                return 3
            else:
                front_right_dist = get_ray_distance(15)
                if front_right_dist < check_distance * 0.5:
                    return 2
                else:
                    return 1

                    # === 场景 B: 左边更空旷 ===
        else:
            if is_critical:
                # 同理，左侧更空旷时返回 6，语义上表示“后退并向左脱离”。
                return 6
            else:
                front_left_dist = get_ray_distance(-15)
                if front_left_dist < check_distance * 0.5:
                    return 5
                else:
                    return 4

def draw_points_with_arrows(
    road_list,
    image_path=DEFAULT_ROUTE_IMAGE_PATH,
    output_path=DEFAULT_ROUTE_OUTPUT_PATH,
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：路线绘图底图不存在，跳过绘图 -> {image_path}")
        return

    point_color = (0, 0, 255)  # 红色点 (BGR格式)
    point_radius = 8  # 点半径
    arrow_color = (0, 255, 0)  # 绿色箭头
    arrow_thickness = 2  # 箭头线宽
    text_color = (255, 0, 0)  # 蓝色文字
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(road_list)):
        x, y = map(int, road_list[i])  # 确保坐标为整数
        cv2.circle(image, (x, y), point_radius, point_color, -1)
        cv2.putText(image, str(i + 1), (x + 10, y - 10),
                    font, 0.7, text_color, 2)
        if i < len(road_list) - 1:
            next_x, next_y = map(int, road_list[i + 1])
            cv2.arrowedLine(image, (x, y), (next_x, next_y),
                            arrow_color, arrow_thickness,
                            tipLength=0.2)  # 箭头头部长度比例
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not write_image_unicode(output_path, image):
        print(f"警告：路线绘图写入失败 -> {output_path}")
        return
    print(f"结果已保存到: {output_path}")

if __name__ == '__main__':

    map_tool = MapNavigator(r"D:\project\Python\auto_pubg\resource\map\hpjy_mask.tif")
    draw_points_with_arrows(map_tool.plan_path((1307,222),(915,1616)), image_path=r"D:\project\Python\auto_pubg\resource\map\hpjy.png", output_path=r"D:\project\Python\auto_pubg\resource\map\hpjy_road.png")
