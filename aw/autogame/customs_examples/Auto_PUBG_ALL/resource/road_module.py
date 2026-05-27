import os
import cv2
import json
import math
import heapq
import numpy as np

def save_color_coords(mask_path, save_json, color='red'):
    # 读取图像（彩色）
    img = cv2.imread(mask_path, cv2.IMREAD_COLOR)

    # OpenCV 默认是 BGR，因此红色是 (0,0,255)
    if color == 'red':
        bgr = (0, 0, 255)
    elif color == 'blue':
        bgr = (255, 0, 0)
    else:
        bgr = (0,0,0)

    coords = {}
    count = 1
    h, w, _ = img.shape

    for y in range(h):
        for x in range(w):
            if tuple(img[y, x]) == bgr:
                coords[f"c{count}"] = [int(x), int(y)]  # (x, y)
                count += 1

    # 保存为 JSON
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(coords, f, indent=4, ensure_ascii=False)

    print(f"已提取 {count-1} 个{color}像素点，结果保存到 {save_json}")

def build_road_matrix(mask_path, json_path, save_path):
    img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    WHITE = (255, 255, 255)
    RED = (0, 0, 255)  # BGR
    BLUE = (255, 0, 0)

    # 加载交点
    with open(json_path, "r", encoding="utf-8") as f:
        intersections = json.load(f)

    nodes = {key: tuple(value) for key, value in intersections.items()}
    node_list = list(nodes.keys())
    node_index = {node: idx for idx, node in enumerate(node_list)}
    n = len(node_list)

    M = [[[] for _ in range(n)] for _ in range(n)]

    def is_white(px):
        return (px == WHITE).all()
    def is_red(px):
        return (px == RED).all()
    def is_blue(px):
        return (px == BLUE).all()
    def is_node(x, y):
        return any((x, y) == tuple(coord) for coord in nodes.values())
    def neighbors(x, y):
        """返回 (x,y) 的上下左右邻居坐标"""
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if is_white(img[ny, nx]) or is_red(img[ny, nx]) or is_blue(img[ny, nx]):
                    yield (nx, ny)

    for i, node in enumerate(node_list):
        sx, sy = nodes[node]

        for nx, ny in neighbors(sx, sy):
            if (nx, ny) == (sx, sy):
                continue
            path = []
            prev = (sx, sy)
            curr = (nx, ny)

            while True:
                if is_node(*curr) and curr != (sx, sy):
                    # 遇到下一个交点
                    j = [k for k,v in nodes.items() if tuple(v) == curr][0]
                    j_idx = node_index[j]
                    if len(path) > 0:
                        if len(M[i][j_idx]) == 0:
                            M[i][j_idx].append([len(path), path.copy()])
                            rev_path = [p for p in reversed(path)]
                            M[j_idx][i].append([len(path),rev_path])
                        elif len(M[i][j_idx]) > 0:
                            path_len_list = [M[i][j_idx][path_i][0] for path_i in range(len(M[i][j_idx]))]
                            if len(path) not in path_len_list:
                                M[i][j_idx].append([len(path),path.copy()])
                                rev_path = [p for p in reversed(path)]
                                M[j_idx][i].append([len(path), rev_path])
                    break

                # 否则继续走
                path.append(curr)
                next_candidates = [p for p in neighbors(*curr) if p != prev]

                if not next_candidates:
                    print(f'出现死路,{curr} 点没有邻居！')
                    break  # 理论上不会出现死路

                prev, curr = curr, next_candidates[0]

    # 对角线设置为 "inf"
    for i in range(n):
        M[i][i] = "inf"

    # 转为 JSON
    M_dict = {}
    for i in range(n):
        for j in range(n):
            key = f"{node_list[i]}->{node_list[j]}"
            M_dict[key] = M[i][j]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(M_dict, f, indent=4, ensure_ascii=False)


class RoadTopo():
    def __init__(self, json_path=r"aw/autogame/customs_examples/Auto_PUBG_ALL/resource/road/road_matrix.json",mask_path=r"aw/autogame/customs_examples/Auto_PUBG_ALL/resource/road/road_mask.png",node_path=r"aw/autogame/customs_examples/Auto_PUBG_ALL/resource/road/red_coords.json", route_node_path = r"aw/autogame/customs_examples/Auto_PUBG_ALL/resource/road/blue_coords.json"):
        self.node_data = None
        self.route_node_data = None
        self.json_path = json_path
        self.mask_path = mask_path
        self.node_path = node_path
        self.route_node_path = route_node_path
        self._road_pixel_coords = None
        self.intersections = self.load_intersections()
        self.get_matrix()

    @staticmethod
    def _is_point(value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        return isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))

    @classmethod
    def _is_path(cls, value):
        return isinstance(value, list) and len(value) > 0 and all(cls._is_point(point) for point in value)

    @classmethod
    def _normalize_path(cls, path):
        return [[int(point[0]), int(point[1])] for point in path if cls._is_point(point)]

    @classmethod
    def _normalize_route_entry(cls, route):
        if (
            isinstance(route, (list, tuple))
            and len(route) >= 2
            and isinstance(route[0], (int, float))
            and cls._is_path(route[1])
        ):
            path = cls._normalize_path(route[1])
            return [int(route[0]), path]

        if isinstance(route, list) and len(route) == 1 and cls._is_path(route[0]):
            path = cls._normalize_path(route[0])
            return [len(path), path]

        if cls._is_path(route):
            path = cls._normalize_path(route)
            return [len(path), path]

        return None

    @classmethod
    def _normalize_neighbors(cls, neighbors):
        if neighbors == "inf":
            return math.inf
        if not neighbors:
            return []

        direct_route = cls._normalize_route_entry(neighbors)
        if direct_route is not None:
            return [direct_route]

        routes = []
        for route in neighbors:
            normalized = cls._normalize_route_entry(route)
            if normalized is not None:
                routes.append(normalized)
        return routes

    def get_matrix(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            road_data = json.load(f)
        with open(self.node_path, "r", encoding="utf-8") as f:
            self.node_dict = json.load(f)
            self.node_data = list(self.node_dict.values())
        with open(self.route_node_path, "r", encoding="utf-8") as f:
            route_node_data = json.load(f)
            self.route_node_data = list(route_node_data.values())
        self.node_number = len(self.node_data)
        M = [[[] for _ in range(self.node_number)] for _ in range(self.node_number)]
        for cij, neighbors in road_data.items():
            i = int(cij.split('->')[0][1:]) - 1
            j = int(cij.split('->')[1][1:]) - 1
            M[i][j] = self._normalize_neighbors(neighbors)
        self.M = M

    def load_intersections(self):
        with open(self.node_path, "r", encoding="utf-8") as f:
            intersections = json.load(f)
        # 转换成 { "c1": (x,y), ... }
        intersections = {k: tuple(v) for k, v in intersections.items()}
        return intersections

    def dijkstra(self,start, end):
        if self.M is None:
            self.get_matrix()
        n = len(self.M)
        dist = [math.inf] * n
        prev = [-1] * n
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end:
                break
            for v in range(n):
                if self.M[u][v] == math.inf or self.M[u][v] == []:
                    continue
                # 如果有多条路线，取最短的那条
                best_len = min(route[0] for route in self.M[u][v])
                nd = d + best_len
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        path_nodes = []
        if dist[end] < math.inf:
            cur = end
            while cur != -1:
                path_nodes.append(cur)
                cur = prev[cur]
            path_nodes.reverse()
        return dist[end] + len(path_nodes), path_nodes

    def shortest_path_from_point(self,px, py, dest):
        for i, (x, y) in enumerate(self.intersections.values()):
            if (px, py) == (x, y):
                # 已在交点
                dist, path_nodes = self.dijkstra( i, dest)
                node_path = [f"c{n + 1}" for n in path_nodes]
                return [node_path, dist, self.build_pixel_path(path_nodes)]

        ci, cj, path = None, None, None
        n = len(self.M)
        for i in range(n):
            for j in range(n):
                if isinstance(self.M[i][j], list) and self.M[i][j]:
                    for route_len, route_path in self.M[i][j]:
                        if [px, py] in route_path:
                            ci, cj, path = i, j, route_path
                            break
                if ci is not None:
                    break
            if ci is not None:
                break

        if ci is None:
            print("给定点不在任何道路上！")
            return None
        idx = path.index([px, py])  # 假设 path 是 [[x,y], ...]
        di = idx
        dj = len(path) - idx - 1

        d_dest_i, path_i = self.dijkstra(ci, dest)
        d_dest_j, path_j = self.dijkstra(cj, dest)

        total_i = d_dest_i + di
        total_j = d_dest_j + dj

        if total_i <= total_j:
            node_path = [f"c{n + 1}" for n in path_i]
            total_dist = total_i
            pixel_path = path[idx::-1]
            pixel_path += self.build_pixel_path(path_i)
        else:
            node_path = [f"c{n + 1}" for n in path_j]
            total_dist = total_j
            pixel_path = path[idx:]  # 起点到cj
            pixel_path += self.build_pixel_path(path_j)
        return node_path, total_dist, pixel_path

    def build_pixel_path(self,path_nodes):
        pixels = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if not self.M[u][v] or self.M[u][v] == math.inf:
                continue
            if len(self.M[u][v]) == 1:
                _, path = self.M[u][v][0]
            else:
                min_path_len = 100000000
                min_path_idx = 0
                for i in range(len(self.M[u][v])):
                    if self.M[u][v][i][0] < min_path_len:
                        min_path_idx = i
                        min_path_len = self.M[u][v][i][0]
                _,path = self.M[u][v][min_path_idx]
            pixels.append(list(self.intersections['c'+str(u+1)]))
            pixels.extend(path)
            if v == path_nodes[-1]:
                pixels.append(list(self.intersections['c'+str(v+1)]))
        return pixels

    def find_nearest_point_from_mask(self, px, py):
        if self._road_pixel_coords is None:
            img = cv2.imread(self.mask_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"错误：无法读取道路 mask {self.mask_path}")
                return None, math.inf

            white = np.all(img == (255, 255, 255), axis=2)
            red = np.all(img == (0, 0, 255), axis=2)
            blue = np.all(img == (255, 0, 0), axis=2)
            ys, xs = np.where(white | red | blue)
            if len(xs) == 0:
                print("道路 mask 中没有可用道路像素")
                return None, math.inf
            self._road_pixel_coords = np.column_stack((xs, ys)).astype(np.int32)

        try:
            point = np.array([int(round(float(px))), int(round(float(py)))], dtype=np.int32)
        except (TypeError, ValueError):
            return None, math.inf

        deltas = self._road_pixel_coords - point
        dist_sq = np.sum(deltas * deltas, axis=1)
        idx = int(np.argmin(dist_sq))
        nearest = self._road_pixel_coords[idx]
        return [int(nearest[0]), int(nearest[1])], math.sqrt(float(dist_sq[idx]))

    def build_path_from_sequence(self,node_sequence):
        full_path = []
        for i in range(len(node_sequence) - 1):
            curr_node = node_sequence[i]
            next_node = node_sequence[i + 1]
            curr_idx = int(curr_node[1:]) - 1
            next_idx = int(next_node[1:]) - 1
            path_info = self.M[curr_idx][next_idx]
            if not path_info:
                print(f'{curr_node} -> {next_node} 并不邻接!')
                return None
            if len(path_info) > 1:
                path_info = [min(path_info,key=lambda item : item[0])]
            if i == 0:
                full_path.append(self.node_dict[node_sequence[i]])
            full_path.extend(path_info[0][1])
            full_path.append(self.node_dict[node_sequence[i+1]])

        return full_path

    def get_piecewise_road_with_point(self,road_list):
        if len(road_list) == 0:
            print('road_list is empty')
            return None
        piecewise_road = [road_list[0]]
        for i in range(1, len(road_list)):
            if road_list[i] in self.node_data or road_list[i] in self.route_node_data:
                piecewise_road.append(road_list[i])
        if piecewise_road[-1] != road_list[-1]:
            piecewise_road.append(road_list[-1])
        return piecewise_road

    def draw_points_with_arrows(self, road_list, image_path = r"resource\map\hpjy.png",output_path = r'temp\road\route.jpg'):
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图像 {image_path}")
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
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        cv2.imwrite(output_path, image)
        print(f"结果已保存到: {output_path}")

    def get_road_list(self,start_point):
        ring_nodes = ['c23', 'c21', 'c28', 'c27', 'c34', 'c33', 'c31', 'c37',
                      'c43', 'c42', 'c44', 'c41', 'c38', 'c40', 'c35', 'c32']
        node_list = np.array(self.node_data + self.route_node_data)
        start_point = np.array(start_point)
        dis = np.linalg.norm(node_list - start_point, axis=1)
        nearest_idx = np.argmin(dis)
        nearest_coord = node_list[nearest_idx]  # [x1, y1]
        ring_coords = np.array([self.node_dict[name] for name in ring_nodes])
        dis_ring = np.linalg.norm(ring_coords - nearest_coord, axis=1)
        nearest_ring_idx = np.argmin(dis_ring)
        nearest_node_name = ring_nodes[nearest_ring_idx]
        road_list1 = self.shortest_path_from_point(nearest_coord[0],nearest_coord[1],int(nearest_node_name[1:]) - 1)
        road_list1 = road_list1[2]
        reordered_ring = ring_nodes[nearest_ring_idx:] + ring_nodes[:nearest_ring_idx + 1]
        road_list2 = self.build_path_from_sequence(reordered_ring)
        road_list = road_list1 + road_list2
        cleaned = [road_list[0]]
        for i in range(1, len(road_list)):
            if road_list[i] != road_list[i - 1]:
                cleaned.append(road_list[i])
        road_list = self.get_piecewise_road_with_point(cleaned)
        return road_list

if __name__ == '__main__':
    road_topo = RoadTopo()
    # M = road_topo.get_matrix()
    # road_topo.draw_node()
    # pass
    # (px, py), _ = road_topo.find_nearest_point_from_mask(1139, 1424)
    # _, _, road_list = road_topo.shortest_path_from_point(px, py, 14)
    # piecewise_road = road_topo.get_piecewise_road_with_point(road_list)
    # road_topo.draw_points_with_arrows(piecewise_road)
    # 使用示例
    # save_color_coords(r"D:\Resource\Image\MiniMap\road_mask_hpjy_with_blue_point.png", r"D:\Resource\Image\MiniMap\blue_coords.json",color='blue')
    # build_road_matrix(mask_path=r"D:\Resource\Image\MiniMap\road_mask_with_blue_point.png", json_path=r"D:\Resource\Image\MiniMap\red_coords.json", save_path=r"D:\Resource\Image\MiniMap\road_matrix.json")

    # circle_sequence = ['c23','c21','c28','c27','c34','c33','c31','c37','c43','c42','c44','c41','c38','c40','c35','c32']
    # path = road_topo.build_path_from_sequence(circle_sequence)
    # path = road_topo.get_piecewise_road_with_point(path)
    # road_topo.draw_points_with_arrows(path)

    road_topo.draw_points_with_arrows(road_topo.get_road_list([1094,1344]))


