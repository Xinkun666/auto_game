"""Legacy configured vehicle-route helpers.

These helpers choose configured intermediate points that monotonically approach
the vehicle destination. They are not obstacle-aware path planners; callers
should fall back to ``MapNavigator.plan_path`` when needed.
"""

import ast
import json
import math


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
    with open(r"config\config.json", "r") as f:
        config = json.load(f)
    car_points = ast.literal_eval(config["car_points"])
    if not car_points:
        return []

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nearest_neighbor_route(start, points):
        unvisited = points.copy()
        route = [start]
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda point: dist(current, point))
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
                    current_dist = (
                        dist(route[i], route[i + 1])
                        + dist(route[j], route[j + 1])
                    )
                    swapped_dist = (
                        dist(route[i], route[j])
                        + dist(route[i + 1], route[j + 1])
                    )
                    if swapped_dist < current_dist:
                        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                        improved = True
        return route

    start = start_point if start_point is not None else car_points[0]
    return two_opt(nearest_neighbor_route(start, car_points))[1:]


def find_path(start, city="r_city", tol=1):
    """Return configured points that progressively approach a car destination."""
    car_cfg = CAR_POINT_CONFIG.get(city)
    if car_cfg is None:
        raise ValueError(f"未知 car_points 配置: {city}")

    target = car_cfg["destination"]
    points = set(car_cfg["road_points"])

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    current = start
    path = [start]
    while True:
        if distance(current, target) <= tol:
            path.append(target)
            return path[1:]

        next_point = next(
            (
                point
                for point in sorted(points, key=lambda item: distance(current, item))
                if distance(point, target) < distance(current, target)
            ),
            None,
        )
        if next_point is None:
            path.append(target)
            return path[1:]

        path.append(next_point)
        points.remove(next_point)
        current = next_point
