class ObstacleAvoidanceAnalyzer:
    # YOLO26 类别定义，用于驱动阶段的类别感知避障
    CLASS_NAMES = {
        0: "door",
        1: "object",
        2: "window",
        3: "pick_menu",
        4: "open_door",
        5: "stair",
        6: "down_stair",
        7: "car",
        8: "house",
        9: "stone_wall",
        10: "stump",
        11: "rock",
        12: "grass_tuft",
        13: "fence",
        14: "water",
        15: "ditch",
        16: "unique_construction",
        17: "wrecked_car",
        18: "box",
        19: "sandband_wall",
    }

    # 仅将真正会挡住车或会让车陷进去的类别纳入视觉避障
    CLASS_CONFIG = {
        7: {
            "name": "car",
            "min_area_ratio": 0.003,
            "min_bottom_ratio": 0.25,
            "min_width_ratio": 0.04,
            "x_padding_ratio": 0.06,
            "priority": "hard",
        },
        8: {
            "name": "house",
            "min_area_ratio": 0.05,
            "min_bottom_ratio": 0.55,
            "min_width_ratio": 0.18,
            "x_padding_ratio": 0.02,
            "priority": "hard",
        },
        9: {
            "name": "stone_wall",
            "min_area_ratio": 0.005,
            "min_bottom_ratio": 0.35,
            "min_width_ratio": 0.05,
            "x_padding_ratio": 0.05,
            "priority": "hard",
        },
        10: {
            "name": "stump",
            "min_area_ratio": 0.0015,
            "min_bottom_ratio": 0.38,
            "min_width_ratio": 0.02,
            "x_padding_ratio": 0.03,
            "priority": "medium",
        },
        11: {
            "name": "rock",
            "min_area_ratio": 0.002,
            "min_bottom_ratio": 0.35,
            "min_width_ratio": 0.03,
            "x_padding_ratio": 0.04,
            "priority": "hard",
        },
        13: {
            "name": "fence",
            "min_area_ratio": 0.004,
            "min_bottom_ratio": 0.4,
            "min_width_ratio": 0.06,
            "x_padding_ratio": 0.05,
            "priority": "medium",
        },
        14: {
            "name": "water",
            "min_area_ratio": 0.04,
            "min_bottom_ratio": 0.58,
            "min_width_ratio": 0.2,
            "x_padding_ratio": 0.08,
            "priority": "hard",
        },
        15: {
            "name": "ditch",
            "min_area_ratio": 0.01,
            "min_bottom_ratio": 0.45,
            "min_width_ratio": 0.08,
            "x_padding_ratio": 0.06,
            "priority": "hard",
        },
        16: {
            "name": "unique_construction",
            "min_area_ratio": 0.012,
            "min_bottom_ratio": 0.38,
            "min_width_ratio": 0.08,
            "x_padding_ratio": 0.05,
            "priority": "hard",
        },
        17: {
            "name": "wrecked_car",
            "min_area_ratio": 0.003,
            "min_bottom_ratio": 0.25,
            "min_width_ratio": 0.04,
            "x_padding_ratio": 0.06,
            "priority": "hard",
        },
        18: {
            "name": "box",
            "min_area_ratio": 0.0018,
            "min_bottom_ratio": 0.38,
            "min_width_ratio": 0.025,
            "x_padding_ratio": 0.03,
            "priority": "medium",
        },
        19: {
            "name": "sandband_wall",
            "min_area_ratio": 0.005,
            "min_bottom_ratio": 0.35,
            "min_width_ratio": 0.05,
            "x_padding_ratio": 0.05,
            "priority": "hard",
        },
    }

    def __init__(
        self,
        width,
        height,
        w_r=0.15,  # 最小可通行比率 (Gap width / Image width)
        conf_thresh=0.25,  # 检测置信度阈值
        center_tolerance=0.1,  # 中心死区 (在这个范围内认为是 Straight)
        slight_thresh=0.25,  # 轻微转向阈值 (偏离比例)
        small_thresh=0.5,  # 小转向阈值
    ):
        self.W = max(1, int(width))
        self.H = max(1, int(height))
        self.w_r = w_r
        self.min_gap_width = self.W * w_r
        self.conf_thresh = conf_thresh

        self.center_tolerance = center_tolerance
        self.slight_thresh = slight_thresh
        self.small_thresh = small_thresh

    def analyze(self, results):
        """
        Input: results = [[x1, y1, x2, y2, conf, cls], ...]
        Output: {'decision': str, 'debug_info': dict}
        """
        obstacles = []
        kept_items = []
        hard_intervals = []

        for det in results:
            parsed = self._parse_detection(det)
            if parsed is None:
                continue

            interval, meta = parsed
            obstacles.append(interval)
            kept_items.append(meta)
            if meta["priority"] == "hard":
                hard_intervals.append(interval)

        if not obstacles:
            return {
                "decision": "straight",
                "target_x": self.W / 2.0,
                "obstacles_count": 0,
                "hard_obstacles_count": 0,
                "coverage_ratio": 0.0,
                "classes": [],
                "center_blocked": False,
            }

        merged_obstacles = self._merge_intervals(obstacles)
        merged_hard_obstacles = self._merge_intervals(hard_intervals)
        gaps = self._find_gaps(merged_obstacles, self.W)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= self.min_gap_width]

        cx = self.W / 2.0
        center_blocked = any(interval[0] <= cx <= interval[1] for interval in merged_hard_obstacles)
        coverage_ratio = sum((obs[1] - obs[0]) for obs in merged_obstacles) / float(self.W)
        class_names = sorted({meta["name"] for meta in kept_items})

        if not valid_gaps:
            largest_gap = max(gaps, key=lambda x: x[1] - x[0]) if gaps else (0, 0)
            gap_center = (largest_gap[0] + largest_gap[1]) / 2.0
            decision = "reverse_and_left" if gap_center < cx else "reverse_and_right"
            return {
                "decision": decision,
                "target_x": gap_center,
                "gap_center": gap_center,
                "obstacles_count": len(merged_obstacles),
                "hard_obstacles_count": len(merged_hard_obstacles),
                "coverage_ratio": coverage_ratio,
                "classes": class_names,
                "center_blocked": center_blocked,
            }

        center_gap = None
        for gap in valid_gaps:
            if gap[0] <= cx <= gap[1]:
                center_gap = gap
                break

        target_x = cx
        decision = "straight"

        if center_gap is None:
            best_gap = min(valid_gaps, key=lambda g: abs(((g[0] + g[1]) / 2.0) - cx))
            target_x = (best_gap[0] + best_gap[1]) / 2.0
            decision = self._decision_from_target(target_x, cx, center_blocked, coverage_ratio)

        return {
            "decision": decision,
            "target_x": target_x,
            "obstacles_count": len(merged_obstacles),
            "hard_obstacles_count": len(merged_hard_obstacles),
            "coverage_ratio": coverage_ratio,
            "classes": class_names,
            "center_blocked": center_blocked,
        }

    def _parse_detection(self, det):
        if len(det) < 6:
            return None

        x1, y1, x2, y2, conf, cls_id = det[:6]
        if conf < self.conf_thresh:
            return None

        cls_id = int(cls_id)
        cfg = self.CLASS_CONFIG.get(cls_id)
        if cfg is None:
            return None

        x1 = max(0.0, min(float(self.W), float(x1)))
        x2 = max(0.0, min(float(self.W), float(x2)))
        y1 = max(0.0, min(float(self.H), float(y1)))
        y2 = max(0.0, min(float(self.H), float(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio = (box_w * box_h) / float(self.W * self.H)
        bottom_ratio = y2 / float(self.H)
        width_ratio = box_w / float(self.W)

        if area_ratio < cfg["min_area_ratio"]:
            return None
        if bottom_ratio < cfg["min_bottom_ratio"]:
            return None
        if width_ratio < cfg["min_width_ratio"]:
            return None

        pad = float(self.W) * cfg.get("x_padding_ratio", 0.0)
        interval = [
            max(0.0, x1 - pad),
            min(float(self.W), x2 + pad),
        ]
        meta = {
            "name": cfg["name"],
            "priority": cfg["priority"],
        }
        return interval, meta

    def _decision_from_target(self, target_x, cx, center_blocked, coverage_ratio):
        offset = target_x - cx
        ratio = offset / (self.W / 2.0)
        abs_ratio = abs(ratio)
        direction = "right" if ratio > 0 else "left"

        if abs_ratio < self.center_tolerance:
            return "straight"

        if abs_ratio < self.slight_thresh:
            level = "slight"
        elif abs_ratio < self.small_thresh:
            level = "small"
        else:
            level = "large"

        if center_blocked or coverage_ratio >= 0.45:
            if level == "slight":
                level = "small"
            elif level == "small":
                level = "large"

        return f"{level}_{direction}"

    def _merge_intervals(self, intervals):
        """
        合并重叠区间的经典算法
        输入: [[10, 50], [40, 80], [100, 120]]
        输出: [[10, 80], [100, 120]]
        """
        if not intervals:
            return []

        # 按左边界排序
        intervals.sort(key=lambda x: x[0])

        merged = []
        for curr in intervals:
            if not merged:
                merged.append(curr)
                continue

            prev = merged[-1]

            # 如果当前区间的开始 < 上一个区间的结束，说明重叠
            if curr[0] < prev[1]:
                # 合并：结束点取两者的最大值
                prev[1] = max(prev[1], curr[1])
            else:
                merged.append(curr)

        return merged

    def _find_gaps(self, merged_obstacles, width):
        """
        根据障碍物区间，反向计算空白区间
        """
        gaps = []
        current_x = 0

        for obs in merged_obstacles:
            obs_start, obs_end = obs
            # 如果当前位置和障碍物开始之间有空隙
            if obs_start > current_x:
                gaps.append((current_x, obs_start))

            # 更新当前位置为障碍物的结束点
            current_x = max(current_x, obs_end)

        # 检查最后一个障碍物到图像右边缘是否有空隙
        if current_x < width:
            gaps.append((current_x, width))

        return gaps
