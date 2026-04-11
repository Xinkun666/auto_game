import numpy as np

class ObstacleAvoidanceAnalyzer:
    def __init__(self,
                 width,
                 height,
                 w_r=0.15,  # 最小可通行比率 (Gap width / Image width)
                 conf_thresh=0.25,  # 检测置信度阈值
                 center_tolerance=0.1,  # 中心死区 (在这个范围内认为是 Straight)
                 slight_thresh=0.25,  # 轻微转向阈值 (偏离比例)
                 small_thresh=0.5,  # 小转向阈值
                 # large_thresh > 0.5    # 大转向
                 ):
        """
        基于1D投影的避障分析器
        """
        self.W = width
        self.H = height
        self.w_r = w_r
        self.min_gap_width = width * w_r  # 转换为像素宽度
        self.conf_thresh = conf_thresh

        # 转向阈值设定
        self.center_tolerance = center_tolerance
        self.slight_thresh = slight_thresh
        self.small_thresh = small_thresh

    def analyze(self, results):
        """
        Input: results = [[x1, y1, x2, y2, conf, cls], ...]
        Output: {'decision': str, 'debug_info': dict}
        """
        # 1. 提取所有障碍物的 X 轴投影区间 [x1, x2]
        obstacles = []
        for det in results:
            if len(det) < 6: continue
            x1, y1, x2, y2, conf, cls = det

            # 过滤低置信度
            if conf < self.conf_thresh:
                continue

            # 边界截断，防止超出图像范围
            x1 = max(0, min(self.W, x1))
            x2 = max(0, min(self.W, x2))

            # 简单的逻辑：只关心在一定高度以下的障碍物 (可选优化)
            # 如果物体完全在天空（例如 y2 < H/3），可能不需要投影
            # 这里严格按照你的要求：全部投影
            obstacles.append([x1, x2])

        # 2. 合并重叠区间 (核心算法)
        merged_obstacles = self._merge_intervals(obstacles)

        # 3. 计算可通行间隙 (Gaps)
        gaps = self._find_gaps(merged_obstacles, self.W)

        # 4. 筛选有效间隙 (宽度 >= w_r * W)
        valid_gaps = [g for g in gaps if (g[1] - g[0]) >= self.min_gap_width]

        # 5. 决策逻辑
        cx = self.W / 2.0
        decision = "straight"  # 默认

        # 情况 A: 没有任何有效间隙 -> 倒车
        if not valid_gaps:
            # 寻找所有间隙（包括不够宽的）里最大的那个，决定倒车方向
            largest_gap = max(gaps, key=lambda x: x[1] - x[0]) if gaps else (0, 0)
            gap_center = (largest_gap[0] + largest_gap[1]) / 2
            if gap_center < cx:
                decision = "reverse_and_left"  # 空间在左边，往左倒/退
            else:
                decision = "reverse_and_right"

            return {
                "decision": decision,
                "gap_center": gap_center
            }

        # 情况 B: 分析中心点
        # 找到包含中心点 cx 的间隙
        center_gap = None
        for g in valid_gaps:
            if g[0] <= cx <= g[1]:
                center_gap = g
                break

        target_x = cx  # 默认目标是中心

        # 逻辑：如果中心点在某个有效间隙内，且不仅包含cx，该间隙本身是有效的
        # (代码逻辑上 valid_gaps 里的都已经足够宽了)
        if center_gap:
            # 中心可通行，直接直行
            decision = "straight"
            target_x = cx
        else:
            # 情况 C: 中心有障碍物，寻找最近的有效间隙
            # 计算每个有效间隙中心到图像中心的距离
            # 策略：选择“距离中心最近”的那个间隙的中心点作为目标
            best_gap = min(valid_gaps, key=lambda g: abs(((g[0] + g[1]) / 2) - cx))

            # 目标点设为该间隙的中心
            target_x = (best_gap[0] + best_gap[1]) / 2.0

            # 计算偏离程度
            offset = target_x - cx
            # 归一化偏离值 (-1.0 ~ 1.0), 负左正右
            ratio = offset / (self.W / 2.0)

            abs_ratio = abs(ratio)
            direction = "right" if ratio > 0 else "left"

            if abs_ratio < self.center_tolerance:
                decision = "straight"
            elif abs_ratio < self.slight_thresh:
                decision = f"slight_{direction}"
            elif abs_ratio < self.small_thresh:
                decision = f"small_{direction}"
            else:
                decision = f"large_{direction}"

        return {
            "decision": decision,
            "target_x": target_x,
            "obstacles_count": len(merged_obstacles)
        }

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
