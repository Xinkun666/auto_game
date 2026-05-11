# import time
# import cv2
# import math
# import numpy as np
# from datetime import datetime
from typing import TYPE_CHECKING
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.map_navigator import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import *
from aw.autogame.tools.Utils import *

if TYPE_CHECKING:
    # 假设你的框架类定义在 framework.py 文件中
    from aw.autogame.tools.GameFrameWorker import FrameWorker


class Searching_House:
    def __init__(self):
        self.map_tool = MapNavigator()
        self.house_data = load_json(
            r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/house_entry/house_entries_summary.json')

        self.completed_houses = set()
        self.current_house_id = None
        self.active_entry = None

        # 状态机: IDLE -> FAST_NAV -> PRECISE_NAV -> SCANNING -> VISUAL_APPROACH -> INTERACT -> FINAL_ENTRY
        self.status = "IDLE"

        # 辅助变量
        self.first_view = False
        self.auto_forward = False
        self.screen_w, self.screen_h = get_resolution()

        # 用于智能选点的临时黑名单 (本轮循环跳过，不永久删除)
        self.temp_skip_houses = set()

        # --- 卡顿检测相关变量 ---
        self.history_locations = []
        self.max_history_len = 5  # 记录最近10次位置
        self.stuck_threshold = 0.5  # 判定卡住的距离阈值

        self.searching_number = 0

        # 用于搜索房屋使用到的辅助变量
        self.supplies = []  # [(绝对角度, 框高)]
        self.doors = []  # [(绝对角度, 框高)]
        self.player_yaw = 0.0  # 累计旋转角度（0° = 进入房间时的朝向）

    def process(self, w: 'FrameWorker'):
        self.start_searching(w)

        # location = check_location(w.get_info('location')[0])
        # direction = w.get_info('direction')
        #
        # if location is None:
        #     print('位置值是None，尝试向前移动一段距离刷新位置...')
        #     w.tap_single('摇杆', y_bias=-300, wait=500)
        #     return
        #
        # # 0. 基础设置
        # if not self.first_view:
        #     w.click('第一人称')
        #     self.first_view = True
        #
        #
        #
        # self.searching_logic(w, location, direction)

    def searching_logic(self, w: 'FrameWorker', current_loc, current_direction):

        if self.searching_number == 5:
            print('已经搜满5个房间，切换到跑图阶段')
            self.searching_number = 0
            w.change_stage('跑图阶段')
            return

        # --- 智能选点 ---
        if self.current_house_id is None:
            self.select_smart_target(current_loc, current_direction)
            if not self.current_house_id:
                print("[Searching] 当前区域无合适目标或已搜完")
                return
            self.status = "FAST_NAV"
            print(f"[Searching] 锁定目标: {self.current_house_id} | 状态: 快速导航")
            self.history_locations = []  # 切换目标时清空历史

        target_loc = self.active_entry['location']
        dist = get_distance(current_loc, target_loc)

        # --- 快速前进 (距离 > 5.0) ---
        if self.status == "FAST_NAV":
            # 卡顿检测逻辑
            if self.update_and_check_stuck(current_loc):
                print("[Nav] 检测到人物卡死，启动避障程序...")
                self.execute_unstuck_logic(w, current_loc)
                self.history_locations = []
                return

            if dist <= 5.0:
                print(f"[Nav] 进入精细导航范围 (距离 {dist:.2f})")
                self.stop_auto_forward(w)
                self.status = "PRECISE_NAV"
                return

            self.align_direction(w, target_loc)

            if not self.auto_forward:
                w.click('自动前进')
                self.auto_forward = True

            self.handle_jump_logic(w)

        # --- 精细逼近 ---
        elif self.status == "PRECISE_NAV":
            # --- [修改 1] 在精细导航阶段加入卡顿检测 ---
            # 原因：即使在慢速移动时，也可能卡在树根或小障碍物上
            if self.update_and_check_stuck(current_loc):
                print("[Nav] (Precise) 检测到人物卡死，启动避障程序...")
                self.execute_unstuck_logic(w, current_loc)
                self.history_locations = []  # 清空历史，防止重复触发
                return
            # ----------------------------------------

            if dist <= 1:
                print(f"[Nav] 已到达进门点 (距离 {dist:.2f})")
                self.status = "SCANNING"
                return

            self.stop_auto_forward(w)
            self.align_direction(w, target_loc)
            press_duration = get_time_from_distance(dist)
            w.tap_single('摇杆', y_bias=-300, dura=300, wait=press_duration - 300)
            w.refresh_frame()
            self.handle_jump_logic(w)

        # --- 进门点扫描 ---
        elif self.status == "SCANNING":
            print("[Scan] 到达点位，开始门检测...")
            ideal_angle = self.active_entry['direction']
            self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)

            if self.check_and_lock_door(w):
                self.status = "VISUAL_APPROACH"
                return

            scan_offsets = [30, -30]
            found_door = False
            for offset in scan_offsets:
                target_angle = (ideal_angle + offset) % 360
                print(f"[Scan] 尝试角度: {target_angle} (偏移 {offset})")
                self.align_direction_blocking(w, w.get_info('direction'), target_angle)
                w.refresh_frame()

                if self.check_and_lock_door(w):
                    found_door = True
                    self.status = "VISUAL_APPROACH"
                    break
                else:
                    print(f"[Data] 角度 {target_angle} 未发现门，保存样本")
                    self.save_dataset_image(w.frame, f"no_door_offset_{offset}")

            if not found_door:
                print("[Scan] All angles scanned, door not found. Discarding current point.")
                self.completed_houses.add(self.current_house_id)
                self.handle_failed_entry_logic(ideal_angle)
                self.status = "IDLE"

        # --- 视觉对齐与推进 ---
        elif self.status == "VISUAL_APPROACH":
            while True:
                door = self.find_largest_door(w)
                if not door:
                    print("[Visual] 丢失目标，重新扫描")
                    self.status = "INTERACT"
                    break

                inf_w, inf_h = get_wh()
                frame_w = max(inf_w, inf_h)
                scale = self.screen_w / frame_w
                door_center_x = (door[0] + door[2]) / 2
                offset_real = (door_center_x - (frame_w / 2)) * scale

                if abs(offset_real) <= 80:
                    print("[Visual] 对齐完成，尝试交互")
                    self.status = "INTERACT"
                    break

                adjust_val = int(offset_real * 0.33)
                adjust_val = max(-400, min(400, adjust_val))
                w.tap_single('视角', x_bias=adjust_val, dura=500, wait=500)
                w.refresh_frame()

        # --- 交互逻辑 ---
        elif self.status == "INTERACT":
            print(f"[Interact] 尝试在 {self.current_house_id} 寻找交互按钮...")
            success = False
            for i in range(10):
                w.refresh_frame()

                # --- [修改 2] 交互前移时加入跳跃检测 ---
                # 原因：门前可能有台阶或门槛，不跳跃无法靠近
                if w.get_info('跳跃'):
                    print("[Interact] 门前检测到障碍，尝试跳跃")
                    self.handle_jump_logic(w)  # 执行跳跃并前冲
                    w.refresh_frame()
                    continue  # 跳跃动作较大，跳过本次微调，直接进入下一次循环检查按钮
                # -----------------------------------

                if w.get_info('开门'):
                    w.click('开门')
                    time.sleep(1)
                    success = True
                    break
                if w.get_info('关门'):
                    w.click('关门')
                    time.sleep(1.2)
                    w.refresh_frame()
                    if w.get_info('开门'):
                        w.click('开门')
                        time.sleep(0.5)
                    success = True
                    break
                w.tap_single('摇杆', y_bias=-300, dura=300, wait=200)

            if success:
                print("[Interact] 交互成功，准备入户")
                self.status = "FINAL_ENTRY"
            else:
                print(f"[Interact] 警告：交互失败，舍弃进门点")
                ideal_angle = self.active_entry['direction']
                self.handle_failed_entry_logic(ideal_angle)
                self.status = "IDLE"
                return

        # --- 最终入户 ---
        elif self.status == "FINAL_ENTRY":
            ideal_angle = self.active_entry['direction']
            print(f"[Entry] 调整至进门角度: {ideal_angle}")
            self.align_direction_blocking(w, w.get_info('direction'), ideal_angle)
            print("[Entry] 进门")
            w.tap_single('摇杆', y_bias=-300, dura=300, wait=1000)
            # w.tap_single('摇杆', y_bias=-500, dura=300, wait=1000)
            # time.sleep(2)
            # if w.get_info('关门') is None:
            #     # 当前可能没进入房屋成功
            #     print(f"[Entry] 第一次进门未成功，左移后重新进门")
            #     w.tap_single('摇杆', x_bias=-45, dura=300, wait=100)
            #     # 左移后重新进入房屋
            #     w.tap_single('摇杆', y_bias=-300, dura=300, wait=100)

            self.start_searching(w)
            self.completed_houses.add(self.current_house_id)
            print(f"[Finish] 房屋 {self.current_house_id} 完成")
            w.refresh_frame()
            exit_direction = w.get_info('direction')
            self.prepare_next_target_logic(exit_direction)
            self.current_house_id = None
            self.status = "IDLE"

    def update_and_check_stuck(self, current_loc):
        self.history_locations.append(current_loc)
        if len(self.history_locations) > self.max_history_len:
            self.history_locations.pop(0)

        if len(self.history_locations) < self.max_history_len:
            return False

        x_coords = [loc[0] for loc in self.history_locations]
        y_coords = [loc[1] for loc in self.history_locations]
        max_dist = math.sqrt((max(x_coords) - min(x_coords)) ** 2 + (max(y_coords) - min(y_coords)) ** 2)
        return max_dist < self.stuck_threshold

    def execute_unstuck_logic(self, w: 'FrameWorker', current_loc):
        self.stop_auto_forward(w)
        if w.get_info('跳跃'):
            print("[Unstuck] 尝试跳跃脱困")
            self.handle_jump_logic(w)
            w.tap_single('摇杆', y_bias=-300, dura=500, wait=1000)
            w.refresh_frame()
            new_loc = check_location(w.get_info('location')[0])
            if new_loc and get_distance(current_loc, new_loc) > self.stuck_threshold:
                print("[Unstuck] 跳跃脱困成功")
                return

        print("[Unstuck] 跳跃无效，进入 U 型避障移动...")
        while True:
            print("[Unstuck] 后退...")
            w.tap_single('摇杆', y_bias=300, dura=300, wait=1500)
            w.refresh_frame()
            loc_after_back = check_location(w.get_info('location')[0])
            if not loc_after_back: continue

            print("[Unstuck] 右移试探...")
            w.tap_single('摇杆', x_bias=300, dura=300, wait=1500)
            w.refresh_frame()
            loc_after_right = check_location(w.get_info('location')[0])

            side_way_clear = False
            last_valid_loc = loc_after_back

            if loc_after_right and get_distance(loc_after_back, loc_after_right) > 0.5:
                print("[Unstuck] 右侧可通行")
                side_way_clear = True
                last_valid_loc = loc_after_right
            else:
                print("[Unstuck] 右侧受阻，左移试探...")
                w.tap_single('摇杆', x_bias=-300, dura=300, wait=1500)
                w.refresh_frame()
                loc_after_left = check_location(w.get_info('location')[0])

                if loc_after_left and get_distance(loc_after_right, loc_after_left) > 0.5:
                    print("[Unstuck] 左侧可通行")
                    side_way_clear = True
                    last_valid_loc = loc_after_left

            if not side_way_clear:
                print("[Unstuck] 左右均受阻 (U型死角)，再次后退...")
                continue

            print("[Unstuck] 尝试向前突破...")
            while True:
                w.tap_single('摇杆', y_bias=-300, dura=300, wait=2000)
                w.refresh_frame()
                loc_after_forward = check_location(w.get_info('location')[0])

                if loc_after_forward and get_distance(last_valid_loc, loc_after_forward) > 0.5:
                    print("[Unstuck] 脱困成功！")
                    return
                else:
                    print("[Unstuck] 前方依然受阻，继续侧向移动...")
                    moved_side = False
                    for bias in [300, -300]:
                        w.tap_single('摇杆', x_bias=bias, dura=300, wait=1500)
                        w.refresh_frame()
                        temp_loc = check_location(w.get_info('location')[0])
                        if temp_loc and get_distance(loc_after_forward, temp_loc) > 0.5:
                            last_valid_loc = temp_loc
                            moved_side = True
                            break

                    if not moved_side:
                        print("[Unstuck] 前方死路，重新执行后退逻辑")
                        break

    def handle_jump_logic(self, w: 'FrameWorker'):
        if w.get_info('跳跃'):
            print("[Jump] 检测到障碍，执行跳跃")
            self.stop_auto_forward(w)
            w.click('跳跃')
            time.sleep(0.2)
            w.tap_single('摇杆', y_bias=-400, dura=600)
            w.refresh_frame()

    def select_smart_target(self, current_loc, current_direction):
        best_dist = float('inf')
        best_id = None
        best_entry = None
        avoid_angle = getattr(self, 'avoid_angle_ref', None)
        avoid_mode = getattr(self, 'avoid_mode', None)

        for house_id, entries in self.house_data.items():
            if house_id in self.completed_houses: continue
            if house_id in self.temp_skip_houses: continue

            for entry in entries:
                dist = get_distance(current_loc, entry['location'])
                if avoid_angle is not None:
                    angle_to_target = calculate_angle(current_loc, entry['location'])
                    diff = abs(angle_to_target - avoid_angle)
                    if diff > 180: diff = 360 - diff
                    if avoid_mode == 'SAME' and diff < 45: continue
                    if avoid_mode == 'OPPOSITE' and diff > 135: continue

                if dist < best_dist:
                    best_dist = dist
                    best_id = house_id
                    best_entry = entry

        self.current_house_id = best_id
        self.active_entry = best_entry
        self.avoid_angle_ref = None
        self.avoid_mode = None
        self.temp_skip_houses.clear()

    def handle_failed_entry_logic(self, failed_entry_angle):
        print(f"[Smart] 进门失败，临时跳过 {self.current_house_id}")
        self.temp_skip_houses.add(self.current_house_id)
        self.current_house_id = None
        self.avoid_angle_ref = failed_entry_angle
        self.avoid_mode = 'SAME'

    def prepare_next_target_logic(self, exit_direction):
        self.avoid_angle_ref = exit_direction
        self.avoid_mode = 'OPPOSITE'

    def check_and_lock_door(self, w):
        if self.find_largest_door(w):
            return True
        return False

    def save_dataset_image(self, frame, suffix):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = f"temp/no_door/{timestamp}_{suffix}.jpg"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"[Data] 已保存图片: {path}")
        except Exception as e:
            print(f"[Data] 保存图片失败: {e}")

    def stop_auto_forward(self, w):
        if self.auto_forward:
            w.click('自动前进')
            self.auto_forward = False

    def align_direction_blocking(self, w, current_dir, target_angle):
        for _ in range(10):
            turn_dir, px, diff = calculate_move_count(current_dir, target_angle)
            if diff <= 5: return True
            x_bias = px if turn_dir == 'right' else - px
            w.tap_single('视角', x_bias=int(x_bias), dura=800, wait=500)
            w.refresh_frame()
            current_dir = w.get_info('direction')
        return False

    def align_direction(self, w, tar_loc, threshold=5):
        while True:
            cur_loc = w.get_info('location')[0]
            cur_dir = w.get_info('direction')
            target_angle = calculate_angle(cur_loc, tar_loc)
            turn_dir, px, diff = calculate_move_count(cur_dir, target_angle)
            if abs(diff) <= threshold: break
            move_px = px if turn_dir == 'right' else -px
            w.tap_single('视角', x_bias=move_px, dura=800, wait=500)
            w.refresh_frame()

    def find_largest_door(self, w):
        """
          0: house
          1: door
          2: window
          3: open_door
          4: door_frame
        """
        scene = w.get_info('forward_scene')
        if not scene: return None
        doors = [obj for obj in scene if int(obj[5]) in [0]]
        if not doors: return None
        return max(doors, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def start_searching(self, w):

        # 旋转360度遍历当前房间，检查当前房间内是否有物资跟联通当前这个房间的门
        # 屏幕滑动430，角度大概旋转60度左右
        picked_points = []  # 当前待搜索的物资点
        not_indoors = []  # 当前进入房间搜索的门

        # 顺时针旋转视角
        time.sleep(2)

        for _ in range(6):
            w.tap_single('视角', x_bias=430, dura=800, wait=500)
            self.update_yaw(60)  # 更新绝对朝向
            time.sleep(1)
            # 获取当前画面中的门跟物资
            scene = w.get_info('forward_scene')
            picked_object = [obj for obj in scene if int(obj[5]) in [1]]
            in_door = [obj for obj in scene if int(obj[5]) in [4]]

            picked_points.extend(picked_object)
            not_indoors.extend(in_door)
            w.refresh_frame()

        time.sleep(5)

        # 获取当前画面中物资和门
        supplies_raw = self.get_targets_info(picked_points)  # [(rel_angle, box_h, det)]
        doors_raw = self.get_targets_info(not_indoors)

        print("[Searching] 搜寻物资。。。。。。。。。。。当前获房间取到物资点信息{}".format(supplies_raw))
        print("[Searching] 搜寻物资。。。。。。。。。。。当前获可进入其他房间门的信息{}".format(doors_raw))

        for rel_ang, box_h, _ in supplies_raw:
            abs_ang = (self.player_yaw + rel_ang) % 360
            if not self.same_target(self.supplies, abs_ang, box_h):
                self.supplies.append((abs_ang, box_h))
                print(f"  > 物资 {abs_ang:.1f}° 框高 {box_h}px")

        for rel_ang, box_h, _ in doors_raw:
            abs_ang = (self.player_yaw + rel_ang) % 360
            if not self.same_target(self.doors, abs_ang, box_h):
                self.doors.append((abs_ang, box_h))
                print(f"  > 门   {abs_ang:.1f}° 框高 {box_h}px")

        print(f"[扫描] 完成。物资: {len(self.supplies)}，门: {len(self.doors)}")

        # 当前房间内存在物资点，开始当前房间内物资点物资的拾取

        if self.supplies:
            supplies_sorted = sorted(self.supplies, key=lambda x: x[1], reverse=True)
            print(f"[物资] 共====== {len(supplies_sorted)} 个")
            for idx, (abs_ang, box_h) in enumerate(supplies_sorted, 1):
                while True:
                    print(f"==========物资{idx} {abs_ang:.1f}° 框高{box_h}px")
                    self.collect_item(w, abs_ang, box_h)

        else:
            print("[物资] 无")

        w.tap_single('视角', x_bias=430, dura=800, wait=500)

    def turn_to_absolute(self, w, target_abs: float):
        """让角色朝向指定的绝对方向，并更新 player_yaw"""
        delta = (target_abs - self.player_yaw + 180) % 360 - 180
        if abs(delta) < 0.3:
            return
        self.turn_by_angle(w, delta)
        self.update_yaw(delta)
        print(f"    [转向] 转动 {delta:.1f}°，当前 player_yaw = {self.player_yaw:.1f}°")

    def turn_by_angle(self, w, delta_angle: float, duration_ms: int = 200):
        """
        滑动右侧屏幕旋转视角，delta_angle > 0 右转，< 0 左转。
        """
        swipe_dist = delta_angle * 7.1
        # start_x = int(self.scr_w * 0.75)
        # start_y = int(self.scr_h * 0.5)
        # end_x = start_x + int(swipe_dist)
        # end_y = start_y
        # self._run_adb(f"adb shell input swipe {start_x} {start_y} {end_x} {end_y} {duration_ms}")
        w.tap_single('视角', x_bias=int(swipe_dist), dura=800, wait=500)
        time.sleep(duration_ms / 800)
        w.refresh_frame()


    def collect_item(self, w, target_abs_angle: float, target_box_h: float):
        """锁定物资，闭环对准并靠近拾取"""
        print(f"    [搜集] 锁定物资 {target_abs_angle:.1f}°")
        self.current_target_abs = target_abs_angle  # 记录用于跟踪

        # 首次转向大致方向
        self.turn_to_absolute(w, target_abs_angle)


        for step in range(20):
            # 检查 UI 拾取按钮
            if w.get_info("拾取首个物资"):
                print("    出现可拾取物资的提示框信息")
                return

            # 重新检测并匹配当前物资，获取当前相对角度
            cur_rel_angle = self.find_target_relative_angle(w, target_box_h, 1)
            print("当前相对角度{}".format(cur_rel_angle))
            if cur_rel_angle is None:
                print("    [丢失] 未找到目标物资，放弃")
                return

            # 若偏差较大，微调朝向（闭环比例控制）
            if abs(cur_rel_angle) > 1.5:
                print(f"    [微调] 偏差 {cur_rel_angle:.1f}°")
                self.turn_by_angle(w, cur_rel_angle, duration_ms=150)
                self.update_yaw(cur_rel_angle)  # 更新 player_yaw
                time.sleep(0.15)
                continue

            # 对准后前进一小步
            w.tap_single('摇杆', y_bias=-400, dura=600)
            time.sleep(0.2)

        print("    -> 超时未拾取，放弃")

    def find_target_relative_angle(self, w, target_box_h, class_id):
        """
        在当前画面中寻找与 self.current_target_abs 匹配的物资，
        返回其相对角度（度），若未找到返回 None。
        """

        scene = w.get_info('forward_scene')
        detections = [obj for obj in scene if int(obj[5]) in [class_id]]
        print("调整人物转向过程中重新获取物资{}".format(detections))

        best = None
        best_diff = 999
        for det in detections:
            if det[5] not in [class_id]:
                continue
            cx = (det[0] + det[2]) / 2
            rel_ang = self.pixel_to_angle(cx)
            abs_ang = (self.player_yaw + rel_ang) % 360
            diff = abs((abs_ang - self.current_target_abs + 180) % 360 - 180)
            box_h = det[3] - det[1]
            # 角度差和框高差都要在阈值内
            if diff < 5 and abs(box_h - target_box_h) < 20:
                if diff < best_diff:
                    print("best_diff信息为{}".format(best_diff))
                    best_diff = diff
                    best = rel_ang
            print("调整人物转向过程中角度偏差{}以及宽高偏差{}".format(diff, abs(box_h - target_box_h)))

        return best

    def get_targets_info(self, targets):
        """
        从当前画面检测指定类别的目标，返回 [(rel_angle, box_height), ...]
        """
        info = []
        for target in targets:
            cx = (target[0] + target[2]) / 2
            bh = target[3] - target[1]
            rel_angle = self.pixel_to_angle(cx)
            info.append((rel_angle, bh, target))  # 保留原始框用于后续
        return info

    def pixel_to_angle(self, px: float) -> float:
        """像素水平坐标 -> 相对角度（度）"""
        center = self.screen_w / 2
        return (px - center) / (self.screen_w / 2) * (80 / 2)

    def update_yaw(self, delta):
        """每次旋转后调用，更新绝对朝向"""
        self.player_yaw = (self.player_yaw + delta) % 360

    def same_target(self, target_list, abs_angle, box_h):
        """角度 & 框高双重去重"""
        for a, h in target_list:
            angle_diff = abs((abs_angle - a + 180) % 360 - 180)
            # TARGET_LOCK_ANGLE_THRESH 绝对方向匹配角度容差（度） 设置偏差5 TARGET_LOCK_BOX_THRESH = 20    # 框高匹配容差（像素）
            if angle_diff < 5 and abs(box_h - h) < 20:
                return True
        return False
