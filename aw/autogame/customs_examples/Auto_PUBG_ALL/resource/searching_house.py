import time
import cv2
import math
import numpy as np
from datetime import datetime
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
        self.house_data = load_json(r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/house_entry/house_entries_summary.json')

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

    def process(self, w: 'FrameWorker'):
        location = check_location(w.get_info('location')[0])
        direction = w.get_info('direction')

        if location is None:
            print('位置值是None，尝试向前移动一段距离刷新位置...')
            w.tap_single('摇杆', y_bias=-300, wait=500)
            return

        # 0. 基础设置
        if not self.first_view:
            w.click('第一人称')
            self.first_view = True

        self.searching_logic(w, location, direction)

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
                self.history_locations = [] # 清空历史，防止重复触发
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
                    self.handle_jump_logic(w) # 执行跳跃并前冲
                    w.refresh_frame()
                    continue # 跳跃动作较大，跳过本次微调，直接进入下一次循环检查按钮
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
            w.tap_single('摇杆', y_bias=-300, dura=300, wait=100)
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
        doors = [obj for obj in scene if int(obj[5]) in [1]]
        if not doors: return None
        return max(doors, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def start_searching(self, w):
        w.tap_single('摇杆', y_bias=-300, dura=500, wait=500)
        time.sleep(5)
        w.tap_single('摇杆', y_bias=300, dura=500, wait=3000)
        tar_dir = (w.get_info('direction') + 180) % 360
        self.align_direction_blocking(w, w.get_info('direction'), tar_dir)
        self.searching_number += 1