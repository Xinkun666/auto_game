# import time
# import cv2
# import math
# import numpy as np
# from datetime import datetime
import random
import time
from typing import TYPE_CHECKING
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.map_navigator import MapNavigator
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.toolkit import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.house_exit import HouseExitManager
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
        self.last_target_bbox = None

        self.rooms_searched = 0

        self.entrance_doors = []  # 入口房间门列表 [(rel_angle, box_h), ...]
        self.a_door_sign = None  # 入口A门特征 (rel_angle, box_h)
        self.sub_rooms_info = []  # 已进入的子房间信息
        self.visited_doors = set()
        self.sub_rooms = []
        self.rooms_done = 0

        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.visited_abs = []
        self.visited_doors_info = []
        self.sub_room_area = None
        self.visited_sub_doors = []
        self.sub_rooms_entered = 0

        self.house_exit_manager = HouseExitManager()
        self.indoor_stuck_frames = 0

    def reset(self):
        self.completed_houses = set()
        self.current_house_id = None
        self.active_entry = None
        self.status = "IDLE"
        self.first_view = False
        self.auto_forward = False
        self.temp_skip_houses = set()
        self.history_locations = []
        self.searching_number = 0
        self.supplies = []
        self.doors = []
        self.player_yaw = 0.0
        self.last_target_bbox = None
        self.rooms_searched = 0
        self.entrance_doors = []
        self.a_door_sign = None
        self.sub_rooms_info = []
        self.visited_doors = set()
        self.sub_rooms = []
        self.rooms_done = 0
        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.visited_abs = []
        self.visited_doors_info = []
        self.sub_room_area = None
        self.visited_sub_doors = []
        self.sub_rooms_entered = 0

        self.house_exit_manager.reset()
        self.indoor_stuck_frames = 0

    def process(self, w: 'FrameWorker'):
        # self.start_searching(w)

        location_raw = w.get_info('location')
        if location_raw is None:
            print('位置值是None，尝试向前移动一段距离刷新位置...')
            w.tap_single('摇杆', y_bias=-300, wait=500)
            return
        location = check_location(location_raw[0])
        direction = w.get_info('direction')

        if location is None:
            print('位置值是None，尝试向前移动一段距离刷新位置...')
            w.tap_single('摇杆', y_bias=-300, wait=500)
            return

        # 0. 基础设置
        if not self.first_view:
            w.click('人称')
            self.first_view = True

        self.searching_logic(w, location, direction)

    def _get_house_scene(self, w: 'FrameWorker'):
        value = w.get_info('house_scene')
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def searching_logic(self, w: 'FrameWorker', current_loc, current_direction):

        if self.searching_number == 5:
            print('已经搜满5个房间，切换到跑图阶段')
            self.searching_number = 0
            w.change_stage('跑图阶段')
            return

        # --- 屋内卡死兜底检测 ---
        house_scene = self._get_house_scene(w)
        if house_scene == 0:
            self.indoor_stuck_frames += 1
            if self.indoor_stuck_frames > 30:
                print('[Searching] 检测到长时间困在屋内 (house_scene=0)，启动兜底出房策略')
                self.house_exit_manager.reset()
                for _ in range(20):
                    if self.house_exit_manager.process(w):
                        print('[Searching] 兜底出房成功，切换到跑图阶段')
                        self.indoor_stuck_frames = 0
                        self.searching_number = 0
                        self.completed_houses.add(self.current_house_id)
                        self.current_house_id = None
                        self.status = "IDLE"
                        w.change_stage('跑图阶段')
                        return
                print('[Searching] 兜底出房失败，强制重置状态切跑图')
                self.indoor_stuck_frames = 0
                self.searching_number = 0
                self.current_house_id = None
                self.status = "IDLE"
                w.change_stage('跑图阶段')
                return
        else:
            self.indoor_stuck_frames = 0

        # --- 智能选点 ---
        if self.current_house_id is None:
            self.select_smart_target(current_loc, current_direction)
            if not self.current_house_id:
                print("[Searching] 当前区域无合适目标或已搜完，切回跑图阶段")
                self.searching_number = 0
                w.change_stage('跑图阶段')
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
            self.searching_number += 1
            print(f"[Finish] 房屋 {self.current_house_id} 完成，已搜 {self.searching_number}/5")
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
            time.sleep(0.2)
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

        self.room_yaw = 0.0
        self.global_yaw = 0.0
        self.sub_rooms_entered = 0
        self.visited_sub_doors.clear()

        print("[搜房]入口房间搜集物资。。。")
        self.collect_supplies_in_room(w)

        self.house_entry_yaw = self.global_yaw
        a_door_abs_yaw = (self.house_entry_yaw + 180) % 360
        self.visited_sub_doors.append((a_door_abs_yaw, 999))
        print("[搜房] 已记录入口A门方向，防止误入")

        door_info = self._find_open_door_in_view(w)
        if not door_info: door_info = self._scan_for_open_door(w, 360)

        while door_info and self.sub_rooms_entered < 2:
            rel_ang, bh = door_info
            if self._enter_sub_room_and_collect(w, rel_ang, bh):
                self.sub_rooms_entered += 1
                door_info = self._find_open_door_in_view(w)
                if not door_info: door_info = self._scan_for_open_door(w, 360)
            else:
                break

        # 4. 退出房屋
        self._exit_house(w)

        time.sleep(10)

    def _find_closed_door_in_view(self, w):
        doors = self.new_targets_of_class(w, [0])
        if not doors: return None
        best = max(doors, key=lambda x: x[1])
        return (best[0], best[1])

    def _scan_for_closed_door(self, w, max_rotate=360):
        total = 0
        while total < max_rotate:
            self._turn(w, 30)
            total += 30
            time.sleep(0.2)
            res = self._find_closed_door_in_view(w)
            if res: return res
        return None

    def _enter_closed_door(self, w, rel_angle, rush_time=1.0):
        # 对关门贴脸时不需要盲冲(传0)，贴脸后点击开门，待门开后再盲冲
        approached = self._robust_pass_through_door(w, rel_angle, [0], rush_time=0.0)
        if approached:
            if w.get_info('开门'):
                w.click('开门')
                time.sleep(1)
            time.sleep(0.5)
            w.tap_single('摇杆', y_bias=-400, dura=1000)
            w.refresh_frame()
            time.sleep(0.2)
            return True
        return False

    def _exit_house(self, w):

        print("\n>>> 准备退出房屋")

        # 策略1：入口房间关闭门
        print("[出口] 策略1：在入口房间寻找关闭的门")
        closed = self._find_closed_door_in_view(w)
        if not closed: closed = self._scan_for_closed_door(w, 360)
        if closed:
            rel_ang, _ = closed
            print(f"[出口] 发现入口房间关闭门，推开离开！")
            self._enter_closed_door(w, rel_ang, rush_time=1.2)
            return

        # 策略2：进子房间找关闭门
        print("[出口] 策略2：入口无关闭门，进入子房间寻找")
        open_door = self._find_open_door_in_view(w)
        if not open_door: open_door = self._scan_for_open_door(w, 360)

        if open_door:
            rel_ang, bh = open_door
            print(f"[出口] 进子房间找关闭门")
            self._pass_through_open_door(w, rel_ang, rush_time=0.8)
            self.room_yaw = 0.0

            closed_in_sub = self._find_closed_door_in_view(w)
            if not closed_in_sub: closed_in_sub = self._scan_for_closed_door(w, 360)

            if closed_in_sub:
                c_rel_ang, _ = closed_in_sub
                print(f"[出口] 发现子房间关闭门，推开离开！")
                self._enter_closed_door(w, c_rel_ang, rush_time=1.2)
                return

            # 子房间没找到出口，退回入口房间
            print("[出口] 子房间无关闭门，扇区快搜退回入口房间")
            exit_door = self._find_open_door_in_view(w, ignore_visited=True)
            if not exit_door: exit_door = self._scan_for_open_door(w, 360, ignore_visited=True)
            if exit_door: self._pass_through_open_door(w, exit_door[0], rush_time=0.8)

        # 策略3：从入口A门原路返回
        print("[出口] 从入口A门原路返回")
        a_door = self._find_open_door_in_view(w, ignore_visited=True)
        if not a_door: a_door = self._scan_for_open_door(w, 360, ignore_visited=True)

        if a_door:
            print("[出口] 发现A门，穿过离开！")
            self._pass_through_open_door(w, a_door[0], rush_time=1.2)
        else:
            print("[出口] 极端情况：找不到A门，防卡死逃逸")
            for _ in range(3):
                w.tap_single('摇杆', y_bias=-400, dura=500)
                w.refresh_frame()
                time.sleep(0.2)
                self._turn(w, random.choice([-45, 45]))

        # 策略4：所有策略均失败，启动HouseExitManager兜底
        w.refresh_frame()
        if self._get_house_scene(w) == 0:
            print("[出口] 策略3后仍在屋内，启动HouseExitManager兜底出房")
            self.house_exit_manager.reset()
            for _ in range(30):
                if self.house_exit_manager.process(w):
                    print("[出口] 兜底出房成功")
                    return
            print("[出口] 兜底出房也失败，强制前进冲出")
            for _ in range(5):
                w.tap_single('摇杆', y_bias=-500, dura=300)
                w.refresh_frame()
                time.sleep(0.3)

    def _calc_abs_angle(self, rel_ang):

        return (self.global_yaw + rel_ang) % 360

    def _robust_pass_through_door(self, w, rel_angle, target_classes=None, rush_time=1.0):

        if target_classes is None:
            target_classes = [4]
        self._visual_align(w, rel_angle, target_classes)
        inf_w, inf_h = get_wh()
        frame_w = max(inf_w, inf_h)
        center_x = frame_w / 2

        for _ in range(30):
            doors = self.new_targets_of_class(w, target_classes)
            if not doors:
                print("  [搜房] 警告：未检测到门，尝试盲冲补救")
                break

            best = max(doors, key=lambda x: x[1])
            rel_ang, bh, _, det = best
            cx = (det[0] + det[2]) / 2
            offset_px = cx - center_x

            inf_w, inf_h = get_wh()
            frame_h = min(inf_w, inf_h)

            # 贴脸判定
            if bh > frame_h * 0.6:
                print(f"  [搜房] 已贴脸门框(高度比:{bh / frame_h:.2f})，准备盲冲穿过！")
                break

            if abs(offset_px) > 5:
                self._turn(w, self.pixel_to_angle(cx) * 0.6)
                time.sleep(0.05)
                continue

            # 轨迹笔直，允许前进
            # self.adb.forward(30)
            w.tap_single('摇杆', y_bias=-400, dura=300)
            w.refresh_frame()
            time.sleep(0.2)

        print(f"  [鲁棒穿门] 执行盲冲，时间: {rush_time}s")
        # self.adb.forward(rush_time)
        w.tap_single('摇杆', y_bias=-500, dura=1000)
        w.refresh_frame()
        time.sleep(0.2)
        return True

    def _pass_through_open_door(self, w, rel_angle, rush_time=1.0):
        return self._robust_pass_through_door(w, rel_angle, [4], rush_time)

    def _enter_sub_room_and_collect(self, w, rel_angle, box_h):
        """子房间完整交互流程：记录特征 -> 鲁棒穿门 -> 战术搜物资 -> 扇区回搜退门"""
        print("\n[子房间] 进入...")
        # 1. 记录进门绝对特征并去重
        abs_ang_enter = self._calc_abs_angle(rel_angle)
        self.visited_sub_doors.append((abs_ang_enter, box_h))

        # 2. 记录进门前的全局朝向，用于退出时计算反向扇区
        enter_yaw = self.global_yaw

        # 3. 穿门进入
        if not self._pass_through_open_door(w, rel_angle, rush_time=1.0):
            print("[错误] 进入失败")
            return False

        self.room_yaw = 0.0  # 重置局部坐标系
        # 4. 搜集物资（内部自带战术复位）
        self._search_supplies(w)

        # 5. 扇区快搜退出门
        print("[子房间] 搜集完毕，扇区快搜退出门...")
        target_exit_yaw = (enter_yaw + 180) % 360  # 计算进门背后的朝向
        # ignore_visited必须为True！因为进来的门已被标记，不忽略会看不到它
        exit_door = self._sector_scan_for_open_door(w, target_exit_yaw, sector_angle=120, ignore_visited=True)

        # 扇区兜底：如果扇区没找到，进行360全图扫描
        if not exit_door:
            print("[子房间] 未找到，360度兜底扫描...")
            exit_door = self._scan_for_open_door(w, 360, ignore_visited=True)

        if exit_door:
            rel_exit, _ = exit_door
            print(f"[子房间] 发现退出门，退出...")
            self._pass_through_open_door(w, rel_exit, rush_time=0.8)

            # 退回入口房间后，更新该门的特征以防重复进入
            time.sleep(0.2)
            doors = self.new_targets_of_class(w, [4])
            if doors:
                best = max(doors, key=lambda x: x[1])
                back_abs = self._calc_abs_angle(best[0])
                if not self._is_door_visited(w, back_abs, best[1]):
                    self.visited_sub_doors.append((back_abs, best[1]))
            return True

        print("[错误] 找不到退出门")
        return False

    def _sector_scan_for_open_door(self, w, center_yaw, sector_angle=120, ignore_visited=True):

        print(f"  [搜房] 中心朝向:{center_yaw:.0f}°, 扫描范围:{sector_angle}°")

        # 计算并转向目标中心朝向（处理最短路径旋转）
        delta = center_yaw - self.global_yaw
        if delta > 180: delta -= 360
        if delta < -180: delta += 360
        self._turn(w, delta)
        time.sleep(0.2)

        # 1. 检查中心点
        res = self._find_open_door_in_view(w, ignore_visited)
        if res: return res

        # 2. 左右扇区扫描
        half_sector = sector_angle // 2
        steps = half_sector // 30

        for i in range(1, steps + 1):  # 向左扫
            self._turn(w, 30)
            time.sleep(0.1)
            res = self._find_open_door_in_view(w, ignore_visited)
            if res: return res

        self._turn(w, - (half_sector))  # 瞬间归位中心
        time.sleep(0.2)
        for i in range(1, steps + 1):  # 向右扫
            self._turn(w, -30)
            time.sleep(0.1)
            res = self._find_open_door_in_view(w, ignore_visited)
            if res: return res

        return None

    def _scan_for_open_door(self, w, max_rotate=360, ignore_visited=False):

        total = 0
        while total < max_rotate:
            self._turn(w, 30)
            total += 30
            time.sleep(0.2)
            res = self._find_open_door_in_view(w, ignore_visited)
            if res: return res
        return None

    def _find_open_door_in_view(self, w, ignore_visited=False):

        doors = self.new_targets_of_class(w, [4])
        if not doors: return None
        doors.sort(key=lambda x: x[1], reverse=True)  # 框高越大越近，优先进入最近的门
        for rel_ang, bh, _, _ in doors:
            abs_ang = self._calc_abs_angle(rel_ang)
            if not ignore_visited and self._is_door_visited(w, abs_ang, bh):
                continue
            return (rel_ang, bh)
        return None

    def _is_door_visited(self, w, abs_ang, bh):

        for v_ang, v_bh in self.visited_sub_doors:
            angle_diff = abs(abs_ang - v_ang)
            angle_diff = min(angle_diff, 360 - angle_diff)  # 处理圆周折返
            if angle_diff < 20 and abs(bh - v_bh) < 50:  # 角度容差20度，框高容差50像素
                return True
        return False

    def collect_supplies_in_room(self, w):

        collected = []  # 已拾取的 (abs_angle, box_h)
        player_yaw = 0.0

        def calc_abs(rel_angle, box_h):
            return ((player_yaw + rel_angle) % 360, box_h)

        def is_duplicate(abs_ang, box_h):
            for a, h in collected:
                angle_diff = abs((abs_ang - a + 180) % 360 - 180)
                if angle_diff < 8 and abs(box_h - h) < 25:
                    return True
            return False

        def pickup_one_in_current_view(w):
            """在当前画面拾取一个未拾取过的物资，成功返回 True，否则 False"""
            # 获取当前画面所有物资，按面积取最近（最大）的一个
            scene = w.get_info('forward_scene')
            supplies = [obj for obj in scene if int(obj[5]) in [1]]

            if not supplies:
                return False
            # 选择面积最大的
            best = max(supplies, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            cx = (best[0] + best[2]) / 2
            rel_ang = self.pixel_to_angle(cx)
            box_h = best[3] - best[1]
            abs_ang = (player_yaw + rel_ang) % 360

            if is_duplicate(abs_ang, box_h):
                return False

            # 执行对准和拾取
            print(f"  发现物资（绝对{abs_ang:.1f}° 框高{box_h}px），开始拾取{best[:4]}")
            success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
            if success:
                collected.append((abs_ang, box_h))
                return True
            return False

        # ---------- 方向序列 ----------
        print("======[搜资] 检查初始方向 (0°)，在刚进入房屋的视角下检查是否有物资，有则搜集======")
        while pickup_one_in_current_view(w):
            time.sleep(0.2)

        print("======[搜资] 左转45°检查是否有物资，有则收集======")
        self.turn_by_angle(w, -45, 300)
        player_yaw = (player_yaw - 45) % 360
        time.sleep(0.3)
        while pickup_one_in_current_view(w):
            time.sleep(0.2)

        print("======[搜资] 左转45°后回正，右转45度检查是否有物资，有则收集======")
        self.turn_by_angle(w, 45, 300)  # 回到 0°
        player_yaw = (player_yaw + 45) % 360
        time.sleep(0.3)
        self.turn_by_angle(w, 45, 300)  # 右转 45°
        player_yaw = (player_yaw + 45) % 360
        time.sleep(0.3)
        while pickup_one_in_current_view(w):
            time.sleep(0.2)

        print(f"[搜资] 结束，共拾取 {len(collected)} 个物资")
        self.turn_by_angle(w, -45, 300)
        print("========回正方向==============")
        return len(collected)

    # def _scan_for_open_door(self, w, max_rotate=360):
    #     """旋转视角寻找开启的门（最多旋转 max_rotate 度），找到后返回 (rel_angle, box_h)"""
    #     total = 0
    #     while total < max_rotate:
    #         self._turn(w, 30)
    #         total += 30
    #         time.sleep(0.2)
    #         res = self._find_open_door_in_view(w)
    #         if res:
    #             return res
    #     return None
    #
    # def _find_open_door_in_view(self, w):
    #     """当前视角中寻找未访问的开启门，返回 (rel_angle, box_h) 或 None"""
    #     doors = self.new_targets_of_class(w, [4])
    #     if not doors:
    #         return None
    #     # 按框高排序，优先处理最近的
    #     doors.sort(key=lambda x: x[1], reverse=True)
    #     for rel_ang, bh, _, _ in doors:
    #         # 去重：与已访问门比较相对角度和框高
    #         if self._is_entry_door(rel_ang, bh):
    #             continue
    #             # 2. 过滤已访问的子房间门（基于绝对角度+框高去重）
    #         abs_ang = (self.room_yaw + rel_ang) % 360
    #         already = False
    #         for v_abs, v_bh in self.visited_sub_doors:
    #             if abs((abs_ang - v_abs + 180) % 360 - 180) < 10 and abs(bh - v_bh) < 30:
    #                 already = True
    #                 break
    #         if not already:
    #             return (rel_ang, bh)
    #     return None

    # def _exit_house(self,w):
    #     print("\n>>> 离开房屋")
    #     # 优先：入口房间存在 class=0 的非入口门
    #     if self.has_closed_exit:
    #         print("[出口] 入口房间存在关闭的非入口门，寻找...")
    #         # 重新扫描当前房间门（可能视角已变）
    #         current_doors = self._scan_current_room_doors(w)
    #         # 排除入口门
    #         if self.entry_door:
    #             e_abs, e_bh, e_cls = self.entry_door
    #             current_doors = [d for d in current_doors if not (
    #                 abs((d[0]-e_abs+180)%360-180) < 10 and abs(d[1]-e_bh) < 30 and d[2]==e_cls
    #             )]
    #         # 选择 关闭着的门中框高最大的（最近）
    #         closed_doors = [d for d in current_doors if d[2] == 0]
    #         if closed_doors:
    #             exit_door = max(closed_doors, key=lambda x: x[1])
    #             rel_exit = (exit_door[0] - self.room_yaw + 180) % 360 - 180
    #             self._enter_door_dynamic(rel_exit, 0)
    #             return
    #         else:
    #             print("[出口] 未找到关闭的门，尝试其他方式...")
    #
    #     # 其次：尝试进入任意子房间寻找 class=0 的门
    #     if self.sub_rooms:
    #         print("[出口] 尝试进入子房间寻找关闭的门...")
    #         for info in self.sub_rooms:
    #             # 这里我们简化：重新扫描入口房间，找到一个非入口门进入（不一定是之前那个）
    #             # 直接进入最近的子房间（假设之前记录了入口角度）
    #             pass
    #         # 由于我们没有保存子房间入口的绝对角度，此处略过，直接走入口门
    #         # 实际使用时可从 self.visited_doors_info 中选取未访问过的关闭门进入
    #
    #     # 最后：从入口门离开
    #     if self.entry_door:
    #         e_abs, _, _ = self.entry_door
    #         rel_exit = (e_abs - self.room_yaw + 180) % 360 - 180
    #         self._enter_door_dynamic(rel_exit, 4)
    #     else:
    #         print("[错误] 无法找到出口")
    #
    #
    # def _enter_door_dynamic(self, w, initial_rel_angle, door_class=None):
    #     """
    #     垂直靠近并穿过门。
    #     若 door_class 已知，按对应策略；否则根据实时检测自动处理。
    #     返回 (是否成功, 记录到的框高)
    #     """
    #     self._visual_align(w, initial_rel_angle, [0,4])
    #     print(f"  [进门] 开始靠近...")
    #     recorded_bh = None
    #     for _ in range(30):
    #         doors = self.targets_of_class(w,[0,4])
    #         front = [d for d in doors if abs(d[0]) < 5]
    #         if not front:
    #             if door_class == 0:
    #                 print("  [进门] 贴门（关闭），开门")
    #                 if w.get_info('开门'):
    #                     w.click('开门')
    #                     time.sleep(1)
    #                 time.sleep(0.5)
    #                 w.tap_single('摇杆', y_bias=-20, dura=300, wait=200)
    #                 w.refresh_frame()
    #                 return True, recorded_bh
    #             else:
    #                 print("  [进门] 前方无门，已穿过")
    #                 w.tap_single('摇杆', y_bias=-40, dura=300, wait=200)
    #                 w.refresh_frame()
    #                 time.sleep(0.5)
    #                 w.tap_single('摇杆', y_bias=-40, dura=300, wait=200)
    #                 w.refresh_frame()
    #                 time.sleep(0.5)
    #
    #                 return True, recorded_bh
    #
    #         best = max(front, key=lambda x: x[1])
    #         rel_ang, bh, cls, _ = best
    #         recorded_bh = bh
    #
    #         current_cls = door_class if door_class is not None else cls
    #
    #         # 开门且框高极大，直接穿过
    #         inf_w, inf_h = get_wh()
    #         frame_h = min(inf_w, inf_h)
    #         if current_cls == 4 and bh > frame_h * 0.7:
    #             print("  [进门] 开着的门已贴面，穿过")
    #             for _ in range(2):
    #                 w.tap_single('摇杆', y_bias=-20, dura=300, wait=200)
    #                 w.refresh_frame()
    #                 time.sleep(0.2)
    #             return True, recorded_bh
    #
    #         if abs(rel_ang) > 1:
    #             self._turn(w, max(-5, min(5, rel_ang * 0.9)))  # 微调视角
    #             time.sleep(0.1)
    #             continue
    #         w.tap_single('摇杆', y_bias=-20, dura=300, wait=200)
    #         w.refresh_frame()
    #         time.sleep(0.2)
    #     print("  [进门] 超时")
    #     return False, recorded_bh
    #
    # def _initial_entrance_scan(self, w):
    #     """
    #     入口房间进入后调用一次，旋转360°记录所有门，
    #     标记入口门（身后框高最大的门），并记录是否存在 class=0 的非入口门。
    #     """
    #     # 1. 扫描所有门
    #     all_doors = self._scan_current_room_doors(w)
    #     if not all_doors:
    #         print("[入口] 未发现任何门")
    #         return
    #
    #     # 2. 标记入口门（身后方向，即绝对角度最接近180°）
    #     entry = None
    #     best_diff = float('inf')
    #     for abs_ang, bh, cls in all_doors:
    #         diff = abs((abs_ang - 180 + 180) % 360 - 180)
    #         # 选择最接近180°且框高最大的
    #         if diff < best_diff or (diff == best_diff and (entry is None or bh > entry[1])):
    #             best_diff = diff
    #             entry = (abs_ang, bh, cls)
    #     self.entry_door = entry
    #     print("当前旋转360度的门的信息{}".format(all_doors))
    #     print(f"[入口门] 绝对角度 {entry[0]:.1f}° 框高 {entry[1]} 类型 {entry[2]}")
    #
    #     # 3. 判断是否存在 class=0 的非入口门
    #     has_closed = False
    #     for abs_ang, bh, cls in all_doors:
    #         if (abs_ang, bh, cls) == entry:
    #             continue
    #         if cls[5] == 0:
    #             has_closed = True
    #             break
    #     self.has_closed_exit = has_closed
    #     print(f"[出口标记] 入口房间有非入口的关闭门: {has_closed}")
    #
    #     # 存储入口房间所有门信息，用于后续去重和出口
    #     self.entrance_all_doors = all_doors
    # def _find_sub_room_doors(self,w):
    #     """
    #     在入口房间内渐进旋转，每30°检测一次门，
    #     过滤掉入口门和已访问的门，发现新门则返回其 (相对角度, 绝对角度, 框高, 类别)。
    #     若旋转满360°仍无新门，返回 None。
    #     """
    #     rotated = 0
    #     while rotated < 360 and self.rooms_done < 2:
    #         self._turn(w, 30)
    #         rotated += 30
    #         time.sleep(0.2)
    #         # 检测当前画面中的门
    #         doors = self.targets_of_class(w,[0,4])
    #         for rel_ang, box_h, cls, _ in doors:
    #             abs_ang = (self.room_yaw + rel_ang) % 360
    #
    #             # 跳过入口门（角度接近180°且框高匹配）
    #             if self.entry_door:
    #                 e_abs, e_bh, e_cls = self.entry_door
    #                 if abs((abs_ang - e_abs + 180) % 360 - 180) < 10 and abs(box_h - e_bh) < 30 and cls == e_cls:
    #                     continue
    #
    #             # 跳过已访问的门
    #             already = False
    #             for a, h, c in self.visited_doors_info:
    #                 if abs((abs_ang - a + 180) % 360 - 180) < 12 and abs(box_h - h) < 40 and c == cls:
    #                     already = True
    #                     break
    #             if already:
    #                 continue
    #
    #             # 找到新门
    #             print(f"  [发现子房间门] 相对角度 {rel_ang:.1f}° 绝对角度 {abs_ang:.1f}° 类型 {cls}")
    #             return (rel_ang, abs_ang, box_h, cls)
    # def _scan_current_room_doors(self,w):
    #     """旋转360°扫描当前房间的门，返回去重后的 (abs_ang, box_h, cls) 列表"""
    #     doors = []
    #     for _ in range(360 // 30):          # 例如 12 步 × 30° = 360°
    #         self._turn(w,30)
    #         time.sleep(0.2)
    #         for rel_ang, box_h, cls, _ in self.targets_of_class(w, [0, 4]):
    #             abs_ang = (self.room_yaw + rel_ang) % 360
    #             # 去重：角度差<8°、框高差<25px、且同类别
    #             dup = False
    #             for a, h, c in doors:
    #                 angle_diff = abs((abs_ang - a + 180) % 360 - 180)
    #                 if angle_diff < 8 and abs(box_h - h) < 25 and c == cls:
    #                     dup = True
    #                     break
    #             if not dup:
    #                 doors.append((abs_ang, box_h, cls))
    #     return doors
    # def collect_supplies_in_room(self, w):
    #     """
    #     按固定方向检查并拾取房间内所有物资。
    #     返回拾取的物资数量。
    #     """
    #     collected = []  # 已拾取的 (abs_angle, box_h)
    #     player_yaw = 0.0
    #
    #     def calc_abs(rel_angle, box_h):
    #         return ((player_yaw + rel_angle) % 360, box_h)
    #
    #     def is_duplicate(abs_ang, box_h):
    #         for a, h in collected:
    #             angle_diff = abs((abs_ang - a + 180) % 360 - 180)
    #             if angle_diff < 8 and abs(box_h - h) < 25:
    #                 return True
    #         return False
    #
    #     def pickup_one_in_current_view(w):
    #         """在当前画面拾取一个未拾取过的物资，成功返回 True，否则 False"""
    #         # 获取当前画面所有物资，按面积取最近（最大）的一个
    #         scene = w.get_info('forward_scene')
    #         supplies = [obj for obj in scene if int(obj[5]) in [1]]
    #
    #         if not supplies:
    #             return False
    #         # 选择面积最大的
    #         best = max(supplies, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
    #         cx = (best[0] + best[2]) / 2
    #         rel_ang = self.pixel_to_angle(cx)
    #         box_h = best[3] - best[1]
    #         abs_ang = (player_yaw + rel_ang) % 360
    #
    #         if is_duplicate(abs_ang, box_h):
    #             return False
    #
    #         # 执行对准和拾取
    #         print(f"  发现物资（绝对{abs_ang:.1f}° 框高{box_h}px），开始拾取{best[:4]}")
    #         success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
    #         if success:
    #             collected.append((abs_ang, box_h))
    #             return True
    #         return False
    #
    #     # ---------- 方向序列 ----------
    #     print("======[搜资] 检查初始方向 (0°)，在刚进入房屋的视角下检查是否有物资，有则搜集======")
    #     while pickup_one_in_current_view(w):
    #         time.sleep(0.2)
    #
    #     print("======[搜资] 左转45°检查是否有物资，有则收集======")
    #     self.turn_by_angle(w, -45, 300)
    #     player_yaw = (player_yaw - 45) % 360
    #     time.sleep(0.3)
    #     while pickup_one_in_current_view(w):
    #         time.sleep(0.2)
    #
    #     print("======[搜资] 左转45°后回正，右转45度检查是否有物资，有则收集======")
    #     self.turn_by_angle(w, 45, 300)  # 回到 0°
    #     player_yaw = (player_yaw + 45) % 360
    #     time.sleep(0.3)
    #     self.turn_by_angle(w, 45, 300)  # 右转 45°
    #     player_yaw = (player_yaw + 45) % 360
    #     time.sleep(0.3)
    #     while pickup_one_in_current_view(w):
    #         time.sleep(0.2)
    #
    #     print(f"[搜资] 结束，共拾取 {len(collected)} 个物资")
    #     self.turn_by_angle(w, -45, 300)
    #     print("========回正方向==============")
    #     return len(collected)

    def approach_and_pickup(self, w, initial_bbox, target_class, rel_ang):
        """
        小步靠近物资，并拾取
        返回是否成功拾取。
        """
        # last_bbox = initial_bbox[:4]
        # pickup_finish = False

        if abs(rel_ang) > 2:
            self.turn_by_angle(w, rel_ang, 200)
            time.sleep(1)

        for i in range(30):
            if i == 30:
                print("当前已移动完成30步或者已经拾取完物资")
                return False
            w.refresh_frame()
            scene = w.get_info('forward_scene')
            pick_menu = [obj for obj in scene if int(obj[5]) in [3]]

            print("当前是否有物资提示信息{}".format(pick_menu))
            if pick_menu:
                print("检查到附近有物资")
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                time.sleep(1)
                # 关闭附近弹窗，不影响继续旋转角度查找物资点
                if w.get_info("关闭附近"):
                    print("检测到关闭附近按钮。。。")
                    w.click(w.get_info("关闭附近"))
                    time.sleep(0.5)
                    w.refresh_frame()
                i = 30
                return True
            # 走到物资点后，检测到
            if w.get_info("关闭附近"):
                print("检查到附近有物资")
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                w.click("拾取首个物资")
                time.sleep(1)
                w.refresh_frame()
                time.sleep(1)
                if w.get_info("关闭附近"):
                    print("检测到关闭附近按钮。。。")
                    w.click(w.get_info("关闭附近"))
                    time.sleep(0.5)
                    w.refresh_frame()
                i = 30
                return True

            else:
                print("======识别到物资后，视角对准，往前靠近{}步，最大移动距离30步======".format(i + 1))
                w.tap_single('摇杆', y_bias=-20, dura=300)
                time.sleep(0.5)
                w.refresh_frame()
                i += 1

            time.sleep(1)

    def pixel_to_angle(self, cx):
        inf_w, inf_h = get_wh()
        frame_w = max(inf_w, inf_h)
        center = frame_w / 2
        if frame_w <= 0: return 0.0
        return (cx - center) / center * (80 / 2)

    def turn_by_angle(self, w, delta_angle, duration_ms=200):
        swipe_dist = delta_angle * 7.16
        if swipe_dist > 0:
            swipe_dist = swipe_dist + 10
        else:
            swipe_dist = swipe_dist - 10

        w.tap_single('视角', x_bias=int(swipe_dist), dura=800, wait=500)
        time.sleep(0.5)
        # 旋转视角后，刷新当前帧
        w.refresh_frame()

    def targets_of_class(self, w, target_class=None):
        if target_class is None:
            target_class = [4]
        scene = w.get_info('forward_scene')
        dets = [obj for obj in scene if int(obj[5]) in target_class]
        # print("[进入子房间]，旋转360过程中，检测到当前画面中开着的门的信息{}".format(dets))
        infos = []
        for d in dets:
            if d[5] in [0, 1, 2, 3, 4]:
                cx = (d[0] + d[2]) / 2
                bh = d[3] - d[1]
                angle = self.pixel_to_angle(cx)
                area = (d[2] - d[0]) * (d[3] - d[1])
                infos.append((angle, bh, d, area))
        return infos

    def new_targets_of_class(self, w, target_class=None):
        if target_class is None:
            target_class = [4]
        scene = w.get_info('forward_scene')
        dets = [obj for obj in scene if int(obj[5]) in target_class]
        # print("[进入子房间]，旋转360过程中，检测到当前画面中开着的门的信息{}".format(dets))
        infos = []
        for d in dets:
            if d[5] in [0, 1, 2, 3, 4]:
                cx = (d[0] + d[2]) / 2
                bh = d[3] - d[1]
                angle = self.pixel_to_angle(cx)
                area = (d[2] - d[0]) * (d[3] - d[1])
                infos.append((angle, bh, d[5], d))
        return infos

    def _approach_door(self, w, rel_ang, is_sub_room=False):
        """
        从 initial_bbox 开始，视觉对准 + 小步靠近 + 拾取。
        返回是否成功拾取。
        """
        print("出子房间的门之前的角度{}".format(rel_ang))
        last_door = []
        # 调整角度
        if abs(rel_ang) > 2:

            if rel_ang > 0:
                print("向右滑动调整视角，角度有偏差，添加5度的偏差")
                rel_ang += 5
            else:
                print("角度微微调整")
                rel_ang += 6
            print("出子房间的门的进行调整的角度{}".format(rel_ang))
            if abs(rel_ang) > 45:

                count = int(abs(rel_ang) / 45)
                count_ang = abs(rel_ang) % 45
                print("角度大于45度，拆分成多次来旋转，拆分成{}次，是否有多余的{}".format(count, count_ang))
                for i in range(count):
                    if rel_ang > 0:
                        self.turn_by_angle(w, 45, 200)
                    else:
                        self.turn_by_angle(w, -45, 200)

                # if rel_ang > 0:
                #     self.turn_by_angle(w, count_ang, 200)
                # else:
                #     self.turn_by_angle(w, -count_ang, 200)

            else:
                self.turn_by_angle(w, rel_ang, 200)

            time.sleep(1)

        # 调整角度结束后，往前移动靠近
        for i in range(30):
            if i == 30:
                print("当前已移动完成30步")
                return False
            w.tap_single('摇杆', y_bias=-20, dura=300)
            i += 1
            w.refresh_frame()
            time.sleep(1)

            scene = w.get_info('forward_scene')
            open_door1 = [obj for obj in scene if int(obj[5]) in [4]]

            if open_door1:
                last_door = max(open_door1, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                # 当前还在画面中可以检测到
                inf_w, inf_h = get_wh()
                frame_w = max(inf_w, inf_h)
                center = frame_w / 2
                print("向门靠近并移动的过程中门的信息{}，门的中心点位置{},屏幕的中心点位置{}".format(open_door1, (
                        open_door1[0][2] - open_door1[0][0]) / 2, center))
                # 移动靠近的过程中y1会逐渐减小，小于等于10 的时候，人物靠近门，这个时候暂停移动

            else:
                # 检测不到当前视角中的门的时候，当前已经靠近门边，直接往前走，可能会出现擦着墙边过的情况
                print("当前已经靠近房间的门,微调角度处理。。。。")
                door = last_door
                if door:
                    inf_w, inf_h = get_wh()
                    frame_w = max(inf_w, inf_h)
                    scale = self.screen_w / frame_w
                    door_center_x = (door[0] + door[2]) / 2
                    offset_real = (door_center_x - (frame_w / 2)) * scale

                    adjust_val = int(offset_real * 0.33)
                    adjust_val = max(-400, min(400, adjust_val))
                    print("当前微调视角，水平滑动{}".format(adjust_val))
                    w.tap_single('视角', x_bias=int(adjust_val), dura=500, wait=500)
                    w.refresh_frame()
                    time.sleep(0.5)
                    # print("当前人物水平微调向相反的方向，水平滑动{}".format((adjust_val / 3) * -1))
                    # w.tap_single('摇杆', x_bias=int(adjust_val / 3)*-1, dura=500, wait=500)
                    # w.refresh_frame()
                    time.sleep(5)

                if w.get_info('开门'):
                    w.click('开门')
                    time.sleep(1)
                print("靠近门后，微调结束，直走进入房间。。。")
                w.tap_single('摇杆', y_bias=-400, dura=300)
                w.refresh_frame()
                w.tap_single('摇杆', y_bias=-400, dura=300)
                w.refresh_frame()
                print("靠近门后往前移动俩步结束，不在往前移动")
                return True
        time.sleep(1)

    def _collect_in_direction(self, w, avoid_door_abs=None):
        collected = []
        # supplies = self.targets_of_class(w, target_class=[4])
        supplies = self.new_targets_of_class(w, target_class=[4])
        print("子房间查找物资的信息{}".format(supplies))
        # 过滤当前在子房间内发现的入口房间的物资
        # if supplies:
        #     filtered = []
        #     for rel_ang, box_h, det, area in supplies:
        #         # 计算该物资的绝对角度
        #         abs_ang = (self.room_yaw + rel_ang) % 360
        #         # 与避开方向的夹角
        #         diff = abs((abs_ang - avoid_door_abs + 180) % 360 - 180)
        #         if diff > 20:  # 夹角大于20°才保留
        #             filtered.append((rel_ang, box_h, det, area))
        #     supplies = filtered
        #     if not supplies:
        #         print("  [物资] 当前方向（避开门口）无物资")
        #         return

        if supplies:

            # 选择面积最大的
            # best = max(supplies, key=lambda d: d[3])
            best = max(supplies, key=lambda d: d[1])
            rel_ang = best[0]
            abs_ang = (self.room_yaw + rel_ang) % 360

            print(f"  发现物资（绝对{abs_ang:.1f}° 框高{best[1]}px），开始拾取{best[:4]}")
            success = self.approach_and_pickup(w, best[:4], [0, 1], rel_ang)
            if success:
                collected.append((abs_ang, best[1]))
        else:
            print("当前子房间内未找到物资信息,继续下一次视角中获取物资...")
            # self.turn_by_angle(w, 45)
            time.sleep(1)

        if len(collected) == 2:
            print("当前物资已拾满")

    def _search_supplies(self, w, avoid_door_abs=None):
        print("[物资] 方向扫描...")
        self._collect_in_direction(w, avoid_door_abs)  # 正前
        self._turn(w, -45)
        self._collect_in_direction(w, avoid_door_abs)  # 左45°
        self._turn(w, 45)
        time.sleep(5)
        self._turn(w, 45)
        self._collect_in_direction(w, avoid_door_abs)  # 右45°
        self._turn(w, -45)  # 回正

    # def _explore_sub_rooms(self, w):
    #     # 查找所有开着的门
    #     all_doors = self._scan_all_doors(w, target_class=[4])
    #     print("当前门信息{}".format(all_doors))
    #     if not all_doors:
    #         print("[门] 入口房间无门")
    #         return
    #
    #     best_entry = None
    #     best_diff = float('inf')
    #     for abs_ang, bh, rel_ang in all_doors:
    #         diff = abs((abs_ang - 180 + 180) % 360 - 180)
    #         if diff < best_diff or (diff == best_diff and (best_entry is None or bh > best_entry[1])):
    #             best_diff = diff
    #             best_entry = (abs_ang, bh)
    #     self.entry_door_abs = best_entry
    #     print(f"[入口门] 绝对角度 {best_entry[0]:.1f}° 框高 {best_entry[1]}px")
    #
    #     # 当前回到进门初始位置
    #
    #     while self.rooms_done < 2:
    #         # 先判断当前视野中是否有子房间的门
    #         doors = self.targets_of_class(w, target_class=[4])
    #         if not doors:
    #             self._turn(w, 45)
    #
    #         else:
    #             print("当前子房间的门的信息{}并选择最大的进入".format(doors))
    #             for rel_ang, box_h, d, area in doors:
    #                 abs_ang = (self.room_yaw + rel_ang) % 360
    #                 if self.entry_door_abs:
    #                     e_abs, e_bh = self.entry_door_abs
    #                     if self._is_same_angle(abs_ang, e_abs, 10) and abs(box_h - e_bh) < 30:
    #                         continue
    #                 if self._is_visited(abs_ang):
    #                     continue
    #                     # 检查是否已进入过（全局已访问）
    #                 if self._is_duplicate_door(abs_ang, box_h, self.visited_doors_info, angle_thresh=12, box_thresh=40):
    #                     continue
    #
    #                 print(f"\n[发现] 子房间门 绝对角度 {abs_ang:.1f}°")
    #                 # self.visited_abs.append(abs_ang)
    #                 self.visited_doors_info.append((abs_ang, box_h, rel_ang))
    #                 self._enter_sub_room(w, rel_ang, abs_ang)
    #                 print(f"\n[发现] 子房间门 绝对角度 {abs_ang:.1f}°已搜索结束")
    #                 print("出当前子房间后往左移动一点。。。")
    #                 w.tap_single('摇杆', x_bias=-10, dura=300)
    #                 w.refresh_frame()
    #                 time.sleep(0.2)
    #                 self.rooms_done += 1
    #                 break
    #
    #
    #
    #     # 探索子房间
    #     self.room_yaw = 0.0
    #     rotated = 0
    #     while rotated < 360 and self.rooms_done < 2:
    #         self._turn(w, 45)
    #         rotated += 45
    #         time.sleep(0.2)
    #         doors = self.targets_of_class(w, target_class=[4])
    #         print("当前子房间的门的信息{}".format(doors))
    #         for rel_ang, box_h, d, area in doors:
    #             abs_ang = (self.room_yaw + rel_ang) % 360
    #             if self.entry_door_abs:
    #                 e_abs, e_bh = self.entry_door_abs
    #                 if self._is_same_angle(abs_ang, e_abs, 10) and abs(box_h - e_bh) < 30:
    #                     continue
    #             if self._is_visited(abs_ang):
    #                 continue
    #                 # 检查是否已进入过（全局已访问）
    #             if self._is_duplicate_door(abs_ang, box_h, self.visited_doors_info, angle_thresh=12, box_thresh=40):
    #                 continue
    #
    #             print(f"\n[发现] 子房间门 绝对角度 {abs_ang:.1f}°")
    #             # self.visited_abs.append(abs_ang)
    #             self.visited_doors_info.append((abs_ang, box_h, rel_ang))
    #             self._enter_sub_room(w, rel_ang, abs_ang)
    #             print(f"\n[发现] 子房间门 绝对角度 {abs_ang:.1f}°已搜索结束")
    #             print("出当前子房间后往左移动一点。。。")
    #             w.tap_single('摇杆', x_bias=-10, dura=300)
    #             w.refresh_frame()
    #             time.sleep(0.2)
    #             self.rooms_done += 1
    #             break
    #     print("[扫描] 子房间探索结束")

    #
    # def _enter_door_straight(self, w, initial_rel_angle):
    #     """
    #     严格正对门进入：全程保持门框中心在屏幕正中（误差<1°），
    #     直到门消失后点击开门。避免斜向进入。
    #     """
    #     # 第一步：精确对准
    #     self._visual_align(w, initial_rel_angle, )
    #     print("  [正对进门] 开始垂直靠近...")
    #     recorded_bh = None
    #     for step in range(30):
    #         # 检测门是否还在正前方
    #         doors = self.targets_of_class(w, target_class=[4])
    #         front_doors = [d for d in doors if abs(d[0]) < 5]
    #
    #         if not front_doors:
    #             # 门已贴面或丢失，尝试开门
    #             print("  [正对进门] 已贴门，开门")
    #             # self.adb.tap_door()
    #             time.sleep(0.5)
    #
    #             print("已在门附近，多往前走俩步")
    #             w.tap_single('摇杆', y_bias=-20, dura=300)
    #             w.refresh_frame()
    #             time.sleep(0.2)
    #             w.tap_single('摇杆', y_bias=-15, dura=300)
    #             w.refresh_frame()
    #             time.sleep(0.2)
    #             return True, recorded_bh
    #
    #         # 取最近的门（框高最大）
    #         best = max(front_doors, key=lambda x: x[1])
    #         recorded_bh = best[1]
    #         rel_ang = best[0]
    #
    #         # 严格要求：偏差大于 1° 时立即微调，不前进
    #         if abs(rel_ang) > 1.0:
    #             # 微调视角（比例控制，限制单次最大5°）
    #             adjust = max(-5, min(5, rel_ang * 0.9))
    #             self._turn(w, adjust)
    #             time.sleep(0.1)
    #             continue  # 调整后重新检测，不移动
    #
    #         # 偏差极小，可以前进一步
    #         w.tap_single('摇杆', y_bias=-20, dura=300)
    #         w.refresh_frame()
    #         time.sleep(0.1)
    #
    #     print("  [正对进门] 超时未贴门")
    #     return False

    # ─────── 视觉闭环对准 ───────
    def _visual_align(self, w, target_angle, target_class=None):
        print("开始调整。。。{}".format(target_angle))
        for _ in range(6):
            if abs(target_angle) <= 1.5:
                return
            step = max(-30, min(30, target_angle))
            self._turn(w, step)
            time.sleep(0.15)
            targets = self.new_targets_of_class(w, target_class=target_class)
            if not targets:
                print("  [对准] 目标丢失")
                return
            best = max(targets, key=lambda x: x[1])
            target_angle = best[0]

    def _turn(self, w, delta):
        self.turn_by_angle(w, delta)
        self.room_yaw = (self.room_yaw + delta) % 360
