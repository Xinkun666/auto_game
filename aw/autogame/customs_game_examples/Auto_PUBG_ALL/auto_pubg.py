import time
from typing import TYPE_CHECKING

from aw.autogame.tools.Utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.parachute import ParachuteManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.running_map import RunningManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.driving_car import DrivingManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.searching_house import (
    Searching_House,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.house_exit import (
    HouseExitManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.phase_timer import (
    PHASE_DRIVING,
    PHASE_RUNNING,
    PhaseTimeManager,
)

"""
1. w.current_stage ： 当前自动化的阶段，可以参考你标注工程导出的info.py里，对应的阶段为True，即表示当前阶段
2. w.get_info() : 获取你标注的区域是否出现
3. w.click() ： 点击操作
4. w.tap_single() : 单指操作
5. w.tap_double() : 双指操作
6. w.click_down() : 按下操作
7. w.change_stage() : 改变你的阶段到你想要的阶段
8. w.refresh_frame() : 刷新帧和帧信息
9. w.stop() ： 结束自动化
10. insert_logs() : 插入日志
"""

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker

PHASE_STAGE_MAP = {
    "跑图阶段": PHASE_RUNNING,
    "开车阶段": PHASE_DRIVING,
}

# 单位：分钟
PHASE_DURATIONS = {
    PHASE_RUNNING: 10,
    PHASE_DRIVING: 10,
}

DROP_TARGET_GARAGE = RunningManager.CAR_ENTRY_POINT
DROP_TARGET_CENTER = (1024, 1024)
SP_SAVE_LONG_PRESS_MS = 3000

start_game = False
final_shutdown_pending = False
parachute_manager = ParachuteManager()
running_manager = RunningManager()
driving_manager = DrivingManager()
searching_house_manager = Searching_House()
house_exit_manager = HouseExitManager()
phase_timer = PhaseTimeManager(PHASE_DURATIONS, PHASE_STAGE_MAP)


def prepare_round():
    phase_timer.start_new_round()

    need_drive = phase_timer.need_drive()
    drop_target = DROP_TARGET_GARAGE if need_drive else DROP_TARGET_CENTER

    parachute_manager.reset()
    parachute_manager.configure(target_pos=drop_target, landing_stage="跑图阶段")

    running_manager.reset(finding_car=need_drive)

    driving_manager.reset()
    house_exit_manager.reset()

    print(
        f"[Round] need_drive={need_drive}, "
        f"running_remaining={phase_timer.get_remaining(PHASE_RUNNING):.2f}s, "
        f"driving_remaining={phase_timer.get_remaining(PHASE_DRIVING):.2f}s, "
        f"drop_target={drop_target}"
    )


def handle_sp_start(w: "FrameWorker"):
    if not phase_timer.should_start_sp():
        return
    if phase_timer.start_game_time is not None:
        running_manager.set_game_time(phase_timer.start_game_time)
        driving_manager.set_game_time(phase_timer.start_game_time)
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_started()


def handle_sp_stop(w: "FrameWorker"):
    if not phase_timer.sp_recording:
        return
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_stopped()


def finalize_automation(w: "FrameWorker"):
    global final_shutdown_pending

    if w.current_stage == "跑图阶段":
        running_manager.stop_auto_forward(w)

    if not phase_timer.sp_saved:
        w.click_down("sp", dura=SP_SAVE_LONG_PRESS_MS)
        time.sleep(1)
        phase_timer.mark_sp_stopped()
        phase_timer.mark_sp_saved()

    final_shutdown_pending = True
    w.change_stage("结束阶段")


def finalize_after_lobby(w: "FrameWorker"):
    global final_shutdown_pending
    final_shutdown_pending = False
    w.stop()


def on_stage(w: "FrameWorker"):
    global start_game, final_shutdown_pending

    previous_stage = phase_timer.last_stage
    stage_events = phase_timer.sync_stage(w.current_stage)
    stage_events |= phase_timer.refresh()

    if previous_stage == "开车阶段" and w.current_stage == "跑图阶段":
        running_manager.notify_vehicle_exit()

    if "landed" in stage_events and not phase_timer.all_done():
        handle_sp_start(w)

    if w.current_stage == "关闭弹窗阶段":
        if w.get_info("关闭公告"):
            w.click(w.get_info("关闭公告"))
            w.refresh_frame()
            return
        
        if w.get_info("提示"):
            w.click("取消")
            w.refresh_frame()
            return
        
        if w.get_info('对局结束'):
            w.click('确定')
            w.refresh_frame()
            return
        
        if w.get_info('关闭预约'):
            w.click(w.get_info('关闭预约'))
            w.refresh_frame()
            return

        if w.get_info("关闭"):
            w.click(w.get_info("关闭"))
            w.refresh_frame()
            return

        if w.get_info("回归"):
            w.click(w.get_info("回归"))
            w.refresh_frame()
            return

        if w.get_info("关闭活动"):
            w.click(w.get_info("关闭活动"))
            w.refresh_frame()
            return

        if w.get_info("关闭新玩法"):
            w.click(w.get_info("关闭新玩法"))
            w.refresh_frame()
            return

        if w.get_info("关闭活动2"):
            w.click(w.get_info("关闭活动2"))
            w.refresh_frame()
            return

        time.sleep(2)
        if w.get_info("房子"):
            if final_shutdown_pending:
                finalize_after_lobby(w)
                return
            w.change_stage("选择地图阶段")
            return

    if w.current_stage == "选择地图阶段":
        w.click("地图")
        time.sleep(2)
        w.click("经典模式")
        time.sleep(2)
        w.click("切换")
        time.sleep(2)
        w.refresh_frame()

        if w.get_info("对号"):
            w.click(w.get_info("对号"))
            time.sleep(2)

        w.click("海岛")
        w.click("确定")
        w.change_stage("开始游戏阶段")
        return

    if w.current_stage == "开始游戏阶段":
        if w.get_info("加速礼包"):
            w.click("放弃")
            w.refresh_frame()

        if w.get_info("房子"):
            if not start_game:
                w.click("开始游戏")
                start_game = True
            w.refresh_frame()

        if w.get_info("提示"):
            w.click("不提示")
            time.sleep(1)
            w.click("不需要")
            time.sleep(1)

        if w.get_info("拳头"):
            prepare_round()
            w.change_stage("跳伞阶段")
            start_game = False
            return

    if w.current_stage == "跳伞阶段":
        parachute_manager.process(w)
        return

    if w.current_stage == "搜房阶段":
        house_scene = w.get_info("house_scene")
        if isinstance(house_scene, (list, tuple)) and len(house_scene) == 1:
            house_scene = house_scene[0]

        if house_exit_manager.process(w):
            w.change_stage("跑图阶段")
        return

    if w.current_stage == "跑图阶段":
        handle_sp_start(w)

        if phase_timer.all_done():
            finalize_automation(w)
            return

        running_manager.process(w)
        return

    if w.current_stage == "开车阶段":
        driving_manager.set_running_fallback_enabled(not phase_timer.is_completed(PHASE_RUNNING))

        if "enter_开车" in stage_events:
            driving_manager.set_remaining_drive_time(phase_timer.get_remaining(PHASE_DRIVING))

        if phase_timer.is_completed(PHASE_DRIVING):
            driving_manager.set_remaining_drive_time(0)

        driving_manager.process(w)
        return

    if w.current_stage == "结束阶段":
        if final_shutdown_pending:
            handle_sp_stop(w)
            w.click("设置")
            time.sleep(1)
            w.click("返回大厅")
            time.sleep(1)
            w.click("确定")
            time.sleep(3)
            w.change_stage("关闭弹窗阶段")
            return

        handle_sp_stop(w)

        w.click("设置")
        time.sleep(1)
        w.click("返回大厅")
        time.sleep(1)
        w.click("确定")
        time.sleep(3)
        w.change_stage("开始游戏阶段")
