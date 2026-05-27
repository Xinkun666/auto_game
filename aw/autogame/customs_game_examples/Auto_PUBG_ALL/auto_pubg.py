import time
from typing import TYPE_CHECKING

from aw.autogame.tools.Utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.parachute import ParachuteManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.running_map import RunningManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.driving_car import DrivingManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.searching_house_yajun import (
    Searching_House,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.house_exit import (
    HouseExitManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.phase_timer import (
    PHASE_DRIVING,
    PHASE_RUNNING,
    PHASE_SEARCHING,
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
    "搜房阶段": PHASE_SEARCHING,
    "跑图阶段": PHASE_RUNNING,
    "开车阶段": PHASE_DRIVING,
}

# 单位：分钟
PHASE_DURATIONS = {
    PHASE_SEARCHING: 10,
    PHASE_RUNNING: 10,
    PHASE_DRIVING: 10,
}

DROP_TARGET_GARAGE = (990, 757)
DROP_TARGET_CENTER = (990, 757)
DROP_TARGET_RUNNING_AFTER_SEARCH = (1094, 790)
SP_SAVE_LONG_PRESS_MS = 3000
START_GAME_VERIFY_DELAY = 5.0

start_game = False
start_game_click_time = None
final_shutdown_pending = False
rank_finish_pending = False
next_phase_report_time = 0.0
all_done_reported = False
searching_view_synced = False
parachute_manager = ParachuteManager()
running_manager = RunningManager()
driving_manager = DrivingManager()
searching_house_manager = Searching_House()
house_exit_manager = HouseExitManager()
phase_timer = PhaseTimeManager(PHASE_DURATIONS, PHASE_STAGE_MAP)


def pause_sp_after_death(w: "FrameWorker"):
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_stopped()


running_manager.pause_sp_callback = pause_sp_after_death
driving_manager.pause_sp_callback = pause_sp_after_death


def prepare_round():
    global next_phase_report_time, all_done_reported, searching_view_synced, rank_finish_pending

    phase_timer.start_new_round()
    searching_view_synced = False
    rank_finish_pending = False
    next_phase_report_time = 0.0
    all_done_reported = False

    need_drive = phase_timer.need_drive()
    need_searching = not phase_timer.is_completed(PHASE_SEARCHING)
    landing_stage = "搜房阶段" if need_searching else "跑图阶段"
    if need_searching:
        drop_target = DROP_TARGET_GARAGE if need_drive else DROP_TARGET_CENTER
    else:
        drop_target = DROP_TARGET_RUNNING_AFTER_SEARCH

    parachute_manager.reset()
    parachute_manager.configure(target_pos=drop_target, landing_stage=landing_stage)

    running_manager.reset(finding_car=need_drive)

    driving_manager.reset()
    searching_house_manager.reset()
    house_exit_manager.reset()

    print(
        f"[Round] need_drive={need_drive}, "
        f"need_searching={need_searching}, "
        f"total_remaining={phase_timer.get_total_remaining():.2f}s, "
        f"searching_remaining={phase_timer.get_remaining(PHASE_SEARCHING):.2f}s, "
        f"running_remaining={phase_timer.get_remaining(PHASE_RUNNING):.2f}s, "
        f"driving_remaining={phase_timer.get_remaining(PHASE_DRIVING):.2f}s, "
        f"drop_target={drop_target}, landing_stage={landing_stage}"
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


def should_abort_searching(w: "FrameWorker"):
    global rank_finish_pending

    if w.current_stage != "搜房阶段":
        return True

    if phase_timer.is_completed(PHASE_SEARCHING):
        print("[Timer] 搜房阶段 600s 已用完，强制切换到跑图阶段")
        searching_house_manager.stop_auto_forward(w)
        w.change_stage("跑图阶段")
        return True

    if w.get_info("变身") or w.get_info("红色血条"):
        print("[Searching] 检测到死亡，进入结束阶段")
        searching_house_manager.stop_auto_forward(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    if w.get_info("个人排名") or w.get_info("队伍排名"):
        print("[Searching] 检测到排名界面，进入结束阶段")
        rank_finish_pending = True
        searching_house_manager.stop_auto_forward(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    return False


searching_house_manager.abort_callback = should_abort_searching
searching_house_manager.can_finish_callback = lambda w: phase_timer.is_completed(PHASE_SEARCHING)


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


def prepare_rank_finish_for_lobby(w: "FrameWorker"):
    global rank_finish_pending

    if not rank_finish_pending and not (w.get_info("个人排名") or w.get_info("队伍排名")):
        return

    print("[End] 检测到排名界面，先点击观战对手再返回大厅")
    w.click("观战对手")
    time.sleep(1)
    w.refresh_frame()
    rank_finish_pending = False


def _format_phase_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def maybe_report_phase_remaining():
    global next_phase_report_time, all_done_reported

    if phase_timer.start_game_time is None:
        return

    now = time.time()
    if next_phase_report_time <= 0.0:
        next_phase_report_time = now + 5.0

    if now >= next_phase_report_time:
        total_remaining = phase_timer.get_total_remaining()
        searching_remaining = phase_timer.get_remaining(PHASE_SEARCHING)
        running_remaining = phase_timer.get_remaining(PHASE_RUNNING)
        driving_remaining = phase_timer.get_remaining(PHASE_DRIVING)
        print(
            "[Timer] 阶段剩余时间 | "
            f"总计={_format_phase_seconds(total_remaining)} | "
            f"搜房={_format_phase_seconds(searching_remaining)} | "
            f"跑图={_format_phase_seconds(running_remaining)} | "
            f"开车={_format_phase_seconds(driving_remaining)}"
        )
        next_phase_report_time = now + 5.0

    if phase_timer.all_done() and not all_done_reported:
        total_remaining = phase_timer.get_total_remaining()
        searching_remaining = phase_timer.get_remaining(PHASE_SEARCHING)
        running_remaining = phase_timer.get_remaining(PHASE_RUNNING)
        driving_remaining = phase_timer.get_remaining(PHASE_DRIVING)
        print(
            "[Timer] 30 分钟总时长已圆满结束 | "
            f"总计剩余={_format_phase_seconds(total_remaining)} | "
            f"搜房剩余={_format_phase_seconds(searching_remaining)} | "
            f"跑图剩余={_format_phase_seconds(running_remaining)} | "
            f"开车剩余={_format_phase_seconds(driving_remaining)}"
        )
        all_done_reported = True


def on_stage(w: "FrameWorker"):
    global start_game, start_game_click_time, final_shutdown_pending, searching_view_synced

    previous_stage = phase_timer.last_stage
    stage_events = phase_timer.sync_stage(w.current_stage)
    stage_events |= phase_timer.refresh()

    if previous_stage == "开车阶段" and w.current_stage == "跑图阶段":
        finding_car = driving_manager.consume_running_transition_finding_car(
            default=phase_timer.need_drive()
        )
        running_manager.notify_vehicle_exit(finding_car=finding_car)

    if previous_stage == "搜房阶段" and w.current_stage == "跑图阶段":
        running_manager.notify_searching_exit(finding_car=phase_timer.need_drive())

    if "landed" in stage_events and not phase_timer.all_done():
        if phase_timer.start_game_time is not None:
            running_manager.set_game_time(phase_timer.start_game_time)
            driving_manager.set_game_time(phase_timer.start_game_time)

    if w.current_stage in {"搜房阶段", "跑图阶段", "开车阶段"}:
        maybe_report_phase_remaining()

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

        if start_game and start_game_click_time is not None:
            if time.time() - start_game_click_time >= START_GAME_VERIFY_DELAY:
                if w.get_info("开始游戏"):
                    print("[StartGame] 开始游戏按钮仍可识别，判定上次点击未生效，准备重试")
                    start_game = False
                    start_game_click_time = None

        if w.get_info("房子"):
            if not start_game:
                w.click("开始游戏")
                start_game = True
                start_game_click_time = time.time()
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
            start_game_click_time = None
            return

    if w.current_stage == "跳伞阶段":
        parachute_manager.process(w)
        return

    if w.current_stage == "搜房阶段":
        handle_sp_start(w)
        if should_abort_searching(w):
            return

        searching_view_synced = True
        searching_house_manager.process(w)
        return
        # house_scene = w.get_info("house_scene")
        # if isinstance(house_scene, (list, tuple)) and len(house_scene) == 1:
        #     house_scene = house_scene[0]
        #
        # if house_exit_manager.process(w):
        #     w.stop()
        # return

    if w.current_stage == "跑图阶段":
        if searching_view_synced:
            running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
            searching_view_synced = False

        handle_sp_start(w)

        if phase_timer.all_done():
            finalize_automation(w)
            return

        running_manager.set_drive_required(phase_timer.need_drive())
        running_manager.process(w)
        return

    if w.current_stage == "开车阶段":
        driving_manager.set_running_fallback_enabled(not phase_timer.is_completed(PHASE_RUNNING))

        if "enter_开车" in stage_events:
            driving_manager.set_remaining_drive_time(phase_timer.get_remaining(PHASE_DRIVING))
            entry_source = running_manager.consume_vehicle_entry_source()
            if entry_source == RunningManager.VEHICLE_ENTRY_ROADSIDE:
                driving_manager.skip_initial_exit_garage("roadside vehicle")

        if phase_timer.is_completed(PHASE_DRIVING):
            driving_manager.set_remaining_drive_time(0)

        driving_manager.process(w)
        return

    if w.current_stage == "结束阶段":
        if final_shutdown_pending:
            handle_sp_stop(w)
            prepare_rank_finish_for_lobby(w)
            w.click("设置")
            time.sleep(1)
            w.click("返回大厅")
            time.sleep(1)
            w.click("确定")
            time.sleep(3)
            w.change_stage("关闭弹窗阶段")
            return

        handle_sp_stop(w)
        prepare_rank_finish_for_lobby(w)

        w.click("设置")
        time.sleep(1)
        w.click("返回大厅")
        time.sleep(1)
        w.click("确定")
        time.sleep(3)
        w.change_stage("开始游戏阶段")
