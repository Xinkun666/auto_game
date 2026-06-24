import os
import time
from typing import TYPE_CHECKING

from aw.autogame.tools.Utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.parachute_manager import ParachuteManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.running_manager import RunningManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.driving_manager import DrivingManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_scene_search_manager import (
    HouseSceneSearchManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_exit_manager import (
    HouseExitManager,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.phase_time_manager import (
    PHASE_DRIVING,
    PHASE_RUNNING,
    PHASE_SEARCHING,
    PhaseTimeManager,
    PhaseTimeReporter,
    parse_case_loop_count,
)
from aw.autogame.tools.GameLaunchProfile import should_use_sp_recording_for_profile

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
    PHASE_SEARCHING: 5,
    PHASE_RUNNING: 5,
    PHASE_DRIVING: 5,
}

DROP_TARGET_R_CITY = (990, 757)
DROP_TARGET_R_CITY_SEARCH_START = (986, 759)
DROP_TARGET_GARAGE = DROP_TARGET_R_CITY
DROP_TARGET_CENTER = DROP_TARGET_R_CITY
DROP_TARGET_RUNNING_AFTER_SEARCH = (1094, 790)
STAGE_PRIORITY_JUMP_FORWARD_Y_BIAS = -400
STAGE_PRIORITY_JUMP_FORWARD_DURA = 100
STAGE_PRIORITY_JUMP_FORWARD_WAIT = 300
STAGE_PRIORITY_JUMP_SETTLE_SECONDS = 0.2
STAGE_PRIORITY_JUMP_REPEAT_SUPPRESS_SECONDS = 1.0
RANK_FINISH_SPECTATE_WAIT_SECONDS = 2.0
SP_SAVE_LONG_PRESS_MS = 3000
SP_RECORDING_ENABLED = should_use_sp_recording_for_profile(
    os.environ.get("AUTOGAME_TEST_PROFILE")
)
START_GAME_VERIFY_DELAY = 5.0
CLOSE_POPUP_SETTLE_DELAY = 1.0
LOBBY_CONFIRM_INTERVAL = 0.7
LOBBY_CONFIRM_REQUIRED = 2
CLOSE_POPUP_INFOS = (
    "关闭公告",
    "提示",
    "对局结束",
    "关闭预约",
    "关闭",
    "回归",
    "关闭活动",
    "关闭新玩法",
    "关闭活动2",
    "关闭活动3",
)

start_game = False
start_game_click_time = None
final_shutdown_pending = False
rank_finish_pending = False
searching_view_synced = False
searching_phase_finishing = False
searching_to_running_notified = False
searching_exit_retry_count = 0
last_popup_close_time = 0.0
lobby_house_confirm_count = 0
last_stage_priority_jump_time = None
parachute_manager = ParachuteManager()
running_manager = RunningManager()
driving_manager = DrivingManager()
searching_house_manager = HouseSceneSearchManager()
house_exit_manager = HouseExitManager()
phase_timer = PhaseTimeManager(PHASE_DURATIONS, PHASE_STAGE_MAP)
phase_timer.configure_case_loop_count(
    parse_case_loop_count(os.environ.get("AUTOGAME_SINGLE_CASE_LOOPS"))
)
phase_reporter = PhaseTimeReporter()


def pause_sp_after_death(w: "FrameWorker"):
    if not SP_RECORDING_ENABLED:
        return
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_stopped()


running_manager.pause_sp_callback = pause_sp_after_death
driving_manager.pause_sp_callback = pause_sp_after_death


def prepare_round():
    global searching_view_synced, rank_finish_pending, last_stage_priority_jump_time
    global searching_phase_finishing, searching_to_running_notified, searching_exit_retry_count

    phase_timer.start_new_round()
    phase_reporter.reset()
    searching_view_synced = False
    searching_phase_finishing = False
    searching_to_running_notified = False
    searching_exit_retry_count = 0
    rank_finish_pending = False
    last_stage_priority_jump_time = None

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
    if not SP_RECORDING_ENABLED:
        return
    if not phase_timer.should_start_sp():
        return
    if phase_timer.start_game_time is not None:
        running_manager.set_game_time(phase_timer.start_game_time)
        driving_manager.set_game_time(phase_timer.start_game_time)
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_started()


def handle_sp_stop(w: "FrameWorker"):
    if not SP_RECORDING_ENABLED:
        return
    if not phase_timer.sp_recording:
        return
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_stopped()


def _has_rank_finish_info(w: "FrameWorker") -> bool:
    return bool(w.get_info("个人排名")) or bool(w.get_info("队伍排名"))


def _has_death_finish_info(w: "FrameWorker") -> bool:
    return bool(w.get_info("变身")) or bool(w.get_info("红色血条"))


def _stop_active_motion(w: "FrameWorker"):
    for manager in (searching_house_manager, running_manager):
        stop_func = getattr(manager, "stop_auto_forward", None)
        if callable(stop_func):
            stop_func(w)

    cancel_drive = getattr(driving_manager, "_cancel_drive_auto_forward", None)
    if callable(cancel_drive):
        cancel_drive(w, "检测到死亡或排名界面，取消车辆自动前进")


def handle_terminal_state(w: "FrameWorker", context: str = "阶段入口") -> bool:
    global rank_finish_pending, searching_phase_finishing

    if _has_rank_finish_info(w):
        print(f"[Terminal] {context} 检测到个人排名或队伍排名，进入结束阶段")
        rank_finish_pending = True
        searching_phase_finishing = False
        _stop_active_motion(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    if _has_death_finish_info(w):
        print(f"[Terminal] {context} 检测到死亡界面，进入结束阶段")
        searching_phase_finishing = False
        _stop_active_motion(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    return False


def should_abort_searching(w: "FrameWorker"):
    if w.current_stage != "搜房阶段":
        return True

    if handle_terminal_state(w, "搜房阶段"):
        return True

    if searching_phase_finishing:
        return False

    if phase_timer.is_completed(PHASE_SEARCHING):
        print("[Timer] 搜房阶段计时已用完，强制切换到跑图阶段")
        finish_searching_and_enter_running(w, "搜房阶段计时已用完")
        return True

    return False


searching_house_manager.abort_callback = should_abort_searching
searching_house_manager.can_finish_callback = lambda w: phase_timer.is_completed(PHASE_SEARCHING)
running_manager.terminal_state_callback = handle_terminal_state
driving_manager.terminal_state_callback = handle_terminal_state


def recover_bad_landing_to_r_city(w: "FrameWorker", target, reason: str):
    global searching_view_synced, searching_to_running_notified

    route_target = tuple(target or DROP_TARGET_R_CITY)
    print(
        f"[Flow] 搜房落点异常，切到跑图阶段恢复到R城: "
        f"reason={reason}, target={route_target}"
    )
    searching_house_manager.stop_auto_forward(w)
    running_manager.start_forced_route(
        target=route_target,
        finish_stage="搜房阶段",
        reason=reason,
        arrival_distance=searching_house_manager.r_city_near_distance,
    )
    running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
    searching_view_synced = True
    searching_to_running_notified = True
    w.change_stage("跑图阶段")
    return True


def route_to_r_city_search_start(
    w: "FrameWorker",
    target,
    reason: str,
    arrival_distance: float,
):
    global searching_view_synced, searching_to_running_notified

    route_target = tuple(target or DROP_TARGET_R_CITY_SEARCH_START)
    print(
        f"[Flow] 搜房前置跑图，先到R城搜房起点: "
        f"reason={reason}, target={route_target}, arrival={arrival_distance:.1f}"
    )
    searching_house_manager.stop_auto_forward(w)
    running_manager.start_forced_route(
        target=route_target,
        finish_stage="搜房阶段",
        reason=reason,
        arrival_distance=arrival_distance,
    )
    running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
    searching_view_synced = True
    searching_to_running_notified = True
    w.change_stage("跑图阶段")
    return True


def _should_find_car_after_searching() -> bool:
    return (
        not phase_timer.is_completed(PHASE_DRIVING)
        and phase_timer.get_remaining(PHASE_DRIVING) > 0
    )


def finish_searching_and_enter_running(w: "FrameWorker", reason: str):
    global searching_view_synced, searching_phase_finishing, searching_to_running_notified
    global searching_exit_retry_count

    if searching_phase_finishing:
        return True

    searching_phase_finishing = True
    print(
        f"[Flow] 搜房结束: {reason} | "
        f"searching_remaining={phase_timer.get_remaining(PHASE_SEARCHING):.2f}s, "
        f"running_remaining={phase_timer.get_remaining(PHASE_RUNNING):.2f}s, "
        f"driving_remaining={phase_timer.get_remaining(PHASE_DRIVING):.2f}s"
    )

    searching_house_manager.stop_auto_forward(w)
    w.refresh_frame()
    house_scene = searching_house_manager._get_house_scene(w)
    if house_scene == searching_house_manager.HOUSE_INDOOR:
        searching_exit_retry_count += 1
        print(
            f"[Flow] 搜房结束时仍在屋内，先执行搜房出房策略，再切跑图 "
            f"(retry={searching_exit_retry_count})"
        )
        exit_ok = searching_house_manager._exit_house(w)
        if w.current_stage != "搜房阶段":
            searching_phase_finishing = False
            return True
        w.refresh_frame()
        if not exit_ok and searching_house_manager._get_house_scene(w) == searching_house_manager.HOUSE_INDOOR:
            print("[Flow] 搜房结束出房未确认，暂不切跑图；下一帧继续用搜房阶段出房")
            searching_phase_finishing = False
            return True
    else:
        searching_exit_retry_count = 0

    finding_car = _should_find_car_after_searching()
    running_manager.notify_searching_exit(finding_car=finding_car)
    running_manager.set_drive_required(finding_car)
    if phase_timer.start_game_time is not None:
        running_manager.set_game_time(phase_timer.start_game_time)
    searching_house_manager.reset()
    searching_view_synced = True
    searching_to_running_notified = True
    searching_exit_retry_count = 0
    searching_phase_finishing = False
    w.change_stage("跑图阶段")
    return True


searching_house_manager.finish_callback = finish_searching_and_enter_running


def finalize_automation(w: "FrameWorker"):
    global final_shutdown_pending

    if w.current_stage == "跑图阶段":
        running_manager.stop_auto_forward(w)

    if SP_RECORDING_ENABLED and not phase_timer.sp_saved:
        w.click_down("sp", dura=SP_SAVE_LONG_PRESS_MS)
        time.sleep(1)
        phase_timer.mark_sp_stopped()
        phase_timer.mark_sp_saved()

    final_shutdown_pending = True
    w.change_stage("结束阶段")


def finish_case_loop_or_finalize(w: "FrameWorker"):
    if not phase_timer.has_next_case_loop():
        finalize_automation(w)
        return

    if w.current_stage == "跑图阶段":
        running_manager.stop_auto_forward(w)

    next_loop_message = (
        "暂停 sp，返回大厅后继续下一次循环"
        if SP_RECORDING_ENABLED
        else "返回大厅后继续下一次循环"
    )
    print(f"[Timer] 第 {phase_timer.case_loop_index}/{phase_timer.case_loop_count} 次循环已完成，{next_loop_message}")
    handle_sp_stop(w)
    phase_timer.advance_case_loop()
    w.change_stage("结束阶段")


def finalize_after_lobby(w: "FrameWorker"):
    global final_shutdown_pending
    final_shutdown_pending = False
    w.stop()


def reset_lobby_confirm(mark_popup_closed: bool = False):
    global last_popup_close_time, lobby_house_confirm_count

    lobby_house_confirm_count = 0
    if mark_popup_closed:
        last_popup_close_time = time.time()


def click_popup_and_refresh(w: "FrameWorker", target):
    w.click(target)
    reset_lobby_confirm(mark_popup_closed=True)
    w.refresh_frame()


def has_close_popup_info(w: "FrameWorker") -> bool:
    return any(w.get_info(info_name) for info_name in CLOSE_POPUP_INFOS)


def confirm_lobby_after_popups(w: "FrameWorker") -> bool:
    global lobby_house_confirm_count

    if last_popup_close_time > 0 and time.time() - last_popup_close_time < CLOSE_POPUP_SETTLE_DELAY:
        return False

    time.sleep(LOBBY_CONFIRM_INTERVAL)
    w.refresh_frame()

    if has_close_popup_info(w):
        if lobby_house_confirm_count:
            print("[Popup] 大厅确认过程中又检测到弹窗，取消本次房子图标确认")
        reset_lobby_confirm()
        return False

    if not w.get_info("房子"):
        if lobby_house_confirm_count:
            print("[Popup] 房子图标未连续稳定出现，取消本次大厅确认")
        reset_lobby_confirm()
        return False

    lobby_house_confirm_count += 1
    print(f"[Popup] 房子图标稳定确认 {lobby_house_confirm_count}/{LOBBY_CONFIRM_REQUIRED}")
    return lobby_house_confirm_count >= LOBBY_CONFIRM_REQUIRED


def prepare_rank_finish_for_lobby(w: "FrameWorker"):
    global rank_finish_pending

    if not rank_finish_pending and not _has_rank_finish_info(w):
        return

    print("[End] 检测到排名界面，等待2s后点击观战对手再返回大厅")
    time.sleep(RANK_FINISH_SPECTATE_WAIT_SECONDS)
    w.click("观战对手")
    w.refresh_frame()
    rank_finish_pending = False


def maybe_report_phase_remaining():
    phase_reporter.maybe_report(phase_timer)


def handle_priority_stage_jump_forward(w: "FrameWorker", stage_label: str) -> bool:
    global last_stage_priority_jump_time

    if not w.get_info("跳跃"):
        return False

    now = time.monotonic()
    if (
        last_stage_priority_jump_time is not None
        and now - last_stage_priority_jump_time < STAGE_PRIORITY_JUMP_REPEAT_SUPPRESS_SECONDS
    ):
        print(f"[Jump] {stage_label} 刚执行过跳跃前推，跳过重复点击")
        return True

    print(f"[Jump] {stage_label} 检测到跳跃按钮，第一优先级点击跳跃并前推")
    searching_house_manager.stop_auto_forward(w)
    running_manager.stop_auto_forward(w)
    w.click("跳跃")
    last_stage_priority_jump_time = now
    time.sleep(STAGE_PRIORITY_JUMP_SETTLE_SECONDS)
    w.tap_single(
        "摇杆",
        y_bias=STAGE_PRIORITY_JUMP_FORWARD_Y_BIAS,
        dura=STAGE_PRIORITY_JUMP_FORWARD_DURA,
        wait=STAGE_PRIORITY_JUMP_FORWARD_WAIT,
    )
    w.refresh_frame()
    searching_house_manager.history_locations = []
    running_manager.history_locations = []
    return True


def on_stage(w: "FrameWorker"):
    global start_game, start_game_click_time, final_shutdown_pending
    global searching_view_synced, searching_to_running_notified

    previous_stage = phase_timer.last_stage
    stage_events = phase_timer.sync_stage(w.current_stage)
    stage_events |= phase_timer.refresh()

    if previous_stage == "开车阶段" and w.current_stage == "跑图阶段":
        finding_car = driving_manager.consume_running_transition_finding_car(
            default=phase_timer.need_drive()
        )
        running_manager.notify_vehicle_exit(finding_car=finding_car)

    if previous_stage == "搜房阶段" and w.current_stage == "跑图阶段":
        if searching_to_running_notified:
            searching_to_running_notified = False
        else:
            running_manager.notify_searching_exit(finding_car=_should_find_car_after_searching())

    if "landed" in stage_events and not phase_timer.all_done():
        if phase_timer.start_game_time is not None:
            running_manager.set_game_time(phase_timer.start_game_time)
            driving_manager.set_game_time(phase_timer.start_game_time)

    if w.current_stage in {"搜房阶段", "跑图阶段", "开车阶段"}:
        if handle_terminal_state(w, f"{w.current_stage}入口"):
            return
        maybe_report_phase_remaining()

    if w.current_stage == "关闭弹窗阶段":
        if w.get_info("关闭公告"):
            click_popup_and_refresh(w, w.get_info("关闭公告"))
            return

        if w.get_info("提示"):
            click_popup_and_refresh(w, "取消")
            return

        if w.get_info('对局结束'):
            click_popup_and_refresh(w, "确定")
            return

        if w.get_info('关闭预约'):
            click_popup_and_refresh(w, w.get_info("关闭预约"))
            return

        if w.get_info("关闭"):
            click_popup_and_refresh(w, w.get_info("关闭"))
            return

        if w.get_info("回归"):
            click_popup_and_refresh(w, w.get_info("回归"))
            return

        if w.get_info("关闭活动"):
            click_popup_and_refresh(w, w.get_info("关闭活动"))
            return

        if w.get_info("关闭新玩法"):
            click_popup_and_refresh(w, w.get_info("关闭新玩法"))
            return

        if w.get_info("关闭活动2"):
            click_popup_and_refresh(w, w.get_info("关闭活动2"))
            return

        if w.get_info("关闭活动3"):
            click_popup_and_refresh(w, w.get_info("关闭活动3"))
            return

        if confirm_lobby_after_popups(w):
            if final_shutdown_pending:
                finalize_after_lobby(w)
                return
            reset_lobby_confirm()
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
        time.sleep(1)
        w.refresh_frame()
        if w.get_info('自动匹配'):
            w.click(w.get_info('自动匹配'))
        time.sleep(1)
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

        if handle_priority_stage_jump_forward(w, "搜房阶段"):
            return

        searching_view_synced = True
        searching_house_manager.process(w)
        return

    if w.current_stage == "跑图阶段":
        if searching_view_synced:
            running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
            searching_view_synced = False

        handle_sp_start(w)

        if phase_timer.all_done():
            finish_case_loop_or_finalize(w)
            return

        if handle_priority_stage_jump_forward(w, "跑图阶段"):
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
