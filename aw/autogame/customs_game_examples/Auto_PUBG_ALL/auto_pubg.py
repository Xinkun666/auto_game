import os
import time
from typing import TYPE_CHECKING

from aw.autogame.tools.Utils import *
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.parachute_manager import ParachuteManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.running_manager import RunningManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.driving_manager import DrivingManager
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_search_manager import (
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
    load_phase_durations_from_config,
    parse_case_loop_count,
)
from aw.autogame.tools.GameLaunchProfile import should_use_sp_recording_for_profile
from aw.autogame.tools.Utils import _read_autogame_config
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.structured_log import autogame_print as print

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

DROP_TARGET_R_CITY = (990, 757)
DROP_TARGET_R_CITY_SEARCH_START = (986, 759)
DROP_TARGET_GARAGE = DROP_TARGET_R_CITY
DROP_TARGET_CENTER = DROP_TARGET_R_CITY
DROP_TARGET_RUNNING_AFTER_SEARCH = (1094, 790)
STAGE_PRIORITY_JUMP_FORWARD_Y_BIAS = -400
STAGE_PRIORITY_JUMP_FORWARD_DURA = 100
STAGE_PRIORITY_JUMP_FORWARD_WAIT = 300
STAGE_PRIORITY_JUMP_SETTLE_SECONDS = 0.2
RANK_FINISH_SPECTATE_WAIT_SECONDS = 2.0
SP_SAVE_LONG_PRESS_MS = 3000
SP_RECORDING_ENABLED = False
START_GAME_VERIFY_DELAY = 5.0
CLOSE_POPUP_SETTLE_DELAY = 1.0
LOBBY_CONFIRM_INTERVAL = 0.7
LOBBY_CONFIRM_REQUIRED = 2
CLOSE_POPUP_INFOS = (
    "关闭公告",
    "重新进入比赛",
    "对局结束",
    "关闭预约",
    "关闭",
    "回归",
    "关闭活动",
    "关闭新玩法",
    "关闭活动2",
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
PHASE_DURATIONS = None
parachute_manager = None
running_manager = None
driving_manager = None
searching_house_manager = None
house_exit_manager = None
phase_timer = None
phase_reporter = None
_runtime_initialized = False


def initialize_runtime():
    global SP_RECORDING_ENABLED, PHASE_DURATIONS, _runtime_initialized
    global parachute_manager, running_manager, driving_manager
    global searching_house_manager, house_exit_manager, phase_timer, phase_reporter

    if _runtime_initialized:
        return

    SP_RECORDING_ENABLED = should_use_sp_recording_for_profile(
        os.environ.get("AUTOGAME_TEST_PROFILE")
    )
    PHASE_DURATIONS = load_phase_durations_from_config(_read_autogame_config())

    parachute_manager = ParachuteManager()
    running_manager = RunningManager()
    driving_manager = DrivingManager()
    searching_house_manager = HouseSceneSearchManager()
    searching_house_manager.configure_r_city_landing_target(DROP_TARGET_R_CITY)
    searching_house_manager.configure_r_city_pre_search_target(
        DROP_TARGET_R_CITY_SEARCH_START,
        arrival_distance=3.0,
    )
    house_exit_manager = HouseExitManager()
    phase_timer = PhaseTimeManager(PHASE_DURATIONS, PHASE_STAGE_MAP)
    print(
        "[Timer] 阶段时间配置: "
        f"搜房={phase_timer.get_duration_minutes_label(PHASE_SEARCHING)}分钟, "
        f"跑图={phase_timer.get_duration_minutes_label(PHASE_RUNNING)}分钟, "
        f"开车={phase_timer.get_duration_minutes_label(PHASE_DRIVING)}分钟"
    )
    phase_timer.configure_case_loop_count(
        parse_case_loop_count(os.environ.get("AUTOGAME_SINGLE_CASE_LOOPS"))
    )
    phase_reporter = PhaseTimeReporter()

    running_manager.pause_sp_callback = pause_sp_after_death
    driving_manager.pause_sp_callback = pause_sp_after_death
    searching_house_manager.abort_callback = should_abort_searching
    searching_house_manager.can_finish_callback = lambda w: phase_timer.is_completed(PHASE_SEARCHING)
    running_manager.terminal_state_callback = handle_terminal_state
    driving_manager.terminal_state_callback = handle_terminal_state
    searching_house_manager.r_city_recovery_route_callback = recover_bad_landing_to_r_city
    searching_house_manager.r_city_pre_search_route_callback = route_to_r_city_search_start
    searching_house_manager.r_city_entry_route_callback = route_to_r_city_entry_point
    searching_house_manager.finish_callback = finish_searching_and_enter_running

    _runtime_initialized = True


def _require_runtime():
    initialize_runtime()


def pause_sp_after_death(w: "FrameWorker"):
    _require_runtime()
    if not SP_RECORDING_ENABLED:
        return
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_stopped()


def prepare_round():
    global searching_view_synced, rank_finish_pending
    global searching_phase_finishing, searching_to_running_notified, searching_exit_retry_count

    _require_runtime()
    phase_timer.start_new_round()
    phase_reporter.reset()
    searching_view_synced = False
    searching_phase_finishing = False
    searching_to_running_notified = False
    searching_exit_retry_count = 0
    rank_finish_pending = False

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
    _require_runtime()
    if not SP_RECORDING_ENABLED:
        return
    if not phase_timer.should_start_sp():
        return
    set_stage_decision(
        w,
        "当前帧达到 sp 录制启动条件",
        "点击 sp 开始录制本局片段",
        action="开始 sp 录制",
        method="w.click(sp)",
        result="后续帧继续当前阶段逻辑",
    )
    if phase_timer.start_game_time is not None:
        running_manager.set_game_time(phase_timer.start_game_time)
        driving_manager.set_game_time(phase_timer.start_game_time)
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_started()


def handle_sp_stop(w: "FrameWorker"):
    _require_runtime()
    if not SP_RECORDING_ENABLED:
        return
    if not phase_timer.sp_recording:
        return
    set_stage_decision(
        w,
        "当前帧需要停止 sp 录制",
        "点击 sp 停止保存片段",
        action="停止 sp 录制",
        method="w.click(sp)",
        result="后续继续结束/回大厅流程",
    )
    w.click("sp")
    time.sleep(0.5)
    phase_timer.mark_sp_stopped()


def _has_rank_finish_info(w: "FrameWorker") -> bool:
    return bool(w.get_info("个人排名")) or bool(w.get_info("队伍排名"))


def _has_death_finish_info(w: "FrameWorker") -> bool:
    return bool(w.get_info("变身")) or bool(w.get_info("红色血条"))


def _stop_active_motion(w: "FrameWorker"):
    _require_runtime()
    for manager in (searching_house_manager, running_manager):
        stop_func = getattr(manager, "stop_auto_forward", None)
        if callable(stop_func):
            stop_func(w)

    cancel_drive = getattr(driving_manager, "_cancel_drive_auto_forward", None)
    if callable(cancel_drive):
        cancel_drive(w, "检测到死亡或排名界面，取消车辆自动前进")


def handle_terminal_state(w: "FrameWorker", context: str = "阶段入口") -> bool:
    global rank_finish_pending, searching_phase_finishing

    _require_runtime()
    if _has_rank_finish_info(w):
        print(f"[Terminal] {context} 检测到个人排名或队伍排名，进入结束阶段")
        set_stage_decision(
            w,
            f"{context} 检测到个人排名或队伍排名",
            "停止当前移动/录制并进入结束阶段",
            action="进入结束阶段",
            method="_stop_active_motion(); handle_sp_stop(); w.change_stage(结束阶段)",
            result="下一帧执行返回大厅流程",
        )
        rank_finish_pending = True
        searching_phase_finishing = False
        _stop_active_motion(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    if _has_death_finish_info(w):
        print(f"[Terminal] {context} 检测到死亡界面，进入结束阶段")
        set_stage_decision(
            w,
            f"{context} 检测到死亡界面",
            "停止当前移动/录制并进入结束阶段",
            action="进入结束阶段",
            method="_stop_active_motion(); handle_sp_stop(); w.change_stage(结束阶段)",
            result="下一帧执行返回大厅流程",
        )
        searching_phase_finishing = False
        _stop_active_motion(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    return False


def should_abort_searching(w: "FrameWorker"):
    _require_runtime()
    if w.current_stage != "搜房阶段":
        return True

    if handle_terminal_state(w, "搜房阶段"):
        return True

    if searching_phase_finishing:
        return False

    if phase_timer.is_completed(PHASE_SEARCHING):
        print(
            f"[Timer] 搜房阶段 {phase_timer.get_duration_minutes_label(PHASE_SEARCHING)} 分钟已用完，"
            "强制切换到跑图阶段"
        )
        finish_searching_and_enter_running(w, "搜房阶段计时已用完")
        return True

    return False


def recover_bad_landing_to_r_city(w: "FrameWorker", target, reason: str):
    global searching_view_synced, searching_to_running_notified

    _require_runtime()
    route_target = tuple(target or DROP_TARGET_R_CITY)
    print(
        f"[Flow] 搜房落点异常，切到跑图阶段恢复到R城: "
        f"reason={reason}, target={route_target}"
    )
    set_stage_decision(
        w,
        f"搜房落点异常，需要恢复到R城：{reason}",
        "切到跑图阶段，强制导航到R城恢复目标",
        action="切换跑图恢复R城",
        method="running_manager.start_forced_route(); w.change_stage(跑图阶段)",
        result="跑图阶段接管恢复路线",
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

    _require_runtime()
    route_target = tuple(target or DROP_TARGET_R_CITY_SEARCH_START)
    print(
        f"[Flow] 搜房前置跑图，先到R城搜房起点: "
        f"reason={reason}, target={route_target}, arrival={arrival_distance:.1f}"
    )
    set_stage_decision(
        w,
        f"需要先前往R城搜房起点：{reason}",
        "切到跑图阶段，强制导航到搜房起点后再回来搜房",
        action="前置跑图到搜房起点",
        method="running_manager.start_forced_route(); w.change_stage(跑图阶段)",
        result="到达后回到搜房阶段",
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


def route_to_r_city_entry_point(
    w: "FrameWorker",
    target,
    reason: str,
    arrival_distance: float,
    approach_location=None,
):
    global searching_view_synced, searching_to_running_notified

    _require_runtime()
    route_target = tuple(target or DROP_TARGET_R_CITY_SEARCH_START)
    print(
        f"[Flow] 落地后最近入门点仍较远，先按跑图阶段冲到入门点附近: "
        f"reason={reason}, target={route_target}, arrival={arrival_distance:.1f}"
    )
    set_stage_decision(
        w,
        f"最近入门点较远，先跑图到入门点附近：{reason}",
        "切到跑图阶段，强制导航到锁定入门点附近后回到搜房",
        action="跑图到入门点附近",
        method="running_manager.start_forced_route(); w.change_stage(跑图阶段)",
        result="到达入门点附近后继续搜房进门",
    )
    searching_house_manager.stop_auto_forward(w)
    running_manager.start_forced_route(
        target=route_target,
        finish_stage="搜房阶段",
        reason=reason,
        arrival_distance=arrival_distance,
        approach_target=approach_location,
        target_resolver=searching_house_manager.get_live_r_city_entry_for_route,
    )
    running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
    searching_view_synced = True
    searching_to_running_notified = True
    w.change_stage("跑图阶段")
    return True


def _should_find_car_after_searching() -> bool:
    _require_runtime()
    return (
        not phase_timer.is_completed(PHASE_DRIVING)
        and phase_timer.get_remaining(PHASE_DRIVING) > 0
    )


def finish_searching_and_enter_running(w: "FrameWorker", reason: str):
    global searching_view_synced, searching_phase_finishing, searching_to_running_notified
    global searching_exit_retry_count

    _require_runtime()
    if searching_phase_finishing:
        return True

    searching_phase_finishing = True
    print(
        f"[Flow] 搜房结束: {reason} | "
        f"searching_remaining={phase_timer.get_remaining(PHASE_SEARCHING):.2f}s, "
        f"running_remaining={phase_timer.get_remaining(PHASE_RUNNING):.2f}s, "
        f"driving_remaining={phase_timer.get_remaining(PHASE_DRIVING):.2f}s"
    )
    set_stage_decision(
        w,
        f"搜房结束：{reason}",
        "先确认是否在屋内，必要时出房，然后切换到跑图阶段",
        action="搜房结束并准备跑图",
        method="finish_searching_and_enter_running()",
        result="出房成功或已在室外后进入跑图阶段",
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
        set_stage_decision(
            w,
            f"搜房结束时仍在屋内，house_scene={house_scene}",
            "先执行出房策略，确认出房后再切跑图",
            action="先出房再跑图",
            method="searching_house_manager._exit_house()",
            result="未确认出房则下一帧继续搜房阶段出房",
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


def finalize_automation(w: "FrameWorker"):
    global final_shutdown_pending

    _require_runtime()
    set_stage_decision(
        w,
        "当前用例/所有循环已完成",
        "停止移动和录制，进入结束阶段返回大厅",
        action="进入结束阶段",
        method="finalize_automation(); w.change_stage(结束阶段)",
        result="结束阶段处理返回大厅",
    )
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
    _require_runtime()
    if not phase_timer.has_next_case_loop():
        finalize_automation(w)
        return

    set_stage_decision(
        w,
        f"第 {phase_timer.case_loop_index}/{phase_timer.case_loop_count} 次循环已完成",
        "返回大厅后继续下一次循环",
        action="进入结束阶段并准备下一轮",
        method="phase_timer.advance_case_loop(); w.change_stage(结束阶段)",
        result="结束阶段返回大厅后重新开始",
    )
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


def set_stage_decision(
    w: "FrameWorker",
    observation: str,
    decision: str,
    action: str = None,
    method: str = "",
    result: str = "等待下一帧确认执行结果",
    target: str = None,
):
    setter = getattr(w, "set_frame_decision", None)
    if not callable(setter):
        return
    setter(
        observation=observation,
        target=target or w.current_stage,
        decision=decision,
        action=action or decision,
        method=method,
        result=result,
    )


def click_popup_info_if_visible(w: "FrameWorker", info_name: str, click_target=None) -> bool:
    target = w.get_info(info_name)
    if not target:
        return False
    control_target = click_target or target
    w.frame_log(f"当前观察到{info_name}弹窗挡住流程，所以先点击{click_target or info_name}关闭它")
    set_stage_decision(
        w,
        f"当前帧出现{info_name}",
        f"决策：点击{info_name}",
        action=f"点击{info_name}",
        method=f"w.click({info_name})",
        result="等待下一帧确认弹窗关闭",
    )
    click_popup_and_refresh(w, control_target)
    return True


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
    _require_runtime()
    phase_reporter.maybe_report(phase_timer)


def _current_position_text(w: "FrameWorker") -> tuple[str, str]:
    location = w.get_info("location")
    if isinstance(location, (list, tuple)) and location:
        location_text = str(location[0])
    else:
        location_text = str(location)
    direction = w.get_info("direction")
    return location_text, str(direction)


def record_manager_decision(w: "FrameWorker", target: str, decision: str, method: str):
    location_text, direction_text = _current_position_text(w)
    w.frame_log(f"当前观察到{target}位置={location_text}、方位={direction_text}，所以交给{method}继续细分决策")
    w.set_frame_decision(
        observation=f"{target}：当前位置={location_text}，当前方位={direction_text}",
        target=target,
        decision=decision,
        action=decision,
        method=method,
        result="等待 manager 根据当前帧继续执行",
    )


def handle_priority_stage_jump_forward(w: "FrameWorker", stage_label: str) -> bool:
    _require_runtime()
    if not w.get_info("跳跃"):
        return False

    print(f"[Jump] {stage_label} 检测到跳跃按钮，第一优先级点击跳跃并前推")
    w.frame_log(f"当前观察到{stage_label}出现跳跃按钮，所以先点击跳跃并轻推摇杆越过障碍")
    w.set_frame_decision(
        observation=f"{stage_label}当前帧出现跳跃按钮",
        target=stage_label,
        decision="点击跳跃并轻微前推",
        action="点击跳跃并轻微前推",
        method="w.click(跳跃) + w.tap_single(摇杆)",
        result="等待下一帧确认是否越过障碍",
    )
    searching_house_manager.stop_auto_forward(w)
    running_manager.stop_auto_forward(w)
    w.click("跳跃")
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

    _require_runtime()
    w.frame_log(f"当前观察到阶段={w.current_stage}，所以进入总流程 on_stage 分发")
    previous_stage = phase_timer.last_stage
    stage_events = phase_timer.sync_stage(w.current_stage)
    stage_events |= phase_timer.refresh()

    if previous_stage == "开车阶段" and w.current_stage == "跑图阶段":
        w.frame_log("当前观察到阶段从开车切回跑图，所以同步车辆退出状态给跑图模块")
        finding_car = driving_manager.consume_running_transition_finding_car(
            default=phase_timer.need_drive()
        )
        running_manager.notify_vehicle_exit(finding_car=finding_car)

    if previous_stage == "搜房阶段" and w.current_stage == "跑图阶段":
        if searching_to_running_notified:
            w.frame_log("当前观察到搜房已经主动通知过跑图恢复，所以只清理通知标记")
            searching_to_running_notified = False
        else:
            w.frame_log("当前观察到搜房阶段切到跑图阶段，所以通知跑图模块接管寻车/进圈目标")
            running_manager.notify_searching_exit(finding_car=_should_find_car_after_searching())

    if "landed" in stage_events and not phase_timer.all_done():
        print("[Flow] 当前人物已经落地，接下来同步落地后的搜房/跑图/开车目标")
        w.frame_log("当前观察到已落地事件，所以同步跑图和开车模块的本局开始时间")
        if phase_timer.start_game_time is not None:
            running_manager.set_game_time(phase_timer.start_game_time)
            driving_manager.set_game_time(phase_timer.start_game_time)

    if w.current_stage in {"搜房阶段", "跑图阶段", "开车阶段"}:
        if handle_terminal_state(w, f"{w.current_stage}入口"):
            return
        maybe_report_phase_remaining()

    if w.current_stage == "关闭弹窗阶段":
        w.frame_log("当前观察到关闭弹窗阶段，所以按优先级检查公告、重进、结算和活动弹窗")
        if click_popup_info_if_visible(w, "关闭公告"):
            return

        if click_popup_info_if_visible(w, "重新进入比赛", "取消"):
            return

        if click_popup_info_if_visible(w, "对局结束", "确定已结束"):
            return

        if click_popup_info_if_visible(w, "关闭预约"):
            return

        if click_popup_info_if_visible(w, "关闭"):
            return

        if click_popup_info_if_visible(w, "回归"):
            return

        if click_popup_info_if_visible(w, "确定获得"):
            return

        if click_popup_info_if_visible(w, "关闭记忆"):
            return

        if click_popup_info_if_visible(w, "关闭活动"):
            return

        if click_popup_info_if_visible(w, "关闭新玩法"):
            return

        if click_popup_info_if_visible(w, "关闭活动2"):
            return

        if confirm_lobby_after_popups(w):
            if final_shutdown_pending:
                w.frame_log("当前观察到大厅已稳定且任务准备结束，所以停止本轮自动化")
                set_stage_decision(
                    w,
                    "关闭弹窗阶段已确认回到大厅，且本用例准备结束",
                    "停止当前用例循环",
                    action="停止任务",
                    method="finalize_after_lobby()",
                    result="结束当前自动化任务",
                )
                finalize_after_lobby(w)
                return
            w.frame_log("当前观察到大厅房子图标连续稳定，所以进入选择地图阶段")
            set_stage_decision(
                w,
                "关闭弹窗阶段已连续确认房子图标，说明大厅可操作",
                "切换到选择地图阶段",
                action="进入选择地图阶段",
                method="w.change_stage(选择地图阶段)",
                result="下一帧开始选择地图",
            )
            reset_lobby_confirm()
            w.change_stage("选择地图阶段")
            return

    if w.current_stage == "选择地图阶段":
        w.frame_log("当前观察到选择地图阶段，所以打开地图选择面板并准备切到海岛")
        set_stage_decision(
            w,
            "当前处于选择地图阶段",
            "依次点击地图、经典模式、切换，准备选择海岛并确认",
            action="打开地图选择面板",
            method="w.click(地图); w.click(经典模式); w.click(切换)",
            result="等待地图选项刷新",
        )
        w.click("地图")
        time.sleep(2)
        w.click("经典模式")
        time.sleep(2)
        w.click("切换")
        time.sleep(2)
        w.refresh_frame()

        if w.get_info("对号"):
            w.frame_log("当前观察到已有地图对号，所以先点击对号清理旧选择")
            set_stage_decision(
                w,
                "当前帧出现对号，说明已有选中项需要取消或确认切换",
                "点击对号，清理当前选中状态后再选择海岛",
                action="点击对号",
                method="w.click(对号)",
                result="继续选择海岛",
            )
            w.click(w.get_info("对号"))
            time.sleep(2)

        set_stage_decision(
            w,
            "地图选择面板已打开",
            "点击海岛并处理自动匹配选项，然后确定进入开始游戏阶段",
            action="选择海岛并确定",
            method="w.click(海岛); w.click(确定)",
            result="下一帧进入开始游戏阶段",
        )
        w.click("海岛")
        time.sleep(1)
        w.refresh_frame()
        if w.get_info('自动匹配'):
            w.frame_log("当前观察到自动匹配选项，所以点击它避免配置阻塞后续确认")
            set_stage_decision(
                w,
                "当前帧出现自动匹配选项",
                "点击自动匹配，关闭/切换该选项后继续确定地图",
                action="点击自动匹配",
                method="w.click(自动匹配)",
                result="继续点击确定",
            )
            w.click(w.get_info('自动匹配'))
        time.sleep(1)
        w.click("确定")
        w.change_stage("开始游戏阶段")
        return

    if w.current_stage == "开始游戏阶段":
        if w.get_info("加速礼包"):
            w.frame_log("当前观察到加速礼包弹窗，所以点击放弃后刷新继续找开始游戏")
            set_stage_decision(
                w,
                "当前帧出现加速礼包",
                "点击放弃，避免礼包弹窗阻塞开始游戏",
                action="点击放弃",
                method="w.click(放弃)",
                result="刷新后继续检查开始游戏按钮",
            )
            w.click("放弃")
            w.refresh_frame()

        if start_game and start_game_click_time is not None:
            if time.time() - start_game_click_time >= START_GAME_VERIFY_DELAY:
                if w.get_info("开始游戏"):
                    print("[StartGame] 开始游戏按钮仍可识别，判定上次点击未生效，准备重试")
                    w.frame_log("当前观察到点击开始游戏后按钮仍存在，所以判定上次点击未生效并准备重试")
                    set_stage_decision(
                        w,
                        "点击开始游戏后仍识别到开始游戏按钮",
                        "判定上次点击未生效，重置状态准备重试",
                        action="重置开始游戏点击状态",
                        method="start_game=False",
                        result="后续帧重新点击开始游戏",
                    )
                    start_game = False
                    start_game_click_time = None

        if w.get_info("房子"):
            if not start_game:
                w.frame_log("当前观察到大厅房子图标且尚未点击开始游戏，所以点击开始游戏")
                set_stage_decision(
                    w,
                    "当前帧出现房子图标，说明在大厅且可开始游戏",
                    "点击开始游戏",
                    action="点击开始游戏",
                    method="w.click(开始游戏)",
                    result="等待拳头/出生岛信号进入跳伞阶段",
                )
                w.click("开始游戏")
                start_game = True
                start_game_click_time = time.time()
            else:
                w.frame_log("当前观察到已点击开始游戏但仍在大厅，所以刷新画面等待进入出生岛")
                set_stage_decision(
                    w,
                    "已点击开始游戏，当前帧仍在大厅房子图标界面",
                    "刷新画面等待进入出生岛",
                    action="等待开始游戏生效",
                    method="w.refresh_frame()",
                    result="后续帧继续确认是否出现拳头",
                )
            w.refresh_frame()

        if w.get_info("提示"):
            w.frame_log("当前观察到提示弹窗，所以点击不提示和不需要避免阻塞匹配")
            set_stage_decision(
                w,
                "当前帧出现提示弹窗",
                "点击不提示和不需要，避免提示弹窗阻塞匹配",
                action="关闭提示弹窗",
                method="w.click(不提示); w.click(不需要)",
                result="继续等待进入出生岛",
            )
            w.click("不提示")
            time.sleep(1)
            w.click("不需要")
            time.sleep(1)

        if w.get_info("拳头"):
            w.frame_log("当前观察到拳头按钮，所以判断已经进入出生岛并准备切到跳伞阶段")
            set_stage_decision(
                w,
                "当前帧出现拳头，说明已经进入出生岛/游戏内",
                "初始化本局状态并切换到跳伞阶段",
                action="进入跳伞阶段",
                method="prepare_round(); w.change_stage(跳伞阶段)",
                result="下一帧开始监控R城距离",
            )
            prepare_round()
            w.change_stage("跳伞阶段")
            start_game = False
            start_game_click_time = None
            return

    if w.current_stage == "跳伞阶段":
        w.frame_log("当前观察到跳伞阶段，所以交给跳伞模块计算取消跟随、航线距离和跳伞时机")
        record_manager_decision(
            w,
            "跳伞阶段",
            "根据当前坐标计算到R城距离，继续监控跳伞时机",
            "parachute_manager.process(w)",
        )
        parachute_manager.process(w)
        return

    if w.current_stage == "搜房阶段":
        w.frame_log("当前观察到搜房阶段，所以先处理录制、终局和跳跃优先级，再交给搜房模块")
        handle_sp_start(w)
        if should_abort_searching(w):
            return

        if handle_priority_stage_jump_forward(w, "搜房阶段"):
            return

        searching_view_synced = True
        record_manager_decision(
            w,
            "搜房阶段",
            "根据当前房屋/门/入门点信息选择搜房、进门或出房动作",
            "searching_house_manager.process(w)",
        )
        searching_house_manager.process(w)
        return

    if w.current_stage == "跑图阶段":
        if searching_view_synced:
            w.frame_log("当前观察到搜房视角已同步标记，所以跑图先切回第一人称视角模式")
            running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
            searching_view_synced = False

        handle_sp_start(w)

        if phase_timer.all_done():
            w.frame_log("当前观察到搜房/跑图/开车阶段时间都已完成，所以结束本轮或准备最终收尾")
            finish_case_loop_or_finalize(w)
            return

        if handle_priority_stage_jump_forward(w, "跑图阶段"):
            return

        running_manager.set_drive_required(phase_timer.need_drive())
        w.frame_log("当前观察到跑图阶段仍需推进，所以更新是否需要找车并交给跑图模块")
        record_manager_decision(
            w,
            "跑图阶段",
            "继续跑图导航，保持自动前进/路线推进",
            "running_manager.process(w)",
        )
        running_manager.process(w)
        return

    if w.current_stage == "开车阶段":
        w.frame_log("当前观察到开车阶段，所以先同步剩余开车时间和上车来源，再交给驾驶模块")
        driving_manager.set_running_fallback_enabled(not phase_timer.is_completed(PHASE_RUNNING))

        if "enter_开车" in stage_events:
            w.frame_log("当前观察到刚进入开车阶段，所以刷新驾驶剩余时间并判断是否跳过首次出库")
            driving_manager.set_remaining_drive_time(phase_timer.get_remaining(PHASE_DRIVING))
            entry_source = running_manager.consume_vehicle_entry_source()
            if entry_source == RunningManager.VEHICLE_ENTRY_ROADSIDE:
                driving_manager.skip_initial_exit_garage("roadside vehicle")

        if phase_timer.is_completed(PHASE_DRIVING):
            w.frame_log("当前观察到开车阶段计时已完成，所以把剩余驾驶时间置零让驾驶模块准备收尾")
            driving_manager.set_remaining_drive_time(0)

        record_manager_decision(
            w,
            "开车阶段",
            "根据当前位置、方位、道路和车辆状态继续驾驶",
            "driving_manager.process(w)",
        )
        driving_manager.process(w)
        return

    if w.current_stage == "结束阶段":
        if final_shutdown_pending:
            w.frame_log("当前观察到结束阶段且最终停止标记已置位，所以返回大厅后进入关闭弹窗阶段确认收尾")
            set_stage_decision(
                w,
                "结束阶段且本用例准备最终停止",
                "停止录制/回到大厅/确认后进入关闭弹窗阶段",
                action="返回大厅并准备结束",
                method="handle_sp_stop(); 设置->返回大厅->确定退出比赛",
                result="关闭弹窗阶段确认大厅后停止任务",
            )
            handle_sp_stop(w)
            prepare_rank_finish_for_lobby(w)
            w.click("设置")
            time.sleep(1)
            w.click("返回大厅")
            time.sleep(1)
            w.click("确定退出比赛")
            time.sleep(3)
            w.change_stage("关闭弹窗阶段")
            return

        w.frame_log("当前观察到结束阶段，所以停止录制、返回大厅并准备下一轮开始")
        set_stage_decision(
            w,
            "当前处于结束阶段",
            "停止录制并返回大厅，然后进入关闭弹窗阶段处理结算/活动弹窗",
            action="返回大厅",
            method="handle_sp_stop(); 设置->返回大厅->确定退出比赛",
            result="下一帧处理大厅弹窗",
        )
        handle_sp_stop(w)
        prepare_rank_finish_for_lobby(w)

        w.click("设置")
        time.sleep(1)
        w.click("返回大厅")
        time.sleep(1)
        w.click("确定退出比赛")
        time.sleep(3)
        w.change_stage("开始游戏阶段")
