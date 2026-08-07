import logging
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
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_latest_house_search import (
    build_nanda_house_search_strategy,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_house_search_strategy import (
    NandaHouseSearchStrategy,
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
from aw.autogame.tools.GameLaunchProfile import (
    TEST_TYPE_MARATHON,
    should_use_sp_recording_for_profile,
)
from aw.autogame.tools.FrameLog import FrameLogType
from aw.autogame.tools.Utils import _read_project_config

LOGGER = logging.getLogger("AutoPUBG")

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
DROP_TARGET_R_CITY_HOUSE = (1042, 748)
DROP_TARGET_M_CITY_HOUSE = (1526, 1218)
DROP_TARGET_L_CITY_HOUSE = (1813, 822)
DROP_TARGET_G_TOWN_HOUSE = (563, 1003)
DROP_TARGETS_BY_CITY = {
    "R城": DROP_TARGET_R_CITY_HOUSE,
    "M城": DROP_TARGET_M_CITY_HOUSE,
    "L城": DROP_TARGET_L_CITY_HOUSE,
    "G镇": DROP_TARGET_G_TOWN_HOUSE,
}
DROP_TARGET_R_CITY_CAR_SEARCH = (1104, 790)
DROP_TARGET_L_CITY_CAR_SEARCH = (1731, 910)
DROP_TARGET_M_CITY_CAR_SEARCH = (1477, 1171)
DROP_TARGET_G_TOWN_CAR_SEARCH = (576, 1127)
DROP_CAR_SEARCH_TARGETS_BY_CITY = {
    "R城": DROP_TARGET_R_CITY_CAR_SEARCH,
    "L城": DROP_TARGET_L_CITY_CAR_SEARCH,
    "M城": DROP_TARGET_M_CITY_CAR_SEARCH,
    "G镇": DROP_TARGET_G_TOWN_CAR_SEARCH,
}
DROP_TARGET_GARAGE = DROP_TARGET_R_CITY
DROP_TARGET_CENTER = DROP_TARGET_R_CITY
STAGE_PRIORITY_JUMP_FORWARD_Y_BIAS = -400
STAGE_PRIORITY_JUMP_FORWARD_DURA = 100
STAGE_PRIORITY_JUMP_FORWARD_WAIT = 300
STAGE_PRIORITY_JUMP_SETTLE_SECONDS = 0.2
RANK_FINISH_SPECTATE_WAIT_SECONDS = 2.0
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


def handle_parachute_target_selected(region_name, target):
    if (
        parachute_manager is not None
        and parachute_manager.landing_stage == "跑图阶段"
    ):
        if running_manager is None:
            return
        finding_car = (
            phase_timer.need_drive()
            if phase_timer is not None
            else True
        )
        running_manager.notify_searching_exit(
            finding_car=finding_car,
            search_region=region_name,
        )
        return

    if searching_house_manager is None:
        return
    searching_house_manager.configure_house_region(region_name, target)
    searching_house_manager.configure_r_city_landing_target(target)


def initialize_runtime():
    global SP_RECORDING_ENABLED, PHASE_DURATIONS, _runtime_initialized
    global parachute_manager, running_manager, driving_manager
    global searching_house_manager, house_exit_manager, phase_timer, phase_reporter

    if _runtime_initialized:
        return

    SP_RECORDING_ENABLED = should_use_sp_recording_for_profile(
        os.environ.get("AUTOGAME_TEST_PROFILE")
    )
    autogame_config = _read_project_config("Auto_PUBG_ALL")
    PHASE_DURATIONS = load_phase_durations_from_config(autogame_config)

    parachute_config = autogame_config.get("parachute", {})
    if not isinstance(parachute_config, dict):
        parachute_config = {}
    parachute_manager = ParachuteManager(
        route_max_distance=parachute_config.get("route_max_distance")
    )
    running_manager = RunningManager()
    driving_manager = DrivingManager()
    try:
        nanda_search_strategy = build_nanda_house_search_strategy(autogame_config)
        if nanda_search_strategy.enabled:
            LOGGER.info(
                "[NandaPreload] 自动化启动前检查 DINOv3、MLP、房型索引和"
                "全部模板门窗结构..."
            )
            preload_error = nanda_search_strategy.validate_ready()
            if preload_error is None:
                LOGGER.info(
                    "[NandaPreload] 南大房型匹配启动预检完成；"
                    "全部模板门窗结构已在本地就绪，允许启动自动化。"
                )
            else:
                LOGGER.error(
                    "[NandaPreload] 南大房型匹配启动预检失败：%s；"
                    "本轮禁用南大管线，继续原搜房逻辑。",
                    preload_error.message,
                )
                nanda_search_strategy = NandaHouseSearchStrategy()
    except Exception as exc:
        LOGGER.exception(
            "[NandaPreload] 南大策略构建/预检异常: %s；"
            "本轮禁用南大管线，继续原搜房逻辑。",
            exc,
        )
        nanda_search_strategy = NandaHouseSearchStrategy()
    searching_house_manager = HouseSceneSearchManager(
        nanda_search_strategy=nanda_search_strategy
    )
    parachute_manager.target_selected_callback = handle_parachute_target_selected
    searching_house_manager.configure_r_city_landing_target(DROP_TARGET_R_CITY)
    searching_house_manager.configure_r_city_pre_search_target(
        DROP_TARGET_R_CITY_SEARCH_START,
        arrival_distance=3.0,
    )
    house_exit_manager = HouseExitManager()
    phase_timer = PhaseTimeManager(PHASE_DURATIONS, PHASE_STAGE_MAP)
    phase_timer.configure_case_loop_count(
        parse_case_loop_count(os.environ.get("AUTOGAME_SINGLE_CASE_LOOPS"))
    )
    phase_reporter = PhaseTimeReporter()

    running_manager.pause_sp_callback = pause_sp_after_death
    driving_manager.pause_sp_callback = pause_sp_after_death
    searching_house_manager.abort_callback = should_abort_searching
    searching_house_manager.replay_abort_callback = should_abort_nanda_replay
    searching_house_manager.can_finish_callback = lambda w: phase_timer.is_completed(PHASE_SEARCHING)
    running_manager.terminal_state_callback = handle_terminal_state
    driving_manager.terminal_state_callback = handle_terminal_state
    searching_house_manager.r_city_recovery_route_callback = recover_bad_landing_to_r_city
    searching_house_manager.r_city_pre_search_route_callback = route_to_r_city_search_start
    searching_house_manager.r_city_entry_route_callback = route_to_r_city_entry_point
    searching_house_manager.finish_callback = finish_searching_and_enter_running

    _runtime_initialized = True


def preload_runtime():
    """在启动游戏和 HOS 抓流前预加载项目运行时。"""
    initialize_runtime()


def _require_runtime():
    initialize_runtime()


def pause_sp_after_death(w: "FrameWorker"):
    _require_runtime()
    if not SP_RECORDING_ENABLED:
        return
    if w.sp_controller.is_recording and w.sp_controller.pause():
        time.sleep(0.5)


def prepare_round(w: "FrameWorker" = None):
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
        drop_target = None
        drop_target_candidates = DROP_TARGETS_BY_CITY
    else:
        drop_target = None
        drop_target_candidates = DROP_CAR_SEARCH_TARGETS_BY_CITY

    parachute_manager.reset()
    parachute_manager.configure(
        target_pos=drop_target,
        target_candidates=drop_target_candidates,
        landing_stage=landing_stage,
    )

    running_manager.reset(finding_car=need_drive)

    driving_manager.reset()
    searching_house_manager.reset()
    house_exit_manager.reset()

    if w is not None:
        drop_target_text = (
            f"候选={drop_target_candidates}"
            if drop_target_candidates
            else f"落点={drop_target}"
        )
        w.frame_log(
            f"本轮配置：需要开车={need_drive}，需要搜房={need_searching}，"
            f"{drop_target_text}，落地后进入={landing_stage}",
            log_type=FrameLogType.LOGIC,
        )


def handle_sp_start(w: "FrameWorker"):
    _require_runtime()
    if not SP_RECORDING_ENABLED:
        return
    if not phase_timer.landed:
        return
    if w.sp_controller.is_recording or w.sp_controller.is_saved:
        return
    if phase_timer.start_game_time is not None:
        running_manager.set_game_time(phase_timer.start_game_time)
        driving_manager.set_game_time(phase_timer.start_game_time)
    if w.sp_controller.start("sp"):
        time.sleep(0.5)
        if is_marathon_test(w):
            w.frame_log(
                f"马拉松 SP 目标 {w.sp_controller.target_duration_seconds / 60:g} 分钟，"
                f"当前有效时间 {w.sp_controller.effective_time / 60:.1f} 分钟",
                log_type=FrameLogType.TIME,
            )


def handle_sp_stop(w: "FrameWorker"):
    _require_runtime()
    if not SP_RECORDING_ENABLED:
        return
    if not w.sp_controller.is_recording:
        return
    if w.sp_controller.pause():
        time.sleep(0.5)


def _has_rank_finish_info(w: "FrameWorker") -> bool:
    return bool(w.get_info("个人排名")) or bool(w.get_info("队伍排名"))


def _has_death_finish_info(w: "FrameWorker") -> bool:
    return bool(w.get_info("变身")) or bool(w.get_info("红色血条"))


def _stop_active_motion(w: "FrameWorker", reason: str = "检测到死亡或排名界面"):
    _require_runtime()
    for manager in (searching_house_manager, running_manager):
        stop_func = getattr(manager, "stop_auto_forward", None)
        if callable(stop_func):
            stop_func(w)

    cancel_drive = getattr(driving_manager, "_cancel_drive_auto_forward", None)
    if callable(cancel_drive):
        cancel_drive(w, f"{reason}，取消车辆自动前进")


def is_marathon_test(w: "FrameWorker") -> bool:
    return w.test_type == TEST_TYPE_MARATHON


def finalize_marathon_if_target_reached(w: "FrameWorker") -> bool:
    _require_runtime()
    if not SP_RECORDING_ENABLED or not is_marathon_test(w):
        return False
    if not w.sp_controller.is_recording or not w.sp_controller.target_reached:
        return False

    w.frame_log(
        f"马拉松 SP 有效时间已达到 "
        f"{w.sp_controller.effective_time / 60:.1f}/"
        f"{w.sp_controller.target_duration_seconds / 60:g} 分钟，结束当前动作并长按保存",
        log_type=FrameLogType.TIME,
    )
    _stop_active_motion(w, "马拉松 SP 有效时间已达标")
    finalize_automation(w)
    return True


def handle_terminal_state(w: "FrameWorker", context: str = "阶段入口") -> bool:
    global rank_finish_pending, searching_phase_finishing

    _require_runtime()
    if _has_rank_finish_info(w):
        w.frame_log(
            f"{context}检测到个人排名或队伍排名，进入结束阶段",
            log_type=FrameLogType.LOGIC,
        )
        rank_finish_pending = True
        searching_phase_finishing = False
        _stop_active_motion(w)
        handle_sp_stop(w)
        w.change_stage("结束阶段")
        return True

    if _has_death_finish_info(w):
        w.frame_log(
            f"{context}检测到死亡界面，进入结束阶段",
            log_type=FrameLogType.LOGIC,
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
        return True

    return False


def should_abort_nanda_replay(w: "FrameWorker"):
    """回放已开始后不再因搜房计时到期中断。

    死亡、用例结束或外部已切走搜房阶段仍属于必须立即中止的情况。
    """
    _require_runtime()
    if w.current_stage != "搜房阶段":
        return True
    return handle_terminal_state(w, "南大回放")


def recover_bad_landing_to_r_city(w: "FrameWorker", target, reason: str):
    global searching_view_synced, searching_to_running_notified

    _require_runtime()
    route_target = tuple(target or DROP_TARGET_R_CITY)
    w.frame_log(
        f"搜房落点异常，切到跑图阶段恢复到R城: "
        f"reason={reason}, target={route_target}",
        log_type=FrameLogType.LOGIC,
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
    w.frame_log(
        f"搜房前置跑图，先到R城搜房起点: "
        f"reason={reason}, target={route_target}, arrival={arrival_distance:.1f}",
        log_type=FrameLogType.LOGIC,
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
    w.frame_log(
        f"落地后最近入门点仍较远，先按跑图阶段冲到入门点附近: "
        f"reason={reason}, target={route_target}, arrival={arrival_distance:.1f}",
        log_type=FrameLogType.LOGIC,
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
    w.frame_log(
        f"搜房结束: {reason} | "
        f"searching_remaining={phase_timer.get_remaining(PHASE_SEARCHING):.2f}s, "
        f"running_remaining={phase_timer.get_remaining(PHASE_RUNNING):.2f}s, "
        f"driving_remaining={phase_timer.get_remaining(PHASE_DRIVING):.2f}s",
        log_type=FrameLogType.LOGIC,
    )

    searching_house_manager.stop_auto_forward(w)
    w.refresh_frame()
    house_scene = searching_house_manager._get_house_scene(w)
    if house_scene == searching_house_manager.HOUSE_INDOOR:
        searching_exit_retry_count += 1
        w.frame_log(
            f"搜房结束时仍在屋内，先执行搜房出房策略，再切跑图 "
            f"(retry={searching_exit_retry_count})",
            log_type=FrameLogType.LOGIC,
        )
        exit_ok = searching_house_manager._exit_house(w)
        if w.current_stage != "搜房阶段":
            searching_phase_finishing = False
            return True
        w.refresh_frame()
        if not exit_ok and searching_house_manager._get_house_scene(w) == searching_house_manager.HOUSE_INDOOR:
            w.frame_log(
                "搜房结束出房未确认，保留搜房阶段并继续出房",
                log_type=FrameLogType.LOGIC,
            )
            searching_phase_finishing = False
            return True
    else:
        searching_exit_retry_count = 0

    finding_car = _should_find_car_after_searching()
    search_region = getattr(searching_house_manager, "house_region", None)
    running_manager.notify_searching_exit(
        finding_car=finding_car,
        search_region=search_region,
    )
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
    w.frame_log(
        "当前用例及全部循环已完成，进入结束阶段",
        log_type=FrameLogType.LOGIC,
    )
    if w.current_stage == "跑图阶段":
        running_manager.stop_auto_forward(w)

    if SP_RECORDING_ENABLED and not w.sp_controller.is_saved:
        if w.sp_controller.stop():
            sp_save_wait_seconds = (
                w.get_sp_save_settle_seconds()
                if is_marathon_test(w)
                else 1.0
            )
            if is_marathon_test(w):
                w.frame_log(
                    f"SP 长按保存指令已执行，等待 {sp_save_wait_seconds:g} 秒"
                    "让数据落盘，随后才退出本轮并拉取 SP 数据",
                    log_type=FrameLogType.TIME,
                )
            time.sleep(sp_save_wait_seconds)

    final_shutdown_pending = True
    w.change_stage("结束阶段")


def finish_case_loop_or_finalize(w: "FrameWorker"):
    _require_runtime()
    marathon_test = is_marathon_test(w)
    if marathon_test:
        if w.sp_controller.target_reached:
            w.frame_log(
                f"马拉松 SP 有效时间已达到 "
                f"{w.sp_controller.effective_time / 60:.1f}/"
                f"{w.sp_controller.target_duration_seconds / 60:g} 分钟，准备长按保存",
                log_type=FrameLogType.TIME,
            )
            finalize_automation(w)
            return
    elif not phase_timer.has_next_case_loop():
        finalize_automation(w)
        return

    if w.current_stage == "跑图阶段":
        running_manager.stop_auto_forward(w)

    if marathon_test:
        next_loop_message = (
            f"马拉松 SP 有效时间 "
            f"{w.sp_controller.effective_time / 60:.1f}/"
            f"{w.sp_controller.target_duration_seconds / 60:g} 分钟，"
            "暂停 sp 并返回大厅继续下一次循环"
        )
    else:
        next_loop_message = (
            "暂停 sp，返回大厅后继续下一次循环"
            if SP_RECORDING_ENABLED
            else "返回大厅后继续下一次循环"
        )
    w.frame_log(
        f"第 {phase_timer.case_loop_index}/{phase_timer.case_loop_count} 次循环已完成，"
        f"{next_loop_message}",
        log_type=FrameLogType.TIME,
    )
    handle_sp_stop(w)
    phase_timer.advance_case_loop(allow_extend=marathon_test)
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




def click_popup_info_if_visible(w: "FrameWorker", info_name: str, click_target=None) -> bool:
    target = w.get_info(info_name)
    if not target:
        return False
    control_target = click_target or target
    w.frame_log(
        f"关闭{info_name}弹窗，点击={click_target or info_name}",
        log_type=FrameLogType.UI_CONTROL,
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
            w.frame_log(
                "大厅确认过程中又检测到弹窗，取消本次房子图标确认",
                log_type=FrameLogType.UI_CONTROL,
            )
        reset_lobby_confirm()
        return False

    if not w.get_info("房子"):
        if lobby_house_confirm_count:
            w.frame_log(
                "房子图标未连续稳定出现，取消本次大厅确认",
                log_type=FrameLogType.UI_CONTROL,
            )
        reset_lobby_confirm()
        return False

    lobby_house_confirm_count += 1
    w.frame_log(
        f"房子图标稳定确认 {lobby_house_confirm_count}/{LOBBY_CONFIRM_REQUIRED}",
        log_type=FrameLogType.UI_CONTROL,
    )
    return lobby_house_confirm_count >= LOBBY_CONFIRM_REQUIRED


def prepare_rank_finish_for_lobby(w: "FrameWorker") -> bool:
    global rank_finish_pending

    if not rank_finish_pending and not _has_rank_finish_info(w):
        return True

    rank_finish_pending = True
    w.frame_log(
        "检测到排名界面，等待2s后通过区域获取观战对手位置",
        log_type=FrameLogType.LOGIC,
    )
    time.sleep(RANK_FINISH_SPECTATE_WAIT_SECONDS)
    if not w.refresh_frame():
        return False

    spectate_opponent = w.get_info("观战对手")
    if not spectate_opponent:
        w.frame_log(
            "未识别到观战对手区域，保留在排名界面等待下一帧",
            log_type=FrameLogType.LOGIC,
        )
        return False

    w.frame_log(
        f"通过区域动态点击观战对手: position={spectate_opponent}",
        log_type=FrameLogType.UI_CONTROL,
    )
    w.click(spectate_opponent)
    if not w.refresh_frame():
        return False
    rank_finish_pending = False
    return True


def maybe_report_phase_remaining():
    _require_runtime()
    phase_reporter.maybe_report(phase_timer)


def click_classic_island_region(w: "FrameWorker") -> bool:
    classic_island = w.get_info("经典海岛")
    if not classic_island:
        w.frame_log(
            "未识别到经典海岛区域，保留在选图页等待下一帧",
            log_type=FrameLogType.LOGIC,
        )
        return False

    w.frame_log(
        f"通过经典海岛区域动态选择海岛: position={classic_island}",
        log_type=FrameLogType.UI_CONTROL,
    )
    w.click(classic_island)
    return True






def handle_priority_stage_jump_forward(w: "FrameWorker", stage_label: str) -> bool:
    _require_runtime()
    if not w.get_info("跳跃"):
        return False

    w.frame_log(
        f"{stage_label}检测到跳跃按钮，点击跳跃并前推",
        log_type=FrameLogType.LOGIC,
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
    phase_timer.set_frame_logger(w.frame_log)
    previous_stage = phase_timer.last_stage
    stage_events = phase_timer.sync_stage(w.current_stage)
    stage_events |= phase_timer.refresh()

    if previous_stage == "开车阶段" and w.current_stage == "跑图阶段":
        w.frame_log(
            "开车阶段切回跑图阶段，同步下车后的寻车状态",
            log_type=FrameLogType.LOGIC,
        )
        finding_car = driving_manager.consume_running_transition_finding_car(
            default=phase_timer.need_drive()
        )
        running_manager.notify_vehicle_exit(finding_car=finding_car)

    if previous_stage == "搜房阶段" and w.current_stage == "跑图阶段":
        if searching_to_running_notified:
            w.frame_log(
                "搜房模块已完成跑图交接，清理交接标记",
                log_type=FrameLogType.LOGIC,
            )
            searching_to_running_notified = False
        else:
            w.frame_log(
                "搜房阶段切到跑图阶段，初始化寻车状态",
                log_type=FrameLogType.LOGIC,
            )
            running_manager.notify_searching_exit(
                finding_car=_should_find_car_after_searching(),
                search_region=getattr(searching_house_manager, "house_region", None),
            )

    if "landed" in stage_events and not phase_timer.all_done():
        w.frame_log(
            "人物已落地，同步搜房、跑图和开车计时",
            log_type=FrameLogType.LOGIC,
        )
        if phase_timer.start_game_time is not None:
            running_manager.set_game_time(phase_timer.start_game_time)
            driving_manager.set_game_time(phase_timer.start_game_time)

    if w.current_stage in {"搜房阶段", "跑图阶段", "开车阶段"}:
        if handle_terminal_state(w, f"{w.current_stage}入口"):
            return
        if finalize_marathon_if_target_reached(w):
            return
        maybe_report_phase_remaining()

    if w.current_stage == "关闭弹窗阶段":
        w.frame_log(
            "检查可关闭弹窗",
            log_type=FrameLogType.UI_CONTROL,
        )
        if click_popup_info_if_visible(w, "关闭公告"):
            return

        if click_popup_info_if_visible(w, "重新进入比赛", "取消"):
            return

        if click_popup_info_if_visible(w, "确定已结束"):
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
                w.frame_log(
                    "大厅确认完成，停止本轮自动化",
                    log_type=FrameLogType.LOGIC,
                )
                finalize_after_lobby(w)
                return
            w.frame_log(
                "大厅确认完成，进入选择地图阶段",
                log_type=FrameLogType.LOGIC,
            )
            reset_lobby_confirm()
            w.change_stage("选择地图阶段")
            return

    if w.current_stage == "选择地图阶段":
        w.frame_log(
            "打开地图选择面板",
            log_type=FrameLogType.UI_CONTROL,
        )
        w.click("地图")
        time.sleep(2)
        w.click("经典模式")
        time.sleep(2)
        w.click("切换")
        time.sleep(2)
        w.refresh_frame()

        if w.get_info("对号"):
            w.frame_log(
                "清理已有地图选择",
                log_type=FrameLogType.UI_CONTROL,
            )
            w.click(w.get_info("对号"))
            time.sleep(2)

        if not w.refresh_frame():
            return
        if not click_classic_island_region(w):
            return
        time.sleep(1)
        w.refresh_frame()
        if w.get_info('自动匹配'):
            w.frame_log(
                "调整自动匹配选项",
                log_type=FrameLogType.UI_CONTROL,
            )
            w.click(w.get_info('自动匹配'))
        time.sleep(1)
        w.click("确定")
        w.change_stage("开始游戏阶段")
        return

    if w.current_stage == "开始游戏阶段":
        if w.get_info("加速礼包"):
            w.frame_log(
                "关闭加速礼包弹窗",
                log_type=FrameLogType.UI_CONTROL,
            )
            w.click("放弃")
            w.refresh_frame()

        if start_game and start_game_click_time is not None:
            if time.time() - start_game_click_time >= START_GAME_VERIFY_DELAY:
                if w.get_info("开始游戏"):
                    w.frame_log(
                        "开始游戏点击未生效，重置点击状态",
                        log_type=FrameLogType.UI_CONTROL,
                    )
                    start_game = False
                    start_game_click_time = None

        if w.get_info("房子"):
            if not start_game:
                w.frame_log(
                    "点击开始游戏",
                    log_type=FrameLogType.UI_CONTROL,
                )
                w.click("开始游戏")
                start_game = True
                start_game_click_time = time.time()
            else:
                w.frame_log(
                    "等待进入出生岛",
                    log_type=FrameLogType.LOGIC,
                )
            w.refresh_frame()

        if w.get_info("提示"):
            w.frame_log(
                "关闭匹配提示弹窗",
                log_type=FrameLogType.UI_CONTROL,
            )
            w.click("不提示")
            time.sleep(1)
            w.click("不需要")
            time.sleep(1)

        if w.get_info("拳头"):
            w.frame_log(
                "已进入出生岛，初始化本轮并进入跳伞阶段",
                log_type=FrameLogType.LOGIC,
            )
            prepare_round(w)
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
            # 南大取景/匹配也会调用 should_abort_searching。计时到期时，
            # 内层只返回中止信号，等触控和感知分组清理完成后，
            # 再由这个最外层阶段入口统一执行出房与跑图交接。
            # 已经开始的南大回放使用独立的中止回调，会先完整回放。
            if (
                w.current_stage == "搜房阶段"
                and phase_timer.is_completed(PHASE_SEARCHING)
            ):
                w.frame_log(
                    f"搜房阶段 "
                    f"{phase_timer.get_duration_minutes_label(PHASE_SEARCHING)} "
                    "分钟已用完，安全结束当前搜房动作后切换到跑图阶段",
                    log_type=FrameLogType.TIME,
                )
                finish_searching_and_enter_running(w, "搜房阶段计时已用完")
            return

        if handle_priority_stage_jump_forward(w, "搜房阶段"):
            return

        searching_view_synced = True
        searching_house_manager.process(w)
        return

    if w.current_stage == "跑图阶段":
        if searching_view_synced:
            w.frame_log(
                "恢复第一人称视角",
                log_type=FrameLogType.LOGIC,
            )
            running_manager.set_view_mode(RunningManager.VIEW_MODE_FIRST)
            searching_view_synced = False

        handle_sp_start(w)

        if phase_timer.all_done():
            w.frame_log(
                "搜房、跑图和开车任务均已完成",
                log_type=FrameLogType.LOGIC,
            )
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
            w.frame_log(
                "初始化驾驶阶段状态",
                log_type=FrameLogType.LOGIC,
            )
            driving_manager.set_remaining_drive_time(phase_timer.get_remaining(PHASE_DRIVING))
            entry_source = running_manager.consume_vehicle_entry_source()
            if entry_source == RunningManager.VEHICLE_ENTRY_ROADSIDE:
                driving_manager.skip_initial_exit_garage("roadside vehicle")

        if phase_timer.is_completed(PHASE_DRIVING):
            w.frame_log(
                "驾驶计时完成",
                log_type=FrameLogType.TIME,
            )
            driving_manager.set_remaining_drive_time(0)

        driving_manager.process(w)
        return

    if w.current_stage == "结束阶段":
        if final_shutdown_pending:
            w.frame_log(
                "返回大厅并完成本轮",
                log_type=FrameLogType.LOGIC,
            )
            handle_sp_stop(w)
            if not prepare_rank_finish_for_lobby(w):
                return
            w.click("设置")
            time.sleep(1)
            w.click("返回大厅")
            time.sleep(1)
            w.click("确定退出比赛")
            time.sleep(3)
            w.change_stage("关闭弹窗阶段")
            return

        w.frame_log(
            "返回大厅并准备下一轮",
            log_type=FrameLogType.LOGIC,
        )
        handle_sp_stop(w)
        if not prepare_rank_finish_for_lobby(w):
            return

        w.click("设置")
        time.sleep(1)
        w.click("返回大厅")
        time.sleep(1)
        w.click("确定退出比赛")
        time.sleep(3)
        w.change_stage("开始游戏阶段")
