import unittest
import importlib
import sys
from unittest import mock


class FakeFrameWorker:
    def __init__(self, stage, info=None):
        self.current_stage = stage
        self.info = dict(info or {})
        self.events = []

    def get_info(self, name):
        return self.info.get(name)

    def click(self, name):
        self.events.append(("click", name))

    def tap_single(self, name, **kwargs):
        self.events.append(("tap_single", name, kwargs))

    def refresh_frame(self):
        self.events.append(("refresh_frame",))

    def change_stage(self, stage):
        self.events.append(("change_stage", stage))
        self.current_stage = stage


class FakePhaseTimer:
    last_stage = "跑图阶段"
    start_game_time = None
    sp_recording = False

    def sync_stage(self, stage):
        return set()

    def refresh(self):
        return set()

    def all_done(self):
        return False

    def need_drive(self):
        return False

    def is_completed(self, phase):
        return False

    def get_remaining(self, phase):
        return 60.0


class FakeRunningManager:
    VIEW_MODE_FIRST = "first"
    VEHICLE_ENTRY_ROADSIDE = "roadside"

    def __init__(self):
        self.process_calls = []
        self.stop_calls = []

    def set_view_mode(self, mode):
        pass

    def notify_vehicle_exit(self, finding_car=False):
        pass

    def notify_searching_exit(self, finding_car=False):
        pass

    def set_game_time(self, game_time):
        pass

    def reset(self, finding_car=False):
        pass

    def start_forced_route(self, **kwargs):
        pass

    def stop_auto_forward(self, w):
        self.stop_calls.append(w)

    def process(self, w):
        self.process_calls.append(w)

    def set_drive_required(self, required):
        pass

    def consume_vehicle_entry_source(self):
        return None


class FakeDrivingManager:
    def __init__(self):
        self.process_calls = []

    def set_game_time(self, game_time):
        pass

    def reset(self):
        pass

    def consume_running_transition_finding_car(self, default=False):
        return default

    def set_running_fallback_enabled(self, enabled):
        pass

    def set_remaining_drive_time(self, remaining):
        pass

    def skip_initial_exit_garage(self, reason):
        pass

    def process(self, w):
        self.process_calls.append(w)


class FakeSearchManager:
    HOUSE_INDOOR = 0
    r_city_near_distance = 30.0

    def __init__(self):
        self.stop_calls = []
        self.history_locations = []

    def configure_r_city_landing_target(self, target):
        pass

    def configure_r_city_pre_search_target(self, target, arrival_distance=3.0):
        pass

    def stop_auto_forward(self, w):
        self.stop_calls.append(w)

    def reset(self):
        pass

    def process(self, w):
        pass


class FakeHouseExitManager:
    def reset(self):
        pass


class FakeParachuteManager:
    def reset(self):
        pass

    def configure(self, **kwargs):
        pass

    def process(self, w):
        pass


def load_auto_pubg_with_fakes():
    module_name = "aw.autogame.customs_game_examples.Auto_PUBG_ALL.auto_pubg"
    sys.modules.pop(module_name, None)
    patches = [
        mock.patch(
            "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.running_manager.RunningManager",
            FakeRunningManager,
        ),
        mock.patch(
            "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.driving_manager.DrivingManager",
            FakeDrivingManager,
        ),
        mock.patch(
            "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_scene_search_manager.HouseSceneSearchManager",
            FakeSearchManager,
        ),
        mock.patch(
            "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_exit_manager.HouseExitManager",
            FakeHouseExitManager,
        ),
        mock.patch(
            "aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.parachute_manager.ParachuteManager",
            FakeParachuteManager,
        ),
    ]
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        return importlib.import_module(module_name)


class PubgTerminalStateTests(unittest.TestCase):
    def test_rank_finish_waits_two_seconds_before_spectating(self):
        auto_pubg = load_auto_pubg_with_fakes()
        w = FakeFrameWorker("结束阶段", {"个人排名": True})
        auto_pubg.rank_finish_pending = False
        events = []

        with mock.patch.object(auto_pubg.time, "sleep", side_effect=lambda seconds: events.append(("sleep", seconds))):
            auto_pubg.prepare_rank_finish_for_lobby(w)

        self.assertEqual(
            [("sleep", 2), ("click", "观战对手"), ("refresh_frame",)],
            events + w.events,
        )
        self.assertFalse(auto_pubg.rank_finish_pending)

    def test_running_stage_terminal_rank_preempts_jump_and_running_logic(self):
        auto_pubg = load_auto_pubg_with_fakes()
        w = FakeFrameWorker("跑图阶段", {"队伍排名": True, "跳跃": True})

        with mock.patch.object(auto_pubg, "phase_timer", FakePhaseTimer()), \
            mock.patch.object(auto_pubg, "handle_sp_start"), \
            mock.patch.object(auto_pubg.running_manager, "process", wraps=auto_pubg.running_manager.process) as running_process, \
            mock.patch.object(auto_pubg.running_manager, "stop_auto_forward", wraps=auto_pubg.running_manager.stop_auto_forward) as stop_running, \
            mock.patch.object(auto_pubg.searching_house_manager, "stop_auto_forward", wraps=auto_pubg.searching_house_manager.stop_auto_forward) as stop_searching:
            auto_pubg.on_stage(w)

        self.assertFalse(running_process.called)
        self.assertNotIn(("click", "跳跃"), w.events)
        self.assertIn(("change_stage", "结束阶段"), w.events)
        stop_running.assert_called_once_with(w)
        stop_searching.assert_called_once_with(w)


if __name__ == "__main__":
    unittest.main()
