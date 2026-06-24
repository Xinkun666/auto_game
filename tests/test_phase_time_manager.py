import unittest
import ast
from pathlib import Path
from unittest import mock

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.phase_time_manager import (
    PHASE_DRIVING,
    PHASE_RUNNING,
    PHASE_SEARCHING,
    PhaseTimeManager,
    PhaseTimeReporter,
    parse_case_loop_count,
)


class PhaseTimeManagerLoopTests(unittest.TestCase):
    def _new_timer(self):
        return PhaseTimeManager(
            {
                PHASE_SEARCHING: 10,
                PHASE_RUNNING: 10,
                PHASE_DRIVING: 10,
            },
            {
                "搜房阶段": PHASE_SEARCHING,
                "跑图阶段": PHASE_RUNNING,
                "开车阶段": PHASE_DRIVING,
            },
        )

    def test_parse_case_loop_count_defaults_invalid_values_to_one(self):
        self.assertEqual(1, parse_case_loop_count(None))
        self.assertEqual(1, parse_case_loop_count(""))
        self.assertEqual(1, parse_case_loop_count("0"))
        self.assertEqual(1, parse_case_loop_count("-2"))
        self.assertEqual(3, parse_case_loop_count("3"))

    def test_advance_case_loop_resets_phase_time_but_keeps_sp_unsaved(self):
        timer = self._new_timer()
        timer.configure_case_loop_count(2)
        timer.mark_sp_started()
        timer.mark_sp_stopped()
        timer.total_elapsed = timer.total_duration
        for state in timer.phase_states.values():
            state.elapsed = state.duration
            state.completed = True

        self.assertTrue(timer.all_done())
        self.assertTrue(timer.has_next_case_loop())
        self.assertTrue(timer.advance_case_loop())

        self.assertEqual(2, timer.case_loop_index)
        self.assertFalse(timer.all_done())
        self.assertFalse(timer.sp_recording)
        self.assertFalse(timer.sp_saved)
        self.assertTrue(timer.sp_started_ever)
        for state in timer.phase_states.values():
            self.assertEqual(0.0, state.elapsed)
            self.assertFalse(state.completed)

    def test_advance_case_loop_returns_false_on_final_loop(self):
        timer = self._new_timer()
        timer.configure_case_loop_count(1)

        self.assertFalse(timer.has_next_case_loop())
        self.assertFalse(timer.advance_case_loop())

    def test_phase_timer_prints_phase_start_and_end(self):
        timer = self._new_timer()

        with mock.patch("builtins.print") as print_mock:
            events = timer.sync_stage("搜房阶段")
            timer.active_since -= timer.phase_states[PHASE_SEARCHING].duration + 1
            events |= timer.refresh()

        self.assertIn(f"enter_{PHASE_SEARCHING}", events)
        self.assertIn(f"completed_{PHASE_SEARCHING}", events)
        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        self.assertIn(
            "[AutoLog][逻辑日志] 当前状态=搜房阶段开始 | 当前目标=时间管理 | "
            "要做什么=记录阶段开始 | 怎么做=启动搜房阶段计时器，计划 10 分钟 | 结果=计时中",
            printed_lines,
        )
        self.assertIn(
            "[AutoLog][逻辑日志] 当前状态=搜房阶段结束 | 当前目标=时间管理 | "
            "要做什么=记录阶段结束 | 怎么做=累计搜房阶段计时 10 分钟 | 结果=阶段计时完成",
            printed_lines,
        )

    def test_pubg_demo_phase_durations_are_five_minutes_each(self):
        auto_pubg_path = (
            Path(__file__).resolve().parents[1]
            / "aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py"
        )
        module = ast.parse(auto_pubg_path.read_text(encoding="utf-8"))
        phase_durations_node = next(
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PHASE_DURATIONS" for target in node.targets)
        )
        phase_durations = {
            key.id: value.value
            for key, value in zip(phase_durations_node.keys, phase_durations_node.values)
        }

        self.assertEqual(
            {
                "PHASE_SEARCHING": 5,
                "PHASE_RUNNING": 5,
                "PHASE_DRIVING": 5,
            },
            phase_durations,
        )

    def test_all_done_report_uses_configured_total_minutes(self):
        timer = PhaseTimeManager(
            {
                PHASE_SEARCHING: 5,
                PHASE_RUNNING: 5,
                PHASE_DRIVING: 5,
            },
            {},
        )
        timer.start_game_time = 1.0
        timer.total_elapsed = timer.total_duration
        reporter = PhaseTimeReporter()

        with mock.patch("builtins.print") as print_mock:
            reporter.maybe_report(timer)

        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        self.assertIn(
            "[AutoLog][逻辑日志] 当前状态=总计=00:00 | 搜房=05:00 | 跑图=05:00 | 开车=05:00 | "
            "当前目标=时间管理 | 要做什么=结束本轮阶段计时 | 怎么做=汇总 15 分钟阶段预算并报告剩余时间 | 结果=总时长已结束",
            printed_lines,
        )


if __name__ == "__main__":
    unittest.main()
