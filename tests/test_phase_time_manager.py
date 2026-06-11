import unittest

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.support.phase_time_manager import (
    PHASE_DRIVING,
    PHASE_RUNNING,
    PHASE_SEARCHING,
    PhaseTimeManager,
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


if __name__ == "__main__":
    unittest.main()
