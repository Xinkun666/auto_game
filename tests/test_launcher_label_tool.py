import unittest
import json
import tempfile
from pathlib import Path

from launcher import (
    CUSTOMS_EXAMPLES_DIR,
    discover_history_outputs,
    format_history_record_summary,
    get_testcase_button_texts,
    is_multiprocessing_child,
    resolve_label_project_dir,
)


class LauncherLabelToolTests(unittest.TestCase):
    def test_resolve_label_project_dir_returns_existing_project_with_info(self):
        project_dir = resolve_label_project_dir("Auto_PUBG_ALL")

        self.assertEqual(CUSTOMS_EXAMPLES_DIR / "Auto_PUBG_ALL", project_dir)

    def test_resolve_label_project_dir_rejects_blank_or_missing_project(self):
        self.assertIsNone(resolve_label_project_dir(""))
        self.assertIsNone(resolve_label_project_dir("Missing_Project"))

    def test_testcase_button_texts_reflect_selection_state(self):
        self.assertEqual(("选择用例", "重选"), get_testcase_button_texts(False))
        self.assertEqual(("已选择", "重选"), get_testcase_button_texts(True))

    def test_is_multiprocessing_child_detects_pyinstaller_worker_argv(self):
        self.assertTrue(
            is_multiprocessing_child(
                [
                    "AutoGameLauncherDebug.exe",
                    "--multiprocessing-fork",
                    "parent_pid=5424",
                    "pipe_handle=2036",
                ]
            )
        )
        self.assertFalse(
            is_multiprocessing_child(
                [
                    "AutoGameLauncherDebug.exe",
                    "--run-testcase",
                    "testcases/pubg/pubg_full_flow/auto_pubg",
                ]
            )
        )

    def test_discover_history_outputs_reads_archive_metadata_and_counts_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            archive_dir = temp_root / "game_cases_20260609120000" / "第1次_20260609120100"
            logs_dir = archive_dir / "logs"
            frames_dir = archive_dir / "process_temp_logs"
            logs_dir.mkdir(parents=True)
            frames_dir.mkdir()
            (logs_dir / "launcher_output.txt").write_text("run log", encoding="utf-8")
            (logs_dir / "device.txt").write_text("device log", encoding="utf-8")
            (frames_dir / "frame_1.jpg").write_text("frame", encoding="utf-8")
            (archive_dir / "preview_10fps.mp4").write_text("video", encoding="utf-8")
            (archive_dir / "archive_info.json").write_text(
                json.dumps(
                    {
                        "archive_time": "2026-06-09 12:01:30",
                        "run_index": 1,
                        "project_case": "Auto_PUBG_ALL",
                        "target_case": "auto_pubg",
                        "testcase_label": "testcases/pubg/和平精英全流程/auto_pubg",
                        "exit_code": 0,
                        "timed_out": False,
                        "stream_disconnected": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            records = discover_history_outputs(temp_root)

            self.assertEqual(1, len(records))
            record = records[0]
            self.assertEqual(archive_dir, record["archive_dir"])
            self.assertEqual("Auto_PUBG_ALL", record["project_case"])
            self.assertEqual("auto_pubg", record["target_case"])
            self.assertEqual("run log", record["launcher_output"])
            self.assertEqual(2, record["log_file_count"])
            self.assertEqual(1, record["process_temp_file_count"])
            self.assertTrue(record["preview_video_exists"])

    def test_format_history_record_summary_includes_status_and_paths(self):
        summary = format_history_record_summary(
            {
                "archive_time": "2026-06-09 12:01:30",
                "run_index": 2,
                "project_case": "Auto_PUBG_ALL",
                "target_case": "auto_pubg",
                "testcase_label": "testcases/pubg/和平精英全流程/auto_pubg",
                "exit_code": 1,
                "timed_out": True,
                "stream_disconnected": False,
                "archive_dir": Path("/tmp/archive"),
                "log_file_count": 3,
                "process_temp_file_count": 4,
                "process_save_frame_count": 5,
                "preview_video_exists": False,
            }
        )

        self.assertIn("Auto_PUBG_ALL", summary)
        self.assertIn("exit_code: 1", summary)
        self.assertIn("timed_out: True", summary)
        self.assertIn("/tmp/archive", summary)


if __name__ == "__main__":
    unittest.main()
