import unittest
import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from launcher import (
    apply_pyinstaller_splash_suppression,
    build_restart_device_commands,
    CUSTOMS_EXAMPLES_DIR,
    discover_history_outputs,
    find_latest_preview_frame,
    format_history_record_summary,
    get_testcase_button_texts,
    HiddenSubprocess,
    is_multiprocessing_child,
    resolve_app_paths,
    resolve_preview_frame_dir,
    resolve_label_project_dir,
    resolve_runtime_temp_dir,
)
from aw.autogame.tools.ProcessUtils import hidden_subprocess_kwargs
from aw.autogame.tools.ProcessUtils import install_hidden_subprocess_patch


class FakeStartupInfo:
    def __init__(self):
        self.dwFlags = 0
        self.wShowWindow = None


class FakePopen:
    pass


class FakeSubprocessModule:
    STARTF_USESHOWWINDOW = 0x01
    CREATE_NO_WINDOW = 0x08000000
    STARTUPINFO = FakeStartupInfo
    Popen = FakePopen


class FakeProcessEnvironment:
    def __init__(self):
        self.values = {}

    def insert(self, key, value):
        self.values[key] = value


class FakeStdout:
    def read(self, _size):
        return b""


class FakeProcess:
    pid = 1234
    stdout = FakeStdout()
    returncode = 0

    def poll(self):
        return None

    def wait(self):
        return 0


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

    def test_apply_pyinstaller_splash_suppression_disables_child_splash(self):
        env = FakeProcessEnvironment()

        apply_pyinstaller_splash_suppression(env)

        self.assertEqual("1", env.values["PYINSTALLER_SUPPRESS_SPLASH_SCREEN"])
        self.assertEqual("0", env.values["_PYI_SPLASH_IPC"])

    def test_hidden_subprocess_starts_with_hidden_kwargs(self):
        process = HiddenSubprocess()
        process.setProgram("AutoGameLauncher.exe")
        process.setArguments(["--run-testcase", "case"])

        with mock.patch(
            "launcher.hidden_subprocess_kwargs",
            return_value={"creationflags": 123},
        ), mock.patch(
            "launcher.subprocess.Popen",
            return_value=FakeProcess(),
        ) as popen:
            process.start()

        self.assertEqual(
            ["AutoGameLauncher.exe", "--run-testcase", "case"],
            popen.call_args.args[0],
        )
        self.assertEqual(123, popen.call_args.kwargs["creationflags"])
        self.assertEqual(subprocess.DEVNULL, popen.call_args.kwargs["stdin"])
        self.assertEqual(subprocess.PIPE, popen.call_args.kwargs["stdout"])
        self.assertEqual(subprocess.STDOUT, popen.call_args.kwargs["stderr"])

    def test_resolve_app_paths_non_frozen_uses_source_file_parent(self):
        paths = resolve_app_paths(
            frozen=False,
            file_path=Path("/repo/launcher.py"),
        )

        self.assertEqual(Path("/repo"), paths.app_dir)
        self.assertEqual(Path("/repo"), paths.internal_dir)
        self.assertEqual(Path("/repo"), paths.root_dir)

    def test_resolve_app_paths_frozen_prefers_internal_dir_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_dir = (Path(temp_dir) / "AutoGameLauncherDebug").resolve()
            internal_dir = app_dir / "_internal"
            internal_dir.mkdir(parents=True)
            exe_path = app_dir / "AutoGameLauncherDebug.exe"

            paths = resolve_app_paths(
                frozen=True,
                executable=exe_path,
            )

            self.assertEqual(app_dir, paths.app_dir)
            self.assertEqual(internal_dir, paths.internal_dir)
            self.assertEqual(internal_dir, paths.root_dir)

    def test_resolve_app_paths_frozen_falls_back_to_app_dir_without_internal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_dir = (Path(temp_dir) / "AutoGameLauncherDebug").resolve()
            app_dir.mkdir()
            exe_path = app_dir / "AutoGameLauncherDebug.exe"

            paths = resolve_app_paths(
                frozen=True,
                executable=exe_path,
            )

            self.assertEqual(app_dir, paths.app_dir)
            self.assertEqual(app_dir / "_internal", paths.internal_dir)
            self.assertEqual(app_dir, paths.root_dir)

    def test_hidden_subprocess_kwargs_returns_create_no_window_for_windows(self):
        kwargs = hidden_subprocess_kwargs(
            os_name="nt",
            subprocess_module=FakeSubprocessModule,
        )

        self.assertEqual(FakeSubprocessModule.CREATE_NO_WINDOW, kwargs["creationflags"])
        self.assertEqual(
            FakeSubprocessModule.STARTF_USESHOWWINDOW,
            kwargs["startupinfo"].dwFlags,
        )
        self.assertEqual(0, kwargs["startupinfo"].wShowWindow)

    def test_hidden_subprocess_kwargs_is_empty_for_non_windows(self):
        self.assertEqual({}, hidden_subprocess_kwargs(os_name="posix"))

    def test_install_hidden_subprocess_patch_is_noop_for_non_windows(self):
        self.assertFalse(install_hidden_subprocess_patch(os_name="posix"))

    def test_install_hidden_subprocess_patch_does_not_replace_popen_on_windows(self):
        class WindowsSubprocessModule(FakeSubprocessModule):
            Popen = FakePopen

        original_popen = WindowsSubprocessModule.Popen

        self.assertFalse(
            install_hidden_subprocess_patch(
                os_name="nt",
                subprocess_module=WindowsSubprocessModule,
            )
        )
        self.assertIs(original_popen, WindowsSubprocessModule.Popen)

        class InheritedPopen(WindowsSubprocessModule.Popen):
            pass

        self.assertTrue(issubclass(InheritedPopen, original_popen))

    def test_restart_device_commands_run_hdc_directly_without_cmd_or_bat(self):
        commands = build_restart_device_commands("hdc")

        self.assertEqual(
            [
                ["hdc", "shell", "reboot", "-D"],
                ["hdc", "wait"],
                ["hdc", "shell", "setenforce", "0"],
                ["hdc", "fport", "tcp:12345", "tcp:12345"],
            ],
            commands,
        )
        flattened = " ".join(" ".join(command) for command in commands)
        self.assertNotIn("cmd", flattened)
        self.assertNotIn("restart.bat", flattened)

    def test_runtime_preview_dir_uses_app_dir_even_when_internal_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            app_dir = (Path(temp_dir) / "AutoGameLauncher").resolve()
            internal_dir = app_dir / "_internal"
            internal_dir.mkdir(parents=True)

            self.assertEqual(
                app_dir / "aw" / "autogame" / "temp",
                resolve_runtime_temp_dir(app_dir),
            )
            self.assertEqual(
                app_dir / "aw" / "autogame" / "temp" / "logs" / "process_temp_logs",
                resolve_preview_frame_dir(app_dir),
            )

    def test_find_latest_preview_frame_supports_common_image_suffixes_by_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preview_dir = Path(temp_dir)
            (preview_dir / "frame_00007.jpg").write_text("old", encoding="utf-8")
            (preview_dir / "frame_00009.png").write_text("new", encoding="utf-8")
            (preview_dir / "frame_00008.jpeg").write_text("middle", encoding="utf-8")
            (preview_dir / "frame_00010.json").write_text("metadata", encoding="utf-8")

            self.assertEqual(
                preview_dir / "frame_00009.png",
                find_latest_preview_frame(preview_dir),
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
