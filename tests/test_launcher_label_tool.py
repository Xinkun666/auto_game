import unittest
import json
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest import mock

from launcher import (
    apply_pyinstaller_splash_suppression,
    build_restart_device_commands,
    build_launcher_plan_env_values,
    classify_output_line,
    check_capture_stream_for_screen_mode,
    CUSTOMS_EXAMPLES_DIR,
    discover_history_outputs,
    filter_output_text,
    find_latest_preview_frame,
    format_history_record_summary,
    get_testcase_button_texts,
    HiddenSubprocess,
    is_multiprocessing_child,
    is_pubg_testcase_keyword_match,
    LauncherWindow,
    LOG_CATEGORY_LOGIC,
    LOG_CATEGORY_OTHER,
    LOG_CATEGORY_SYSTEM,
    LOG_CATEGORY_TIME,
    LOG_CATEGORY_UI,
    LOG_FILTER_ALL,
    PUBG_CASE_DEFAULT_LOOP_COUNT,
    PUBG_CASE_RUNTIME_DESCRIPTION,
    resolve_app_paths,
    resolve_preview_frame_dir,
    resolve_test_profile_from_radio_selection,
    resolve_screen_mode_for_test_profile,
    resolve_label_project_dir,
    resolve_runtime_temp_dir,
    run_testcase_entry,
    run_hdc_shell,
    write_screen_mode_config,
    WindowsProcessLaunchTracer,
)
from aw.autogame.tools.ProcessUtils import is_window_suppression_enabled
from aw.autogame.tools.ProcessUtils import hidden_subprocess_kwargs
from aw.autogame.tools.ProcessUtils import hidden_subprocess_context
from aw.autogame.tools.ProcessUtils import hdc_command_args
from aw.autogame.tools.ProcessUtils import install_hidden_subprocess_patch
from aw.autogame.tools.ProcessUtils import start_hidden_subprocess_window_suppressor
from aw.autogame.tools.ProcessUtils import WindowsSubprocessWindowSuppressor
from aw.autogame.tools import Utils as autogame_utils


class FakeStartupInfo:
    def __init__(self):
        self.dwFlags = 0
        self.wShowWindow = None


class FakePopen:
    calls = []

    def __init__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeSubprocessModule:
    STARTF_USESHOWWINDOW = 0x01
    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008
    STARTUPINFO = FakeStartupInfo
    Popen = FakePopen
    DEVNULL = subprocess.DEVNULL
    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    call = mock.Mock(return_value=0)


class FakeOsModule:
    name = "nt"
    system = mock.Mock(return_value=99)


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


class BlockingStdout:
    def __init__(self):
        self.closed = False
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def read(self, _size):
        self.read_started.set()
        self.release_read.wait(timeout=5)
        return b""

    def close(self):
        self.closed = True
        self.release_read.set()


class KillableFakeProcess:
    pid = 987654321

    def __init__(self):
        self.stdout = BlockingStdout()
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


class FakeTimer:
    def __init__(self):
        self.started = False
        self.stopped = False

    def isActive(self):
        return self.started

    def start(self, *args):
        self.started = True

    def stop(self):
        self.stopped = True


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

    def test_pubg_testcase_keyword_match_detects_pubg_and_chinese_names(self):
        self.assertTrue(
            is_pubg_testcase_keyword_match(
                "testcases/pubg/和平精英全流程/auto_pubg.py",
                "Auto_PUBG_ALL",
            )
        )
        self.assertTrue(is_pubg_testcase_keyword_match("PUBG mobile smoke test"))
        self.assertTrue(is_pubg_testcase_keyword_match("和平精英功能测试"))
        self.assertFalse(is_pubg_testcase_keyword_match("testcases/racing/demo.py"))

    def test_pubg_case_runtime_description_documents_default_loop_plan(self):
        self.assertEqual(2, PUBG_CASE_DEFAULT_LOOP_COUNT)
        self.assertIn("10分钟搜房", PUBG_CASE_RUNTIME_DESCRIPTION)
        self.assertIn("10分钟开车", PUBG_CASE_RUNTIME_DESCRIPTION)
        self.assertIn("10分钟跑图", PUBG_CASE_RUNTIME_DESCRIPTION)
        self.assertIn("总测试时长60分钟", PUBG_CASE_RUNTIME_DESCRIPTION)
        self.assertIn("循环2次", PUBG_CASE_RUNTIME_DESCRIPTION)

    def test_classify_output_line_groups_launcher_logs(self):
        self.assertEqual(LOG_CATEGORY_SYSTEM, classify_output_line("[Launcher] 开始执行"))
        self.assertEqual(LOG_CATEGORY_SYSTEM, classify_output_line("hdc shell aa force-stop demo.package"))
        self.assertEqual(LOG_CATEGORY_TIME, classify_output_line("[Timer] 搜房阶段开始"))
        self.assertEqual(LOG_CATEGORY_LOGIC, classify_output_line("[Parachute] 检测到跳伞按钮，开始监控航线距离"))
        self.assertEqual(
            LOG_CATEGORY_LOGIC,
            classify_output_line("[AutoLog][逻辑日志] 观察现象=发现房体 | 当前目标=进门点 | 要做的事=绕行 | 结果=等待"),
        )
        self.assertEqual(LOG_CATEGORY_UI, classify_output_line("执行点击: open_door"))
        self.assertEqual(LOG_CATEGORY_OTHER, classify_output_line("plain unclassified line"))

    def test_filter_output_text_keeps_requested_category_only(self):
        text = (
            "[Launcher] 开始执行\n"
            "[Timer] 搜房阶段开始\n"
            "[AutoLog][逻辑日志] 观察现象=发现房体 | 当前目标=进门点 | 要做的事=绕行 | 结果=等待\n"
            "执行点击: open_door\n"
            "plain unclassified line\n"
        )

        self.assertEqual(text, filter_output_text(text, LOG_FILTER_ALL))
        self.assertEqual("[Timer] 搜房阶段开始\n", filter_output_text(text, LOG_CATEGORY_TIME))
        self.assertEqual("执行点击: open_door\n", filter_output_text(text, LOG_CATEGORY_UI))

    def test_output_log_buffer_preserves_all_text_when_filter_changes(self):
        window = LauncherWindow.__new__(LauncherWindow)
        window.output_log_filter = LOG_FILTER_ALL
        window.output_log_entries = []
        text = (
            "[Launcher] 开始执行\n"
            "[Timer] 搜房阶段开始\n"
            "[AutoLog][逻辑日志] 观察现象=发现房体 | 当前目标=进门点 | 要做的事=绕行 | 结果=等待\n"
        )

        LauncherWindow._record_output_text(window, text)
        window.output_log_filter = LOG_CATEGORY_TIME

        self.assertEqual("[Timer] 搜房阶段开始\n", LauncherWindow._filtered_output_text(window))
        self.assertEqual(text, LauncherWindow._all_output_text(window))

    def test_archive_run_artifacts_skips_preview_video_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            process_temp_dir = log_dir / "process_temp_logs"
            process_save_dir = log_dir / "process_save_frames"
            process_temp_dir.mkdir(parents=True)
            (process_temp_dir / "frame_1.jpg").write_text("frame", encoding="utf-8")

            with mock.patch.object(autogame_utils, "TEMP_DIR", temp_root), mock.patch.object(
                autogame_utils,
                "LOG_DIR",
                log_dir,
            ), mock.patch.object(autogame_utils, "PROCESS_TEMP_LOGS_DIR", process_temp_dir), mock.patch.object(
                autogame_utils,
                "PROCESS_SAVE_FRAMES_DIR",
                process_save_dir,
            ), mock.patch.object(autogame_utils, "_create_preview_video") as create_video:
                archive_dir = autogame_utils.archive_run_artifacts(
                    run_index=1,
                    source="test",
                    extra_metadata={
                        "batch_start_timestamp": "20260615120000",
                        "run_start_timestamp": "20260615120100",
                    },
                )

            metadata = json.loads((archive_dir / "archive_info.json").read_text(encoding="utf-8"))
            create_video.assert_not_called()
            self.assertIsNone(metadata["preview_video"])
            self.assertIsNone(metadata["preview_video_source"])
            self.assertFalse((archive_dir / "preview_10fps.mp4").exists())

    def test_archive_run_artifacts_generates_preview_video_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            log_dir = temp_root / "logs"
            process_temp_dir = log_dir / "process_temp_logs"
            process_save_dir = log_dir / "process_save_frames"
            process_temp_dir.mkdir(parents=True)
            (process_temp_dir / "frame_1.jpg").write_text("frame", encoding="utf-8")

            with mock.patch.object(autogame_utils, "TEMP_DIR", temp_root), mock.patch.object(
                autogame_utils,
                "LOG_DIR",
                log_dir,
            ), mock.patch.object(autogame_utils, "PROCESS_TEMP_LOGS_DIR", process_temp_dir), mock.patch.object(
                autogame_utils,
                "PROCESS_SAVE_FRAMES_DIR",
                process_save_dir,
            ), mock.patch.object(
                autogame_utils,
                "_create_preview_video",
                return_value="preview_10fps.mp4",
            ) as create_video:
                archive_dir = autogame_utils.archive_run_artifacts(
                    run_index=1,
                    source="test",
                    extra_metadata={
                        "batch_start_timestamp": "20260615120000",
                        "run_start_timestamp": "20260615120100",
                    },
                    generate_preview_video=True,
                )

            metadata = json.loads((archive_dir / "archive_info.json").read_text(encoding="utf-8"))
            create_video.assert_called_once()
            self.assertEqual("preview_10fps.mp4", metadata["preview_video"])
            self.assertEqual("process_temp_logs", metadata["preview_video_source"])

    def test_launcher_archive_passes_generate_video_toggle_to_archiver(self):
        window = LauncherWindow.__new__(LauncherWindow)
        window.current_plan = {
            "mode": "direct",
            "project_case": "Auto_PUBG_ALL",
            "target_case": "auto_pubg",
            "testcase_label": None,
            "inactivity_timeout_minutes": 5.0,
            "generate_preview_video": True,
        }
        window.output_log_entries = [(LOG_CATEGORY_SYSTEM, "[Launcher] run\n")]
        window.current_run_output_start = 0
        window.current_run_timed_out = False
        window.current_run_stream_disconnected = False
        window.current_run_stream_disconnect_startup = False
        window.current_run_stream_disconnect_message = ""
        window.current_run_stream_started = True
        window.current_run_sp_started = False
        window.current_run_sp_state = {}
        window.current_batch_start_timestamp = "20260615120000"
        window.current_run_start_timestamp = "20260615120100"
        window.messages = []
        window._log_message = lambda text, level=None: window.messages.append(text)

        with mock.patch("launcher.archive_run_artifacts", return_value=Path("/tmp/archive")) as archive_mock:
            LauncherWindow._archive_run_outputs(window, 1, 0)

        self.assertTrue(archive_mock.call_args.kwargs["generate_preview_video"])

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

    def test_test_profile_maps_to_expected_screen_mode(self):
        self.assertEqual("0", resolve_screen_mode_for_test_profile("power"))
        self.assertEqual("1", resolve_screen_mode_for_test_profile("function"))

    def test_test_profile_radio_selection_defaults_to_power(self):
        self.assertEqual("power", resolve_test_profile_from_radio_selection(True, False))
        self.assertEqual("function", resolve_test_profile_from_radio_selection(False, True))
        self.assertEqual("power", resolve_test_profile_from_radio_selection(False, False))

    def test_write_screen_mode_config_preserves_other_config_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "screen_mode": "0",
                        "touch_backend": "sendevent",
                        "width": 768,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            write_screen_mode_config("1", config_path)

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual("1", config["screen_mode"])
            self.assertEqual("sendevent", config["touch_backend"])
            self.assertEqual(768, config["width"])

    def test_launcher_plan_env_values_include_profile_and_case_loop_count(self):
        env_values = build_launcher_plan_env_values(
            {
                "test_profile": "function",
                "screen_mode": "1",
                "case_loop_count": 2,
            }
        )

        self.assertEqual("function", env_values["AUTOGAME_TEST_PROFILE"])
        self.assertEqual("1", env_values["AUTOGAME_SCREEN_MODE"])
        self.assertEqual("2", env_values["AUTOGAME_SINGLE_CASE_LOOPS"])

    def test_safety_check_logs_retry_when_device_status_unavailable(self):
        window = LauncherWindow.__new__(LauncherWindow)
        window.batch_active = True
        window.current_plan = {
            "run_count": 1,
            "safe_temp": 40.0,
            "safe_battery": 30,
        }
        window.process = None
        window.stop_requested = False
        window.current_run_index = 0
        window.safety_timer = FakeTimer()
        window.status_messages = []
        window.runtime_messages = []
        window.output_messages = []
        window._set_status = lambda text: window.status_messages.append(text)
        window._set_runtime = lambda text: window.runtime_messages.append(text)
        window._log_message = lambda text, level=None: window.output_messages.append(text)
        window._finish_batch = lambda message: self.fail(f"unexpected finish: {message}")
        window._launch_iteration = lambda *args: self.fail("should not launch without device status")

        with mock.patch("launcher.get_battery_temperature_c", return_value=None), mock.patch(
            "launcher.get_battery_capacity",
            return_value=None,
        ):
            LauncherWindow._check_and_start_if_safe(window)

        self.assertTrue(window.safety_timer.started)
        self.assertIn("无法读取手机温度或电量，稍后重试。", window.status_messages)
        self.assertTrue(any("安全检查" in text and "无法读取" in text for text in window.output_messages))

    def test_run_hdc_shell_uses_short_timeout_for_launcher_responsiveness(self):
        observed = {}

        def fake_run(_cmd, **kwargs):
            observed["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(_cmd, 0, stdout="ok\n", stderr="")

        with mock.patch("launcher.resolve_hdc_executable", return_value="hdc"), mock.patch(
            "launcher.subprocess.run",
            side_effect=fake_run,
        ):
            result = run_hdc_shell("echo ok")

        self.assertEqual("ok", result)
        self.assertLessEqual(observed["timeout"], 5)

    def test_capture_stream_check_for_hdc_mode_validates_snapshot_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            created_image = temp_root / "created.jpeg"

            def fake_run(command, **_kwargs):
                if command[:3] == ["hdc", "shell", "snapshot_display"]:
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
                if command[:3] == ["hdc", "file", "recv"]:
                    from PIL import Image

                    local_path = Path(command[4])
                    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(local_path)
                    created_image.write_text("ok", encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
                if command[:3] == ["hdc", "shell", "rm"]:
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
                raise AssertionError(f"unexpected command: {command}")

            with mock.patch("launcher.resolve_hdc_executable", return_value="hdc"), mock.patch(
                "launcher.subprocess.run",
                side_effect=fake_run,
            ):
                result = check_capture_stream_for_screen_mode("1", temp_root=temp_root)

            self.assertTrue(result.ok, result.message)
            self.assertTrue(created_image.exists())

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

    def test_hidden_subprocess_kill_closes_stdout_pipe_to_unblock_reader(self):
        process = HiddenSubprocess()
        process.setProgram("AutoGameLauncher.exe")
        fake_process = KillableFakeProcess()

        with mock.patch(
            "launcher.hidden_subprocess_kwargs",
            return_value={},
        ), mock.patch(
            "launcher.subprocess.Popen",
            return_value=fake_process,
        ):
            process.start()
            self.assertTrue(fake_process.stdout.read_started.wait(timeout=1))
            process.kill()

        self.assertTrue(fake_process.killed)
        self.assertTrue(fake_process.stdout.closed)

    def test_process_launch_tracer_is_disabled_off_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracer = WindowsProcessLaunchTracer(
                log_dir=Path(temp_dir),
                os_name="posix",
            )

            self.assertIsNone(tracer.start("test"))

    def test_process_launch_tracer_starts_hidden_powershell_on_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_process = mock.Mock()
            fake_process.pid = 4321
            fake_process.poll.return_value = None
            tracer = WindowsProcessLaunchTracer(
                log_dir=Path(temp_dir),
                os_name="nt",
                root_pid=1234,
            )

            with mock.patch(
                "launcher.hidden_subprocess_kwargs",
                return_value={"creationflags": 123},
            ), mock.patch(
                "launcher.subprocess.Popen",
                return_value=fake_process,
            ) as popen:
                log_path = tracer.start("testcase")

            self.assertIsNotNone(log_path)
            self.assertEqual(Path(temp_dir), log_path.parent)
            self.assertIn("process_launch_trace_", log_path.name)
            self.assertEqual("powershell.exe", popen.call_args.args[0][0])
            self.assertIn("-EncodedCommand", popen.call_args.args[0])
            self.assertEqual(123, popen.call_args.kwargs["creationflags"])
            self.assertEqual(subprocess.DEVNULL, popen.call_args.kwargs["stdin"])
            self.assertEqual(subprocess.DEVNULL, popen.call_args.kwargs["stdout"])
            self.assertEqual(subprocess.DEVNULL, popen.call_args.kwargs["stderr"])

    def test_process_launch_tracer_script_includes_polling_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracer = WindowsProcessLaunchTracer(
                log_dir=Path(temp_dir),
                os_name="nt",
                root_pid=1234,
            )

            script = tracer._build_powershell_script(Path(temp_dir) / "trace.log", "test")

        self.assertIn("EVENT_CREATE", script)
        self.assertIn("POLL_CREATE", script)
        self.assertIn("Get-AutoGameProcesses", script)
        self.assertIn("Start-Sleep -Milliseconds 200", script)

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

        self.assertEqual(
            FakeSubprocessModule.CREATE_NO_WINDOW,
            kwargs["creationflags"],
        )
        self.assertEqual(
            FakeSubprocessModule.STARTF_USESHOWWINDOW,
            kwargs["startupinfo"].dwFlags,
        )
        self.assertEqual(0, kwargs["startupinfo"].wShowWindow)

    def test_hidden_subprocess_kwargs_is_empty_for_non_windows(self):
        self.assertEqual({}, hidden_subprocess_kwargs(os_name="posix"))

    def test_install_hidden_subprocess_patch_is_noop_for_non_windows(self):
        self.assertFalse(install_hidden_subprocess_patch(os_name="posix"))

    def test_window_suppression_is_disabled_off_windows(self):
        self.assertFalse(is_window_suppression_enabled(os_name="posix"))
        self.assertFalse(start_hidden_subprocess_window_suppressor(os_name="posix"))

    def test_window_suppressor_hides_any_descendant_window(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")
        suppressor._process_snapshot = {
            100: (1, "autogamelauncher.exe"),
            200: (100, "workerhelper.exe"),
            300: (200, "unknownpopup.exe"),
        }

        self.assertEqual((False, "root"), suppressor._should_hide_process(100, "autogamelauncher.exe"))
        self.assertEqual((True, "descendant"), suppressor._should_hide_process(300, "unknownpopup.exe"))

    def test_window_suppressor_hides_direct_hdc_without_parent_chain(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")
        suppressor._process_snapshot = {
            400: (4, "hdc.exe"),
        }

        self.assertEqual((True, "direct"), suppressor._should_hide_process(400, "hdc.exe"))

    def test_window_suppressor_hides_xdc_temp_path_without_parent_chain(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")
        process_path = r"C:\Temp\XDC\abc\unknownpopup.exe"

        self.assertEqual(
            (True, "xdc-temp-path"),
            suppressor._should_hide_process(500, "unknownpopup.exe", process_path),
        )

    def test_window_suppressor_hides_xdc_temp_title_without_parent_chain(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")
        process_path = r"C:\Windows\System32\conhost.exe"
        window_title = r"C:\Temp\XDC\76d836f2xxxx\hdc.lnk"

        self.assertEqual(
            (True, "xdc-temp-title"),
            suppressor._should_hide_process(501, "conhost.exe", process_path, window_title),
        )

    def test_window_suppressor_respects_excluded_process(self):
        suppressor = WindowsSubprocessWindowSuppressor(
            root_pid=100,
            os_name="nt",
            excluded_processes=("keep.exe",),
        )
        suppressor._process_snapshot = {
            100: (1, "autogamelauncher.exe"),
            500: (100, "keep.exe"),
        }

        self.assertEqual((False, "excluded"), suppressor._should_hide_process(500, "keep.exe"))

    def test_window_suppressor_logs_new_descendant_processes(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")

        suppressor._update_process_snapshot({
            100: (1, "autogamelauncher.exe"),
            200: (100, "workerhelper.exe"),
        })
        with mock.patch("aw.autogame.tools.ProcessUtils.LOGGER") as logger:
            suppressor._update_process_snapshot({
                100: (1, "autogamelauncher.exe"),
                200: (100, "workerhelper.exe"),
                300: (200, "unknownpopup.exe"),
            })

        logger.info.assert_called()
        message = logger.info.call_args.args[0]
        self.assertIn("subprocess process create", message)
        self.assertEqual(300, logger.info.call_args.args[1])
        self.assertEqual(200, logger.info.call_args.args[2])
        self.assertEqual("unknownpopup.exe", logger.info.call_args.args[3])
        self.assertEqual("", logger.info.call_args.args[4])
        self.assertEqual("descendant", logger.info.call_args.args[5])

    def test_window_suppressor_logs_direct_hdc_without_parent_chain(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")

        suppressor._update_process_snapshot({100: (1, "autogamelauncher.exe")})
        with mock.patch("aw.autogame.tools.ProcessUtils.LOGGER") as logger:
            suppressor._update_process_snapshot({
                100: (1, "autogamelauncher.exe"),
                400: (4, "hdc.exe"),
            })

        logger.info.assert_called()
        self.assertEqual(400, logger.info.call_args.args[1])
        self.assertEqual(4, logger.info.call_args.args[2])
        self.assertEqual("hdc.exe", logger.info.call_args.args[3])
        self.assertEqual("", logger.info.call_args.args[4])
        self.assertEqual("direct", logger.info.call_args.args[5])

    def test_window_suppressor_logs_new_process_path(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")
        suppressor._process_path = mock.Mock(return_value=r"C:\Temp\XDC\abc\hdc.lnk")

        suppressor._update_process_snapshot({100: (1, "autogamelauncher.exe")})
        with mock.patch("aw.autogame.tools.ProcessUtils.LOGGER") as logger:
            suppressor._update_process_snapshot({
                100: (1, "autogamelauncher.exe"),
                400: (4, "hdc.exe"),
            })

        logger.info.assert_called()
        self.assertIn("path=%s", logger.info.call_args.args[0])
        self.assertEqual(r"C:\Temp\XDC\abc\hdc.lnk", logger.info.call_args.args[4])

    def test_window_suppressor_logs_xdc_temp_process_without_parent_chain(self):
        suppressor = WindowsSubprocessWindowSuppressor(root_pid=100, os_name="nt")
        suppressor._process_path = mock.Mock(return_value=r"C:\Temp\XDC\abc\unknownpopup.exe")

        suppressor._update_process_snapshot({100: (1, "autogamelauncher.exe")})
        with mock.patch("aw.autogame.tools.ProcessUtils.LOGGER") as logger:
            suppressor._update_process_snapshot({
                100: (1, "autogamelauncher.exe"),
                500: (4, "unknownpopup.exe"),
            })

        logger.info.assert_called()
        self.assertEqual(500, logger.info.call_args.args[1])
        self.assertEqual(4, logger.info.call_args.args[2])
        self.assertEqual("unknownpopup.exe", logger.info.call_args.args[3])
        self.assertEqual(r"C:\Temp\XDC\abc\unknownpopup.exe", logger.info.call_args.args[4])
        self.assertEqual("xdc-temp-path", logger.info.call_args.args[5])

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

    def test_hidden_subprocess_context_hides_target_command_and_restores_popen(self):
        class WindowsSubprocessModule(FakeSubprocessModule):
            class Popen(FakePopen):
                calls = []

        original_popen = WindowsSubprocessModule.Popen
        command = r"C:\Program Files (x86)\ISEP\bin\icpm_xdc.exe --version"

        with hidden_subprocess_context(
            os_name="nt",
            subprocess_module=WindowsSubprocessModule,
            target_executables=("icpm_xdc.exe",),
        ):
            self.assertTrue(issubclass(WindowsSubprocessModule.Popen, original_popen))
            WindowsSubprocessModule.Popen(command)

        self.assertIs(original_popen, WindowsSubprocessModule.Popen)
        args, kwargs = original_popen.calls[-1]
        self.assertEqual((command,), args)
        self.assertEqual(
            FakeSubprocessModule.CREATE_NO_WINDOW,
            kwargs["creationflags"],
        )
        self.assertEqual(
            FakeSubprocessModule.STARTF_USESHOWWINDOW,
            kwargs["startupinfo"].dwFlags,
        )
        self.assertEqual(0, kwargs["startupinfo"].wShowWindow)

    def test_hidden_subprocess_context_leaves_non_target_command_unchanged(self):
        class WindowsSubprocessModule(FakeSubprocessModule):
            class Popen(FakePopen):
                calls = []

        original_popen = WindowsSubprocessModule.Popen

        with hidden_subprocess_context(
            os_name="nt",
            subprocess_module=WindowsSubprocessModule,
            target_executables=("icpm_xdc.exe",),
        ):
            WindowsSubprocessModule.Popen(["hdc", "list", "targets"])

        args, kwargs = original_popen.calls[-1]
        self.assertEqual((["hdc", "list", "targets"],), args)
        self.assertNotIn("creationflags", kwargs)
        self.assertNotIn("startupinfo", kwargs)

    def test_hidden_subprocess_context_hide_all_hides_hdc_command(self):
        class WindowsSubprocessModule(FakeSubprocessModule):
            class Popen(FakePopen):
                calls = []

        original_popen = WindowsSubprocessModule.Popen

        with hidden_subprocess_context(
            os_name="nt",
            subprocess_module=WindowsSubprocessModule,
            hide_all=True,
        ):
            WindowsSubprocessModule.Popen(["hdc", "list", "targets"])

        args, kwargs = original_popen.calls[-1]
        self.assertEqual((["hdc", "list", "targets"],), args)
        self.assertEqual(
            FakeSubprocessModule.CREATE_NO_WINDOW,
            kwargs["creationflags"],
        )
        self.assertEqual(
            FakeSubprocessModule.STARTF_USESHOWWINDOW,
            kwargs["startupinfo"].dwFlags,
        )

    def test_hidden_subprocess_context_routes_os_system_through_hidden_call(self):
        class WindowsSubprocessModule(FakeSubprocessModule):
            class Popen(FakePopen):
                calls = []
            call = mock.Mock(return_value=7)

        class WindowsOsModule:
            name = "nt"
            system = mock.Mock(return_value=99)

        original_system = WindowsOsModule.system

        with hidden_subprocess_context(
            os_name="nt",
            subprocess_module=WindowsSubprocessModule,
            os_module=WindowsOsModule,
            hide_all=True,
        ):
            result = WindowsOsModule.system("hdc list targets")

        self.assertEqual(7, result)
        WindowsSubprocessModule.call.assert_called_once()
        self.assertEqual("hdc list targets", WindowsSubprocessModule.call.call_args.args[0])
        self.assertTrue(WindowsSubprocessModule.call.call_args.kwargs["shell"])
        self.assertEqual(
            FakeSubprocessModule.CREATE_NO_WINDOW,
            WindowsSubprocessModule.call.call_args.kwargs["creationflags"],
        )
        self.assertIs(original_system, WindowsOsModule.system)

    def test_hdc_command_args_converts_shell_command_without_local_cmd(self):
        self.assertEqual(
            ["D:/tools/hdc.exe", "shell", "echo 1 > /sys/class/hiz"],
            hdc_command_args(
                'hdc shell "echo 1 > /sys/class/hiz"',
                hdc_executable="D:/tools/hdc.exe",
            ),
        )

    def test_hdc_command_args_preserves_target_selector(self):
        self.assertEqual(
            ["hdc", "-t", "SERIAL", "shell", "uinput -T -c 2290 204"],
            hdc_command_args(
                "hdc -t SERIAL shell uinput -T -c 2290 204",
                hdc_executable="hdc",
            ),
        )

    def test_run_shell_uses_direct_hdc_args_without_shell_true(self):
        from aw.autogame.tools import Utils

        with mock.patch(
            "aw.autogame.tools.Utils.hdc_command_args",
            return_value=["hdc", "shell", "uinput -T -c 1 2"],
        ), mock.patch(
            "aw.autogame.tools.Utils.hidden_subprocess_kwargs",
            return_value={"creationflags": 123},
        ), mock.patch(
            "aw.autogame.tools.Utils.subprocess.run",
        ) as run:
            Utils.run_shell("hdc shell uinput -T -c 1 2")

        run.assert_called_once()
        self.assertEqual(["hdc", "shell", "uinput -T -c 1 2"], run.call_args.args[0])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(123, run.call_args.kwargs["creationflags"])

    def test_run_testcase_entry_wraps_xdevice_with_icpm_xdc_hidden_context(self):
        xdevice_module = types.ModuleType("xdevice")
        xdevice_main_module = types.ModuleType("xdevice.__main__")
        xdevice_main_module.main_process = mock.Mock()
        context = mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {
                "xdevice": xdevice_module,
                "xdevice.__main__": xdevice_main_module,
            },
        ), mock.patch(
            "launcher.hidden_subprocess_context",
            return_value=context,
        ) as hidden_context:
            with mock.patch(
                "launcher.start_hidden_subprocess_window_suppressor",
                return_value=True,
            ) as suppressor:
                run_testcase_entry("testcases/pubg/pubg_full_flow/auto_pubg")

        suppressor.assert_called_once_with()
        hidden_context.assert_called_once_with(
            target_executables=("icpm_xdc.exe", "hdc.exe", "hdc"),
            hide_all=True,
        )
        context.__enter__.assert_called_once_with()
        context.__exit__.assert_called_once()
        xdevice_main_module.main_process.assert_called_once_with(
            "run -l testcases/pubg/pubg_full_flow/auto_pubg"
        )

    def test_run_testcase_entry_wraps_xdevice_with_icpm_xdc_hidden_context_without_window_suppressor_patch(self):
        xdevice_module = types.ModuleType("xdevice")
        xdevice_main_module = types.ModuleType("xdevice.__main__")
        xdevice_main_module.main_process = mock.Mock()
        context = mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {
                "xdevice": xdevice_module,
                "xdevice.__main__": xdevice_main_module,
            },
        ), mock.patch(
            "launcher.hidden_subprocess_context",
            return_value=context,
        ), mock.patch(
            "launcher.start_hidden_subprocess_window_suppressor",
            return_value=False,
        ):
            run_testcase_entry("testcases/pubg/pubg_full_flow/auto_pubg")

        xdevice_main_module.main_process.assert_called_once_with(
            "run -l testcases/pubg/pubg_full_flow/auto_pubg"
        )

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
