import unittest
import json
import subprocess
import sys
import tempfile
import types
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
    run_testcase_entry,
    WindowsProcessLaunchTracer,
)
from aw.autogame.tools.ProcessUtils import is_window_suppression_enabled
from aw.autogame.tools.ProcessUtils import hidden_subprocess_kwargs
from aw.autogame.tools.ProcessUtils import hidden_subprocess_context
from aw.autogame.tools.ProcessUtils import hdc_command_args
from aw.autogame.tools.ProcessUtils import install_hidden_subprocess_patch
from aw.autogame.tools.ProcessUtils import start_hidden_subprocess_window_suppressor
from aw.autogame.tools.ProcessUtils import WindowsSubprocessWindowSuppressor


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
            FakeSubprocessModule.CREATE_NO_WINDOW | FakeSubprocessModule.DETACHED_PROCESS,
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
            FakeSubprocessModule.CREATE_NO_WINDOW | FakeSubprocessModule.DETACHED_PROCESS,
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
            FakeSubprocessModule.CREATE_NO_WINDOW | FakeSubprocessModule.DETACHED_PROCESS,
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
            FakeSubprocessModule.CREATE_NO_WINDOW | FakeSubprocessModule.DETACHED_PROCESS,
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
