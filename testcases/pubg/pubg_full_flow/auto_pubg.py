# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import json
import os
import time
from pathlib import Path

project_case = "Auto_PUBG_ALL"  # label-tool exported resource directory
target_case = "auto_pubg"  # runtime script under customs_game_examples/<project_case>
GAME_DISPLAY_NAME = "和平精英"  # name shown inside SP app selector
GAME_PACKAGE_NAME = "com.tencent.tmgp.pubgmhd.hw"  # package launched for function tests
STARTUP_WAIT_SECONDS = 10

os.environ["TARGET_PROJECT_CASE"] = project_case
os.environ["TARGET_GAME_CASE"] = target_case

from devicetest.core.test_case import TestCase
from hypium import BY, UiDriver
from hypium.action.os_hypium.device_logger import DeviceLogger
from aw.autogame.tools.GameAutomator import GameAutomator
from aw.autogame.tools.GameLaunchProfile import (
    DEFAULT_SP_PACKAGE,
    cleanup_packages_for_test_profile,
    should_use_sp_recording_for_profile,
)
from aw.autogame.tools.Utils import (
    analyze_txt,
    get_display_rotation,
    normalize_rotation,
    resolve_process_save_frames_dir,
)

PERF_TOOL_PACKAGE = DEFAULT_SP_PACKAGE

class auto_pubg(TestCase):
    def __init__(self, controllers):
        self.TAG = self.__class__.__name__
        TestCase.__init__(self, self.TAG, controllers)

        self.tests = ["test_step"]
        self.driver = UiDriver(self.device1)
        self.automator = None
        self.task_name = os.environ.get("TARGET_GAME_CASE") or target_case
        self.device_logger = DeviceLogger(self.driver)
        self.device_log_started = False
        self.log_path = os.environ.get("AUTOGAME_DEVICE_LOG_PATH") or f"aw/autogame/temp/logs/{self.task_name}.txt"
        self.frame_path = str(resolve_process_save_frames_dir())
        self.game_display_name = GAME_DISPLAY_NAME
        self.game_package = GAME_PACKAGE_NAME
        self.perf_tool_package = PERF_TOOL_PACKAGE
        self.test_profile = os.environ.get("AUTOGAME_TEST_PROFILE")

    def _use_sp_recording(self) -> bool:
        return should_use_sp_recording_for_profile(self.test_profile)

    def _preserve_game_process(self) -> bool:
        return os.environ.get("AUTOGAME_PRESERVE_GAME_PROCESS", "0") == "1"

    def _validate_runtime_entry(self):
        runtime_file = Path("aw") / "autogame" / "customs_game_examples" / project_case / f"{target_case}.py"
        info_file = Path("aw") / "autogame" / "customs_examples" / project_case / "info.py"
        missing = [str(path) for path in (runtime_file, info_file) if not path.exists()]
        if missing:
            raise RuntimeError(
                "testcase template values are not ready. "
                "Copy this template, then update project_case/target_case and export label resources. "
                f"Missing: {', '.join(missing)}"
            )

    def _write_device_log_state(self, event_name, stop_ok=None, error=""):
        archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
        if not archive_dir:
            return

        try:
            log_exists = os.path.exists(self.log_path)
            payload = {
                "event": event_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "log_path": self.log_path,
                "log_exists": log_exists,
                "log_size": os.path.getsize(self.log_path) if log_exists else 0,
                "device_log_started": self.device_log_started,
                "stop_ok": stop_ok,
                "error": str(error or ""),
            }
            os.makedirs(archive_dir, exist_ok=True)
            signal_path = os.path.join(archive_dir, "device_log_state.json")
            tmp_path = signal_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, signal_path)
        except Exception as exc:
            print(f"写入设备日志状态失败: {exc}")

    def setup(self):
        self.log.info("预置条件：设置常亮")
        self.driver.hdc("shell power-shell timeout -o 86400000")
        self.driver.Screen.set_brightness(brightness=130)

    def start_perf_tool(self):
        """封装性能工具启动逻辑"""
        max_attempts = 5
        text_input = None
        last_error = None

        for attempt in range(1, max_attempts + 1):
            print(f"正在启动性能工具并选择应用: {self.game_display_name}，attempt={attempt}/{max_attempts}")
            if attempt == 1:
                self.driver.start_app(self.perf_tool_package)
                self._dismiss_reboot_prompt_if_needed()
            else:
                self._restart_perf_tool()

            try:
                text_input = self._open_perf_tool_app_selector()
                break
            except RuntimeError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise
                print(f"未找到性能工具应用搜索输入框，重启性能工具后重试: {exc}")

        if text_input is None:
            raise last_error or RuntimeError("未找到性能工具应用搜索输入框")

        text_input.inputText(self.game_display_name)
        time.sleep(1)

        print("启动游戏并开始测试")
        self.driver.touch((0.27, 0.17))  # 点击搜索出的游戏
        time.sleep(10)
        self.driver.touch((0.49, 0.94))  # 点击开始测试
        time.sleep(1)

    def _restart_perf_tool(self):
        print("重启性能工具应用")
        self.driver.hdc(f"shell aa force-stop {self.perf_tool_package}")
        time.sleep(1)
        self.driver.start_app(self.perf_tool_package)
        time.sleep(1)

    def _dismiss_reboot_prompt_if_needed(self):
        if os.environ.get("AUTOGAME_DISMISS_REBOOT_PROMPT") != "1":
            return

        print("[Launcher] 重启后首次打开 sp，点击屏幕左下方关闭充电弹窗")
        time.sleep(1)
        self.driver.touch((0.02, 0.92))
        time.sleep(1)

    def _open_perf_tool_app_selector(self):
        comp1 = self.driver.wait_for_component(
            BY.text("应用测试").isAfter(BY.text('性能/功耗测试')),
            timeout=5,
        )
        if comp1 is not None:
            self.driver.touch(comp1)

        comp2 = self.driver.wait_for_component(
            BY.text("请选择一个应用"),
            timeout=5,
        )
        if comp2 is not None:
            self.driver.touch(comp2)

        return self._wait_for_component(
            BY.type("TextInput"),
            timeout=12,
            interval=1.0,
            desc="性能工具应用搜索输入框",
        )

    def _wait_for_component(self, selector, timeout=10, interval=1.0, desc="目标控件"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            component = self.driver.find_component(selector)
            if component is not None:
                return component
            time.sleep(interval)
        raise RuntimeError(f"未找到{desc}，请检查当前页面是否已正确进入目标界面")

    def start_device_log(self):
        print('和平精英-启动日志采集!!!')
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
            print(f"检测到旧日志，已成功删除: {self.log_path}")
        if os.path.exists(f'aw/autogame/temp/results/{self.task_name}/time.txt'):
            os.remove(f'aw/autogame/temp/results/{self.task_name}/time.txt')
            print(f'检测到旧的时间日志，已成功删除: {self.log_path}')

        print('开始抓取日志!')
        self.device_logger.start_log(self.log_path)
        self.device_log_started = True
        self._write_device_log_state("device_log_started")

    def stop_device_log(self):
        if not self.device_log_started:
            return
        stop_ok = False
        stop_error = ""
        try:
            print('自动化结束，结束抓取日志!')
            try:
                self.device_logger.stop_log()
                stop_ok = True
                print(f'日志文件保存在: {self.log_path}')
            except Exception as exc:
                stop_error = exc
                print(f'停止设备日志失败: {exc}')
        finally:
            self._write_device_log_state("device_log_stopped", stop_ok=stop_ok, error=stop_error)
            self.device_log_started = False

    def start_yuanshen(self):
        print('和平精英-启动!!!')
        # self.driver.start_app('com.tencent.tmgp.pubgmhd.hw')
        self.start_device_log()
        # time.sleep(30)

    def start_game_package(self):
        print(f"{self.game_display_name}-通过 HAP 包直接启动: {self.game_package}")
        self.driver.start_app(self.game_package)
        time.sleep(STARTUP_WAIT_SECONDS)

    def _wait_for_game_rotation(self, timeout=20, stable_rounds=3, interval=1.0):
        """
        等待从 sp 竖屏切到游戏横屏后，再初始化自动化。
        横屏一般是 90/270；为了避免正在切换中，要求连续多次读到相同横屏值。
        """
        print("等待游戏横屏旋转稳定...")
        deadline = time.time() + timeout
        last_rotation = None
        stable_count = 0

        while time.time() < deadline:
            rotation = normalize_rotation(get_display_rotation())
            print(f"[Rotation] 当前旋转角: {rotation}")

            if rotation in (90, 270):
                if rotation == last_rotation:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_rotation = rotation

                if stable_count >= stable_rounds:
                    print(f"[Rotation] 横屏已稳定: {rotation}")
                    return rotation
            else:
                last_rotation = rotation
                stable_count = 0

            time.sleep(interval)

        final_rotation = normalize_rotation(get_display_rotation())
        print(f"[Rotation] 等待超时，继续执行，当前旋转角: {final_rotation}")
        return final_rotation

    def _ensure_automator(self):
        if self.automator is not None:
            return

        self._wait_for_game_rotation()
        self.automator = GameAutomator(driver=self.driver, logger=self.log)

    def test_step(self):
        automation_completed = False
        try:
            self._validate_runtime_entry()
            # 1. 启动本地设备日志。即使后续 gRPC 断流被 launcher 杀进程，
            #    已生成的日志文件也会被 launcher 归档到本次运行目录。
            self.start_device_log()
            if self._use_sp_recording():
                self.start_perf_tool()
            else:
                self.start_game_package()
            self._ensure_automator()

            # 2. 运行自动化逻辑（现在执行完会返回了）
            print('开始游戏自动化!')
            self.automator.start()
            automation_completed = True
        finally:
            self.stop_device_log()

            if automation_completed and os.path.exists(f'aw/autogame/temp/results/{self.task_name}/time.txt'):
                result_path = f'aw/autogame/temp/results/{self.task_name}/results.txt'
                if os.path.exists(result_path):
                    os.remove(result_path)
                    print(f'检测到旧的结果日志，已成功删除: {result_path}')
                analyze_txt(self.log_path, self.frame_path, time_txt_path=f'aw/autogame/temp/results/{self.task_name}/time.txt', result_path=result_path)
                print(f'分析完成, 结果保存在 aw/autogame/temp/results/{self.task_name}/results.txt 中')

            cleanup_apps = cleanup_packages_for_test_profile(
                self.test_profile,
                game_package=self.game_package,
                sp_package=self.perf_tool_package,
            )
            if self._preserve_game_process():
                cleanup_apps = ()
                print("当前测试处于保留进程模式：仅清理自动化资源，不强杀应用进程。")
            if self.automator is not None:
                self.automator.cleanup(cleanup_apps)
            else:
                for package_name in cleanup_apps:
                    self.driver.hdc(f"shell aa force-stop {package_name}")
