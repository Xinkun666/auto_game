import time
import shutil
import atexit
import threading
from aw.autogame.tools.GameFrameWorker import FrameWorker
from aw.autogame.tools.GameLaunchProfile import should_preserve_game_process
from aw.autogame.tools.Utils import *
from aw.autogame.stream_client.stream_client import global_buffer, StreamClient, HDCSnapshotClient, HOSScrcpyStreamClient


def _resolve_stream_rotation(screen_w, screen_h, display_rotation):
    rotation_mode = normalize_rotation(display_rotation)
    if rotation_mode is None:
        rotation_mode = infer_landscape_rotation(screen_w, screen_h)
        print(f"[Rotation] 未获取到屏幕旋转，拉流使用兜底 rotation={rotation_mode}")
    return rotation_mode


def create_stream_client_for_mode(
    screen_mode,
    buffer,
    screen_w,
    screen_h,
    width,
    height,
    display_rotation=None,
):
    mode = str(screen_mode)
    if mode == "0":
        rotation_mode = _resolve_stream_rotation(screen_w, screen_h, display_rotation)
        return StreamClient(buffer, rotation_mode=rotation_mode)
    if mode == "1":
        return HDCSnapshotClient(buffer)
    if mode == "2":
        return HOSScrcpyStreamClient(buffer)
    raise ValueError(f"unsupported screen_mode: {screen_mode}")

class GameAutomator:
    def __init__(self, driver, logger, device_sn=None):
        self.driver = driver
        self.logger = logger
        self.device_sn = str(device_sn or "").strip()

        self.screen_w, self.screen_h = get_resolution(
            device_sn=self.device_sn or None
        )
        set_runtime_screen_resolution_env(self.screen_w, self.screen_h)
        self.W, self.H = get_wh()
        self.screen_mode = get_screen_mode()
        display_rotation = get_display_rotation() if str(self.screen_mode) == "0" else None
        self.client = create_stream_client_for_mode(
            self.screen_mode,
            global_buffer,
            self.screen_w,
            self.screen_h,
            self.W,
            self.H,
            display_rotation,
        )
        if self.screen_mode == "0":
            self.client.start_backend(lowh=0, highh=10000, skip=20, width=self.W, height=self.H)
        self.client.set_save_frame(True)
        self.processor = FrameWorker(
            global_buffer,
            driver=self.driver,
            logger=self.logger,
            stream_client=self.client,
        )
        self.is_cleaned_up = False
        self._cleanup_lock = threading.Lock()
        self._client_stop_lock = threading.Lock()
        self._monitor_stop_event = threading.Event()
        self._monitor_thread = None

        atexit.register(self.cleanup)

        self._clear_temp_logs()

    def _clear_temp_logs(self):
        temp_log_dir = str(resolve_process_temp_logs_dir())
        temp_save_dir = str(resolve_process_save_frames_dir())

        try:
            if os.path.exists(temp_log_dir):
                print(f"【系统】正在清空临时日志目录: {temp_log_dir}")
                # 遍历目录内的所有内容并删除
                for filename in os.listdir(temp_log_dir):
                    file_path = os.path.join(temp_log_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # 删除文件或链接
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # 删除子目录
                    except Exception as e:
                        print(f"无法删除 {file_path}: {e}")
            else:
                # 如果目录不存在，则创建它，确保后续写入不报错
                os.makedirs(temp_log_dir, exist_ok=True)
                print(f"【系统】创建临时日志目录: {temp_log_dir}")
        except Exception as e:
            print(f"清空日志目录时出错: {e}")

        try:
            if not bool(getattr(self.client, "save_frame", False)):
                if os.path.isdir(temp_save_dir):
                    shutil.rmtree(temp_save_dir)
                    print(f"【系统】已移除未启用的保存帧目录: {temp_save_dir}")
                elif os.path.exists(temp_save_dir):
                    os.unlink(temp_save_dir)
                return
            if os.path.exists(temp_save_dir):
                print(f"【系统】正在清空临时保存目录: {temp_save_dir}")
                # 遍历目录内的所有内容并删除
                for filename in os.listdir(temp_save_dir):
                    file_path = os.path.join(temp_save_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # 删除文件或链接
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # 删除子目录
                    except Exception as e:
                        print(f"无法删除 {file_path}: {e}")
            else:
                # 如果目录不存在，则创建它，确保后续写入不报错
                os.makedirs(temp_save_dir, exist_ok=True)
                print(f"【系统】创建临时保存目录: {temp_save_dir}")
        except Exception as e:
            print(f"清空保存目录时出错: {e}")

    def _set_hiz_mode(self, active: bool):
        """控制硬件 HIZ 模式，并同步控制充电开关。仅供测试使用。"""
        try:
            if active:
                print("【系统】启用 HIZ 模式并关闭充电...")
                run_shell('hdc shell "echo 1 > /sys/class/hw_power/charger/charge_data/enable_hiz"')
                run_shell('hdc shell "echo stopsink > /sys/class/hw_power/charger/charge_data/plugusb"')
            else:
                print("【系统】关闭 HIZ 模式并开启充电...")
                run_shell('hdc shell "echo 0 > /sys/class/hw_power/charger/charge_data/enable_hiz"')
                run_shell('hdc shell "echo startsink > /sys/class/hw_power/charger/charge_data/plugusb"')
        except Exception as e:
            print(f"设置硬件状态失败: {e}")

    def cleanup(self, app_list = ()):
        """彻底清理：恢复硬件模式并关闭所有应用。通常在所有测试结束后手动调用。"""
        cleanup_lock = getattr(self, "_cleanup_lock", None)
        if cleanup_lock is None:
            cleanup_lock = threading.Lock()
            self._cleanup_lock = cleanup_lock
        with cleanup_lock:
            if self.is_cleaned_up:
                return
            self.is_cleaned_up = True

        print("\n>>> 开始执行深度清理程序...")
        monitor_stop_event = getattr(self, "_monitor_stop_event", None)
        if monitor_stop_event is not None:
            monitor_stop_event.set()

        try:
            self.processor.stop()
        except Exception as exc:
            print(f"[Cleanup] FrameWorker 停止失败: {exc}")

        self._stop_client("Cleanup")

        monitor_thread = getattr(self, "_monitor_thread", None)
        if (
            monitor_thread is not None
            and monitor_thread.is_alive()
            and threading.current_thread() is not monitor_thread
        ):
            monitor_thread.join(timeout=1.0)

        self._set_hiz_mode(False)

        try:
            if should_preserve_game_process():
                print("当前测试处于保留进程模式：仅清理自动化资源，不强杀应用进程。")
                app_list = ()
            if len(app_list) > 0:
                for app in app_list:
                    print(f"停止应用: {app}")
                    run_shell(f'hdc shell aa force-stop {app}')
        except:
            pass
        print(">>> 环境已恢复。")

    def _stop_client(self, source):
        client_stop_lock = getattr(self, "_client_stop_lock", None)
        if client_stop_lock is None:
            client_stop_lock = threading.Lock()
            self._client_stop_lock = client_stop_lock
        with client_stop_lock:
            try:
                self.client.stop()
                return True
            except Exception as exc:
                print(f"[{source}] 视频流停止失败: {exc}")
                return False

    def _monitor_worker(self):
        print("[监控] 任务状态监控已启动...")
        stop_event = getattr(self, "_monitor_stop_event", None)
        while stop_event is None or not stop_event.is_set():
            # 只要 processor 停了，不管是 finished 还是意外中断
            if getattr(self.processor, 'finished', False) or not self.processor.running:
                print("\n[业务通知] 流程结束，正在中断流连接...")
                self._stop_client("监控")
                break
            if stop_event is not None:
                stop_event.wait(0.5)
            else:
                time.sleep(0.5)

    def start(self):
        try:
            self._set_hiz_mode(True)

            self.processor.start()

            self._monitor_stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_worker,
                daemon=True,
                name="GameAutomatorMonitor",
            )
            self._monitor_thread.start()

            print(">>> 正在启动视频流服务（阻塞中）...")
            if self.screen_mode == "0":
                self.client.run(lowh=0, highh=10000, skip=20, width=self.W, height=self.H)
            elif self.screen_mode in {"1", "2"}:
                self.client.run()

        except Exception as e:
            print(f"\n[运行异常] GameAutomator 遇到错误: {e}")
            raise
        finally:
            self.processor.stop()
            print(">>> 自动化处理阶段已结束，正在释放控制权回传主脚本...")

        if getattr(self.processor, "failed", False):
            raise RuntimeError(self.processor.failure_reason or "自动化执行失败")


if __name__ == "__main__":
    # 仅用于独立调试
    automator = GameAutomator()
    automator.start()
