import time
import shutil
import atexit
import threading
from aw.autogame.tools.GameFrameWorker import FrameWorker
from aw.autogame.tools.Utils import *
from aw.autogame.stream_client.stream_client import global_buffer, StreamClient, HDCSnapshotClient

class GameAutomator:
    def __init__(self, driver, logger):
        self.driver = driver
        self.logger = logger

        self.screen_w, self.screen_h = wait_for_landscape_resolution_stable()
        self.W, self.H = get_wh()
        if get_screen_mode() == "0":
            rotation_mode = get_display_rotation()
            self.client = StreamClient(global_buffer, rotation_mode=rotation_mode)
            self.client.start_backend(lowh=0, highh=10000, skip=20, width=self.W, height=self.H)
        elif get_screen_mode() == "1":
            self.client = HDCSnapshotClient(global_buffer)
        self.client.set_save_frame(True)
        self.processor = FrameWorker(global_buffer, driver=self.driver, logger=self.logger)
        self.is_cleaned_up = False

        atexit.register(self.cleanup)

        self._clear_temp_logs()

    def _clear_temp_logs(self):
        log_dir = r'aw/autogame/temp/logs'
        temp_log_dir = os.path.join(log_dir, "process_temp_logs")
        temp_save_dir = os.path.join(log_dir, "process_save_frames")

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
                os.makedirs(temp_log_dir, exist_ok=True)
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
        if self.is_cleaned_up: return
        self.is_cleaned_up = True

        print("\n>>> 开始执行深度清理程序...")
        try:
            self.processor.stop()
            self.client.stop()
        except:
            pass

        self._set_hiz_mode(False)

        try:
            if len(app_list) > 0:
                for app in app_list:
                    print(f"停止应用: {app}")
                    run_shell(f'hdc shell aa force-stop {app}')
        except:
            pass
        print(">>> 环境已恢复。")

    def _monitor_worker(self):
        print("[监控] 任务状态监控已启动...")
        while True:
            # 只要 processor 停了，不管是 finished 还是意外中断
            if getattr(self.processor, 'finished', False) or not self.processor.running:
                print("\n[业务通知] 流程结束，正在中断流连接...")
                self.client.stop()
                break
            time.sleep(0.5)

    def start(self):
        try:
            self._set_hiz_mode(True)

            self.processor.start()

            monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
            monitor_thread.start()

            print(">>> 正在启动视频流服务（阻塞中）...")
            if get_screen_mode() == "0":
                self.client.run(lowh=0, highh=10000, skip=20, width=self.W, height=self.H)
            elif get_screen_mode() == "1":
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
