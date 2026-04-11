import gc
import time
import threading
import importlib
import multiprocessing as mp

from aw.autogame.tools.Utils import *
from aw.autogame.tools.GameSceneHandler import StageLogicController


class Controller:
    """操作层：负责绝对坐标换算与HDC指令发送"""

    def __init__(self, driver, worker, stage_info_raw):
        self.buttons = extract_absolute_points(stage_info_raw)
        self.driver = driver
        self.worker = worker

    def _run_hdc(self, cmd):
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        proc.wait()

    def _get_abs_pos(self, btn_input):
        if isinstance(btn_input, (list, tuple)) and len(btn_input) == 2:
            val_x, val_y = btn_input
            res_w, res_h = get_resolution()

            # 判断逻辑：如果两个值都在 [0, 1] 范围内，则视为归一化坐标，需要转换
            # 如果任意一个值 > 1，则视为已经是绝对像素坐标
            if 0 <= val_x <= 1.0 and 0 <= val_y <= 1.0:
                abs_x, abs_y = int(val_x * res_w), int(val_y * res_h)
                desc = f"Norm({val_x}, {val_y})"
            else:
                abs_x, abs_y = int(val_x), int(val_y)
                desc = f"Abs({abs_x}, {abs_y})"

            return (abs_x, abs_y), desc

        elif isinstance(btn_input, str):
            stage = self.worker.get_stage()
            if not stage:
                return None, None
            full_key = f"{stage}_{btn_input}"
            btn_data = self.buttons.get(full_key)
            if isinstance(btn_data, dict):
                pos = btn_data.get("pos")
            else:
                pos = btn_data
            return pos, full_key

        return None, None

    def tap_double(self, btn1, btn2, wait=100, dura=500, x1_bias=0, y1_bias=1, x2_bias=0, y2_bias=1):
        pos1, label1 = self._get_abs_pos(btn1)
        pos2, label2 = self._get_abs_pos(btn2)
        if pos1 and pos2:
            x1, y1 = pos1
            x2, y2 = pos2
            print(f'执行双指操作: {label1} @({x1},{y1}), {label2} @({x2},{y2})')
            cmd = f"hdc shell uinput -T -m {x1} {y1} {x1 + x1_bias} {y1 + y1_bias} {x2} {y2} {x2 + x2_bias} {y2 + y2_bias} -k {wait} {dura}"
            self._run_hdc(cmd)

    def tap_single(self, btn, wait=100, dura=500, x_bias=0, y_bias=1):
        pos, label = self._get_abs_pos(btn)
        if pos:
            x, y = pos
            print(f'执行单指操作: {label} @({x},{y})')
            cmd = f"hdc shell uinput -T -m {x} {y} {x + x_bias} {y + y_bias} -k {wait} {dura}"
            self._run_hdc(cmd)

    def click_down(self, btn, x_bias=0, y_bias=0, dura=0):
        pos, label = self._get_abs_pos(btn)
        if pos:
            x, y = pos
            print(f'执行按下: {label} @({x + x_bias},{y + y_bias})')
            if dura == 0:
                cmd = f"hdc shell uinput -T -d {x + x_bias} {y + y_bias}"
            else:
                cmd = f"hdc shell uinput -T -d {x + x_bias} {y + y_bias} -i {dura} -u {x} {y}"
            self._run_hdc(cmd)

    def click(self, btn, x_bias=0, y_bias=0):
        pos, label = self._get_abs_pos(btn)
        if pos:
            x, y = pos
            print(f'执行点击: {label} @({x + x_bias},{y + y_bias})')
            cmd = f"hdc shell uinput -T -c {x + x_bias} {y + y_bias}"
            self._run_hdc(cmd)


class FrameWorker(threading.Thread):
    def __init__(self, buffer, driver=None, logger=None):
        super().__init__()
        self.frame_index = 0
        self.viz_queue = mp.Queue(maxsize=5)  # 限制长度防止积压
        self.viz_proc = None
        self.thread = None

        project_case = os.environ.get("TARGET_PROJECT_CASE")
        if not project_case: raise ValueError("TARGET_PROJECT_CASE 未设置")
        info_path = f"aw.autogame.customs_examples.{project_case}.info"
        info_module = importlib.import_module(info_path)
        self.stage_dict = getattr(info_module, "STAGE_DICT")
        raw_stage_info = getattr(info_module, "STAGE_INFO")

        case_name = os.environ.get("TARGET_GAME_CASE")
        if not case_name: raise ValueError("TARGET_GAME_CASE 未设置")
        logic_path = f"aw.autogame.customs_game_examples.{project_case}.{case_name}"
        try:
            logic_module = importlib.import_module(logic_path)
            # 获取用例定义的 on_stage 函数
            self.on_stage_logic = getattr(logic_module, "on_stage")
            print(f"成功加载业务逻辑: {logic_path}")
        except Exception as e:
            print(f"加载业务逻辑失败: {e}")
            raise e

        self.buffer = buffer
        self.driver = driver
        self.logger = logger
        self.running = False
        self.finished = False  # 新增：任务是否完成的标志
        self.controller = Controller(driver, self, raw_stage_info)
        self.stage_resolver = StageLogicController()

        self.stage_info = {}
        self.current_stage = None
        self.frame = None
        self.last_gc_time = time.time()

        # 快捷方法映射
        self.click = self.controller.click
        self.click_down = self.controller.click_down
        self.tap_single = self.controller.tap_single
        self.tap_double = self.controller.tap_double

    def loop(self):
        print("GameFrameWorker 引擎已启动")
        while self.running:
            # 1. 抓取最新帧
            frame = self.buffer.get_latest()
            if frame is None:
                time.sleep(0.1)  # 适当等待，不要空转
                continue

            # 2. 内存管理：定期强制回收 (每30秒执行一次)
            current_time = time.time()
            if current_time - self.last_gc_time > 30:
                gc.collect()
                self.last_gc_time = current_time

            try:
                self.frame = np.array(frame, copy=True)

                # 4. 业务逻辑
                self.current_stage = self.get_stage()
                self.stage_info = self.stage_resolver.process_frame(self.frame, self.current_stage)

                if not self.viz_queue.full() and self.viz_proc:
                    self.viz_queue.put((self.frame.copy(), self.current_stage, self.stage_info, self.frame_index))
                    self.frame_index += 1

                # 5. 执行业务逻辑
                self.on_stage_logic(self)

                time.sleep(0.05)  # 约 20 FPS

            except Exception as e:
                print(f"[Loop Error] 运行时异常: {e}")
                time.sleep(1)  # 报错后稍微停顿防止无限刷报错

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

        self.viz_proc = mp.Process(target=visualizer_process, args=(self.viz_queue,), daemon=True)
        self.viz_proc.start()

    def stop(self):
        print('主动结束游戏自动化中......')
        if not self.running:
            return  # 防止 finally 和 atexit 重复调用导致异常

        self.running = False
        self.finished = True

        if self.viz_proc:
            # 不要立刻 terminate，给它一点时间消费 "STOP"
            try:
                self.viz_queue.put_nowait("STOP")
            except Exception:
                pass

            # 等待可视化进程自己优雅退出
            self.viz_proc.join(timeout=1.0)

            # 如果它还没死，再下狠手
            if self.viz_proc.is_alive():
                self.viz_proc.terminate()
                self.viz_proc.join(timeout=0.5)  # 回收僵尸进程

            # 【最关键的一句】强行告诉 Python 退出时不要等待这个队列的后台线程
            self.viz_queue.cancel_join_thread()

        print("GameFrameWorker 已停止")

    def get_stage(self):
        for stage, active in self.stage_dict.items():
            if active: return stage
        return None

    def get_info(self, area_name):
        suffix = f"__{area_name}"
        for k, v in self.stage_info.items():
            if k.endswith(suffix):
                return v
        return None

    def change_stage(self, stage_name):
        # 检查阶段是否存在
        if stage_name not in self.stage_dict:
            print(f"\n[ERROR] 切换失败：阶段 '{stage_name}' 不在 STAGE_DICT 中！")
            return

        # 记录旧阶段用于打印显示
        old_stage = self.current_stage

        # 执行切换逻辑
        for k in self.stage_dict.keys():
            self.stage_dict[k] = False
        self.stage_dict[stage_name] = True
        self.current_stage = stage_name

        # 打印格式化的切换信息
        print(f"\n" + ">" * 40)
        print(f"  STATUS CHANGE: [{old_stage}] -> [{stage_name}]")
        print(">" * 40 + "\n")

    def refresh_frame(self):
        """
        强制刷新当前帧及其关联的所有信息，并同步至可视化队列。
        """
        frame = self.buffer.get_latest(must_new=True)
        if frame is None:
            print("[FrameWorker] 刷新失败：缓冲区暂无数据")
            return False

        self.frame = np.array(frame, copy=True)

        self.current_stage = self.get_stage()
        self.stage_info = self.stage_resolver.process_frame(self.frame, self.current_stage)

        if not self.viz_queue.full():
            self.viz_queue.put((self.frame.copy(), self.current_stage, self.stage_info, self.frame_index))
            self.frame_index += 1

        return True
