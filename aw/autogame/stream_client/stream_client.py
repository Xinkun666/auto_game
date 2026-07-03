import os
import json
import time
import threading
import queue
import atexit
from datetime import datetime
import subprocess
import grpc
import numpy as np
from PIL import Image
from aw.autogame.tools.ProcessUtils import hdc_command_args, hidden_subprocess_kwargs
from aw.autogame.tools.Utils import (
    _read_autogame_config,
    resolve_process_save_frames_dir,
    resolve_tmp_frames_dir,
)

# 假设这些是你的本地 proto 生成文件
PROTO_IMPORT_ERROR = None
try:
    import faststream_pb2
    import faststream_pb2_grpc
except ImportError as e:
    faststream_pb2 = None
    faststream_pb2_grpc = None
    PROTO_IMPORT_ERROR = e
    print("Warning: gRPC proto files not found. StreamClient will be unavailable.")


def _value_truthy(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_truthy(name, default=False):
    return _value_truthy(os.environ.get(name), default)


def _config_truthy(config, name, default=False):
    if not isinstance(config, dict):
        return default
    return _value_truthy(config.get(name), default)


def _config_float(config, name, default):
    if not isinstance(config, dict) or name not in config:
        return default
    try:
        return float(config.get(name))
    except (TypeError, ValueError):
        return default


def _config_int(config, name, default):
    if not isinstance(config, dict) or name not in config:
        return default
    try:
        return int(config.get(name))
    except (TypeError, ValueError):
        return default


def _resolve_bool_option(env_name, config, config_name, default=False):
    return _env_truthy(env_name, _config_truthy(config, config_name, default))


def _resolve_float_option(env_name, config, config_name, default):
    return _env_float(env_name, _config_float(config, config_name, default))


def _resolve_int_option(env_name, config, config_name, default):
    value = os.environ.get(env_name)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return _config_int(config, config_name, default)


def _resolve_str_option(env_name, config, config_name, default=""):
    value = os.environ.get(env_name)
    if value is not None:
        return str(value).strip()
    if isinstance(config, dict) and config_name in config:
        return str(config.get(config_name) or "").strip()
    return str(default or "").strip()


class FrameBuffer:
    def __init__(self, size=5):
        self.size = size
        self.frames = [None] * size
        self.write_idx = 0
        self.count = 0  # 追踪总共写入了多少帧（即当前最新的 Frame ID 为 count）

        self.condition = threading.Condition()
        self._last_read_id = 0

    def push(self, frame):
        with self.condition:
            try:
                # 统一在缓冲区内保存独立 numpy 帧，避免 PIL/buffer 在多线程间共享
                self.frames[self.write_idx] = np.array(frame, copy=True)
            except Exception:
                self.frames[self.write_idx] = frame
            self.write_idx = (self.write_idx + 1) % self.size
            self.count += 1
            self.condition.notify_all()

    def get_latest(self, timeout=5.0, must_new=False):
        """
        获取最新帧。
        :param must_new: 如果为 True，则忽略缓冲区现有帧，强制等待下一帧产生。
        """
        with self.condition:
            # --- 核心逻辑变化 ---
            if must_new:
                # 强制将“已读标记”同步为“当前最新 ID”
                # 这样下方的 while 循环会立即成立，进入 wait 状态
                self._last_read_id = self.count

            start_wait_time = time.time()
            # 只有当 count 增加（即 push 被调用）并超过 _last_read_id 时，才跳出循环
            while self.count <= self._last_read_id:
                remaining = timeout - (time.time() - start_wait_time)
                if remaining <= 0:
                    return None

                if not self.condition.wait(timeout=remaining):
                    return None  # 超时返回

            # 更新读取记录，确保下次不传 must_new 时不会重复读同一帧
            self._last_read_id = self.count

            # 返回物理内存中的最新帧
            latest_idx = (self.write_idx - 1) % self.size
            return self.frames[latest_idx]

# 创建全局缓冲实例
global_buffer = FrameBuffer(size=5)

def apply_rotation(frame, rotation):
    """
    根据 rotation 对 PIL.Image 做旋转
    rotation: 0 / 90 / 180 / 270
    """
    if rotation == 0:
        return frame

    elif rotation == 90:
        # 逆时针 90°
        return frame.rotate(90, expand=True)

    elif rotation == 180:
        return frame.rotate(180, expand=True)

    elif rotation == 270:
        # 顺时针 90° == 逆时针 270°
        return frame.rotate(270, expand=True)

    else:
        print("[Stream] Invalid rotation:", rotation)
        return frame

class StreamClient:
    def __init__(self, buffer, save_frame=False, rotation_mode=0):
        self.buffer = buffer

        # ---------- 运行状态 ----------
        self.running = False
        self.main_thread = None
        self._loop_active = False
        self._loop_owner = None
        self._stop_event = threading.Event()
        self.rotation_mode = rotation_mode  # 0: 不旋转, 1: 旋转90度, 2: 旋转180度, 3: 旋转270度

        # ---------- gRPC 相关 ----------
        self.stub_ = None
        self.channel = None
        self._responses = None
        self._channel_ready = None
        self.host = '127.0.0.1:12345'
        self.reconnect_base_delay = 1.0
        self.reconnect_max_delay = 5.0

        # ---------- 图像参数 ----------
        self.width = 720
        self.height = 1280

        config = _read_autogame_config()

        # ---------- 保存相关 ----------
        self.save_frame_disabled = _resolve_bool_option(
            "AUTOGAME_DISABLE_SAVE_FRAMES",
            config,
            "stream_disable_save_frames",
            False,
        )
        self.save_frame = bool(save_frame and not self.save_frame_disabled)
        self.save_queue = queue.Queue(maxsize=100)
        self.save_worker = None

        self.save_dir = str(resolve_process_save_frames_dir())

        # ---------- 诊断统计 ----------
        self.diagnostics_enabled = _resolve_bool_option(
            "AUTOGAME_STREAM_DIAGNOSTICS",
            config,
            "stream_diagnostics",
            False,
        )
        self.diagnostics_interval = max(
            0.0,
            _resolve_float_option(
                "AUTOGAME_STREAM_DIAGNOSTICS_INTERVAL",
                config,
                "stream_diagnostics_interval",
                1.0,
            ),
        )
        self._diag_lock = threading.Lock()
        self._diag_window_start = time.monotonic()
        self._diag = self._new_diag_stats()

        # ---------- 线程/状态保护 ----------
        self._state_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._transport_lock = threading.Lock()

        if self.save_frame:
            os.makedirs(self.save_dir, exist_ok=True)

        atexit.register(self._atexit_cleanup)

    def _ensure_proto_ready(self):
        if faststream_pb2 is None or faststream_pb2_grpc is None:
            raise RuntimeError(
                "faststream_pb2 / faststream_pb2_grpc 未准备好，无法启动 StreamClient"
            ) from PROTO_IMPORT_ERROR

    # =========================================================
    # 对外接口
    # =========================================================
    def set_save_frame(self, enable):
        """动态开关保存功能"""
        if enable and self.save_frame_disabled:
            self.save_frame = False
            print("[Stream] Save frame mode disabled by AUTOGAME_DISABLE_SAVE_FRAMES=1")
            return

        self.save_frame = enable
        if enable:
            os.makedirs(self.save_dir, exist_ok=True)
            self._start_save_worker()
        print("[Stream] Save frame mode set to: %s" % enable)

    def start_backend(self, lowh, highh, skip, width, height, layerid=-1):
        """
        后台启动拉流线程，接口风格与 HDCSnapshotClient 对齐
        """
        self._ensure_proto_ready()
        with self._state_lock:
            if self.main_thread is not None and self.main_thread.is_alive():
                print("[Stream] Backend already running.")
                return

            self._stop_event.clear()
            self.main_thread = threading.Thread(
                target=self.run,
                args=(lowh, highh, skip, width, height, layerid),
                name="StreamMainThread",
                daemon=True
            )
            self.main_thread.start()
            print("[Stream] Backend started.")

    def stop(self):
        """优雅关闭，支持重复调用"""
        loop_owner = None
        with self._state_lock:
            has_transport = any(
                handle is not None
                for handle in (self.stub_, self.channel, self._responses, self._channel_ready)
            )
            if not self.running and not self._loop_active and not has_transport:
                return
            self.running = False
            self._stop_event.set()
            loop_owner = self._loop_owner

        print("[Stream] Stopping client...")

        # 1. 通知服务端结束流
        self._close_transport(send_end_stream=True)

        # 3. 结束保存线程
        try:
            self.save_queue.put(None, timeout=1)
        except Exception:
            pass

        if self.save_worker and self.save_worker.is_alive():
            self.save_worker.join(timeout=1)

        # 4. 回收主线程（避免自己 join 自己）
        if (
            self.main_thread
            and self.main_thread.is_alive()
            and threading.current_thread() is not self.main_thread
        ):
            self.main_thread.join(timeout=1)
        elif (
            loop_owner
            and loop_owner.is_alive()
            and threading.current_thread() is not loop_owner
            and loop_owner is not self.main_thread
        ):
            loop_owner.join(timeout=1)

        # 5. 清空句柄引用
        self.stub_ = None
        self.channel = None

        print("[Stream] Client stopped.")

    def _atexit_cleanup(self):
        try:
            self.stop()
        except Exception:
            pass

    # =========================================================
    # 内部主流程
    # =========================================================
    def run(self, lowh, highh, skip, width, height, layerid=-1):
        """
        主循环：
        1. 建立 gRPC 连接
        2. 接收图像流
        3. 解码后推入 buffer
        4. 可选异步落盘
        """
        self.width = width
        self.height = height
        self._ensure_proto_ready()

        current_thread = threading.current_thread()
        wait_owner = None
        with self._state_lock:
            if self._loop_active:
                if self._loop_owner is current_thread:
                    print("[Stream] Stream loop already active in current thread.")
                    return
                wait_owner = self._loop_owner
            else:
                self.running = True
                self._stop_event.clear()
                self._loop_active = True
                self._loop_owner = current_thread

        if wait_owner is not None:
            print("[Stream] Stream loop already active. Waiting for owner thread to exit...")
            while wait_owner.is_alive() and not self._stop_event.is_set():
                wait_owner.join(timeout=0.2)
            return

        if self.save_frame:
            self._start_save_worker()

        options = [
            ('grpc.max_receive_message_length', 5 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', 1),
        ]

        expected_size = self.width * self.height * 4  # RGBX

        try:
            reconnect_attempt = 0
            while self.running and not self._stop_event.is_set():
                try:
                    self.channel = grpc.insecure_channel(self.host, options=options)
                    self._channel_ready = grpc.channel_ready_future(self.channel)
                    self._channel_ready.result(timeout=5)
                    self.stub_ = faststream_pb2_grpc.StreamServiceStub(self.channel)

                    stream_config = faststream_pb2.StreamConfig(
                        lowh=lowh,
                        highh=highh,
                        skip=skip,
                        width=width,
                        height=height,
                        layerid=layerid
                    )

                    print("[Stream] Start receiving...", flush=True)
                    responses = self.stub_.StartStream(stream_config)
                    self._responses = responses
                    reconnect_attempt = 0

                    for message in responses:
                        if not self.running or self._stop_event.is_set():
                            break

                        frame_start = time.perf_counter()
                        data = message.data
                        if not data:
                            continue

                        data = bytes(data)

                        # 风险控制 1：长度不符直接丢弃，避免无意义解码异常
                        if len(data) != expected_size:
                            print("[Stream] Frame size mismatch: got=%d, expected=%d" % (len(data), expected_size))
                            continue

                        decode_start = time.perf_counter()
                        frame = self.decode_frame(data)
                        frame = apply_rotation(frame, self.rotation_mode)
                        decode_ms = (time.perf_counter() - decode_start) * 1000.0
                        if frame is None:
                            continue

                        # 统一出口，便于后续扩展
                        buffer_start = time.perf_counter()
                        self.on_frame(frame)
                        buffer_ms = (time.perf_counter() - buffer_start) * 1000.0

                        enqueue_ms = 0.0
                        if self.save_frame:
                            enqueue_start = time.perf_counter()
                            self._enqueue_save(frame)
                            enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0

                        receive_ms = (decode_start - frame_start) * 1000.0
                        self._record_stream_frame(
                            receive_ms=receive_ms,
                            decode_ms=decode_ms,
                            buffer_ms=buffer_ms,
                            enqueue_ms=enqueue_ms,
                        )
                        self._maybe_print_stream_diagnostics()

                    if not self.running or self._stop_event.is_set():
                        print("[Stream] Receive loop exited.")
                        break

                    reconnect_attempt += 1
                    delay = self._get_reconnect_delay(reconnect_attempt)
                    message = (
                        "[Stream] Receive loop ended unexpectedly. "
                        "Reconnect in %.1fs (attempt=%d)." % (delay, reconnect_attempt)
                    )
                    print(message, flush=True)
                    self._write_disconnect_signal("receive_loop_ended", message, reconnect_attempt)
                    if self._should_exit_on_disconnect():
                        print("[Stream] Exit on stream disconnect requested.", flush=True)
                        break
                    if self._stop_event.wait(delay):
                        break

                except grpc.FutureTimeoutError:
                    if not self.running or self._stop_event.is_set():
                        break
                    reconnect_attempt += 1
                    delay = self._get_reconnect_delay(reconnect_attempt)
                    message = "[Stream] Channel ready timeout. Reconnect in %.1fs (attempt=%d)." % (delay, reconnect_attempt)
                    print(message, flush=True)
                    self._write_disconnect_signal("channel_ready_timeout", message, reconnect_attempt)
                    if self._should_exit_on_disconnect():
                        print("[Stream] Exit on stream disconnect requested.", flush=True)
                        break
                    if self._stop_event.wait(delay):
                        break
                except grpc.RpcError as e:
                    if not self.running or self._stop_event.is_set():
                        break
                    reconnect_attempt += 1
                    delay = self._get_reconnect_delay(reconnect_attempt)
                    status = e.code() if hasattr(e, "code") else "UNKNOWN"
                    details = e.details() if hasattr(e, "details") else str(e)
                    message = "[Stream] gRPC Error: code=%s details=%s" % (status, details)
                    print(message, flush=True)
                    print("[Stream] Reconnect in %.1fs (attempt=%d)." % (delay, reconnect_attempt), flush=True)
                    self._write_disconnect_signal("grpc_error", message, reconnect_attempt)
                    if self._should_exit_on_disconnect():
                        print("[Stream] Exit on stream disconnect requested.", flush=True)
                        break
                    if self._stop_event.wait(delay):
                        break
                except Exception as e:
                    if not self.running or self._stop_event.is_set():
                        break
                    reconnect_attempt += 1
                    delay = self._get_reconnect_delay(reconnect_attempt)
                    message = "[Stream] Runtime Error: %s" % e
                    print(message, flush=True)
                    print("[Stream] Reconnect in %.1fs (attempt=%d)." % (delay, reconnect_attempt), flush=True)
                    self._write_disconnect_signal("runtime_error", message, reconnect_attempt)
                    if self._should_exit_on_disconnect():
                        print("[Stream] Exit on stream disconnect requested.", flush=True)
                        break
                    if self._stop_event.wait(delay):
                        break
                finally:
                    self._close_transport(send_end_stream=False)
        finally:
            # 注意：这里不直接 self.stop()，避免后台线程中自我 join 风险
            self._cleanup_after_run()
            print("[Stream] Main loop exited.")

    # =========================================================
    # 内部功能函数
    # =========================================================
    def on_frame(self, frame):
        """
        与 HDCSnapshotClient 对齐：统一帧分发出口
        """
        try:
            if frame is not None:
                self.buffer.push(frame)
        except Exception as e:
            print("[Stream] Buffer push error: %s" % e)

    def decode_frame(self, data):
        """
        严格解码：
        - 解码失败立即返回 None
        - convert('RGB').copy() 保证后续线程使用时数据独立
        """
        try:
            raw_bytes = bytes(data)
            img = Image.frombytes(
                "RGBX",
                (self.width, self.height),
                raw_bytes,
                "raw",
                "RGBX",
                0,
                1
            )
            return img.convert("RGB").copy()
        except Exception as e:
            print("[Stream] Decode error: %s" % e)
            return None

    def _enqueue_save(self, frame):
        """
        非阻塞入队：
        - 满了就丢，不能反压主拉流线程
        """
        ts_str = datetime.now().strftime("%m-%d %H-%M-%S.%f")[:-3]
        try:
            self.save_queue.put_nowait((frame.copy(), ts_str))
        except queue.Full:
            # 风险控制：保存线程来不及，直接丢弃保存任务，保证主流程实时性
            self._record_save_drop()
            pass
        except Exception as e:
            print("[Stream] Save queue error: %s" % e)

    def _start_save_worker(self):
        with self._save_lock:
            if self.save_worker is None or not self.save_worker.is_alive():
                self.save_worker = threading.Thread(
                    target=self._save_worker_logic,
                    name="StreamSaveWorker",
                    daemon=True
                )
                self.save_worker.start()

    def _save_worker_logic(self):
        """
        保存线程：
        - 阻塞取队列
        - 收到 None 退出
        - 单帧失败不影响整体
        """
        while True:
            item = None
            try:
                item = self.save_queue.get(timeout=1)
                if item is None:
                    break

                frame, time_str = item
                save_path = os.path.join(self.save_dir, "%s.jpg" % time_str)
                save_start = time.perf_counter()
                frame.save(save_path, "JPEG")
                self._record_save_complete((time.perf_counter() - save_start) * 1000.0)

            except queue.Empty:
                # 主线程已停止且队列没活，允许退出
                if not self.running:
                    break
            except Exception as e:
                print("[Stream] Save disk error: %s" % e)
            finally:
                if item is not None:
                    try:
                        self.save_queue.task_done()
                    except Exception:
                        pass

        print("[Stream] Save worker exited.")

    def _new_diag_stats(self):
        return {
            "frames": 0,
            "receive_ms": 0.0,
            "decode_ms": 0.0,
            "buffer_ms": 0.0,
            "enqueue_ms": 0.0,
            "save_count": 0,
            "save_ms": 0.0,
            "save_dropped": 0,
        }

    def _record_stream_frame(self, receive_ms, decode_ms, buffer_ms, enqueue_ms):
        if not self.diagnostics_enabled:
            return
        with self._diag_lock:
            self._diag["frames"] += 1
            self._diag["receive_ms"] += max(0.0, float(receive_ms))
            self._diag["decode_ms"] += max(0.0, float(decode_ms))
            self._diag["buffer_ms"] += max(0.0, float(buffer_ms))
            self._diag["enqueue_ms"] += max(0.0, float(enqueue_ms))

    def _record_save_drop(self):
        if not self.diagnostics_enabled:
            return
        with self._diag_lock:
            self._diag["save_dropped"] += 1

    def _record_save_complete(self, save_ms):
        if not self.diagnostics_enabled:
            return
        with self._diag_lock:
            self._diag["save_count"] += 1
            self._diag["save_ms"] += max(0.0, float(save_ms))

    def _maybe_print_stream_diagnostics(self):
        if not self.diagnostics_enabled:
            return

        now = time.monotonic()
        with self._diag_lock:
            elapsed = max(0.001, now - self._diag_window_start)
            if elapsed < self.diagnostics_interval:
                return
            stats = self._diag
            self._diag = self._new_diag_stats()
            self._diag_window_start = now

        frames = stats["frames"]
        saves = stats["save_count"]
        avg = lambda key, count: (stats[key] / count) if count else 0.0
        try:
            save_qsize = self.save_queue.qsize()
        except Exception:
            save_qsize = -1

        print(
            "[StreamDiag] frames=%d fps=%.1f recv_avg_ms=%.2f decode_avg_ms=%.2f "
            "buffer_avg_ms=%.2f enqueue_avg_ms=%.2f save_count=%d save_avg_ms=%.2f "
            "save_q=%s save_dropped=%d"
            % (
                frames,
                frames / elapsed,
                avg("receive_ms", frames),
                avg("decode_ms", frames),
                avg("buffer_ms", frames),
                avg("enqueue_ms", frames),
                saves,
                avg("save_ms", saves),
                save_qsize,
                stats["save_dropped"],
            ),
            flush=True,
        )

    def _cleanup_after_run(self):
        """
        run() 退出时的轻量清理：
        - 不做 join
        - 不做重复 stop
        - 只做句柄安全释放
        """
        with self._state_lock:
            self.running = False
            self._loop_active = False
            self._loop_owner = None
            self._stop_event.set()

        self._close_transport(send_end_stream=True)

        # 通知保存线程收尾
        try:
            self.save_queue.put_nowait(None)
        except Exception:
            pass

    def _write_disconnect_signal(self, reason, message, attempt):
        archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
        if not archive_dir:
            return

        try:
            os.makedirs(archive_dir, exist_ok=True)
            payload = {
                "event": "stream_disconnected",
                "reason": str(reason),
                "message": str(message),
                "attempt": int(attempt),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "pid": os.getpid(),
            }
            signal_path = os.path.join(archive_dir, "stream_disconnect_signal.json")
            tmp_path = signal_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, signal_path)
        except Exception as exc:
            print("[Stream] Write disconnect signal failed: %s" % exc, flush=True)

    def _should_exit_on_disconnect(self):
        value = os.environ.get("AUTOGAME_EXIT_ON_STREAM_DISCONNECT", "").strip().lower()
        return value in ("1", "true", "yes", "on")

    def _close_transport(self, send_end_stream: bool = False):
        with self._transport_lock:
            responses = self._responses
            channel_ready = self._channel_ready
            stub = self.stub_
            channel = self.channel

            self._responses = None
            self._channel_ready = None
            self.stub_ = None
            self.channel = None

        if responses is not None:
            try:
                if hasattr(responses, "cancel"):
                    responses.cancel()
            except Exception as e:
                if self.running and not self._stop_event.is_set():
                    print("[Stream] Response cancel warning: %s" % e)
            try:
                if hasattr(responses, "close"):
                    responses.close()
            except Exception:
                pass

        if send_end_stream:
            try:
                if stub is not None and faststream_pb2 is not None:
                    stub.EndStream(faststream_pb2.Empty(), timeout=1)
            except Exception as e:
                if self.running and not self._stop_event.is_set():
                    print("[Stream] EndStream warning: %s" % e)

        if channel_ready is not None:
            try:
                channel_ready.cancel()
            except Exception:
                pass

        if channel is not None:
            try:
                channel.close()
            except Exception as e:
                if self.running and not self._stop_event.is_set():
                    print("[Stream] Channel close warning: %s" % e)

    def _get_reconnect_delay(self, attempt: int) -> float:
        attempt = max(1, int(attempt))
        delay = self.reconnect_base_delay * (2 ** (attempt - 1))
        return min(self.reconnect_max_delay, delay)


class PyAVH264Decoder:
    def __init__(self):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "HOScrcpy 流输出 H.264 ByteBuffer，需要安装 PyAV：python -m pip install av"
            ) from exc
        self.codec = av.CodecContext.create("h264", "r")

    def decode(self, data):
        frames = []
        for packet in self.codec.parse(bytes(data)):
            for frame in self.codec.decode(packet):
                frames.append(frame.to_image().convert("RGB").copy())
        return frames


class HOSScrcpyStreamClient:
    def __init__(
        self,
        buffer,
        save_frame=False,
        rotation_mode=0,
        decoder=None,
        device_factory=None,
    ):
        self.buffer = buffer
        self.running = False
        self.main_thread = None
        self._stop_event = threading.Event()
        self.rotation_mode = rotation_mode
        self.decoder = decoder
        self.device_factory = device_factory
        self.device = None
        self._last_error = None
        self._first_frame_received = False

        config = _read_autogame_config()
        self.sn = _resolve_str_option("AUTOGAME_HOSCRCPY_SN", config, "hoscrcpy_sn", "")
        self.ip = _resolve_str_option("AUTOGAME_HOSCRCPY_IP", config, "hoscrcpy_ip", "127.0.0.1") or "127.0.0.1"
        self.port = _resolve_int_option("AUTOGAME_HOSCRCPY_PORT", config, "hoscrcpy_port", 8710)
        self.scale = max(1, _resolve_int_option("AUTOGAME_HOSCRCPY_SCALE", config, "hoscrcpy_scale", 1))
        self.frame_rate = max(1, _resolve_int_option("AUTOGAME_HOSCRCPY_FRAME_RATE", config, "hoscrcpy_frame_rate", 60))
        self.bit_rate = max(1, _resolve_int_option("AUTOGAME_HOSCRCPY_BIT_RATE", config, "hoscrcpy_bit_rate", 30))
        self.device_port = max(1, _resolve_int_option("AUTOGAME_HOSCRCPY_DEVICE_PORT", config, "hoscrcpy_device_port", 5000))
        self.encoder_type = _resolve_str_option("AUTOGAME_HOSCRCPY_ENCODER_TYPE", config, "hoscrcpy_encoder_type", "0") or "0"

        self.save_frame_disabled = _resolve_bool_option(
            "AUTOGAME_DISABLE_SAVE_FRAMES",
            config,
            "stream_disable_save_frames",
            False,
        )
        self.save_frame = bool(save_frame and not self.save_frame_disabled)
        self.save_queue = queue.Queue(maxsize=100)
        self.save_worker = None
        self.save_dir = str(resolve_process_save_frames_dir())
        if self.save_frame:
            os.makedirs(self.save_dir, exist_ok=True)
        atexit.register(self._atexit_cleanup)

    def set_save_frame(self, enable):
        if enable and self.save_frame_disabled:
            self.save_frame = False
            print("[HOS] Save frame mode disabled by AUTOGAME_DISABLE_SAVE_FRAMES=1")
            return
        self.save_frame = bool(enable)
        if self.save_frame:
            os.makedirs(self.save_dir, exist_ok=True)
            self._start_save_worker()
        print("[HOS] Save frame mode set to: %s" % self.save_frame)

    def start_backend(self):
        if self.main_thread is not None and self.main_thread.is_alive():
            print("[HOS] Backend already running.")
            return
        self._stop_event.clear()
        self.main_thread = threading.Thread(target=self.run, name="HOSMainThread", daemon=True)
        self.main_thread.start()
        print("[HOS] Backend started.")

    def run(self):
        self.running = True
        self._stop_event.clear()
        self._last_error = None
        if self.save_frame:
            self._start_save_worker()

        try:
            self.device = self._create_device()
            callback = self._create_callback()
            print(
                "[HOS] Starting HOScrcpy stream: sn=%s ip=%s port=%s scale=%s fps=%s bitrate=%s"
                % (self.sn, self.ip, self.port, self.scale, self.frame_rate, self.bit_rate),
                flush=True,
            )
            self.device.start_capture_screen(callback)
            while self.running and not self._stop_event.is_set():
                if self._last_error is not None:
                    break
                self._stop_event.wait(0.2)
            if self._last_error is not None:
                raise RuntimeError("HOScrcpy stream failed: %s" % self._last_error)
        except Exception as exc:
            self._last_error = exc
            message = "[HOS] Runtime Error: %s" % exc
            print(message, flush=True)
            self._write_disconnect_signal("hoscrcpy_runtime_error", message, 1)
            raise
        finally:
            self.running = False
            self._stop_event.set()
            self._stop_device()
            try:
                self.save_queue.put_nowait(None)
            except Exception:
                pass
            print("[HOS] Main loop exited.", flush=True)

    def stop(self):
        if not self.running and self.device is None:
            return
        print("[HOS] Stopping client...")
        self.running = False
        self._stop_event.set()
        self._stop_device()
        try:
            self.save_queue.put(None, timeout=1)
        except Exception:
            pass
        if self.save_worker and self.save_worker.is_alive():
            self.save_worker.join(timeout=1)
        if (
            self.main_thread
            and self.main_thread.is_alive()
            and threading.current_thread() is not self.main_thread
        ):
            self.main_thread.join(timeout=1)
        print("[HOS] Client stopped.")

    def _atexit_cleanup(self):
        try:
            self.stop()
        except Exception:
            pass

    def _create_device(self):
        if not self.sn:
            self.sn = self._resolve_default_device_sn()
        if self.device_factory is not None:
            return self.device_factory(self)
        from aw.autogame.stream_client.hos_sdk.HosRemoteConfig import HosRemoteConfig
        from aw.autogame.stream_client.hos_sdk.HosRemoteDevice import HosRemoteDevice

        config = HosRemoteConfig(
            sn=self.sn,
            ip=self.ip,
            port=self.port,
            scale=self.scale,
            frame_rate=self.frame_rate,
            bit_rate=self.bit_rate,
            device_port=self.device_port,
            encoder_type=self.encoder_type,
        )
        return HosRemoteDevice(config)

    def _resolve_default_device_sn(self):
        args = hdc_command_args("hdc list targets")
        if not args:
            raise RuntimeError("无法构造 hdc list targets 命令，请在 config.json 写入 hoscrcpy_sn")
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=5,
            text=True,
            **hidden_subprocess_kwargs(),
        )
        if result.returncode != 0:
            raise RuntimeError("自动获取 HOScrcpy 设备 SN 失败: %s" % ((result.stderr or result.stdout or "").strip()))
        devices = [
            line.strip()
            for line in (result.stdout or "").splitlines()
            if line.strip() and "Empty" not in line and not line.startswith("ErrorMessage:")
        ]
        if not devices:
            raise RuntimeError("未发现可用于 HOScrcpy 的 hdc 设备，请在 config.json 写入 hoscrcpy_sn")
        if len(devices) > 1:
            print("[HOS] Multiple hdc devices found, using first: %s" % devices[0], flush=True)
        return devices[0]

    def _create_callback(self):
        from aw.autogame.stream_client.hos_sdk.ScreenCapCallback import ScreenCapCallback

        client = self

        class _Callback(ScreenCapCallback):
            def on_data(self, byte_buffer: bytes):
                client._handle_stream_bytes(byte_buffer)

            def on_exception(self, err: Exception):
                client._handle_stream_exception(err)

            def on_ready(self):
                print("[HOS] HOScrcpy stream ready.", flush=True)

        return _Callback()

    def _handle_stream_bytes(self, data):
        if not data:
            return
        if self.decoder is None:
            self.decoder = PyAVH264Decoder()
        frames = self.decoder.decode(data) or []
        for frame in frames:
            frame = apply_rotation(frame, self.rotation_mode)
            self.on_frame(frame)
            if self.save_frame:
                self._enqueue_save(frame)
            if not self._first_frame_received:
                self._first_frame_received = True
                print("[HOS] First frame received.", flush=True)

    def _handle_stream_exception(self, err):
        if not self.running:
            return
        self._last_error = err
        message = "[HOS] Stream Error: %s" % err
        print(message, flush=True)
        self._write_disconnect_signal("hoscrcpy_stream_error", message, 1)
        self._stop_event.set()

    def on_frame(self, frame):
        try:
            if frame is not None:
                self.buffer.push(frame)
        except Exception as exc:
            print("[HOS] Buffer push error: %s" % exc)

    def _enqueue_save(self, frame):
        ts_str = datetime.now().strftime("%m-%d %H-%M-%S.%f")[:-3]
        try:
            self.save_queue.put_nowait((frame.copy(), ts_str))
        except queue.Full:
            pass
        except Exception as exc:
            print("[HOS] Save queue error: %s" % exc)

    def _start_save_worker(self):
        if self.save_worker is None or not self.save_worker.is_alive():
            self.save_worker = threading.Thread(
                target=self._save_worker_logic,
                name="HOSSaveWorker",
                daemon=True,
            )
            self.save_worker.start()

    def _save_worker_logic(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                try:
                    self.save_queue.task_done()
                except Exception:
                    pass
                break
            frame, time_str = item
            try:
                save_path = os.path.join(self.save_dir, "%s.jpg" % time_str)
                frame.save(save_path, "JPEG")
            except Exception as exc:
                print("[HOS] Save disk error: %s" % exc)
            finally:
                try:
                    self.save_queue.task_done()
                except Exception:
                    pass
        print("[HOS] Save worker exited.")

    def _stop_device(self):
        device = self.device
        self.device = None
        if device is not None:
            try:
                device.stop_capture_screen()
            except Exception as exc:
                print("[HOS] Stop capture warning: %s" % exc)

    def _write_disconnect_signal(self, reason, message, attempt):
        archive_dir = os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR", "").strip()
        if not archive_dir:
            return
        try:
            os.makedirs(archive_dir, exist_ok=True)
            payload = {
                "event": "stream_disconnected",
                "reason": str(reason),
                "message": str(message),
                "attempt": int(attempt),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "pid": os.getpid(),
            }
            signal_path = os.path.join(archive_dir, "stream_disconnect_signal.json")
            tmp_path = signal_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, signal_path)
        except Exception as exc:
            print("[HOS] Write disconnect signal failed: %s" % exc, flush=True)


class HDCSnapshotClient:
    def __init__(self, buffer, save_frame=False):
        self.buffer = buffer
        self.running = False
        self.main_thread = None
        self.save_frame = save_frame

        self.save_queue = queue.Queue(maxsize=100)
        self.save_worker = None

        self.local_tmp_dir = str(resolve_tmp_frames_dir())
        self.save_dir = str(resolve_process_save_frames_dir())
        self.first_frame_received = False
        self.consecutive_capture_failures = 0
        self.startup_frame_timeout = float(os.environ.get("AUTOGAME_HDC_FRAME_READY_TIMEOUT", "10"))
        self.max_consecutive_capture_failures = int(os.environ.get("AUTOGAME_HDC_MAX_CAPTURE_FAILURES", "8"))

        for d in [self.local_tmp_dir, self.save_dir]:
            if not os.path.exists(d): os.makedirs(d, exist_ok=True)

    def set_save_frame(self, enable):
        self.save_frame = enable
        if enable:
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir, exist_ok=True)
            self._start_save_worker()
        print("[HDC] Save frame mode set to: %s" % enable)

    def _start_save_worker(self):
        if self.save_worker is None or not self.save_worker.is_alive():
            self.save_worker = threading.Thread(target=self._save_worker_logic, name="HDCSaveWorker", daemon=True)
            self.save_worker.start()

    def _save_worker_logic(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                try:
                    self.save_queue.task_done()
                except Exception:
                    pass
                break
            frame, time_str = item
            try:
                save_path = os.path.join(self.save_dir, "%s.jpg" % time_str)
                frame.save(save_path, "JPEG")
            except Exception as e:
                print("[HDC] Save Disk Error: %s" % e)
            finally:
                self.save_queue.task_done()

    def _get_frame_strictly(self, path):
        """尝试读取当前帧，失败则立即返回 None，不阻塞"""
        try:
            # 快速检查：文件是否存在且有数据
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with Image.open(path) as img:
                    # 使用 convert("RGB").copy() 确保图片数据完整加载到内存
                    return img.convert("RGB").copy()
        except:
            pass  # 读取失败或图片格式损坏
        return None

    def run(self, interval=0.3):
        self.running = True
        if self.save_frame: self._start_save_worker()

        print("[HDC] Main loop started.")
        loop_start_time = time.time()
        while self.running:
            start_time = time.time()
            ts = int(start_time * 1000)
            remote_path = "/data/local/tmp/snap_%d.jpeg" % ts
            local_path = os.path.join(self.local_tmp_dir, "snap_%d.jpeg" % ts)

            need_remote_rm = False

            try:
                # 1. 截图与接收 (同步阻塞执行)
                r_snap = subprocess.run(
                    hdc_command_args("hdc shell snapshot_display -f %s" % remote_path),
                    capture_output=True,
                    timeout=5,
                    **hidden_subprocess_kwargs(),
                )
                if r_snap.returncode == 0:
                    need_remote_rm = True
                    r_recv = subprocess.run(
                        hdc_command_args("hdc file recv %s %s" % (remote_path, local_path)),
                        capture_output=True,
                        timeout=5,
                        **hidden_subprocess_kwargs(),
                    )

                    if r_recv.returncode == 0:
                        # 2. 核心改动：尝试拿取图片，拿不到就直接跳过
                        frame = self._get_frame_strictly(local_path)

                        if frame:
                            # 3. 只有成功拿到帧才执行后续推送和保存逻辑
                            if self.save_frame:
                                ts_str = datetime.now().strftime("%m-%d %H-%M-%S.%f")[:-3]
                                try:
                                    self.save_queue.put_nowait((frame.copy(), ts_str))
                                except queue.Full:
                                    pass

                            self.on_frame(frame)
                            self.consecutive_capture_failures = 0
                            if not self.first_frame_received:
                                self.first_frame_received = True
                                print("[HDC] First frame received.", flush=True)
                        else:
                            self.consecutive_capture_failures += 1
                    else:
                        self.consecutive_capture_failures += 1
                else:
                    self.consecutive_capture_failures += 1

                if (
                    not self.first_frame_received
                    and time.time() - loop_start_time >= self.startup_frame_timeout
                ):
                    print("[HDC] Frame ready timeout.", flush=True)
                    break

                if self.consecutive_capture_failures >= self.max_consecutive_capture_failures:
                    print("[HDC] Consecutive capture failures exceeded.", flush=True)
                    break

            except Exception as e:
                self.consecutive_capture_failures += 1
                print("[HDC] Run Loop Error: %s" % e)
                if self.consecutive_capture_failures >= self.max_consecutive_capture_failures:
                    print("[HDC] Consecutive capture failures exceeded.", flush=True)
                    break
            finally:
                # 4. 无论如何都清理环境，保证下一帧开始时是干净的
                if need_remote_rm:
                    subprocess.run(
                        hdc_command_args("hdc shell rm %s" % remote_path),
                        capture_output=True,
                        timeout=2,
                        **hidden_subprocess_kwargs(),
                    )
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except:
                        pass

            # 严格控制循环间隔
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))
        print("[HDC] Main loop exited.")

    def on_frame(self, frame):
        if frame: self.buffer.push(frame)

    def start_backend(self, interval=0.1):
        self.main_thread = threading.Thread(target=self.run, args=(interval,), name="HDCMainThread", daemon=True)
        self.main_thread.start()

    def stop(self):
        print("[HDC] Stopping client...")
        self.running = False
        try:
            self.save_queue.put(None, timeout=1)
        except:
            pass
        if self.save_worker: self.save_worker.join(timeout=1)
        if self.main_thread and threading.current_thread() is not self.main_thread:
            self.main_thread.join(timeout=1)
        print("[HDC] Client stopped.")

if __name__ == "__main__":
    # client = HDCSnapshotClient(global_buffer, save_frame=True)
    # # 注意：run 是阻塞的，实际使用时通常在 Thread 中运行或作为主循环
    # client.run()

    client = StreamClient(global_buffer)
    client.set_save_frame(True)
    client.run(lowh=0, highh=10000, skip=10, width=384, height=762)
