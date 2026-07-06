import grpc
import json
import os
import threading
from aw.autogame.stream_client.hos_sdk.communication.proto import scrcpy_pb2, scrcpy_pb2_grpc
from aw.autogame.stream_client.hos_sdk.ScreenCapCallback import ScreenCapCallback
from aw.autogame.stream_client.hos_sdk.utils.logger import get_logger


logger = get_logger(__name__)


DEFAULT_FIRST_FRAME_TIMEOUT = 15.0  # 首帧超时时间（秒）
CONFIG_PATH = os.path.join("aw", "autogame", "config", "config.json")


def _read_autogame_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _coerce_timeout_seconds(value, default):
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default
    if timeout <= 0:
        return default
    return timeout


def resolve_first_frame_timeout():
    config = _read_autogame_config()
    value = os.environ.get("AUTOGAME_HOSCRCPY_FIRST_FRAME_TIMEOUT")
    if value is None and isinstance(config, dict):
        value = config.get("hoscrcpy_first_frame_timeout")
    return _coerce_timeout_seconds(value, DEFAULT_FIRST_FRAME_TIMEOUT)


class RpcManager(object):

    def __init__(self, host: str, port: int, on_first_frame_timeout=None) -> None:
        logger.info("RpcManager port: %s", port)
        # max 10M
        self.channel = grpc.insecure_channel(target="{}:{}".format(host, port),
                                             options=[('grpc_max_receive_message_length', 10485760)])
        self.stub = scrcpy_pb2_grpc.ScrcpyServiceStub(self.channel)
        self.screenCapCallback = None
        self.on_first_frame_timeout = on_first_frame_timeout
        self.first_frame_timeout = resolve_first_frame_timeout()
        self._timeout_timer = None
        self._first_frame_timeout_triggered = False
        self._rpc_call = None

    def _on_first_frame_timeout(self):
        """首帧超时回调，取消gRPC流"""
        logger.warning("首帧超时（%.1f秒），取消gRPC流...", self.first_frame_timeout)
        self._first_frame_timeout_triggered = True
        if self._rpc_call:
            self._rpc_call.cancel()

    def start_scrcpy(self, screen_cap_callback: ScreenCapCallback) -> bool:
        """
        start screen copy
        """
        self._first_frame_timeout_triggered = False
        self._timeout_timer = None
        try:
            self.screenCapCallback = screen_cap_callback
            self._rpc_call = self.stub.onStart(scrcpy_pb2.Empty())
            logger.info("scrcpy grpc onStart called")
            # 仅首次投屏启动首帧超时计时器，重试时不启动
            if self.on_first_frame_timeout:
                self._timeout_timer = threading.Timer(self.first_frame_timeout, self._on_first_frame_timeout)
                self._timeout_timer.daemon = True
                self._timeout_timer.start()
            first_frame_received = False
            for response in self._rpc_call:
                frame_data = response.payload['data'].val_bytes
                if not first_frame_received:
                    first_frame_received = True
                    logger.info("first scrcpy grpc payload received: bytes=%s", len(frame_data))
                    if self._timeout_timer:
                        self._timeout_timer.cancel()
                        self._timeout_timer = None
                screen_cap_callback.on_data(frame_data)
            return True
        except grpc.RpcError as e:
            if self._timeout_timer:
                self._timeout_timer.cancel()
                self._timeout_timer = None
            if self._first_frame_timeout_triggered and self.on_first_frame_timeout:
                logger.info("首帧超时，执行重试回调...")
                self.on_first_frame_timeout()
                return False
            logger.error("start scrcpy error: %s", e)
            self.screenCapCallback.on_exception(e)
            return False

    def stop_scrcpy(self) -> None:
        """
        stop screen copy
        """
        if self._timeout_timer:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        try:
            self.stub.onEnd(scrcpy_pb2.Empty())
        except grpc.RpcError as e:
            logger.error("stop scrcpy error: %s", e)
            if self.screenCapCallback is not None:
                self.screenCapCallback.on_exception(e)
