import threading
from aw.autogame.stream_client.hos_sdk.communication.rpc_manager import RpcManager
from aw.autogame.stream_client.hos_sdk.ScreenCapCallback import ScreenCapCallback


def start_scrcpy(rpc_manager: RpcManager, screen_cap_callback: ScreenCapCallback) -> None:
    rpc_manager.start_scrcpy(screen_cap_callback=screen_cap_callback)


class DeviceVideoProxy(object):
    RPC_THREAD_JOIN_TIMEOUT_SECONDS = 3.0

    def __init__(self, host, port, on_first_frame_timeout=None) -> None:
        self.host = host
        self.port = port
        self.m_rpc_manager = None
        self.on_first_frame_timeout = on_first_frame_timeout
        self._rpc_thread = None

    def close(self) -> None:
        self.stop_video_screen()

    def create_video_screen_copy_request(self, screen_cap_callback: ScreenCapCallback) -> None:
        # 拉起进程
        self.m_rpc_manager = RpcManager(self.host, self.port, self.on_first_frame_timeout)
        screen_cap_callback.on_ready()
        self._rpc_thread = threading.Thread(
            target=start_scrcpy,
            args=(self.m_rpc_manager, screen_cap_callback),
            daemon=True,
            name="HOSScrcpyRpcReceiver",
        )
        self._rpc_thread.start()

    def stop_video_screen(self) -> None:
        rpc_manager = self.m_rpc_manager
        rpc_thread = self._rpc_thread
        if rpc_manager is not None:
            rpc_manager.stop_scrcpy()
        if (
            rpc_thread is not None
            and rpc_thread.is_alive()
            and threading.current_thread() is not rpc_thread
        ):
            rpc_thread.join(timeout=self.RPC_THREAD_JOIN_TIMEOUT_SECONDS)
            if rpc_thread.is_alive():
                return
        self.m_rpc_manager = None
        self._rpc_thread = None
