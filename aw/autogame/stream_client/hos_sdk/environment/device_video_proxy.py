import threading
from aw.autogame.stream_client.hos_sdk.communication.rpc_manager import RpcManager
from aw.autogame.stream_client.hos_sdk.ScreenCapCallback import ScreenCapCallback


def start_scrcpy(rpc_manager: RpcManager, screen_cap_callback: ScreenCapCallback) -> None:
    rpc_manager.start_scrcpy(screen_cap_callback=screen_cap_callback)


class DeviceVideoProxy(object):
    def __init__(self, host, port, on_first_frame_timeout=None) -> None:
        self.host = host
        self.port = port
        self.m_rpc_manager = None
        self.on_first_frame_timeout = on_first_frame_timeout

    def close(self) -> None:
        if self.m_rpc_manager is not None:
            self.m_rpc_manager.stop_scrcpy()

    def create_video_screen_copy_request(self, screen_cap_callback: ScreenCapCallback) -> None:
        # 拉起进程
        self.m_rpc_manager = RpcManager(self.host, self.port, self.on_first_frame_timeout)
        screen_cap_callback.on_ready()
        rpc_thread = threading.Thread(target=start_scrcpy, args=(self.m_rpc_manager, screen_cap_callback))
        rpc_thread.daemon = True
        rpc_thread.start()

    def stop_video_screen(self) -> None:
        if self.m_rpc_manager is not None:
            self.m_rpc_manager.stop_scrcpy()
