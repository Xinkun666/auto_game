import socket
from hoscrcpy_sdk.environment.device import Device
from hoscrcpy_sdk.ScreenCapCallback import ScreenCapCallback
from hoscrcpy_sdk.Size import Size
from typing import Union
from hoscrcpy_sdk.HosRemoteConfig import HosRemoteConfig


class HosRemoteDevice:
    device_sn = ''
    device = None
    config = None
    size = None

    def __init__(self, config: HosRemoteConfig) -> None:
        if not isinstance(config, HosRemoteConfig):
            raise TypeError("config must be of type HosRemoteConfig")
        self.config = config
        self.device_sn = self.config.get_sn()
        self.ip = self.config.get_ip()
        self.port = self.config.get_port()
        self.size = Size()
        self.device = Device(self.device_sn, self.ip, self.port)

    def start_capture_screen(self, screen_cap_callback: ScreenCapCallback) -> None:
        """
        通过传入视频流回调函数来开始获取视频流以及开启实时反控服务
        Args:
            screen_cap_callback: ScreenCapCallback回调类
        """
        if not isinstance(screen_cap_callback, ScreenCapCallback):
            raise TypeError("screen_cap_callback must be of type ScreenCapCallback")
        if self.device.setup(self.config):
            self.device.create_driver()
            self.device.stop_video_screen_copy()
            self.device.start_video_screen_copy(screen_cap_callback)

    def stop_capture_screen(self) -> None:
        """
        停止获取视频流
        """
        self.device.is_setup = False
        self.device.stop_video_screen_copy()

    def get_sn(self) -> str:
        """
        返回设备sn号
        """
        return self.device_sn

    def execute_shell_command(self, command: Union[str, list], timeout: int = 5 * 60) -> str:
        """
        执行hdc shell命令
        Args:
            command: hdc shell 命令
            timeout: 超时时间
        """
        if not isinstance(command, str):
            if not isinstance(command, list):
                raise TypeError("command must be of type str or list")
        if not isinstance(timeout, int):
            raise TypeError("timeOut must be of type int")
        return self.device.connector_shell_command(command, timeout)

    def get_screen_size(self, need_update: bool) -> Size:
        """
        获取当前设备的分辨率，传入true时会重新获取分辨率信息，传入false时会使用之前的缓存分辨率信息
        Args:
            need_update:传入true时会重新获取分辨率信息，传入false时会使用之前的缓存分辨率信息
        """
        if not isinstance(need_update, bool):
            raise TypeError("needUpdate must be of type bool")
        if need_update:
            if self.device.is_setup:
                result = self.device.get_screen_size()
                if "x" in result and "y" in result:
                    self.size.width = result['x']
                    self.size.height = result['y']
                else:
                    # print(result)
                    raise Exception(result)
        return self.size

    def on_touch_down(self, x: int, y: int) -> None:
        """
        注入手指按下事件, xy为手指按下的坐标(需要调用startCaptureScreen方法后再调用)
        """
        if self.device.is_setup:
            self.device.touch_down(x, y)

    def on_touch_up(self, x: int, y: int) -> None:
        """
        注入手指抬起事件, xy为手指抬起的坐标(需要调用startCaptureScreen方法后再调用)
        """
        if self.device.is_setup:
            self.device.touch_up(x, y)

    def on_touch_move(self, x: int, y: int) -> None:
        """
        注入手指移动事件, xy为手指移动的坐标(需要调用startCaptureScreen方法后再调用)
        """
        if self.device.is_setup:
            self.device.touch_move(x, y)

    def set_rotation_horizontal(self) -> None:
        """
        设置当前屏幕为横屏 rotation = 1 横屏
        """
        if self.device.is_setup:
            self.device.set_screen_rotation(rotation=1)

    def set_rotation_vertical(self) -> None:
        """
        设置当前屏幕为竖屏 rotation = 1 竖屏
        """
        if self.device.is_setup:
            self.device.set_screen_rotation(rotation=0)
