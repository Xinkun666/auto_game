from abc import ABC, abstractmethod


class ScreenCapCallback(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def on_data(self, byte_buffer: bytes):
        """
        当sdk获取到视频流数据后，会回调此方法，并传入视频流的ByteBuffer，开发者可以根据此ByteBuffer进行画面显示
        """
        pass

    @abstractmethod
    def on_exception(self, err: Exception):
        """
        当获取视频流出错后，会回调此方法，传入报错信息
        """
        pass

    @abstractmethod
    def on_ready(self):
        """
        因为需要设备画面变动才能获取视频流，所以如果设备已经处于一个亮屏且画面没有变动的状态时，
        onData方法是不会被调用的。onReady方法是为了通知开发者当前已经处于了获取视频流就绪状态，
        开发者可以在此做一些使设备画面变动的动作，比如：按电源键点亮关闭屏幕
        """
        pass
