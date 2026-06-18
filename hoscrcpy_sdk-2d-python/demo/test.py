from hoscrcpy_sdk.ScreenCapCallback import ScreenCapCallback

from hoscrcpy_sdk.HosRemoteDevice import HosRemoteDevice
from hoscrcpy_sdk.HostRemoteConfig import HostRemoteConfig



# 定义一个类继承ScreenCapCallback
class VideoStreamTest(ScreenCapCallback):

    def __init__(self):
        super(VideoStreamTest, self).__init__()

    def on_data(self, byteBuffer):
        print(type(byteBuffer))
        print("重写onData方法")

    def on_exception(self, err):
        print("重写onException")

    def on_ready(self):
        print("重写onReady")


if __name__ == '__main__':
    config = HostRemoteConfig(sn='FMR0123423000069', use_old_version=False)  #填写手机sn
    screen_cap_callback = VideoStreamTest()
    host_device = HosRemoteDevice(config)
    host_device.start_capture_screen(screen_cap_callback)

    host_device.get_screen_size(True)
    host_device.set_rotation_horizontal()
    host_device.set_rotation_vertical()
    result = host_device.execute_shell_command('ls /data/local/tmp', 300)
    print("----shell cmd return-----")
    print(result)
    print("-------------------------")
    try:
        while True:
            pass
    except KeyboardInterrupt:
            print("结束")
            host_device.stop_capture_screen()