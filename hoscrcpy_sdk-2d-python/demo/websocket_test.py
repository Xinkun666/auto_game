import asyncio
import websockets
from hoscrcpy_sdk.ScreenCapCallback import ScreenCapCallback
from hoscrcpy_sdk.HosRemoteDevice import HosRemoteDevice
from hoscrcpy_sdk.HosRemoteConfig import HosRemoteConfig
import json
import threading
from concurrent.futures import ThreadPoolExecutor

async def echo(websocket, path):
    async for message in websocket:
        # print(f"Received: {message}")
        json_message = json.loads(message)
        if json_message["type"] == "screen":
            class VideoStreamTest(ScreenCapCallback):

                def __init__(self):
                    super(VideoStreamTest, self).__init__()
                    self.loop = asyncio.get_running_loop()
                    self.executor = ThreadPoolExecutor()

                def on_data(self, byteBuffer):

                    # async def send_data(frame_data):
                    #     await websocket.send(frame_data)
                    # self.loop.create_task(send_data(byteBuffer))

                    def send_data_sync(frame_data):
                        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(websocket.send(frame_data)))

                    self.loop.run_in_executor(self.executor, send_data_sync, byteBuffer)


                def on_exception(self, err):
                    print("重写onException")

                def on_ready(self):
                    print("重写onReady")

            screen_cap_callback = VideoStreamTest()
            host_device.start_capture_screen(screen_cap_callback)

        elif json_message["type"] == "touchEvent":
            x = json_message["message"]["x"]
            y = json_message["message"]["y"]
            event = json_message["message"]["event"]
            if event == "down":
                host_device.on_touch_down(x, y)
            elif event == "up":
                host_device.on_touch_up(x, y)
            elif event == "move":
                host_device.on_touch_move(x, y)

        elif json_message["type"] == "keyEvent":
            pass

# send_video_data()
if __name__ == '__main__':
    config = HosRemoteConfig(sn='9CN0223A17002964')
    host_device = HosRemoteDevice(config)
    start_server = websockets.serve(echo, "localhost", 8888)
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt as e:
        host_device.stop_capture_screen()
