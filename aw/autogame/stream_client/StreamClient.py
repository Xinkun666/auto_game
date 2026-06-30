import grpc
import faststream_pb2
import faststream_pb2_grpc
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import time
import threading
import sys
import signal


class StreamClient:
    def __init__(self) -> None:
        self.image = None
        self.last_time = time.time()
        self.call_count = 0
        self.root = tk.Tk()
        self.root.title("Image Stream Display")
        self.label = ttk.Label(self.root)
        self.label.pack()
        self.thread_ = None
        self.stub_ = None
        self.streamConfig = None
        self.width = 720
        self.height = 1544
        signal.signal(signal.SIGINT, self.signal_handler)

    def update_image(self):
        if self.image:
            tk_image = ImageTk.PhotoImage(self.image)
            self.label.config(image=tk_image)
            self.label.image = tk_image

    def start_gui(self):
        self.root.mainloop()

    def run(self, lowh, highh, skip, width, height, layerid=-1):
        self.streamConfig = faststream_pb2.StreamConfig()
        self.streamConfig.skip = skip
        self.streamConfig.lowh = lowh
        self.streamConfig.highh = highh
        self.streamConfig.width = width
        self.streamConfig.height = height
        self.streamConfig.layerid = layerid
        self.width = width
        self.height = height
        self.layerid = layerid
        try:
            channel = grpc.insecure_channel('127.0.0.1:12345',
                                            options=[('grpc.max_receive_message_length', 5 * 1024 * 1024)])
            grpc.channel_ready_future(channel).result(timeout=1)
            self.stub_ = faststream_pb2_grpc.StreamServiceStub(channel)
        except grpc.RpcError as e:
            print(f"Connection failed: {e}")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            return

        self.thread_ = threading.Thread(target=self.run_client)
        self.thread_.start()
        self.start_gui()

    def run_client(self):
        if not self.stub_:
            return
        try:
            response_stream = self.stub_.StartStream(self.streamConfig)
            for message in response_stream:
                current_time = time.time()
                self.call_count += 1
                if current_time - self.last_time >= 1.0:
                    rate = self.call_count / (current_time - self.last_time)
                    print(f"Rate {rate:.2f} fps")
                    self.call_count = 0
                    self.last_time = current_time
                data = message.data
                width = self.width
                height = self.height
                stride = 0
                print(len(data))
                self.image = Image.frombytes("RGBX", (width, height), data, "raw", "RGBX", stride, 1)
                self.root.after(0, self.update_image)
        except grpc.RpcError as e:
            print(f"Stream error: {e}")
        finally:
            self.stop()

    def stop(self):
        print("stopping...")
        if self.stub_:
            self.stub_.EndStream(faststream_pb2.Empty())
            print("server stop signal sent")
            self.stub_ = None
        self.root.destroy()
        sys.exit(0)

    def signal_handler(self, sig, frame):
        self.stop()

    def get_levels(self):
        try:
            channel = grpc.insecure_channel('127.0.0.1:12345',
                                            options=[('grpc.max_receive_message_length', 5 * 1024 * 1024)])
            grpc.channel_ready_future(channel).result(timeout=1)
            self.stub_ = faststream_pb2_grpc.StreamServiceStub(channel)
        except grpc.RpcError as e:
            print(f"Connection failed: {e}")
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            return

        try:
            response_stream = self.stub_.GetLayers(faststream_pb2.Empty())
            for message in response_stream:
                print(message)
        except grpc.RpcError as e:
            print(f"Stream error: {e}")
        finally:
            self.stop()


client = StreamClient()
client.run(0, 10000, 12, 768, 1520)
# client.get_levels()
client.stop()