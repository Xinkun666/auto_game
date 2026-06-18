import socket
import json
import struct
from hoscrcpy_sdk.utils.logger import get_logger

logger = get_logger(__name__)


class DeviceProxy(object):
    def __init__(self, host: str, port: int) -> None:
        # 建立socket连接
        self.host = host
        self.port = port

        self.stop_scrcpy_flag = False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
        except Exception as e:
            logger.error("Failed to create socket connection!, error: %s", e)

    def _send_once(self, request):
        if self.sock is not None:
            self.sock.send(request.encode("utf-8") + b'\n')
        return self.sock.recv(4 * 1024 * 1024)

    def close(self) -> None:
        if self.sock is not None:
            try:
                logger.info("close socket for device")
                self.sock.close()
            except Exception as exception:
                logger.exception("Close socket error: %s", exception)
                pass
            self.sock = None

    def create_driver(self):
        args = {
            "api": "Driver.create",
            "args": []
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "callHypiumApi",
                'params': args}
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        self._send_once(request)

    def create_click_request(self, x: int, y: int):
        args = {
            "api": "Driver.click",
            "this": "Driver#0",
            "args": [x, y]
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "callHypiumApi",
                'params': args}
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        self._send_once(request)

    def create_touch_request(self, mode: str, x: int, y: int):
        args = {
            "api": mode,
            "args": {"x": x, "y": y}
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "Gestures",
                'params': args}
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        res = self._send_once(request)
        try:
            result = json.loads(res.decode('utf-8'))
            if "pts" in result:
                return result["pts"]
        except Exception as ex:
            logger.error("parse result fail: %s", ex)
        return None

    def create_swipe_request(self, start_x: int, start_y: int, end_x: int, end_y: int, step: int):
        args = {
            "api": "Driver.swipe",
            "this": "Driver#0",
            "args": [start_x, start_y, end_x, end_y, step]
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "callHypiumApi",
                'params': args}
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        self._send_once(request)

    def get_layout(self):
        args = {
            "api": "captureLayout"
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "Captures",
                'params': args
                }
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))

        result = ""

        if self.sock is not None:
            bytes_request = request.encode('utf-8')
            head = "_uitestkit_rpc_message_head_".encode('utf-8')
            tail = "_uitestkit_rpc_message_tail_".encode('utf-8')
            buffer = bytearray()
            buffer.extend(head)
            buffer.extend(struct.pack('>i', 1145141919))
            buffer.extend(struct.pack('>i', len(bytes_request)))
            buffer.extend(bytes_request)
            buffer.extend(tail)
            self.sock.sendall(buffer)

        while True:
            data = self.sock.recv(1024 * 1024 * 4)
            data = data.decode("utf-8", "ignore")
            if data.startswith("_uitestkit_rpc_message_head_"):
                result += data[data.index("{\"result"):]
            else:
                result += data
            if data.endswith("_uitestkit_rpc_message_tail_"):
                result = result[:result.index("_uitestkit_rpc_message_tail_")]
                break
        return result

    def stop_scrcpy(self):
        self.stop_scrcpy_flag = True
        args = {
            "api": "stopCaptureScreen"
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "Captures",
                'params': args
                }
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        self._send_once(request)

    def create_screen_size_request(self):
        args = {
            "api": "Driver.getDisplaySize",
            "this": "Driver#0",
            "args": []
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "callHypiumApi",
                'params': args}
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        res = self._send_once(request)
        # print(res)
        try:
            result = json.loads(res.decode('utf-8'))
            if "pts" in result:
                return result
        except Exception as ex:
            logger.error("parse result fail: %s", ex)
        return None

    def change_screen_rotation_request(self, rotation: int):
        args = {
            "api": "Driver.setDisplayRotation",
            "this": "Driver#0",
            "args": [rotation]
        }
        data = {'module': "com.ohos.devicetest.hypiumApiHelper",
                'method': "callHypiumApi",
                'params': args}
        request = json.dumps(data,
                             ensure_ascii=False,
                             separators=(',', ':'))

        # print("Send request: {}".format(request))
        res = self._send_once(request)
        # print(res)
        try:
            result = json.loads(res.decode('utf-8'))
            if "pts" in result:
                return result
        except Exception as ex:
            logger.error("parse result fail: %s", ex)
        return None
