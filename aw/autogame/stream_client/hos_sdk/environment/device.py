import os
import hashlib
import time
import re
from typing import Union
from aw.autogame.stream_client.hos_sdk.ScreenCapCallback import ScreenCapCallback
from aw.autogame.stream_client.hos_sdk.environment.device_video_proxy import DeviceVideoProxy
from aw.autogame.stream_client.hos_sdk.utils.variables import Connector
from aw.autogame.stream_client.hos_sdk.utils.variables import ActionMode
from aw.autogame.stream_client.hos_sdk.utils.util import exec_cmd
from aw.autogame.stream_client.hos_sdk.utils.util import get_forward_port
from aw.autogame.stream_client.hos_sdk.utils.util import parse_version
from aw.autogame.stream_client.hos_sdk.environment.device_proxy import DeviceProxy
from aw.autogame.stream_client.hos_sdk.HosRemoteConfig import HosRemoteConfig
from aw.autogame.stream_client.hos_sdk.utils.logger import get_logger

logger = get_logger(__name__)

MODIFY_TIME_STR = "Modify: "
BASE_TIME = "2023-10-01 00:00:00"
AGENT_CLEAR_PATH = ["app", "commons-", "agent", "libagent_antry"]
FPORT_RETRY_ATTEMPTS = int(os.environ.get("AUTOGAME_HOSCRCPY_FPORT_RETRIES", "10"))
FPORT_RETRY_DELAY_SECONDS = float(os.environ.get("AUTOGAME_HOSCRCPY_FPORT_RETRY_DELAY", "1"))

PATH = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(os.path.dirname(PATH), "res")


class Device(object):

    def __init__(self, device_sn, host: str = "127.0.0.1", port: int = "8710") -> None:
        self.device_sn = device_sn
        self.cmd = None

        self.agent_port = None
        self.guest_port = None
        self.layout_port = None
        self.video_port = None

        self.agent_server_port = 8012
        self.video_server_port = 5000
        self.host = host
        self.port = port
        self.proxy = None
        self.guest_proxy = None
        self.layout_proxy = None
        self.video_proxy = None
        self.is_setup = False
        self.device_helper = DeviceHelper(self)
        self.last_action_time = 0
        # 先计算出当前本地所有投屏so的md5
        self.so_name_list = ["libscrcpy_server0.z.so", "libscrcpy_server1.z.so", "libscrcpy_server2.z.so",
                             "libscrcpy_server3.z.so", "libscrcpy_server_5.10-20260114.z.so",
                             "libscrcpy_server_unix_6.3.1-20260113.z.so", "libscrcpy_server_unix_6.4-20260113.z.so",
                             "libscrcpy_server_unix_6.5-20260313.z.so", "libscrcpy_server_unix_6.6-20260418.z.so"]
        self.so_md5_map = dict()
        self._init_so_md5_map()
        self.is_use_unix_socket_video_so = False
        self.is_use_unix_socket_agent_so = False
        self._config = None
        self._video_has_retried = False
        self._screen_cap_callback = None

    def _init_so_md5_map(self):
        folder_path = os.path.join(RESOURCE_PATH)
        for so_name in self.so_name_list:
            path = os.path.join(folder_path, "video" + os.path.sep + so_name)
            self.so_md5_map[self.device_helper._calculate_md5(path)] = so_name

    def _check_device_status(self) -> bool:
        ret = self.connector_command(["list", "targets"])
        logger.info("Device list: %s", ret)
        if "\r\n" in ret:
            device_list = ret.strip().split("\r\n")
        else:
            device_list = ret.strip().split("\n")
        for sn in device_list:
            if self.device_sn == sn:
                return True
        return False

    def setup(self, config: HosRemoteConfig) -> bool:
        self._config = config
        self._video_has_retried = False
        # 1、先执行获取设备的SN
        if not self._check_device_status():
            logger.error("Can not find device [%s], please check...", self.device_sn)
            return False
        # 杀进程
        # 查找指定的so进程
        pid_list = self.device_helper.get_video_pid_list()
        for pid in pid_list:
            self.connector_shell_command("kill -9 {}".format(pid))
        start_result = self._start_uitest()
        self._start_video_server(config)
        self.is_setup = start_result
        return start_result

    def close(self) -> None:
        if self.video_proxy is not None:
            self.video_proxy.stop_video_screen()
        if self.video_port:
            if self.is_use_unix_socket_video_so:
                self.connector_command("fport rm tcp:{} localabstract:scrcpy_grpc_socket".format(self.video_port))
            else:
                self.connector_command("fport rm tcp:{} tcp:{}".format(self.video_port, self.video_server_port))
        if self.agent_port:
            if self.is_use_unix_socket_agent_so:
                ret = self.connector_command("fport rm tcp:{} localabstract:uitest_socket".format(self.agent_port))
            else:
                ret = self.connector_command("fport rm tcp:{} tcp:{}".format(self.agent_port, self.agent_server_port))
            logger.info("Close agent port result: %s", ret)
        if self.guest_port:
            if self.is_use_unix_socket_agent_so:
                ret = self.connector_command("fport rm tcp:{} localabstract:uitest_socket".format(self.guest_port))
            else:
                ret = self.connector_command("fport rm tcp:{} tcp:{}".format(self.guest_port, self.agent_server_port))
            logger.info("Close guest port result: %s", ret)
        if self.layout_port:
            if self.is_use_unix_socket_agent_so:
                ret = self.connector_command("fport rm tcp:{} localabstract:uitest_socket".format(self.layout_port))
            else:
                ret = self.connector_command("fport rm tcp:{} tcp:{}".format(self.layout_port, self.agent_server_port))
            logger.info("Close layout port result: %s", ret)
        if self.proxy:
            self.proxy.close()
        if self.guest_proxy:
            self.guest_proxy.close()
        if self.layout_proxy:
            self.layout_proxy.close()

    def _get_uitest_process(self, extension: bool):
        result = self.connector_shell_command('\"ps -ef | grep singleness\"')
        proc_running = result.split("\n")
        for data in proc_running:
            if extension:
                if "singleness" in data and "grep" not in data and "extension-name" in data:
                    data = data.split()
                    return data[1]
            else:
                if "singleness" in data and "grep" not in data and "extension-name" not in data:
                    data = data.split()
                    return data[1]
        return None

    def _retry_fport(self, is_agent_so: bool, device_port: int):
        last_result = ""
        retry_attempts = max(1, FPORT_RETRY_ATTEMPTS)
        for attempt in range(1, retry_attempts + 1):
            port = get_forward_port(self)
            logger.info("Trying to forward port: %s (attempt %s/%s)", port, attempt, retry_attempts)
            if is_agent_so:
                if self.is_use_unix_socket_agent_so:
                    ret = self.connector_command("fport tcp:{} localabstract:uitest_socket".format(port))
                else:
                    ret = self.connector_command("fport tcp:{} tcp:{}".format(port, device_port))
            else:
                if self.is_use_unix_socket_video_so:
                    ret = self.connector_command("fport tcp:{} localabstract:scrcpy_grpc_socket".format(port))
                else:
                    ret = self.connector_command("fport tcp:{} tcp:{}".format(port, device_port))
            last_result = str(ret or "").strip()
            logger.info("Forward port result: %s", last_result)
            if "OK" in last_result.upper():
                logger.info("Forward port success: local tcp:%s -> device port %s", port, device_port)
                return port
            if attempt < retry_attempts and FPORT_RETRY_DELAY_SECONDS > 0:
                time.sleep(FPORT_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            "cannot fport after {} attempts, device_port={}, last_result={}".format(
                retry_attempts,
                device_port,
                last_result or "<empty>",
            )
        )

    def _start_uitest(self) -> bool:
        # 再检测agent是否需要更新
        self.device_helper.init_agent_resource()
        # 与uitest建立连接
        self.agent_port = self._retry_fport(True, self.agent_server_port)
        self.guest_port = self._retry_fport(True, self.agent_server_port)
        self.layout_port = self._retry_fport(True, self.agent_server_port)
        logger.info(
            "Agent fports ready: agent=%s guest=%s layout=%s device_port=%s",
            self.agent_port,
            self.guest_port,
            self.layout_port,
            self.agent_server_port,
        )
        # 若abc已经启动，则不需要重复启动
        pid = self._get_uitest_process(extension=False)
        if pid is None:
            logger.info("Start device control service...")
            self.connector_shell_command("/system/bin/uitest start-daemon singleness &")
            # 检测uitest是否正常启动
            pid = self._get_uitest_process(extension=False)
            if pid is None:
                logger.error("Start device control service failed.")
                return False
        else:
            logger.info("Device control service already start...")
        # print("Uitest pid: {}".format(pid))
        time.sleep(1)
        self.proxy = DeviceProxy(self.host, self.agent_port)
        self.guest_proxy = DeviceProxy(self.host, self.guest_port)
        self.layout_proxy = DeviceProxy(self.host, self.layout_port)

        return True if self.proxy.sock else False

    def _start_video_server(self, config: HosRemoteConfig) -> None:
        """推送并开启投屏so服务"""
        # 拉起进程
        video_params = config.get_params()
        # 先获取当前设备中有没推送过资源,如果已经推送过了则优先使用设备中的资源进行启动
        folder_path = os.path.join(RESOURCE_PATH)
        device_agent_path = "/data/local/tmp/libscreen_casting.z.so"
        # 获取设备端的资源md5值
        device_so_md5_info = self.connector_shell_command(
            "md5sum {}".format(device_agent_path)).split(" ")[0].strip()
        if device_so_md5_info in self.so_md5_map:
            current_so_name = self.so_md5_map.get(device_so_md5_info)
            logger.info("try to use exist so : %s", device_so_md5_info)
            # 直接进行启动
            self.start_video_so_server(video_params)
            pid_list = self.device_helper.get_video_pid_list()
            if pid_list:
                logger.info("Video server started with existing so=%s pid_list=%s", current_so_name, pid_list)
                # 判断当前推送的so是不是要使用unix_socket连接方式
                self.is_use_unix_socket_video_so = "unix" in current_so_name
                return
        # 启动不成功再逐个遍历尝试
        for so_name in reversed(self.so_name_list):
            logger.info("try to use %s", so_name)
            folder_path = os.path.join(RESOURCE_PATH)
            path = os.path.join(folder_path, "video" + os.path.sep + so_name)
            self.device_helper.init_video_so_resource(path)
            self.start_video_so_server(video_params)
            pid_list = self.device_helper.get_video_pid_list()
            if pid_list:
                logger.info("Video server started with pushed so=%s pid_list=%s", so_name, pid_list)
                # 判断当前推送的so是不是要使用unix_socket连接方式
                self.is_use_unix_socket_video_so = "unix" in so_name
                return
        # 全部试完都不行就报错
        raise Exception("Init scrcpy service failed!")

    def start_video_so_server(self, video_params: str) -> None:
        """拉起投屏so服务"""
        self.connector_shell_command(
            r"/system/bin/uitest start-daemon singleness --extension-name \
            libscreen_casting.z.so {} &".format(video_params))

    def _exec_cmd(self, command: Union[str, list], timeout: int = 5 * 60):
        if isinstance(command, list):
            self.cmd.extend(command)
        else:
            command = command.strip()
            self.cmd.extend(command.split(" "))
        return exec_cmd(self.cmd, join_result=True, timeout=timeout)

    def connector_command(self, command: Union[str, list], timeout: int = 5 * 60):
        self.cmd = [Connector.name, "-s", "{}:{}".format(self.host, self.port), "-t", self.device_sn]
        return self._exec_cmd(command, timeout=timeout)

    def connector_shell_command(self, command: Union[str, list], timeout: int = 5 * 60):
        self.cmd = [Connector.name, "-s", "{}:{}".format(self.host, self.port), "-t", self.device_sn, "shell"]
        return self._exec_cmd(command, timeout)

    def push_file(self, local: str, remote: str) -> None:
        if not os.path.exists(local):
            raise FileNotFoundError("HOScrcpy resource not found: {}".format(local))
        local = "\"{}\"".format(local)
        remote = "\"{}\"".format(remote)
        res = self.connector_command("file send {} {}".format(local, remote))
        logger.info("Push file result: %s", res)

    def perform_action(self, x: int, y: int, action: ActionMode) -> None:
        self.proxy.create_action_request(x, y, action)

    def click(self, x: int, y: int) -> None:
        self.proxy.create_click_request(x, y)

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        self.proxy.create_swipe_request(start_x, start_y, end_x, end_y)

    def touch_up(self, x: int, y: int) -> None:
        result = self.guest_proxy.create_touch_request("touchUp", x, y)
        if result is not None:
            self.last_action_time = result

    def touch_down(self, x: int, y: int) -> None:
        result = self.guest_proxy.create_touch_request("touchDown", x, y)
        if result is not None:
            self.last_action_time = result

    def touch_move(self, x: int, y: int) -> None:
        result = self.guest_proxy.create_touch_request("touchMove", x, y)
        if result is not None:
            self.last_action_time = result

    def get_screen_size(self):
        result = self.guest_proxy.create_screen_size_request()
        if result is not None:
            if "pts" in result:
                self.last_action_time = result["pts"]
                return result["result"]
            else:
                return result

    def set_screen_rotation(self, rotation: int):
        result = self.guest_proxy.change_screen_rotation_request(rotation)
        if result is not None:
            self.last_action_time = result["pts"]
            return result

    def start_video_screen_copy(self, screen_cap_callback: ScreenCapCallback) -> None:
        self._screen_cap_callback = screen_cap_callback
        # 设置端口转发
        self.video_port = self._retry_fport(False, self.video_server_port)
        logger.info(
            "Video fport ready: local tcp:%s -> device port %s, unix_socket=%s",
            self.video_port,
            self.video_server_port,
            self.is_use_unix_socket_video_so,
        )
        self.video_proxy = DeviceVideoProxy(self.host, self.video_port, self._on_first_frame_timeout)
        self.video_proxy.create_video_screen_copy_request(screen_cap_callback)

    def _on_first_frame_timeout(self) -> None:
        """首帧超时回调：删除设备端投屏SO并重走投屏流程"""
        if self._video_has_retried:
            logger.warning("已经重试过一次，不再重试")
            return
        self._video_has_retried = True
        logger.info("首帧超时，删除投屏SO并重走投屏流程...")
        # 停止当前视频代理
        if self.video_proxy is not None:
            self.video_proxy.stop_video_screen()
            self.video_proxy = None
        # 清理端口转发
        if self.video_port:
            try:
                if self.is_use_unix_socket_video_so:
                    self.connector_command("fport rm tcp:{} localabstract:scrcpy_grpc_socket".format(self.video_port))
                else:
                    self.connector_command("fport rm tcp:{} tcp:{}".format(self.video_port, self.video_server_port))
            except Exception as e:
                logger.error("清理端口转发失败: %s", e)
            self.video_port = None
        # 删除设备端投屏SO
        self.connector_shell_command("rm -rf /data/local/tmp/libscreen_casting.z.so")
        # 杀掉投屏进程
        pid_list = self.device_helper.get_video_pid_list()
        for pid in pid_list:
            self.connector_shell_command("kill -9 {}".format(pid))
        # 重新启动投屏服务（推送SO并拉起进程）
        self._start_video_server(self._config)
        # 重新端口转发并创建视频流连接
        self.video_port = self._retry_fport(False, self.video_server_port)
        self.video_proxy = DeviceVideoProxy(self.host, self.video_port)
        self.video_proxy.create_video_screen_copy_request(self._screen_cap_callback)

    def stop_video_screen_copy(self) -> None:
        if self.is_setup:
            if self.video_proxy is not None:
                self.video_proxy.stop_video_screen()
            if self.video_port:
                self.connector_command("fport rm tcp:{} tcp:{}".format(self.video_port, self.video_server_port))
        else:
            self.close()

    def get_layout(self):
        return self.layout_proxy.get_layout()

    def create_driver(self) -> None:
        self.proxy.create_driver()

    def press_power_key(self) -> None:
        self.connector_shell_command("uinput -K -d 18 -u 18")

    def wake_up(self) -> None:
        self.connector_shell_command("power-shell wakeup")


class DeviceHelper(object):

    def __init__(self, device: Device) -> None:
        self.device = device

    def init_agent_resource(self) -> None:
        self._init_so_resource()

    @staticmethod
    def _resolve_resource_path(*paths: str) -> str:
        for path in paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("HOScrcpy resource not found: {}".format(" or ".join(paths)))

    def _init_so_resource(self) -> None:
        folder_path = os.path.join(RESOURCE_PATH)
        file_postfix = ".so"
        device_agent_path = "/data/local/tmp/agent.so"
        logger.info("Start init resource...")
        agent_filename = ""
        normal_agent_path = self._resolve_resource_path(
            os.path.join(folder_path, "uitest_agent_1.1.4.so"),
            os.path.join(folder_path, "uitest_agent_v1.1.4.so"),
        )
        unix_agent_path = self._resolve_resource_path(os.path.join(folder_path, "uitest_agent_1.2.2.so"))
        local_link = "1.1.4"
        agent_path = normal_agent_path
        if self.need_unix_socket_agent_so():
            self.device.is_use_unix_socket_agent_so = True
            agent_path = unix_agent_path
            local_link = "1.2.2"
        # 获取设备端的版本号
        device_ver_info = self.device.connector_shell_command(
            "cat {} | grep -a UITEST_AGENT_LIBRARY".format(device_agent_path))
        # print("{}".format(device_ver_info))
        if "#" in device_ver_info:
            # 要获取#号之后的内容才是实际的版本号
            index = device_ver_info.index("#")
            device_ver_info = device_ver_info[index + 1:]
        matcher = re.search(r'\d{1,3}[.]\d{1,3}[.]\d{1,3}', device_ver_info)
        device_link = matcher.group(0) if matcher else "0.0.0"
        device_link = parse_version(device_link)
        local_link = parse_version(local_link)
        logger.info("local service version %s, device service version %s", local_link, device_link)
        need_update = False
        # 如果当前设备是6.0.2.2以下的uitest版本,同时agent.so是1.2.X以上的版本,则也要重新推送
        target_agent_so_version = parse_version("1.2.0")
        if not self.need_unix_socket_agent_so() and target_agent_so_version < device_link:
            need_update = True
        if device_link < local_link:
            need_update = True
        if need_update:
            logger.info("Start update device control service...")
            # if uitest running kill first
            self.device.connector_shell_command('\"kill -9 $(pidof uitest)\"')
            for file in AGENT_CLEAR_PATH:
                self.device.connector_shell_command("rm /data/local/tmp/{}*".format(file))
            self.device.push_file(agent_path, device_agent_path)
            logger.info("Update device control service finish.")
        else:
            logger.info("device control service is up to date!")

    def init_video_so_resource(self, file_path: str) -> None:
        """推送投屏so服务"""
        device_agent_path = "/data/local/tmp/libscreen_casting.z.so"
        logger.info("Init scrcpy service...")
        # 获取设备端的资源md5值
        self.device.connector_shell_command("rm -rf {}".format(device_agent_path))
        # 推送资源
        self.device.push_file(file_path, device_agent_path)

    def need_unix_socket_agent_so(self) -> bool:
        # 检查uitest的版本是否大于6.0.2.2
        uitest_version = self.device.connector_shell_command("uitest --version", 5)
        matcher = re.search(r'\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3}', uitest_version)
        uitest_version = matcher.group(0) if matcher else "0.0.0.0"
        uitest_version = parse_version(uitest_version)
        target_version = parse_version("6.0.2.1")
        return uitest_version > target_version

    @classmethod
    def _calculate_md5(cls, file_path: str):
        """
        获取文件的md5值
        """
        # 打开文件
        with open(file_path, 'rb') as file:
            # 创建MD5哈希对象
            md5 = hashlib.md5()
            # 读取文件内容，并更新哈希对象
            for chunk in iter(lambda: file.read(4096), b""):
                md5.update(chunk)
            # 返回MD5值
            return md5.hexdigest()

    def get_video_pid_list(self):
        pid_list = []
        res = self.device.connector_shell_command("\"ps -ef | grep singleness\"")
        for s in res.split(os.linesep):
            if "libscreen_casting" in s and "extension-name" in s:
                # pids.append(s.split("\\s+")[1])
                pid_list.append(s.split()[1])
        return pid_list
