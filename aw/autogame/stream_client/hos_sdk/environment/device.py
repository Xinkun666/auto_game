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
VIDEO_PROCESS_START_TIMEOUT_SECONDS = float(
    os.environ.get("AUTOGAME_HOSCRCPY_VIDEO_PROCESS_START_TIMEOUT", "1.0")
)
VIDEO_PROCESS_POLL_INTERVAL_SECONDS = 0.1

PATH = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(os.path.dirname(PATH), "res")
VIDEO_SO_NAME_PATTERN = re.compile(r"^libscrcpy_server.*[.]z[.]so$")
VIDEO_SO_DATE_PATTERN = re.compile(r"-(\d{8})[.]z[.]so$")
VIDEO_SO_VERSION_PATTERN = re.compile(r"libscrcpy_server(?:_unix)?_?([0-9]+(?:[.][0-9]+)*)")


def _compact_log_value(value, limit=500):
    text = "" if value is None else str(value)
    text = text.strip().replace("\r", "\\r").replace("\n", " | ")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _format_video_attempt(attempt):
    keys = (
        "source",
        "so",
        "local_md5",
        "device_md5",
        "push_result",
        "start_output",
        "pid_list",
        "unix_socket",
        "skipped",
        "error",
    )
    return ", ".join(
        "{}={}".format(key, _compact_log_value(attempt.get(key)))
        for key in keys
        if key in attempt
    )


def _video_so_sort_key(name):
    date_match = VIDEO_SO_DATE_PATTERN.search(name)
    date_value = int(date_match.group(1)) if date_match else 0
    version_match = VIDEO_SO_VERSION_PATTERN.search(name)
    if version_match:
        version_value = tuple(int(part) for part in version_match.group(1).split("."))
    else:
        version_value = ()
    return (date_value, version_value, name)


def discover_video_so_names(resource_path=RESOURCE_PATH):
    video_path = os.path.join(resource_path, "video")
    try:
        names = [
            name
            for name in os.listdir(video_path)
            if VIDEO_SO_NAME_PATTERN.match(name)
        ]
    except OSError:
        logger.warning("Video so directory is unavailable: %s", video_path)
        return []
    names.sort(key=_video_so_sort_key)
    return names


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
        self.so_name_list = discover_video_so_names()
        logger.info(
            "Discovered scrcpy so candidates old_to_new=%s push_order=%s",
            self.so_name_list,
            list(reversed(self.so_name_list)),
        )
        self.so_md5_map = dict()
        self._init_so_md5_map()
        self.is_use_unix_socket_video_so = False
        self.is_use_unix_socket_agent_so = False
        self._config = None
        self._video_has_retried = False
        self._screen_cap_callback = None
        self.cleanup_command_timeout_seconds = 5.0
        self.selected_video_so = ""
        self.selected_video_so_source = ""
        self.video_start_pids = []
        self.video_start_attempts = []

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
        self.cleanup_command_timeout_seconds = max(
            0.1,
            float(config.get_cleanup_command_timeout_seconds()),
        )
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

    def _remove_fport(self, local_port, remote_node, label):
        if not local_port:
            return None
        command = "fport rm tcp:{} {}".format(local_port, remote_node)
        try:
            result = self.connector_command(
                command,
                timeout=self.cleanup_command_timeout_seconds,
            )
            logger.info("Close %s fport result: %s", label, result)
            return result
        except Exception as exc:
            logger.warning(
                "Close %s fport failed after %.1fs: %s",
                label,
                self.cleanup_command_timeout_seconds,
                exc,
            )
            raise

    def close(self) -> None:
        cleanup_errors = []
        if self.video_proxy is not None:
            try:
                self.video_proxy.stop_video_screen()
            except Exception as exc:
                cleanup_errors.append(("video rpc", exc))
            finally:
                self.video_proxy = None

        cleanup_specs = (
            (
                "video_port",
                "localabstract:scrcpy_grpc_socket"
                if self.is_use_unix_socket_video_so
                else "tcp:{}".format(self.video_server_port),
                "video",
            ),
            (
                "agent_port",
                "localabstract:uitest_socket"
                if self.is_use_unix_socket_agent_so
                else "tcp:{}".format(self.agent_server_port),
                "agent",
            ),
            (
                "guest_port",
                "localabstract:uitest_socket"
                if self.is_use_unix_socket_agent_so
                else "tcp:{}".format(self.agent_server_port),
                "guest",
            ),
            (
                "layout_port",
                "localabstract:uitest_socket"
                if self.is_use_unix_socket_agent_so
                else "tcp:{}".format(self.agent_server_port),
                "layout",
            ),
        )
        for attr_name, remote_node, label in cleanup_specs:
            local_port = getattr(self, attr_name, None)
            try:
                self._remove_fport(local_port, remote_node, label)
            except Exception as exc:
                cleanup_errors.append(("%s fport" % label, exc))
            finally:
                setattr(self, attr_name, None)

        for attr_name in ("proxy", "guest_proxy", "layout_proxy"):
            proxy = getattr(self, attr_name, None)
            try:
                if proxy is not None:
                    proxy.close()
            except Exception as exc:
                cleanup_errors.append((attr_name, exc))
            finally:
                setattr(self, attr_name, None)

        if cleanup_errors:
            detail = "; ".join(
                "%s: %s" % (label, error)
                for label, error in cleanup_errors
            )
            raise RuntimeError("HOS cleanup failed: %s" % detail)

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
        attempts = self.video_start_attempts = []
        # 先获取当前设备中有没推送过资源,如果已经推送过了则优先使用设备中的资源进行启动
        device_agent_path = "/data/local/tmp/libscreen_casting.z.so"
        # 获取设备端的资源md5值
        device_so_md5_output = self.connector_shell_command("md5sum {}".format(device_agent_path))
        device_so_md5_info = device_so_md5_output.split(" ")[0].strip()
        matched_so_name = self.so_md5_map.get(device_so_md5_info)
        logger.info(
            "Device scrcpy so md5: parsed=%s matched_so=%s raw=%s",
            device_so_md5_info,
            matched_so_name,
            _compact_log_value(device_so_md5_output),
        )
        forced_so = str(getattr(config, "get_force_video_so", lambda: "")() or "").strip()
        auto_mode = forced_so == "auto"
        excluded_video_sos = set(
            getattr(config, "get_excluded_video_sos", lambda: ())() or ()
        )
        attempted_video_sos = set()
        if forced_so and not auto_mode:
            if forced_so == "latest":
                if not self.so_name_list:
                    raise FileNotFoundError("No local HOScrcpy video so candidates were found")
                forced_so = self.so_name_list[-1]
            if os.path.basename(forced_so) != forced_so or forced_so not in self.so_name_list:
                raise ValueError(
                    "Forced HOScrcpy video so is unavailable: {}. candidates={}".format(
                        forced_so,
                        self.so_name_list,
                    )
                )
            logger.info(
                "Force push scrcpy so requested: requested=%s resolved=%s",
                config.get_force_video_so(),
                forced_so,
            )
            path = os.path.join(RESOURCE_PATH, "video" + os.path.sep + forced_so)
            attempt = {"source": "forced", "so": forced_so}
            try:
                attempt["local_md5"] = self.device_helper._calculate_md5(path)
                attempt["push_result"] = self.device_helper.init_video_so_resource(path)
                attempt["start_output"] = self.start_video_so_server(video_params)
                pid_list = self._wait_for_video_pid()
                attempt["pid_list"] = pid_list
                attempt["unix_socket"] = "unix" in forced_so
            except Exception as exc:
                attempt["error"] = repr(exc)
                attempts.append(attempt)
                raise RuntimeError(
                    "Forced HOScrcpy video so failed: {}".format(
                        _format_video_attempt(attempt)
                    )
                ) from exc
            if not pid_list:
                attempts.append(attempt)
                raise RuntimeError(
                    "Forced HOScrcpy video so did not expose a process: {}".format(
                        _format_video_attempt(attempt)
                    )
                )
            self.selected_video_so = forced_so
            self.selected_video_so_source = "forced"
            self.video_start_pids = list(pid_list)
            attempts.append(attempt)
            self.is_use_unix_socket_video_so = "unix" in forced_so
            logger.info(
                "Video server started with forced so=%s pid_list=%s",
                forced_so,
                pid_list,
            )
            return

        if device_so_md5_info in self.so_md5_map and not (
            auto_mode and matched_so_name in excluded_video_sos
        ):
            current_so_name = matched_so_name
            attempted_video_sos.add(current_so_name)
            logger.info("try to use exist so: md5=%s so=%s", device_so_md5_info, current_so_name)
            # 直接进行启动
            start_output = self.start_video_so_server(video_params)
            pid_list = self._wait_for_video_pid()
            attempt = {
                "source": "existing",
                "so": current_so_name,
                "device_md5": device_so_md5_info,
                "start_output": start_output,
                "pid_list": pid_list,
                "unix_socket": "unix" in current_so_name,
            }
            attempts.append(attempt)
            if pid_list:
                logger.info("Video server started with existing so=%s pid_list=%s", current_so_name, pid_list)
                # 判断当前推送的so是不是要使用unix_socket连接方式
                self.is_use_unix_socket_video_so = "unix" in current_so_name
                self.selected_video_so = current_so_name
                self.selected_video_so_source = "existing"
                self.video_start_pids = list(pid_list)
                return
            logger.warning("Existing scrcpy so did not expose process: %s", _format_video_attempt(attempt))
        elif device_so_md5_info in self.so_md5_map:
            attempts.append(
                {
                    "source": "existing",
                    "so": matched_so_name,
                    "device_md5": device_so_md5_info,
                    "skipped": "excluded_after_runtime_failure",
                }
            )
            logger.info(
                "Skip previously failed scrcpy so in auto mode: %s",
                matched_so_name,
            )
        else:
            logger.warning(
                "Device scrcpy so is missing or unknown: parsed=%s raw=%s",
                device_so_md5_info,
                _compact_log_value(device_so_md5_output),
            )
        # 启动不成功再逐个遍历尝试
        for so_name in reversed(self.so_name_list):
            if auto_mode and (
                so_name in excluded_video_sos or so_name in attempted_video_sos
            ):
                continue
            path = os.path.join(RESOURCE_PATH, "video" + os.path.sep + so_name)
            attempt = {
                "source": "auto-pushed" if auto_mode else "pushed",
                "so": so_name,
            }
            try:
                attempt["local_md5"] = self.device_helper._calculate_md5(path)
                logger.info("try to use %s local_md5=%s path=%s", so_name, attempt["local_md5"], path)
                attempt["push_result"] = self.device_helper.init_video_so_resource(path)
                attempt["start_output"] = self.start_video_so_server(video_params)
                pid_list = self._wait_for_video_pid()
                attempt["pid_list"] = pid_list
                attempt["unix_socket"] = "unix" in so_name
            except Exception as exc:
                attempt["error"] = repr(exc)
                attempts.append(attempt)
                logger.exception("Video server attempt failed before process detection: %s", _format_video_attempt(attempt))
                continue
            attempts.append(attempt)
            if pid_list:
                logger.info("Video server started with pushed so=%s pid_list=%s", so_name, pid_list)
                # 判断当前推送的so是不是要使用unix_socket连接方式
                self.is_use_unix_socket_video_so = "unix" in so_name
                self.selected_video_so = so_name
                self.selected_video_so_source = attempt["source"]
                self.video_start_pids = list(pid_list)
                return
            logger.warning("Pushed scrcpy so did not expose process: %s", _format_video_attempt(attempt))
        # 全部试完都不行就报错
        raise Exception(
            "Init scrcpy service failed! attempts=%s"
            % " || ".join(_format_video_attempt(attempt) for attempt in attempts)
        )

    def start_video_so_server(self, video_params: str):
        """拉起投屏so服务"""
        command = (
            r"/system/bin/uitest start-daemon singleness --extension-name \
            libscreen_casting.z.so {} &".format(video_params)
        )
        result = self.connector_shell_command(command)
        logger.info("Start video so command result: %s", _compact_log_value(result))
        return result

    def _wait_for_video_pid(self):
        timeout = max(0.0, VIDEO_PROCESS_START_TIMEOUT_SECONDS)
        deadline = time.monotonic() + timeout
        while True:
            pid_list = self.device_helper.get_video_pid_list()
            if pid_list or time.monotonic() >= deadline:
                return pid_list
            time.sleep(VIDEO_PROCESS_POLL_INTERVAL_SECONDS)

    def collect_disconnect_diagnostics(self, timeout: float = 1.0):
        """在清理抓流前采集设备端进程、视频端点和 HDC 转发状态。"""
        timeout = max(0.1, float(timeout))
        remote_node = (
            "localabstract:scrcpy_grpc_socket"
            if self.is_use_unix_socket_video_so
            else "tcp:{}".format(self.video_server_port)
        )
        result = {
            "selected_video_so": self.selected_video_so,
            "selected_video_so_source": self.selected_video_so_source,
            "video_start_pids": list(self.video_start_pids),
            "video_start_attempts": list(self.video_start_attempts),
            "video_local_port": self.video_port,
            "video_device_port": self.video_server_port,
            "video_remote_node": remote_node,
            "unix_socket": self.is_use_unix_socket_video_so,
        }

        try:
            target_probe = self.connector_shell_command(
                "echo __AUTOGAME_HDC_TARGET_OK__",
                timeout=timeout,
            )
            result["hdc_target_probe"] = _compact_log_value(target_probe, limit=1000)
            result["hdc_target_reachable"] = "__AUTOGAME_HDC_TARGET_OK__" in str(
                target_probe or ""
            )
        except Exception as exc:
            result["hdc_target_reachable"] = False
            result["hdc_target_probe_error"] = str(exc)

        try:
            process_output = self.connector_shell_command(
                '"ps -ef | grep singleness"',
                timeout=timeout,
            )
            result["video_process_listing"] = _compact_log_value(process_output, limit=4000)
            result["video_pid_list_at_disconnect"] = [
                line.split()[1]
                for line in str(process_output or "").splitlines()
                if "libscreen_casting" in line
                and "extension-name" in line
                and len(line.split()) > 1
            ]
            result["video_process_alive"] = bool(result["video_pid_list_at_disconnect"])
        except Exception as exc:
            result["video_process_error"] = str(exc)

        endpoint_command = (
            '"cat /proc/net/unix | grep scrcpy_grpc_socket"'
            if self.is_use_unix_socket_video_so
            else '"netstat -an | grep :{}"'.format(self.video_server_port)
        )
        try:
            endpoint_output = self.connector_shell_command(endpoint_command, timeout=timeout)
            result["video_endpoint_listing"] = _compact_log_value(endpoint_output, limit=4000)
            result["video_endpoint_present"] = bool(str(endpoint_output or "").strip())
        except Exception as exc:
            result["video_endpoint_error"] = str(exc)

        try:
            fport_output = self.connector_command("fport ls", timeout=timeout)
            result["hdc_fport_listing"] = _compact_log_value(fport_output, limit=4000)
            local_marker = "tcp:{}".format(self.video_port) if self.video_port else ""
            result["video_fport_present"] = bool(
                local_marker and local_marker in str(fport_output or "")
            )
        except Exception as exc:
            result["hdc_fport_error"] = str(exc)
        return result

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

    def push_file(self, local: str, remote: str) -> str:
        if not os.path.exists(local):
            raise FileNotFoundError("HOScrcpy resource not found: {}".format(local))
        local = "\"{}\"".format(local)
        remote = "\"{}\"".format(remote)
        res = self.connector_command("file send {} {}".format(local, remote))
        logger.info("Push file result: %s", res)
        return res

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
                    self._remove_fport(
                        self.video_port,
                        "localabstract:scrcpy_grpc_socket",
                        "first-frame-timeout video",
                    )
                else:
                    self._remove_fport(
                        self.video_port,
                        "tcp:{}".format(self.video_server_port),
                        "first-frame-timeout video",
                    )
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
                self.video_proxy = None
            if self.video_port:
                remote_node = (
                    "localabstract:scrcpy_grpc_socket"
                    if self.is_use_unix_socket_video_so
                    else "tcp:{}".format(self.video_server_port)
                )
                try:
                    self._remove_fport(self.video_port, remote_node, "video")
                finally:
                    self.video_port = None
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

    def init_video_so_resource(self, file_path: str) -> str:
        """推送投屏so服务"""
        device_agent_path = "/data/local/tmp/libscreen_casting.z.so"
        logger.info("Init scrcpy service...")
        # 获取设备端的资源md5值
        self.device.connector_shell_command("rm -rf {}".format(device_agent_path))
        # 推送资源
        return self.device.push_file(file_path, device_agent_path)

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
