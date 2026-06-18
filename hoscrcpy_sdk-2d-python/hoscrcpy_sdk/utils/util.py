import subprocess
import platform
import os
import signal
import socket
import secrets
import time
import random
from hoscrcpy_sdk.utils.logger import get_logger

logger = get_logger(__name__)


def get_decode(stream) -> str:
    if not isinstance(stream, str) and not isinstance(stream, bytes):
        ret = str(stream)
    else:
        try:
            ret = stream.decode("utf-8", errors="ignore")
        except (ValueError, AttributeError, TypeError) as _:
            ret = str(stream)
    return ret


def parse_version(version):
    return tuple(map(int, version.split(".")))


def exec_cmd(command, timeout=5 * 60, error_print=True, join_result=False, redirect=False):
    """
    Executes commands in a new shell. Directing stderr to PIPE.

    This is fastboot's own exe_cmd because of its peculiar way of writing
    non-error info to stderr.

    Args:
        command: A sequence of commands and arguments.
        timeout: timeout for exe cmd.
        error_print: print error output or not.
        join_result: join error and out
        redirect: redirect output
    Returns:
        The output of the command run.
    """
    cmd = list()
    if isinstance(command, list):
        cmd.extend(command)
    else:
        command = command.strip()
        cmd.extend(command.split(" "))

    # PIPE本身可容纳的量比较小，所以程序会卡死，所以一大堆内容输出过来的时候，会导致PIPE不足够处理这些内容，因此需要将输出内容定位到其他地方，例如临时文件等
    import tempfile
    out_temp = tempfile.SpooledTemporaryFile(max_size=10 * 1000)
    fileno = out_temp.fileno()

    sys_type = platform.system()
    if sys_type == "Linux" or sys_type == "Darwin":
        if redirect:
            proc = subprocess.Popen(cmd, stdout=fileno,
                                    stderr=fileno, shell=False,
                                    preexec_fn=os.setsid)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, shell=False,
                                    preexec_fn=os.setsid)
    else:
        if redirect:
            proc = subprocess.Popen(cmd, stdout=fileno,
                                    stderr=fileno, shell=False)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, shell=False)
    try:
        (out, err) = proc.communicate(timeout=timeout)
        err = get_decode(err).strip()
        out = get_decode(out).strip()
        if err and error_print:
            logger.error("%s", err)
        if join_result:
            return "%s\n %s" % (out, err) if err else out
        else:
            return err if err else out

    except (TimeoutError, KeyboardInterrupt, AttributeError, ValueError,  # pylint:disable=undefined-variable
            EOFError, IOError) as _:
        sys_type = platform.system()
        if sys_type == "Linux" or sys_type == "Darwin":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            os.kill(proc.pid, signal.SIGINT)
        raise


def get_process_pid(process_name, device):
    cmd = ["ps", "-ef", "|", "grep", process_name]
    ret = device.connector_shell_command(cmd)
    ret = ret.strip()
    pid_list = ret.split("\n")
    for pid in pid_list:
        if "grep" not in pid:
            pid = pid.split()
            return pid[1]
    return None


def get_forward_ports(device) -> list:
    try:
        ports_list = []
        cmd = "fport ls"
        out = get_decode(device.connector_command(cmd)).strip()
        clean_lines = out.split('\n')
        for line_text in clean_lines:
            # clear reverse port first  Example: 'tcp:8011 tcp:9963'     [Reverse]
            if "Reverse" in line_text and "fport" in cmd:
                connector_tokens = line_text.split()
                device.connector_command(["fport", "rm", connector_tokens[0].replace("'", ""),
                                          connector_tokens[1].replace("'", "")])
                continue
            connector_tokens = line_text.split("tcp:")
            if len(connector_tokens) != 3:
                continue
            ports_list.append(int(connector_tokens[1]))
        return ports_list
    except Exception as e:
        logger.error("get_forward_ports error, %s", e)
        return []


def is_idle_port(host, port) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.connect((host, port))
    except Exception as e:
        return True
    finally:
        sock.close()
    return False


def get_forward_port(device, max_attempts: int = 100):
    """
    获取一个可用的转发端口
    通过 TCP socket connect 检测远端端口是否可用，适用于本地和远程设备

    Args:
        device: 设备对象（需要有 ip 和 device_port 属性或 config.get_device_port() 方法）
        max_attempts: 最大尝试次数

    Returns:
        可用的端口号，失败抛出异常
    """
    try:
        # 获取设备 IP 和起始端口
        if hasattr(device, 'ip'):
            host = device.ip
        else:
            host = "127.0.0.1"

        base_port = 20000

        # 随机偏移，避免多个设备同时启动时端口冲突
        start_port = base_port + random.randint(0, 1000)

        for i in range(max_attempts):
            test_port = start_port + i
            if is_idle_port(host, test_port):
                return test_port

        raise Exception(f"No available port found after {max_attempts} attempts")
    except Exception as error:
        logger.error("get_forward_port error, %s", error)
        raise error
