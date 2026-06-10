import logging
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)


def hidden_subprocess_kwargs(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> Dict[str, Any]:
    if (os_name or os.name) != "nt":
        return {}

    startupinfo = subprocess_module.STARTUPINFO()
    startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "startupinfo": startupinfo,
        "creationflags": _hidden_creationflags(subprocess_module),
    }


def resolve_hdc_executable() -> str:
    return shutil.which("hdc") or shutil.which("hdc.exe") or "hdc"


def hdc_command_args(command: str, hdc_executable: Optional[str] = None) -> Optional[List[str]]:
    text = str(command or "").strip()
    if not text:
        return None

    try:
        parts = shlex.split(text, posix=False)
    except ValueError:
        return None
    if not parts:
        return None

    executable = str(parts[0]).strip("\"'")
    executable_name = os.path.basename(executable).lower()
    if executable_name not in {"hdc", "hdc.exe"}:
        return None

    hdc = hdc_executable or resolve_hdc_executable()
    args = [hdc]
    remaining = [_strip_wrapping_quotes(part) for part in parts[1:]]
    shell_index = next((index for index, part in enumerate(remaining) if part.lower() == "shell"), None)
    if shell_index is None:
        args.extend(remaining)
        return args

    args.extend(remaining[:shell_index])
    args.append("shell")
    remote_command = " ".join(remaining[shell_index + 1 :]).strip()
    if remote_command:
        args.append(remote_command)
    return args


def _strip_wrapping_quotes(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _command_text(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _hidden_creationflags(subprocess_module=subprocess) -> int:
    flags = getattr(subprocess_module, "CREATE_NO_WINDOW", 0)
    flags |= getattr(subprocess_module, "DETACHED_PROCESS", 0)
    return flags


def _should_hide_command(command: Any, target_executables: Iterable[str]) -> bool:
    text = _command_text(command).lower()
    return any(str(target).lower() in text for target in target_executables)


def _with_hidden_kwargs(
    kwargs: Dict[str, Any],
    os_name: Optional[str],
    subprocess_module=subprocess,
) -> Dict[str, Any]:
    next_kwargs = dict(kwargs)
    hidden_kwargs = hidden_subprocess_kwargs(
        os_name=os_name,
        subprocess_module=subprocess_module,
    )
    startupinfo = next_kwargs.get("startupinfo") or hidden_kwargs.get("startupinfo")
    if startupinfo is not None:
        startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        next_kwargs["startupinfo"] = startupinfo
    next_kwargs["creationflags"] = (
        (next_kwargs.get("creationflags") or 0)
        | _hidden_creationflags(subprocess_module)
    )
    return next_kwargs


@contextmanager
def hidden_subprocess_context(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
    os_module=os,
    target_executables: Iterable[str] = ("icpm_xdc.exe",),
    hide_all: bool = False,
):
    if (os_name or os.name) != "nt":
        yield False
        return

    original_popen = subprocess_module.Popen
    original_system = getattr(os_module, "system", None)
    original_popen_os = getattr(os_module, "popen", None)

    def should_hide(command: Any, executable: Any = None) -> bool:
        return (
            hide_all
            or _should_hide_command(command, target_executables)
            or _should_hide_command(executable, target_executables)
        )

    class HiddenTargetPopen(original_popen):
        def __init__(self, *args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            if should_hide(command, kwargs.get("executable")):
                kwargs = _with_hidden_kwargs(
                    kwargs,
                    os_name=os_name,
                    subprocess_module=subprocess_module,
                )
                LOGGER.info("hidden subprocess context applied: command=%s", _command_text(command))
            super().__init__(*args, **kwargs)

    def hidden_system(command):
        if should_hide(command):
            LOGGER.info("hidden os.system context applied: command=%s", _command_text(command))
            return subprocess_module.call(
                command,
                shell=True,
                **_with_hidden_kwargs(
                    {},
                    os_name=os_name,
                    subprocess_module=subprocess_module,
                ),
            )
        return original_system(command)

    try:
        subprocess_module.Popen = HiddenTargetPopen
        if original_system is not None:
            os_module.system = hidden_system
        yield True
    finally:
        subprocess_module.Popen = original_popen
        if original_system is not None:
            os_module.system = original_system
        if original_popen_os is not None:
            os_module.popen = original_popen_os


def install_hidden_subprocess_patch(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> bool:
    LOGGER.info("global subprocess patch disabled")
    return False
