import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

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
        "creationflags": subprocess_module.CREATE_NO_WINDOW,
    }


def resolve_hdc_executable() -> str:
    return shutil.which("hdc") or shutil.which("hdc.exe") or "hdc"


def _command_text(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _should_hide_command(command: Any, target_executables: Iterable[str]) -> bool:
    text = _command_text(command).lower()
    return any(str(target).lower() in text for target in target_executables)


@contextmanager
def hidden_subprocess_context(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
    target_executables: Iterable[str] = ("icpm_xdc.exe",),
):
    if (os_name or os.name) != "nt":
        yield False
        return

    original_popen = subprocess_module.Popen

    class HiddenTargetPopen(original_popen):
        def __init__(self, *args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            if _should_hide_command(command, target_executables):
                kwargs = dict(kwargs)
                hidden_kwargs = hidden_subprocess_kwargs(
                    os_name=os_name,
                    subprocess_module=subprocess_module,
                )
                startupinfo = kwargs.get("startupinfo") or hidden_kwargs.get("startupinfo")
                if startupinfo is not None:
                    startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0
                    kwargs["startupinfo"] = startupinfo
                kwargs["creationflags"] = (
                    (kwargs.get("creationflags") or 0)
                    | subprocess_module.CREATE_NO_WINDOW
                )
                LOGGER.info("hidden subprocess context applied: command=%s", _command_text(command))
            super().__init__(*args, **kwargs)

    try:
        subprocess_module.Popen = HiddenTargetPopen
        yield True
    finally:
        subprocess_module.Popen = original_popen


def install_hidden_subprocess_patch(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> bool:
    LOGGER.info("global subprocess patch disabled")
    return False
