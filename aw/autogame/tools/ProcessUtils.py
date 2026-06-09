import os
import shutil
import subprocess
from typing import Any, Dict, Optional


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


def install_hidden_subprocess_patch(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> bool:
    if (os_name or os.name) != "nt":
        return False
    if getattr(subprocess_module, "_autogame_hidden_popen_patch", False):
        return False

    original_popen = subprocess_module.Popen

    def hidden_popen(*args, **kwargs):
        startupinfo = kwargs.get("startupinfo")
        if startupinfo is None:
            startupinfo = subprocess_module.STARTUPINFO()
        startupinfo.dwFlags |= subprocess_module.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = (
            (kwargs.get("creationflags") or 0)
            | subprocess_module.CREATE_NO_WINDOW
        )
        return original_popen(*args, **kwargs)

    subprocess_module.Popen = hidden_popen
    subprocess_module._autogame_hidden_popen_patch = True
    return True
