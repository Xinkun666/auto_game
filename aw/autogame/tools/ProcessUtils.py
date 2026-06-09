import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

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


def install_hidden_subprocess_patch(
    os_name: Optional[str] = None,
    subprocess_module=subprocess,
) -> bool:
    LOGGER.info("global subprocess patch disabled")
    return False
