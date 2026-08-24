#!/usr/bin/env python3
"""Game_Recording 统一入口：绑定一次后在同一窗口录制或回放。"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> int:
    from aw.autogame.customs_examples.Game_Recording.resource.main_app import run
    from aw.autogame.customs_game_examples.Game_Recording.start_record import (
        main as start_record_main,
    )

    return int(start_record_main(argv, runner=run))


if __name__ == "__main__":
    raise SystemExit(main())
