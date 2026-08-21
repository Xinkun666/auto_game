# -*- coding: utf-8 -*-
"""Game_Recording 的空白标注工程。

用标注工具加载本目录后：
1. 在默认的“录制阶段”中新建一个场景；
2. 把键盘键位标为“控点”，名称直接写 w、a、s、d、space、f 等；
3. 导出回本工程。

录制由 start_record 窗口中的按钮开关，q 和 e 也可以作为普通游戏控点使用。
"""

PROJECT_NAME = "Game_Recording"
STAGE_DICT = {
    "录制阶段": True,
}
STAGE_INFO = {
    "录制阶段": {
        "groups": {"默认": {"all": True}},
        "scenes": {},
    },
}
SCENE_POOL = {
    "groups": {"未分组场景": []},
    "scenes": {},
}

# start_record 的按键绑定窗口会自动读写此字段。
KEY_BINDINGS = {}
