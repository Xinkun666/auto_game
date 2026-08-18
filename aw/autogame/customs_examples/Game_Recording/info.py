# -*- coding: utf-8 -*-
"""Game_Recording 的空白标注工程。

用标注工具加载本目录后：
1. 新建一个阶段和一个场景；
2. 把键盘键位标为“控点”，名称直接写 w、a、s、d、space、f 等；
3. 导出回本工程。

q 和 e 是录制开关，不能作为游戏控点使用。
"""

PROJECT_NAME = "Game_Recording"
STAGE_DICT = {}
STAGE_INFO = {}
SCENE_POOL = {
    "groups": {"未分组场景": []},
    "scenes": {},
}
