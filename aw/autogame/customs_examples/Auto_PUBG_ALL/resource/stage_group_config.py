"""Auto_PUBG_ALL 的运行时初始分组配置。

分组内容完全使用标注工具导出的 info.py；这里只声明进入搜房阶段时默认启用
other，避免落入内置的“默认”全量组而常态执行 SAM3。
"""

STAGE_GROUP_OVERRIDES = {
    "搜房阶段": {
        "initial_group": "other",
    },
}
