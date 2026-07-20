"""Auto_PUBG_ALL 的运行时感知分组覆盖。

info.py 由标注工具自动生成；这里单独保存按需模型的运行策略，避免重新导出
场景后让高耗时的 SAM3 又落回搜房常态感知。
"""

SAM3_GROUP_ITEM = {"scene": "sam3", "type": "special_area", "name": "sam3"}

STAGE_GROUP_OVERRIDES = {
    "搜房阶段": {
        "initial_group": "搜房",
        "groups": {
            "搜房": {"exclude_items": [SAM3_GROUP_ITEM]},
            "sam3": {"items": [SAM3_GROUP_ITEM]},
        },
    },
}
