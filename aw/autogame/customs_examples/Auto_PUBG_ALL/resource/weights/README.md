# 目标检测权重

| 场景 | 文件 | 处理入口 |
| --- | --- | --- |
| 搜房阶段 | `house_search.pt` | `house_forward_scene` |
| 开车、跑图阶段 | `driving.pt` | `forward_scene` |

两份权重现均为 2026-09-15 的 1752 张数据六类别训练 best，内容相同，分别按需加载并缓存。
Windows 部署时备份旧文件，再将这两份 `.pt` 复制到本目录，同时更新代码并重启用例。
权重文件由外部部署，Git 不包含 `.pt`。发布包会检查这两份权重。

新权重 SHA-256：`5b8435c37e69886855fecb08aaa8be842b88798b1c28de8ebd62d7db7beed25f`。
模型原始类别为 house=0、door=1、open_door=2、window=3、car=4、wrecked_car=5。
两个入口均校验类别后转换为业务统一编号 house=8、door=0、open_door=4、window=2、car=7、wrecked_car=17，
因此搜房、进出门和找车控制器可继续使用原有编号。
两个模型仅识别上述六类，不输出旧模型的楼梯、石墙、岩石、水面等类别；开车视觉避障范围也随之缩小。

阶段配置仍以 `forward_scene` 为结果名；搜房各分辨率通过
`handler_name: house_forward_scene` 选择新模型。重新导出 `info.py` 时应保留该设置。
