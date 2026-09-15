# 目标检测权重

| 场景 | 文件 | 处理入口 |
| --- | --- | --- |
| 搜房阶段 | `house_search.pt`（原 `best_yolo26_1189.pt`） | `house_forward_scene` |
| 开车、跑图阶段 | `driving.pt`（Windows 原 `best.pt`） | `forward_scene` |

两份权重都放在本目录，分别按需加载并缓存。Windows 部署时复制
`house_search.pt` 到本目录，并将原有 `best.pt` 重命名为 `driving.pt`；同时更新代码并重启用例。
开车权重仅改文件名，内容不变；不要用搜房权重替换它。
权重文件由外部部署，Git 不包含 `.pt`。发布包会检查这两份权重。

新权重 SHA-256：`0d56c6deba076bf1415296d16b312afeeb6eae267ba78a73e06a1e4c5db59138`。
搜房模型原始类别为 house=0、door=1、open_door=2、window=3、car=4。
入口校验类别后转换为业务统一编号 house=8、door=0、open_door=4、window=2、car=7，
因此搜房、进出门和找车控制器可继续使用原有编号。
搜房模型仅识别上述五类，不输出旧模型的楼梯、石墙等类别。

阶段配置仍以 `forward_scene` 为结果名；搜房各分辨率通过
`handler_name: house_forward_scene` 选择新模型。重新导出 `info.py` 时应保留该设置。
