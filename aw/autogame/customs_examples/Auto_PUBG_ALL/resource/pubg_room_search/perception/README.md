# Embedded PUBG Perception

This package embeds the perception pieces used by PUBG room search inside
`auto_game`. It is loaded from `SpecialSceneHandler.py` and does not start
`pubg_test`, `gametest_proxy.main`, YOLO server, or SAM3 server.

Default model locations:

- `models/yolo/26x_det.pt`
- `models/yolo/26x_cls.pt`
- `models/sam3/sam3.pt`

Environment overrides:

- `AUTOGAME_PUBG_YOLO_DETECT_MODEL`
- `AUTOGAME_PUBG_YOLO_CLASSIFY_MODEL`
- `AUTOGAME_PUBG_SAM3_CHECKPOINT`
- `AUTOGAME_PUBG_SAM3_BPE`
- `AUTOGAME_PUBG_SAM3_DEVICE`
- `AUTOGAME_PUBG_SAM3_LOAD_FROM_HF=1`

Special area handler names:

- `pubg_yolo_detect`
- `pubg_yolo_detect_detail`
- `pubg_yolo_classify`
- `pubg_yolo_detect_and_classify`
- `pubg_yolo_reset_tracker`
- `pubg_sam3_segment_house`
- `pubg_sam3_segment_door`
- `pubg_sam3_segment_door_all`

