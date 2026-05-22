# PUBG Perception

This package keeps lightweight local perception glue inside `auto_game`.
YOLO still runs locally from model files. SAM3 is treated as a remote HTTP
algorithm service and is called by uploading the current frame to `/segment`.

Default model locations:

- `models/yolo/26x_det.pt`
- `models/yolo/26x_cls.pt`

Environment overrides:

- `AUTOGAME_PUBG_YOLO_DETECT_MODEL`
- `AUTOGAME_PUBG_YOLO_CLASSIFY_MODEL`
- `AUTOGAME_PUBG_SAM3_HTTP_URL`
- `AUTOGAME_PUBG_SAM3_HTTP_TIMEOUT`

SAM3 segmentation results return `__visualizations__` metadata. The common
`visualizer_process` draws these masks and boxes on the current full frame,
the same way special-area detection results are shown.

Special area handler names:

- `pubg_yolo_detect`
- `pubg_yolo_detect_detail`
- `pubg_yolo_classify`
- `pubg_yolo_detect_and_classify`
- `pubg_yolo_reset_tracker`
- `pubg_sam3_segment_house`
- `pubg_sam3_segment_door`
- `pubg_sam3_segment_door_all`
