"""Standalone-capable SAM3 remote client/server helpers."""

from . import config

__all__ = [
    "MaskGeometry",
    "DoorFrameGeometry",
    "Sam3Segmenter",
    "config",
    "SegmentationResult",
    "extract_door_frame_geometries",
    "extract_door_frame_geometry",
    "extract_mask_geometry",
    "fit_top_profile_line",
]


def __getattr__(name: str):
    if name in {
        "DoorFrameGeometry",
        "extract_door_frame_geometries",
        "MaskGeometry",
        "extract_door_frame_geometry",
        "extract_mask_geometry",
        "fit_top_profile_line",
    }:
        from . import geometry

        return getattr(geometry, name)
    if name in {"Sam3Segmenter", "SegmentationResult"}:
        from . import segmenter

        return getattr(segmenter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
