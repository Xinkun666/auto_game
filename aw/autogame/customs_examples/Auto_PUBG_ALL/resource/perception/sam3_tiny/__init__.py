"""Auto_PUBG_ALL local EfficientSAM3 special-area integration."""

from .local_segmenter import get_sam3_segmenter, segment_sam3

__all__ = ["get_sam3_segmenter", "segment_sam3"]
