"""
Embeddings package for the semantic image search system.
"""
from semantic_img.embeddings.encoder_base import ImageEncoder
from semantic_img.embeddings.clip_encoder import ClipEncoder

__all__ = ["ImageEncoder", "ClipEncoder"] 