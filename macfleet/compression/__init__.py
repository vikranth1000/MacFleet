"""Gradient compression utilities for MacFleet.

Ported from v1 with proven TopK + FP16 pipeline (~20x compression).
"""

from macfleet.compression.pipeline import (
    CompressionPipeline,
    CompressedGradient,
    Compressor,
    FP16Stage,
    TopKStage,
    create_pipeline,
)
from macfleet.compression.quantize import FP16Quantizer, Int8Quantizer
from macfleet.compression.topk import TopKCompressor
from macfleet.compression.adaptive import (
    AdaptiveCompressor,
    AdaptiveCompressionConfig,
    CompressedArray,
    CompressionLevel,
    NumpyTopKCompressor,
    NumpyFP16Compressor,
)

__all__ = [
    # Torch-based pipeline (v1 compat)
    "CompressionPipeline",
    "CompressedGradient",
    "Compressor",
    "FP16Stage",
    "TopKStage",
    "create_pipeline",
    "TopKCompressor",
    "FP16Quantizer",
    "Int8Quantizer",
    # Numpy-native adaptive compression (v2)
    "AdaptiveCompressor",
    "AdaptiveCompressionConfig",
    "CompressedArray",
    "CompressionLevel",
    "NumpyTopKCompressor",
    "NumpyFP16Compressor",
]
