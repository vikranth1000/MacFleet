"""Pluggable training engines (PyTorch, MLX).

Both engines implement the same Engine protocol so the pool/comm layer
stays framework-agnostic. Gradients flow as numpy arrays.
"""

from macfleet.engines.base import (
    Engine,
    EngineCapabilities,
    EngineType,
    HardwareProfile,
    ThermalPressure,
    TrainingMetrics,
)
from macfleet.engines.torch_engine import TorchEngine

__all__ = [
    "Engine",
    "EngineCapabilities",
    "EngineType",
    "HardwareProfile",
    "ThermalPressure",
    "TrainingMetrics",
    "TorchEngine",
]

# MLX is optional — only available on Apple Silicon with mlx installed
try:
    from macfleet.engines.mlx_engine import MLXEngine

    __all__.append("MLXEngine")
except ImportError:
    pass
