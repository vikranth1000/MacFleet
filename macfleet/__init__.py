"""MacFleet: Distributed ML training across Apple Silicon Macs over Thunderbolt."""

__version__ = "0.1.0"

from macfleet.core.config import (
    ClusterConfig,
    ClusterState,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)
from macfleet.core.coordinator import Coordinator
from macfleet.core.worker import Worker
from macfleet.training.trainer import Trainer

__all__ = [
    "__version__",
    "ClusterConfig",
    "ClusterState",
    "NodeConfig",
    "NodeRole",
    "TrainingConfig",
    "Coordinator",
    "Worker",
    "Trainer",
]
