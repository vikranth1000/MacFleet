"""MacFleet: Distributed ML training across Apple Silicon Macs over Thunderbolt."""

import logging

__version__ = "0.2.0"

# Configure a NullHandler so library users can control logging.
# Applications (CLI, scripts) should call logging.basicConfig() to see output.
logging.getLogger(__name__).addHandler(logging.NullHandler())

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
