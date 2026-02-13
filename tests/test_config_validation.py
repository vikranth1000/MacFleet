"""Tests for configuration validation."""

import pytest

from macfleet.core.config import (
    ClusterConfig,
    CompressionType,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)


class TestNodeConfigValidation:
    def test_valid_config(self):
        cfg = NodeConfig(
            hostname="test", ip_address="10.0.0.1",
            gpu_cores=10, ram_gb=16, memory_bandwidth_gbps=200.0,
        )
        assert cfg.hostname == "test"

    def test_empty_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname"):
            NodeConfig(
                hostname="", ip_address="10.0.0.1",
                gpu_cores=10, ram_gb=16, memory_bandwidth_gbps=200.0,
            )

    def test_invalid_tensor_port(self):
        with pytest.raises(ValueError, match="tensor_port"):
            NodeConfig(
                hostname="test", ip_address="10.0.0.1",
                gpu_cores=10, ram_gb=16, memory_bandwidth_gbps=200.0,
                tensor_port=0,
            )

    def test_from_dict_ignores_unknown_keys(self):
        cfg = NodeConfig.from_dict({
            "hostname": "test", "ip_address": "10.0.0.1",
            "gpu_cores": 10, "ram_gb": 16, "memory_bandwidth_gbps": 200.0,
            "unknown_key": "ignored",
        })
        assert cfg.hostname == "test"


class TestClusterConfigValidation:
    def test_valid_config(self):
        cfg = ClusterConfig(role=NodeRole.MASTER)
        assert cfg.master_port == 50051

    def test_invalid_port(self):
        with pytest.raises(ValueError, match="master_port"):
            ClusterConfig(role=NodeRole.MASTER, master_port=0)

    def test_heartbeat_timeout_must_exceed_interval(self):
        with pytest.raises(ValueError, match="heartbeat_timeout_sec"):
            ClusterConfig(
                role=NodeRole.MASTER,
                heartbeat_interval_sec=5.0,
                heartbeat_timeout_sec=3.0,
            )

    def test_min_workers_positive(self):
        with pytest.raises(ValueError, match="min_workers"):
            ClusterConfig(role=NodeRole.MASTER, min_workers=0)

    def test_from_dict_ignores_unknown_keys(self):
        cfg = ClusterConfig.from_dict({
            "role": "master",
            "future_field": True,
        })
        assert cfg.role == NodeRole.MASTER


class TestTrainingConfigValidation:
    def test_valid_config(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 10

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=-0.1)

    def test_invalid_topk_ratio(self):
        with pytest.raises(ValueError, match="topk_ratio"):
            TrainingConfig(topk_ratio=1.5)

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="device"):
            TrainingConfig(device="tpu")

    def test_from_dict_ignores_unknown_keys(self):
        cfg = TrainingConfig.from_dict({
            "compression": "none",
            "extra": 42,
        })
        assert cfg.compression == CompressionType.NONE
