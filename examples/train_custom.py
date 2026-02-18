#!/usr/bin/env python3
"""Template for training any custom model with MacFleet.

This is the recommended starting point for new projects. Replace the
MyModel and MyDataset classes with your own, then launch with:

    # Single node (testing):
    python examples/train_custom.py

    # Distributed (master Mac):
    python examples/train_custom.py --distributed --role master --host 10.0.0.1

    # Distributed (worker Mac):
    python examples/train_custom.py --distributed --role worker --master 10.0.0.1 --host 10.0.0.2

    # With YAML config (recommended):
    python examples/train_custom.py --config macfleet.yaml --distributed --role master

Or use the CLI directly after macfleet init:
    macfleet launch --config macfleet.yaml --role master
    macfleet launch --config macfleet.yaml --role worker
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from macfleet import ClusterConfig, NodeRole, TrainingConfig
from macfleet.training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Replace MyModel with your model
# ─────────────────────────────────────────────────────────────────────────────

class MyModel(nn.Module):
    """Example model — replace with your own architecture.

    Any nn.Module works: ResNet, Transformer, custom architectures, etc.
    MacFleet wraps it in a DDP layer automatically when distributed=True.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Replace MyDataset with your dataset
# ─────────────────────────────────────────────────────────────────────────────

class MyDataset(Dataset):
    """Example dataset — replace with your own data loading logic.

    Any PyTorch Dataset works. MacFleet's WeightedDistributedSampler
    splits batches across nodes proportionally to their GPU capacity.
    """

    def __init__(self, num_samples: int = 10000, num_classes: int = 10):
        # Replace with real data loading, e.g.:
        #   self.data = torch.load("my_data.pt")
        #   self.labels = torch.load("my_labels.pt")
        self.data = torch.randn(num_samples, 784)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 (optional): Custom loss function
# ─────────────────────────────────────────────────────────────────────────────

def get_criterion() -> nn.Module:
    """Return the loss function. Replace with your own if needed."""
    return nn.CrossEntropyLoss()


# ─────────────────────────────────────────────────────────────────────────────
# Main training script
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a custom model with MacFleet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (recommended)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to macfleet.yaml (use 'macfleet init' to generate one)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--compression",
        choices=["none", "topk", "fp16", "topk_fp16"],
        default="topk_fp16",
        help="Gradient compression (topk_fp16 recommended for Thunderbolt)",
    )

    # Cluster setup
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--role", choices=["master", "worker"], default="master")
    parser.add_argument("--master", type=str, default=None, help="Master node IP (e.g., 10.0.0.1)")
    parser.add_argument("--host", type=str, default=None, help="This node's IP (e.g., Thunderbolt bridge IP)")
    parser.add_argument("--port", type=int, default=50051)

    # Data and checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5)

    args = parser.parse_args()

    # ── Build configs ─────────────────────────────────────────────────────────
    if args.config:
        from macfleet.cli.config_loader import cluster_config_from_yaml, training_config_from_yaml
        cluster_config = cluster_config_from_yaml(args.config, role_override=args.role)
        training_config = training_config_from_yaml(args.config)
        # CLI flags override YAML
        if args.master:
            cluster_config.master_addr = args.master
        if args.host:
            cluster_config.host = args.host
    else:
        from macfleet.core.config import CompressionType
        cluster_config = ClusterConfig(
            role=NodeRole.MASTER if args.role == "master" else NodeRole.WORKER,
            master_addr=args.master or "",
            master_port=args.port,
            host=args.host,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            compression=CompressionType(args.compression),
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            device="mps" if torch.backends.mps.is_available() else "cpu",
        )

    # ── Create model and datasets ─────────────────────────────────────────────
    NUM_CLASSES = 10
    model = MyModel(num_classes=NUM_CLASSES)
    train_dataset = MyDataset(num_samples=10000, num_classes=NUM_CLASSES)
    val_dataset = MyDataset(num_samples=2000, num_classes=NUM_CLASSES)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model.__class__.__name__} ({num_params:,} parameters)")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Device: {training_config.device}")
    print(f"Distributed: {args.distributed}")
    if args.distributed:
        print(f"Role: {cluster_config.role.value}")
        print(f"Master: {cluster_config.master_addr}:{cluster_config.master_port}")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        training_config=training_config,
        cluster_config=cluster_config,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"lr": training_config.learning_rate, "weight_decay": 1e-4},
        criterion=get_criterion(),
        val_dataset=val_dataset,
        distributed=args.distributed,
    )

    state = trainer.fit()

    print(f"\nTraining complete!")
    print(f"Final epoch: {state.epoch + 1}/{training_config.epochs}")
    print(f"Total steps: {state.global_step}")
    if os.path.isdir(training_config.checkpoint_dir):
        print(f"Checkpoints saved to: {training_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
