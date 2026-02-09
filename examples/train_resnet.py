#!/usr/bin/env python3
"""Train ResNet-50 on CIFAR-10 with MacFleet distributed training.

This example demonstrates how to use MacFleet for distributed training
across multiple Apple Silicon Macs connected via Thunderbolt.

Usage:
    # Single node (for testing):
    python examples/train_resnet.py

    # Distributed (on master Mac):
    python examples/train_resnet.py --distributed --role master

    # Distributed (on worker Mac):
    python examples/train_resnet.py --distributed --role worker --master 10.0.0.1
"""

import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from macfleet import ClusterConfig, NodeRole, TrainingConfig
from macfleet.training.trainer import Trainer


def get_cifar10_datasets(data_dir: str = "./data"):
    """Load CIFAR-10 train and test datasets."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return train_dataset, test_dataset


def get_resnet50_for_cifar(num_classes: int = 10):
    """Get ResNet-50 adapted for CIFAR-10 (32x32 images)."""
    model = torchvision.models.resnet50(weights=None)

    # Modify first conv for smaller input (CIFAR is 32x32, not 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images

    # Modify final FC for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on CIFAR-10 with MacFleet")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Total batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--role", choices=["master", "worker"], default="master", help="Node role")
    parser.add_argument("--master", type=str, default="10.0.0.1", help="Master IP address")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--compression", choices=["none", "topk", "fp16", "topk_fp16"],
                       default="topk_fp16", help="Gradient compression")
    parser.add_argument("--topk-ratio", type=float, default=0.1, help="Top-K ratio")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    print("=" * 60)
    print("MacFleet ResNet-50 Training on CIFAR-10")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Distributed: {args.distributed}")
    if args.distributed:
        print(f"Role: {args.role}")
        print(f"Master: {args.master}:{args.port}")
    print(f"Compression: {args.compression}")
    print("=" * 60)

    # Load datasets
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, test_dataset = get_cifar10_datasets(args.data_dir)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("\nCreating ResNet-50 model...")
    model = get_resnet50_for_cifar(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training config
    from macfleet.core.config import CompressionType
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        compression=CompressionType(args.compression),
        topk_ratio=args.topk_ratio,
        checkpoint_every=2,
        checkpoint_dir=args.checkpoint_dir,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

    # Cluster config
    cluster_config = ClusterConfig(
        role=NodeRole.MASTER if args.role == "master" else NodeRole.WORKER,
        master_addr=args.master,
        master_port=args.port,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        training_config=training_config,
        cluster_config=cluster_config,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={
            "lr": args.lr,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        val_dataset=test_dataset,
    )

    # Train!
    print("\nStarting training...")
    state = trainer.fit()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final epoch: {state.epoch + 1}")
    print(f"Total steps: {state.global_step}")
    print("=" * 60)


if __name__ == "__main__":
    main()
