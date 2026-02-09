"""High-level Trainer API for MacFleet distributed training.

Provides a simple interface for training PyTorch models across
multiple Macs with automatic distributed coordination.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from macfleet.comm.collectives import AllReduce, Broadcast, CollectiveGroup
from macfleet.comm.transport import TensorTransport
from macfleet.compression.pipeline import create_pipeline
from macfleet.core.config import (
    ClusterConfig,
    ClusterState,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)
from macfleet.training.data_parallel import MacFleetDDP
from macfleet.training.distributed_sampler import WeightedDistributedSampler


console = Console()


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    samples_per_sec: float = 0.0
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    compression_ratio: float = 1.0


@dataclass
class TrainerState:
    """State of the trainer for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_accuracy: float = 0.0


class Trainer:
    """High-level trainer for distributed training with MacFleet.

    Example:
        trainer = Trainer(
            model=model,
            dataset=train_dataset,
            training_config=TrainingConfig(epochs=10),
            cluster_config=ClusterConfig(role=NodeRole.MASTER),
        )
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        training_config: TrainingConfig,
        cluster_config: ClusterConfig,
        optimizer_cls: Type[Optimizer] = torch.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        criterion: Optional[nn.Module] = None,
        val_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[Callable]] = None,
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            dataset: Training dataset.
            training_config: Training configuration.
            cluster_config: Cluster configuration.
            optimizer_cls: Optimizer class.
            optimizer_kwargs: Optimizer keyword arguments.
            criterion: Loss function (default: CrossEntropyLoss).
            val_dataset: Optional validation dataset.
            callbacks: Optional list of callback functions.
        """
        self.model = model
        self.dataset = dataset
        self.training_config = training_config
        self.cluster_config = cluster_config
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": training_config.learning_rate}
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.val_dataset = val_dataset
        self.callbacks = callbacks or []

        # State
        self.state = TrainerState()
        self._metrics = TrainingMetrics()

        # Distributed components (initialized in setup)
        self._transport: Optional[TensorTransport] = None
        self._collective_group: Optional[CollectiveGroup] = None
        self._ddp_model: Optional[MacFleetDDP] = None
        self._optimizer: Optional[Optimizer] = None
        self._dataloader: Optional[DataLoader] = None
        self._sampler: Optional[WeightedDistributedSampler] = None

        # Device
        self._device = self._get_device()

    def _get_device(self) -> str:
        """Get the appropriate device."""
        if self.training_config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif self.training_config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    async def setup(self) -> None:
        """Set up distributed training components."""
        console.print("[bold blue]Setting up distributed training...[/bold blue]")

        # Move model to device
        self.model = self.model.to(self._device)
        console.print(f"  Model moved to {self._device}")

        # Set up transport
        self._transport = TensorTransport(
            host="0.0.0.0",
            port=self.cluster_config.tensor_port,
        )
        await self._transport.start_server()

        # For single-node training (testing/development)
        if self.cluster_config.role == NodeRole.MASTER:
            rank = 0
            world_size = 1  # Will be updated when workers join
            weights = [1.0]
        else:
            # Worker - would connect to master here
            rank = 1
            world_size = 2
            weights = [0.62, 0.38]

        # Create collective group
        self._collective_group = CollectiveGroup(
            rank=rank,
            world_size=world_size,
            transport=self._transport,
        )
        self._collective_group.set_device(self._device)

        # Create compression pipeline
        compression = create_pipeline(
            self.training_config.compression.value,
            self.training_config.topk_ratio,
        )

        # Wrap model in DDP
        self._ddp_model = MacFleetDDP(
            module=self.model,
            collective_group=self._collective_group,
            compression_pipeline=compression,
        )

        # Create optimizer
        self._optimizer = self.optimizer_cls(
            self._ddp_model.parameters(),
            **self.optimizer_kwargs,
        )

        # Create sampler and dataloader
        self._sampler = WeightedDistributedSampler(
            dataset=self.dataset,
            num_replicas=world_size,
            rank=rank,
            weights=weights,
            shuffle=True,
        )

        # Compute per-node batch size
        weight = weights[rank] if rank < len(weights) else 1.0 / world_size
        local_batch_size = max(1, int(self.training_config.batch_size * weight))

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=local_batch_size,
            sampler=self._sampler,
            num_workers=0,  # MPS doesn't work well with multiprocessing
            pin_memory=False,
        )

        console.print(f"  Rank: {rank}, World size: {world_size}")
        console.print(f"  Local batch size: {local_batch_size}")
        console.print(f"  Samples per epoch: {len(self._sampler)}")
        console.print("[bold green]Setup complete![/bold green]")

    async def teardown(self) -> None:
        """Clean up distributed components."""
        if self._transport:
            await self._transport.stop_server()

    def fit(self) -> TrainerState:
        """Train the model.

        Returns:
            Final trainer state.
        """
        return asyncio.get_event_loop().run_until_complete(self._fit_async())

    async def _fit_async(self) -> TrainerState:
        """Async implementation of fit."""
        await self.setup()

        try:
            console.print("\n[bold blue]Starting training...[/bold blue]")

            for epoch in range(self.training_config.epochs):
                self.state.epoch = epoch
                self._sampler.set_epoch(epoch)

                # Train one epoch
                epoch_metrics = await self._train_epoch(epoch)

                # Print epoch summary
                self._print_epoch_summary(epoch, epoch_metrics)

                # Validation
                if self.val_dataset is not None:
                    val_metrics = await self._validate()
                    console.print(f"  Validation loss: {val_metrics['loss']:.4f}")

                # Checkpoint
                if (epoch + 1) % self.training_config.checkpoint_every == 0:
                    self._save_checkpoint(epoch)

            console.print("\n[bold green]Training complete![/bold green]")

        finally:
            await self.teardown()

        return self.state

    async def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self._ddp_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_compute_time = 0.0
        total_comm_time = 0.0

        num_batches = len(self._dataloader)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch + 1}", total=num_batches)

            for batch_idx, (inputs, targets) in enumerate(self._dataloader):
                # Move to device
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # Forward pass (timed)
                compute_start = time.perf_counter()

                self._ddp_model.zero_grad()
                outputs = self._ddp_model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Sync MPS
                if self._device == "mps":
                    torch.mps.synchronize()

                compute_time = (time.perf_counter() - compute_start) * 1000

                # Gradient sync (timed)
                comm_start = time.perf_counter()
                await self._ddp_model.sync_gradients()
                comm_time = (time.perf_counter() - comm_start) * 1000

                # Optimizer step
                self._optimizer.step()

                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
                total_compute_time += compute_time
                total_comm_time += comm_time

                self.state.global_step += 1

                progress.update(task, advance=1)

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        samples_per_sec = total_samples / (total_compute_time / 1000) if total_compute_time > 0 else 0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples_per_sec": samples_per_sec,
            "compute_time_ms": total_compute_time,
            "comm_time_ms": total_comm_time,
        }

    async def _validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataset is None:
            return {}

        self._ddp_model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs = self._ddp_model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        }

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print epoch summary."""
        table = Table(title=f"Epoch {epoch + 1} Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Loss", f"{metrics['loss']:.4f}")
        table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
        table.add_row("Throughput", f"{metrics['samples_per_sec']:.1f} samples/s")
        table.add_row("Compute Time", f"{metrics['compute_time_ms']:.0f} ms")
        table.add_row("Comm Time", f"{metrics['comm_time_ms']:.0f} ms")

        comm_overhead = metrics['comm_time_ms'] / (metrics['compute_time_ms'] + metrics['comm_time_ms']) * 100
        table.add_row("Comm Overhead", f"{comm_overhead:.1f}%")

        console.print(table)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save a training checkpoint."""
        checkpoint_dir = self.training_config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self._ddp_model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "training_config": self.training_config.to_dict(),
        }

        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, path)
        console.print(f"[green]Checkpoint saved: {path}[/green]")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self._device)

        self._ddp_model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state.epoch = checkpoint["epoch"]
        self.state.global_step = checkpoint["global_step"]

        console.print(f"[green]Checkpoint loaded: {path}[/green]")
