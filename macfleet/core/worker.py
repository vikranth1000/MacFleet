"""Worker node for MacFleet distributed training.

The worker runs on secondary nodes (e.g., MacBook Air) and handles:
- Registration with the coordinator
- Heartbeat reporting
- Training execution
- Gradient communication
"""

import asyncio
import subprocess
import time
from typing import Optional

from rich.console import Console

from macfleet.comm.discovery import discover_master
from macfleet.comm.grpc_service import ClusterControlClient
from macfleet.comm.transport import TensorTransport
from macfleet.core.config import (
    ClusterConfig,
    NodeConfig,
    NodeRole,
    ThermalState,
    TrainingConfig,
)
from macfleet.core.node import BaseNode


console = Console()


class Worker(BaseNode):
    """Worker node for the MacFleet cluster.

    Handles:
    - Connection and registration with coordinator
    - Periodic heartbeat sending
    - Tensor communication with coordinator
    - Local training execution
    """

    def __init__(
        self,
        cluster_config: ClusterConfig,
        node_config: Optional[NodeConfig] = None,
        auto_discover: bool = True,
    ):
        """Initialize the worker.

        Args:
            cluster_config: Cluster configuration (role should be WORKER).
            node_config: Optional pre-configured node info.
            auto_discover: Whether to auto-discover the master if not specified.
        """
        # Ensure role is worker
        if cluster_config.role != NodeRole.WORKER:
            cluster_config.role = NodeRole.WORKER

        super().__init__(cluster_config, node_config)

        self._auto_discover = auto_discover
        self._grpc_client: Optional[ClusterControlClient] = None
        self._master_tensor_addr: Optional[str] = None
        self._master_tensor_port: Optional[int] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._current_throughput = 0.0
        self._current_step = 0

    @property
    def master_address(self) -> str:
        """Get the master's gRPC address."""
        return self._cluster_config.master_grpc_address

    async def start(self) -> None:
        """Start the worker and connect to the coordinator."""
        self._running = True
        self._setup_signal_handlers()

        console.print(f"[bold blue]Starting MacFleet Worker[/bold blue]")
        console.print(f"  Hostname: {self.hostname}")
        console.print(f"  IP Address: {self.ip_address}")
        console.print(f"  Tensor Port: {self._node_config.tensor_port}")

        # Try to auto-discover master if enabled
        if self._auto_discover and not self._cluster_config.master_addr:
            console.print("  [yellow]Searching for master node...[/yellow]")
            master = discover_master(timeout=10.0)
            if master:
                self._cluster_config.master_addr = master.ip_address
                self._cluster_config.master_port = master.grpc_port
                console.print(
                    f"  [green]Found master: {master.hostname} "
                    f"({master.ip_address}:{master.grpc_port})[/green]"
                )
            else:
                console.print("  [red]No master found via discovery[/red]")

        console.print(f"  Master: {self.master_address}")

        # Set up tensor transport
        await self._setup_transport()
        await self._transport.start_server()
        console.print(f"  [green]Tensor transport started[/green]")

        # Set up service discovery
        await self._setup_discovery()
        if self._service_registry:
            console.print(f"  [green]Service discovery started[/green]")

        # Connect to coordinator
        await self._connect_to_coordinator()

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        console.print(f"\n[bold green]Worker ready (rank={self.rank})[/bold green]")

    async def stop(self) -> None:
        """Stop the worker and disconnect from coordinator."""
        console.print("\n[bold yellow]Shutting down worker...[/bold yellow]")

        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Disconnect from coordinator
        if self._grpc_client:
            self._grpc_client.disconnect()
            self._grpc_client = None
            console.print("  [yellow]Disconnected from coordinator[/yellow]")

        # Tear down transport
        await self._teardown_transport()
        console.print("  [yellow]Tensor transport stopped[/yellow]")

        # Tear down discovery
        await self._teardown_discovery()
        console.print("  [yellow]Service discovery stopped[/yellow]")

        console.print("[bold green]Worker shutdown complete[/bold green]")

    async def run(self) -> None:
        """Main run loop for the worker.

        Waits for commands from the coordinator.
        """
        await self.start()

        try:
            # Main loop - wait for shutdown or commands
            while self._running:
                await asyncio.sleep(1.0)

        finally:
            await self.stop()

    async def _connect_to_coordinator(self) -> None:
        """Connect to the coordinator and register."""
        console.print("  Connecting to coordinator...")

        self._grpc_client = ClusterControlClient(
            master_addr=self._cluster_config.master_addr,
            master_port=self._cluster_config.master_port,
        )

        # Try to connect with retries
        max_retries = 10
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                self._grpc_client.connect()

                # Register with coordinator
                (
                    rank,
                    weight,
                    world_size,
                    master_tensor_addr,
                    master_tensor_port,
                ) = self._grpc_client.register(
                    hostname=self._node_config.hostname,
                    ip_address=self._node_config.ip_address,
                    gpu_cores=self._node_config.gpu_cores,
                    ram_gb=self._node_config.ram_gb,
                    memory_bandwidth_gbps=self._node_config.memory_bandwidth_gbps,
                    tensor_port=self._node_config.tensor_port,
                )

                # Update node config
                self._node_config.rank = rank
                self._node_config.workload_weight = weight
                self._master_tensor_addr = master_tensor_addr
                self._master_tensor_port = master_tensor_port

                console.print(f"  [green]Registered with coordinator[/green]")
                console.print(f"    Assigned rank: {rank}")
                console.print(f"    Workload weight: {weight:.1%}")
                console.print(f"    World size: {world_size}")

                # Connect to master's tensor channel
                await self._transport.connect(
                    master_tensor_addr,
                    master_tensor_port,
                )
                console.print(f"  [green]Connected to master tensor channel[/green]")

                return

            except Exception as e:
                if attempt < max_retries - 1:
                    console.print(
                        f"  [yellow]Connection attempt {attempt + 1} failed: {e}[/yellow]"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect to coordinator after {max_retries} attempts"
                    )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the coordinator."""
        interval = self._cluster_config.heartbeat_interval_sec

        while self._running:
            try:
                thermal = self._get_thermal_state()
                acknowledged, new_weight, should_stop = self._grpc_client.heartbeat(
                    rank=self.rank,
                    throughput=self._current_throughput,
                    thermal_state=thermal,
                    current_step=self._current_step,
                )

                # Update weight if changed
                if new_weight != self._node_config.workload_weight:
                    old_weight = self._node_config.workload_weight
                    self._node_config.workload_weight = new_weight
                    console.print(
                        f"[yellow]Workload weight changed: "
                        f"{old_weight:.1%} -> {new_weight:.1%}[/yellow]"
                    )

                # Check for stop signal
                if should_stop:
                    console.print("[yellow]Received stop signal from coordinator[/yellow]")
                    self._running = False
                    break

            except Exception as e:
                console.print(f"[red]Heartbeat failed: {e}[/red]")
                # Try to reconnect
                try:
                    await self._connect_to_coordinator()
                except Exception:
                    pass

            await asyncio.sleep(interval)

    def _get_thermal_state(self) -> str:
        """Get the current thermal state from macOS.

        Returns one of: "nominal", "fair", "serious", "critical"
        """
        try:
            # Try to get thermal state from powermetrics (requires sudo)
            # For now, return nominal as fallback
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                output = result.stdout.lower()
                if "critical" in output:
                    return "critical"
                elif "serious" in output:
                    return "serious"
                elif "fair" in output:
                    return "fair"

            return "nominal"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "nominal"

    def update_throughput(self, samples_per_sec: float) -> None:
        """Update the current throughput measurement.

        Args:
            samples_per_sec: Current training throughput.
        """
        self._current_throughput = samples_per_sec

    def update_step(self, step: int) -> None:
        """Update the current training step.

        Args:
            step: Current step number.
        """
        self._current_step = step

    async def send_tensor_to_master(
        self,
        tensor,
        msg_type=None,
    ) -> None:
        """Send a tensor to the master node.

        Args:
            tensor: Tensor to send.
            msg_type: Optional message type.
        """
        from macfleet.utils.tensor_utils import MessageType

        if msg_type is None:
            msg_type = MessageType.TENSOR_GRADIENT

        conn_key = f"{self._master_tensor_addr}:{self._master_tensor_port}"
        await self._transport.send_tensor(tensor, conn_key, msg_type)

    async def recv_tensor_from_master(
        self,
        device: Optional[str] = None,
    ):
        """Receive a tensor from the master node.

        Args:
            device: Target device for the tensor.

        Returns:
            Tuple of (tensor, message_type).
        """
        conn_key = f"{self._master_tensor_addr}:{self._master_tensor_port}"
        return await self._transport.recv_tensor(conn_key, device)

    async def sync_barrier(self, barrier_id: str) -> bool:
        """Wait at a synchronization barrier.

        Args:
            barrier_id: Unique identifier for this barrier.

        Returns:
            True when all nodes have reached the barrier.
        """
        if not self._grpc_client:
            raise RuntimeError("Not connected to coordinator")

        # Poll until barrier is released
        while True:
            proceed, nodes = self._grpc_client.sync_barrier(
                rank=self.rank,
                step=self._current_step,
                barrier_id=barrier_id,
            )

            if proceed:
                return True

            await asyncio.sleep(0.01)  # Small delay between polls


async def run_worker(
    cluster_config: Optional[ClusterConfig] = None,
    master_addr: Optional[str] = None,
    master_port: int = 50051,
) -> None:
    """Run a worker node.

    Convenience function for starting a worker.

    Args:
        cluster_config: Optional cluster configuration.
        master_addr: Master node address (overrides config).
        master_port: Master node port (overrides config).
    """
    if cluster_config is None:
        cluster_config = ClusterConfig(
            role=NodeRole.WORKER,
            master_addr=master_addr or "10.0.0.1",
            master_port=master_port,
        )

    if master_addr:
        cluster_config.master_addr = master_addr
    if master_port:
        cluster_config.master_port = master_port

    worker = Worker(cluster_config=cluster_config)
    await worker.run()
