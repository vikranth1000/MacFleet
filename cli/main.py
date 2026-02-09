"""Command-line interface for MacFleet.

Provides the `macfleet` CLI with commands for:
- launch: Start a coordinator or worker node
- status: Check cluster status
- benchmark: Run performance benchmarks
"""

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from macfleet import __version__
from macfleet.core.config import ClusterConfig, NodeRole, TrainingConfig


console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="macfleet")
def cli():
    """MacFleet: Distributed ML training across Apple Silicon Macs.

    Use 'macfleet launch' to start a node, 'macfleet status' to check
    cluster status, and 'macfleet benchmark' to run performance tests.
    """
    pass


@cli.command()
@click.option(
    "--role",
    type=click.Choice(["master", "worker"]),
    required=True,
    help="Role of this node in the cluster.",
)
@click.option(
    "--port",
    type=int,
    default=50051,
    help="gRPC port for control messages (default: 50051).",
)
@click.option(
    "--tensor-port",
    type=int,
    default=50052,
    help="Port for tensor transfers (default: 50052).",
)
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address for workers (e.g., 10.0.0.1 or 10.0.0.1:50051).",
)
@click.option(
    "--no-discovery",
    is_flag=True,
    help="Disable Bonjour/zeroconf discovery.",
)
def launch(
    role: str,
    port: int,
    tensor_port: int,
    master: str,
    no_discovery: bool,
):
    """Launch a MacFleet node (coordinator or worker).

    Examples:

        # On MacBook Pro (master):
        macfleet launch --role master --port 50051

        # On MacBook Air (worker):
        macfleet launch --role worker --master 10.0.0.1

        # Worker with specific ports:
        macfleet launch --role worker --master 10.0.0.1:50051 --tensor-port 50053
    """
    # Parse master address
    master_addr = "10.0.0.1"
    master_port = port

    if master:
        if ":" in master:
            parts = master.split(":")
            master_addr = parts[0]
            master_port = int(parts[1])
        else:
            master_addr = master

    # Create cluster config
    cluster_config = ClusterConfig(
        role=NodeRole.MASTER if role == "master" else NodeRole.WORKER,
        master_addr=master_addr,
        master_port=master_port,
        tensor_port=tensor_port,
        discovery_enabled=not no_discovery,
    )

    # Print banner
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║[/bold blue]     [bold white]MacFleet[/bold white] - Distributed Training   [bold blue]║[/bold blue]")
    console.print("[bold blue]║[/bold blue]     [dim]Apple Silicon over Thunderbolt[/dim]    [bold blue]║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]")
    console.print()

    try:
        if role == "master":
            _run_coordinator(cluster_config)
        else:
            _run_worker(cluster_config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        sys.exit(1)


def _run_coordinator(config: ClusterConfig):
    """Run the coordinator node."""
    from macfleet.core.coordinator import Coordinator

    coordinator = Coordinator(cluster_config=config)
    asyncio.run(coordinator.run())


def _run_worker(config: ClusterConfig):
    """Run the worker node."""
    from macfleet.core.worker import Worker

    worker = Worker(cluster_config=config)
    asyncio.run(worker.run())


@cli.command()
@click.option(
    "--master",
    type=str,
    default="10.0.0.1:50051",
    help="Master address (default: 10.0.0.1:50051).",
)
def status(master: str):
    """Check the status of a MacFleet cluster.

    Connects to the coordinator and displays cluster information.

    Example:

        macfleet status --master 10.0.0.1:50051
    """
    from macfleet.comm.grpc_service import ClusterControlClient

    # Parse master address
    if ":" in master:
        parts = master.split(":")
        master_addr = parts[0]
        master_port = int(parts[1])
    else:
        master_addr = master
        master_port = 50051

    console.print(f"Connecting to {master_addr}:{master_port}...")

    try:
        client = ClusterControlClient(master_addr, master_port)
        client.connect()
        state = client.get_cluster_state()
        client.disconnect()

        # Display cluster state
        console.print()
        console.print(f"[bold green]Cluster Status[/bold green]")
        console.print(f"  World Size: {state['world_size']}")
        console.print(f"  Training: {state['training_status']}")

        if state['training_active']:
            console.print(f"  Epoch: {state['current_epoch']}")
            console.print(f"  Step: {state['current_step']}")

        console.print()

        # Node table
        table = Table(title="Nodes")
        table.add_column("Rank", style="cyan")
        table.add_column("Hostname", style="green")
        table.add_column("IP Address")
        table.add_column("GPU Cores", justify="right")
        table.add_column("RAM (GB)", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Status", style="yellow")

        for node in state['nodes']:
            table.add_row(
                str(node['rank']),
                node['hostname'],
                node['ip_address'],
                str(node['gpu_cores']),
                str(node['ram_gb']),
                f"{node['workload_weight']:.1%}",
                node['status'],
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("[yellow]Make sure the coordinator is running.[/yellow]")
        sys.exit(1)


@cli.command()
@click.option(
    "--type",
    "bench_type",
    type=click.Choice(["bandwidth", "allreduce", "latency"]),
    default="bandwidth",
    help="Type of benchmark to run.",
)
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address for distributed benchmarks.",
)
@click.option(
    "--sizes",
    type=str,
    default="1,10,50,100,500",
    help="Comma-separated tensor sizes in MB (default: 1,10,50,100,500).",
)
def benchmark(bench_type: str, master: str, sizes: str):
    """Run MacFleet performance benchmarks.

    Examples:

        # Test local bandwidth:
        macfleet benchmark --type bandwidth

        # Test bandwidth with specific sizes:
        macfleet benchmark --type bandwidth --sizes 10,50,100

        # Test AllReduce with master:
        macfleet benchmark --type allreduce --master 10.0.0.1
    """
    import torch

    console.print(f"[bold blue]Running {bench_type} benchmark...[/bold blue]")
    console.print()

    # Parse sizes
    size_list = [int(s) for s in sizes.split(",")]

    if bench_type == "bandwidth":
        _run_bandwidth_benchmark(size_list)
    elif bench_type == "latency":
        _run_latency_benchmark()
    elif bench_type == "allreduce":
        if not master:
            console.print("[red]AllReduce benchmark requires --master[/red]")
            sys.exit(1)
        _run_allreduce_benchmark(master)


def _run_bandwidth_benchmark(sizes_mb: list[int]):
    """Run local bandwidth benchmark."""
    import time
    import torch

    console.print("Testing tensor serialization bandwidth...")
    console.print()

    from macfleet.utils.tensor_utils import tensor_to_bytes, bytes_to_tensor

    table = Table(title="Bandwidth Results")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Serialize (ms)", justify="right")
    table.add_column("Deserialize (ms)", justify="right")
    table.add_column("Throughput (GB/s)", justify="right")

    for size_mb in sizes_mb:
        # Create tensor
        numel = (size_mb * 1024 * 1024) // 4  # FP32 = 4 bytes
        tensor = torch.randn(numel)

        # Benchmark serialization
        start = time.perf_counter()
        data = tensor_to_bytes(tensor)
        serialize_time = (time.perf_counter() - start) * 1000

        # Benchmark deserialization
        start = time.perf_counter()
        tensor2, _ = bytes_to_tensor(data)
        deserialize_time = (time.perf_counter() - start) * 1000

        # Calculate throughput
        total_time_sec = (serialize_time + deserialize_time) / 1000
        throughput_gbps = (size_mb * 2) / (total_time_sec * 1024) if total_time_sec > 0 else 0

        table.add_row(
            str(size_mb),
            f"{serialize_time:.2f}",
            f"{deserialize_time:.2f}",
            f"{throughput_gbps:.2f}",
        )

    console.print(table)


def _run_latency_benchmark():
    """Run loopback latency benchmark."""
    import time

    console.print("Testing loopback latency...")
    console.print()

    async def run():
        from macfleet.comm.transport import TensorTransport
        import torch

        # Start server
        server = TensorTransport("127.0.0.1", 50099)
        received = []

        async def on_recv(tensor, msg_type, addr):
            received.append(time.perf_counter())

        await server.start_server(on_recv)

        # Connect and send
        client = TensorTransport()
        conn_key = await client.connect("127.0.0.1", 50099)

        latencies = []
        for _ in range(100):
            tensor = torch.randn(1000)  # Small tensor
            start = time.perf_counter()
            await client.send_tensor(tensor, conn_key)
            await asyncio.sleep(0.01)  # Wait for receipt
            if received:
                latency = (received[-1] - start) * 1000
                latencies.append(latency)

        await client.disconnect(conn_key)
        await server.stop_server()

        if latencies:
            avg = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            console.print(f"  Average latency: {avg:.2f} ms")
            console.print(f"  Min latency: {min_lat:.2f} ms")
            console.print(f"  Max latency: {max_lat:.2f} ms")
        else:
            console.print("[yellow]No latency measurements collected[/yellow]")

    asyncio.run(run())


def _run_allreduce_benchmark(master: str):
    """Run AllReduce benchmark (requires cluster)."""
    console.print("[yellow]AllReduce benchmark requires a running cluster.[/yellow]")
    console.print("[yellow]This benchmark will be implemented in Phase 2.[/yellow]")


@cli.command()
def info():
    """Display system information for MacFleet.

    Shows GPU, memory, and network configuration.
    """
    from macfleet.utils.network import (
        get_gpu_info,
        get_hostname,
        get_local_ip,
        get_memory_bandwidth,
        get_memory_info,
        get_thunderbolt_bridge_ip,
    )

    console.print("[bold blue]System Information[/bold blue]")
    console.print()

    console.print(f"  Hostname: {get_hostname()}")
    console.print(f"  Local IP: {get_local_ip()}")

    tb_ip = get_thunderbolt_bridge_ip()
    if tb_ip:
        console.print(f"  Thunderbolt IP: [green]{tb_ip}[/green]")
    else:
        console.print(f"  Thunderbolt IP: [yellow]Not detected[/yellow]")

    console.print()

    gpu_info = get_gpu_info()
    console.print(f"  GPU: {gpu_info.get('gpu_name', 'Unknown')}")
    console.print(f"  GPU Cores: {gpu_info.get('gpu_cores', 0)}")

    memory_info = get_memory_info()
    console.print(f"  RAM: {memory_info.get('total_gb', 0)} GB")
    console.print(f"  Memory Bandwidth: ~{get_memory_bandwidth():.0f} GB/s")

    console.print()

    # Check PyTorch/MPS
    try:
        import torch
        console.print(f"  PyTorch: {torch.__version__}")
        console.print(f"  MPS Available: {torch.backends.mps.is_available()}")
        console.print(f"  MPS Built: {torch.backends.mps.is_built()}")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")


if __name__ == "__main__":
    cli()
