"""Command-line interface for MacFleet.

Provides the `macfleet` CLI with commands for:
- init: Scaffold a new project with config template
- launch: Start a coordinator or worker node
- status: Check cluster status
- diagnose: Check system and connectivity
- benchmark: Run performance benchmarks
- info: Show system information
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

    Quick start:

        macfleet init myproject       # create a new project
        macfleet diagnose             # check system & connectivity
        macfleet launch --role master # start coordinator
        macfleet launch --role worker # start worker
    """
    pass


# ── init ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("directory", default=".")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files.",
)
def init(directory: str, force: bool):
    """Scaffold a new MacFleet project.

    Creates a starter macfleet.yaml config and train.py template
    in DIRECTORY (defaults to current directory).

    Examples:

        macfleet init                  # scaffold in current directory
        macfleet init myproject        # scaffold in ./myproject/
    """
    import os

    from macfleet.cli.config_loader import _TRAIN_TEMPLATE, _YAML_TEMPLATE

    os.makedirs(directory, exist_ok=True)

    yaml_path = os.path.join(directory, "macfleet.yaml")
    train_path = os.path.join(directory, "train.py")

    created = []
    skipped = []

    for path, content in [(yaml_path, _YAML_TEMPLATE), (train_path, _TRAIN_TEMPLATE)]:
        if os.path.exists(path) and not force:
            skipped.append(path)
        else:
            with open(path, "w") as f:
                f.write(content)
            created.append(path)

    console.print()
    console.print("[bold green]MacFleet project initialized![/bold green]")
    console.print()

    if created:
        console.print("[bold]Created:[/bold]")
        for path in created:
            console.print(f"  [green]{path}[/green]")

    if skipped:
        console.print("[bold]Skipped (already exist):[/bold]")
        for path in skipped:
            console.print(f"  [yellow]{path}[/yellow] (use --force to overwrite)")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Edit [cyan]macfleet.yaml[/cyan] — set master_addr to your master's IP")
    console.print("  2. Edit [cyan]train.py[/cyan] — replace MyModel and MyDataset with yours")
    console.print("  3. Run [cyan]macfleet diagnose[/cyan] to verify connectivity")
    console.print("  4. On master:  [cyan]macfleet launch --config macfleet.yaml --role master[/cyan]")
    console.print("  5. On worker:  [cyan]macfleet launch --config macfleet.yaml --role worker[/cyan]")
    console.print()


# ── launch ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--role",
    type=click.Choice(["master", "worker"]),
    default=None,
    help="Role of this node (master or worker). Required unless --config provides it.",
)
@click.option(
    "--config",
    "config_path",
    type=str,
    default=None,
    help="Path to macfleet.yaml config file.",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="gRPC port (default: 50051). Overrides config file.",
)
@click.option(
    "--tensor-port",
    type=int,
    default=None,
    help="Tensor transfer port (default: 50052). Overrides config file.",
)
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address for workers (e.g., 10.0.0.1 or 10.0.0.1:50051). Overrides config file.",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="IP address to bind to (e.g., 169.254.83.200 for Thunderbolt bridge). Overrides config file.",
)
@click.option(
    "--no-discovery",
    is_flag=True,
    help="Disable Bonjour/zeroconf discovery.",
)
@click.option(
    "--min-workers",
    type=int,
    default=None,
    help="Minimum workers required before training starts. Overrides config file.",
)
def launch(
    role: str,
    config_path: str,
    port: int,
    tensor_port: int,
    master: str,
    host: str,
    no_discovery: bool,
    min_workers: int,
):
    """Launch a MacFleet node (coordinator or worker).

    Can be configured via a YAML file (--config) or command-line flags.
    CLI flags override values in the config file.

    Examples:

        # From config file:
        macfleet launch --config macfleet.yaml --role master
        macfleet launch --config macfleet.yaml --role worker

        # Manual (Thunderbolt IPs):
        macfleet launch --role master --host 10.0.0.1
        macfleet launch --role worker --master 10.0.0.1 --host 10.0.0.2

        # Auto-discovery (no IPs needed if mDNS works):
        macfleet launch --role master
        macfleet launch --role worker
    """
    # Load config: YAML file first, then apply CLI overrides
    if config_path:
        try:
            from macfleet.cli.config_loader import cluster_config_from_yaml
            cluster_config = cluster_config_from_yaml(config_path, role_override=role)
        except (FileNotFoundError, ImportError, ValueError) as e:
            console.print(f"[bold red]Config error: {e}[/bold red]")
            sys.exit(1)
    else:
        if role is None:
            console.print("[bold red]Error: --role is required (master or worker).[/bold red]")
            console.print("Or use --config to load from a YAML file.")
            sys.exit(1)

        cluster_config = ClusterConfig(
            role=NodeRole.MASTER if role == "master" else NodeRole.WORKER,
            master_addr="",
            master_port=port or 50051,
            tensor_port=tensor_port or 50052,
            discovery_enabled=not no_discovery,
            host=host,
        )

    # Apply CLI overrides on top of config
    if role is not None:
        cluster_config.role = NodeRole.MASTER if role == "master" else NodeRole.WORKER
    if no_discovery:
        cluster_config.discovery_enabled = False
    if host is not None:
        cluster_config.host = host
    if min_workers is not None:
        cluster_config.min_workers = min_workers
    if port is not None:
        cluster_config.master_port = port
    if tensor_port is not None:
        cluster_config.tensor_port = tensor_port

    # Parse --master flag (overrides config)
    if master:
        if ":" in master:
            parts = master.split(":", 1)
            cluster_config.master_addr = parts[0]
            try:
                cluster_config.master_port = int(parts[1])
            except ValueError:
                console.print(f"[bold red]Invalid port in --master '{master}'[/bold red]")
                sys.exit(1)
        else:
            cluster_config.master_addr = master

    # Validate that worker has a master address (via flag, config, or discovery)
    if cluster_config.role == NodeRole.WORKER and not cluster_config.master_addr:
        if not cluster_config.discovery_enabled:
            console.print(
                "[bold red]Worker requires a master address.[/bold red]\n"
                "Provide it with [cyan]--master 10.0.0.1[/cyan] or enable discovery."
            )
            sys.exit(1)
        # Discovery is enabled — worker will auto-find master

    # Print banner
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║[/bold blue]     [bold white]MacFleet[/bold white] - Distributed Training   [bold blue]║[/bold blue]")
    console.print("[bold blue]║[/bold blue]     [dim]Apple Silicon over Thunderbolt[/dim]    [bold blue]║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]")
    console.print()

    try:
        if cluster_config.role == NodeRole.MASTER:
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


# ── status ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address (e.g., 10.0.0.1 or 10.0.0.1:50051).",
)
@click.option(
    "--config",
    "config_path",
    type=str,
    default=None,
    help="Path to macfleet.yaml to read master address from.",
)
def status(master: str, config_path: str):
    """Check the status of a MacFleet cluster.

    Connects to the coordinator and displays cluster information.

    Examples:

        macfleet status --master 10.0.0.1
        macfleet status --config macfleet.yaml
    """
    from macfleet.comm.grpc_service import ClusterControlClient

    # Resolve master address
    master_addr = "127.0.0.1"
    master_port = 50051

    if config_path:
        try:
            from macfleet.cli.config_loader import cluster_config_from_yaml
            cfg = cluster_config_from_yaml(config_path)
            master_addr = cfg.master_addr or master_addr
            master_port = cfg.master_port
        except (FileNotFoundError, ImportError, ValueError) as e:
            console.print(f"[bold red]Config error: {e}[/bold red]")
            sys.exit(1)

    if master:
        if ":" in master:
            parts = master.split(":", 1)
            master_addr = parts[0]
            try:
                master_port = int(parts[1])
            except ValueError:
                console.print(f"[bold red]Invalid port in --master '{master}'[/bold red]")
                sys.exit(1)
        else:
            master_addr = master

    console.print(f"Connecting to {master_addr}:{master_port}...")

    try:
        client = ClusterControlClient(master_addr, master_port)
        client.connect()
        state = client.get_cluster_state()
        client.disconnect()

        # Display cluster state
        console.print()
        console.print("[bold green]Cluster Status[/bold green]")
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


# ── diagnose ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address to test connectivity (e.g., 10.0.0.1).",
)
@click.option(
    "--config",
    "config_path",
    type=str,
    default=None,
    help="Path to macfleet.yaml to read settings from.",
)
def diagnose(master: str, config_path: str):
    """Diagnose MacFleet system and connectivity.

    Checks:
    - PyTorch and MPS availability
    - Thunderbolt bridge detection
    - Network port availability
    - Master node reachability (if --master provided)

    Examples:

        macfleet diagnose                       # local system check
        macfleet diagnose --master 10.0.0.1     # also ping master
        macfleet diagnose --config macfleet.yaml
    """
    import socket

    from macfleet.core.config import DEFAULT_GRPC_PORT, DEFAULT_TENSOR_PORT
    from macfleet.utils.network import (
        get_thunderbolt_bridge_ip,
        get_local_ip,
        get_hostname,
        get_gpu_info,
        get_memory_info,
        is_port_available,
        is_reachable,
    )

    all_ok = True

    def check(label: str, ok: bool, detail: str = "", fix: str = ""):
        nonlocal all_ok
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        line = f"  {icon}  {label}"
        if detail:
            line += f"  [dim]{detail}[/dim]"
        console.print(line)
        if not ok and fix:
            console.print(f"      [yellow]→ {fix}[/yellow]")
        if not ok:
            all_ok = False

    console.print()
    console.print("[bold blue]MacFleet Diagnostics[/bold blue]")
    console.print()

    # ── System ───────────────────────────────────────────────────────────────
    console.print("[bold]System[/bold]")

    hostname = get_hostname()
    console.print(f"  [dim]Hostname: {hostname}[/dim]")

    # PyTorch
    try:
        import torch
        check("PyTorch installed", True, torch.__version__)
    except ImportError:
        check("PyTorch installed", False, fix="pip install torch")

    # MPS
    try:
        import torch
        mps_built = torch.backends.mps.is_built()
        mps_avail = torch.backends.mps.is_available()
        if mps_avail:
            check("MPS (Apple GPU) available", True)
        elif mps_built:
            check("MPS (Apple GPU) available", False,
                  detail="built but not available",
                  fix="Make sure you are on Apple Silicon macOS 12.3+")
        else:
            check("MPS (Apple GPU) available", False,
                  detail="not built — will use CPU",
                  fix="Install PyTorch with MPS support")
    except Exception:
        check("MPS (Apple GPU) available", False)

    # GPU info
    try:
        gpu_info = get_gpu_info()
        gpu_cores = gpu_info.get("gpu_cores", 0)
        gpu_name = gpu_info.get("gpu_name", "Unknown")
        check("GPU detected", gpu_cores > 0, f"{gpu_name} ({gpu_cores} cores)")
    except Exception:
        check("GPU detected", False)

    # Memory
    try:
        mem_info = get_memory_info()
        ram_gb = mem_info.get("total_gb", 0)
        check("Memory", ram_gb > 0, f"{ram_gb} GB RAM")
    except Exception:
        check("Memory", False)

    console.print()

    # ── Network ───────────────────────────────────────────────────────────────
    console.print("[bold]Network[/bold]")

    local_ip = get_local_ip()
    console.print(f"  [dim]Local IP: {local_ip}[/dim]")

    tb_ip = get_thunderbolt_bridge_ip()
    if tb_ip:
        check("Thunderbolt bridge IP", True, tb_ip)
    else:
        check(
            "Thunderbolt bridge IP",
            False,
            detail="not detected",
            fix=(
                "Set up Thunderbolt Bridge in System Settings → Network. "
                "Assign static IPs (e.g., 10.0.0.1 / 10.0.0.2)."
            ),
        )

    # Port availability
    grpc_ok = is_port_available(DEFAULT_GRPC_PORT)
    tensor_ok = is_port_available(DEFAULT_TENSOR_PORT)
    check(f"gRPC port {DEFAULT_GRPC_PORT} available", grpc_ok,
          fix=f"Another process is using port {DEFAULT_GRPC_PORT}. Kill it or use --port.")
    check(f"Tensor port {DEFAULT_TENSOR_PORT} available", tensor_ok,
          fix=f"Another process is using port {DEFAULT_TENSOR_PORT}. Kill it or use --tensor-port.")

    # Master connectivity (if requested)
    master_addr = None
    master_port = DEFAULT_GRPC_PORT

    if config_path:
        try:
            from macfleet.cli.config_loader import cluster_config_from_yaml
            cfg = cluster_config_from_yaml(config_path)
            master_addr = cfg.master_addr or None
            master_port = cfg.master_port
        except Exception:
            pass

    if master:
        if ":" in master:
            parts = master.split(":", 1)
            master_addr = parts[0]
            try:
                master_port = int(parts[1])
            except ValueError:
                master_addr = master
        else:
            master_addr = master

    if master_addr:
        console.print()
        console.print(f"[bold]Master Connectivity[/bold] [dim]({master_addr})[/dim]")

        # Ping check
        from macfleet.utils.network import ping_host
        ping_ok = ping_host(master_addr, timeout=2.0)
        check(
            f"Ping {master_addr}",
            ping_ok,
            fix="Check network cable and IP configuration. Try: ping " + master_addr,
        )

        # gRPC port check
        grpc_reachable = is_reachable(master_addr, master_port, timeout=2.0)
        check(
            f"gRPC port {master_port} reachable",
            grpc_reachable,
            fix=f"Start master first: macfleet launch --role master --host {master_addr}",
        )

        # Tensor port check
        tensor_reachable = is_reachable(master_addr, DEFAULT_TENSOR_PORT, timeout=2.0)
        check(
            f"Tensor port {DEFAULT_TENSOR_PORT} reachable",
            tensor_reachable,
        )

        # Latency estimate
        if ping_ok:
            import time
            try:
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        s.connect((master_addr, master_port if grpc_reachable else 7))
                    times.append((time.perf_counter() - t0) * 1000)
                avg_ms = sum(times) / len(times)
                check("TCP latency", True, f"{avg_ms:.1f} ms avg")
            except Exception:
                pass

    console.print()
    if all_ok:
        console.print("[bold green]All checks passed! MacFleet is ready.[/bold green]")
    else:
        console.print("[bold yellow]Some checks failed. See fixes above.[/bold yellow]")
    console.print()


# ── benchmark ────────────────────────────────────────────────────────────────

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

        macfleet benchmark --type bandwidth
        macfleet benchmark --type bandwidth --sizes 10,50,100
        macfleet benchmark --type allreduce
        macfleet benchmark --type latency
    """
    console.print(f"[bold blue]Running {bench_type} benchmark...[/bold blue]")
    console.print()

    # Parse sizes
    try:
        size_list = [int(s) for s in sizes.split(",")]
    except ValueError:
        console.print("[bold red]Invalid --sizes format. Use comma-separated integers, e.g. 10,50,100[/bold red]")
        sys.exit(1)

    if bench_type == "bandwidth":
        _run_bandwidth_benchmark(size_list)
    elif bench_type == "latency":
        _run_latency_benchmark()
    elif bench_type == "allreduce":
        _run_allreduce_benchmark(size_list)


def _run_bandwidth_benchmark(sizes_mb: list[int]):
    """Run local tensor serialization bandwidth benchmark."""
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
        numel = (size_mb * 1024 * 1024) // 4  # FP32 = 4 bytes
        tensor = torch.randn(numel)

        start = time.perf_counter()
        data = tensor_to_bytes(tensor)
        serialize_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        tensor2, _ = bytes_to_tensor(data)
        deserialize_ms = (time.perf_counter() - start) * 1000

        # Throughput: read + write of size_mb, convert ms→s, MB→GB
        total_sec = (serialize_ms + deserialize_ms) / 1000
        throughput_gbps = (size_mb * 2 / 1024) / total_sec if total_sec > 0 else 0

        table.add_row(
            str(size_mb),
            f"{serialize_ms:.2f}",
            f"{deserialize_ms:.2f}",
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
        from macfleet.utils.network import find_available_port
        import torch

        port = find_available_port(50099)
        server = TensorTransport("127.0.0.1", port)
        received = []

        async def on_recv(tensor, msg_type, addr):
            received.append(time.perf_counter())

        await server.start_server(on_recv)

        client = TensorTransport()
        conn_key = await client.connect("127.0.0.1", port)

        latencies = []
        for _ in range(100):
            tensor = torch.randn(1000)
            start = time.perf_counter()
            await client.send_tensor(tensor, conn_key)
            await asyncio.sleep(0.01)
            if received:
                latency = (received[-1] - start) * 1000
                latencies.append(latency)

        await client.disconnect(conn_key)
        await server.stop_server()

        if latencies:
            avg = sum(latencies) / len(latencies)
            console.print(f"  Average latency: {avg:.2f} ms")
            console.print(f"  Min latency:     {min(latencies):.2f} ms")
            console.print(f"  Max latency:     {max(latencies):.2f} ms")
        else:
            console.print("[yellow]No latency measurements collected[/yellow]")

    asyncio.run(run())


def _run_allreduce_benchmark(sizes_mb: list[int]):
    """Run AllReduce benchmark using loopback simulation."""
    import time
    import torch

    console.print("Running loopback AllReduce benchmark...")
    console.print()

    async def run():
        from macfleet.comm.transport import TensorTransport
        from macfleet.comm.collectives import CollectiveGroup, AllReduce
        from macfleet.utils.network import find_available_port

        port0 = find_available_port(50100)
        port1 = find_available_port(port0 + 1)
        t0 = TensorTransport("127.0.0.1", port0)
        t1 = TensorTransport("127.0.0.1", port1)
        await t0.start_server()
        await t1.start_server()

        g0 = CollectiveGroup(rank=0, world_size=2, transport=t0)
        g1 = CollectiveGroup(rank=1, world_size=2, transport=t1)
        await g0.connect_to_peer(1, "127.0.0.1", port1)
        await g1.connect_to_peer(0, "127.0.0.1", port0)

        table = Table(title="AllReduce Benchmark Results")
        table.add_column("Size (MB)", justify="right", style="cyan")
        table.add_column("Compression", style="yellow")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Throughput (Gbps)", justify="right", style="green")

        for comp in ["none", "topk_fp16"]:
            ar0 = AllReduce(g0)
            ar1 = AllReduce(g1)

            for size_mb in sizes_mb:
                numel = int((size_mb * 1024 * 1024) / 4)
                tensor0 = torch.randn(numel)
                tensor1 = torch.randn(numel)
                actual_mb = (numel * 4) / (1024 * 1024)

                latencies = []
                for _ in range(10):
                    start = time.perf_counter()
                    await asyncio.gather(
                        ar0(tensor0.clone()), ar1(tensor1.clone())
                    )
                    latencies.append((time.perf_counter() - start) * 1000)

                avg_lat = sum(latencies) / len(latencies)
                throughput = (actual_mb * 1024 * 1024 * 2 * 8) / (avg_lat / 1000 * 1e9)

                table.add_row(
                    f"{actual_mb:.1f}", comp,
                    f"{avg_lat:.2f}", f"{throughput:.2f}",
                )

        console.print(table)

        await g0.disconnect_all()
        await g1.disconnect_all()
        await t0.stop_server()
        await t1.stop_server()

    asyncio.run(run())


# ── info ─────────────────────────────────────────────────────────────────────

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
        console.print("  Thunderbolt IP: [yellow]Not detected[/yellow]")

    console.print()

    gpu_info = get_gpu_info()
    console.print(f"  GPU: {gpu_info.get('gpu_name', 'Unknown')}")
    console.print(f"  GPU Cores: {gpu_info.get('gpu_cores', 0)}")

    memory_info = get_memory_info()
    console.print(f"  RAM: {memory_info.get('total_gb', 0)} GB")
    console.print(f"  Memory Bandwidth: ~{get_memory_bandwidth():.0f} GB/s")

    console.print()

    try:
        import torch
        console.print(f"  PyTorch: {torch.__version__}")
        console.print(f"  MPS Available: {torch.backends.mps.is_available()}")
        console.print(f"  MPS Built: {torch.backends.mps.is_built()}")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")


if __name__ == "__main__":
    cli()
