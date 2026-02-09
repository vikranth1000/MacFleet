#!/usr/bin/env python3
"""Thunderbolt 4 bandwidth benchmark for MacFleet.

Measures tensor transfer throughput at various sizes to characterize
the performance of the Thunderbolt bridge connection.

Usage:
    # Local benchmark (serialization only):
    python benchmarks/bandwidth_test.py

    # Network benchmark (requires two machines):
    python benchmarks/bandwidth_test.py --server  # On one machine
    python benchmarks/bandwidth_test.py --client 10.0.0.1  # On other machine
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from rich.console import Console
from rich.table import Table


console = Console()


@dataclass
class BandwidthResult:
    """Result of a bandwidth measurement."""
    size_mb: float
    serialize_time_ms: float
    deserialize_time_ms: float
    transfer_time_ms: Optional[float]
    throughput_gbps: float


def benchmark_serialization(
    sizes_mb: List[float],
    iterations: int = 5,
    dtype: torch.dtype = torch.float32,
) -> List[BandwidthResult]:
    """Benchmark tensor serialization throughput.

    Args:
        sizes_mb: List of tensor sizes in MB to test.
        iterations: Number of iterations per size.
        dtype: Tensor data type.

    Returns:
        List of BandwidthResult for each size.
    """
    from macfleet.utils.tensor_utils import tensor_to_bytes, bytes_to_tensor

    results = []

    console.print("[bold blue]Benchmarking Tensor Serialization[/bold blue]")
    console.print(f"Iterations per size: {iterations}")
    console.print()

    for size_mb in sizes_mb:
        # Calculate number of elements
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        numel = int((size_mb * 1024 * 1024) / bytes_per_elem)

        # Create tensor
        tensor = torch.randn(numel, dtype=dtype)
        actual_size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)

        serialize_times = []
        deserialize_times = []

        # Warmup
        data = tensor_to_bytes(tensor)
        _ = bytes_to_tensor(data)

        # Benchmark
        for _ in range(iterations):
            # Serialize
            start = time.perf_counter()
            data = tensor_to_bytes(tensor)
            serialize_times.append((time.perf_counter() - start) * 1000)

            # Deserialize
            start = time.perf_counter()
            _ = bytes_to_tensor(data)
            deserialize_times.append((time.perf_counter() - start) * 1000)

        avg_serialize = sum(serialize_times) / len(serialize_times)
        avg_deserialize = sum(deserialize_times) / len(deserialize_times)
        total_time_sec = (avg_serialize + avg_deserialize) / 1000

        # Throughput: bytes/sec -> Gbps
        bytes_transferred = actual_size_mb * 1024 * 1024 * 2  # serialize + deserialize
        throughput_gbps = (bytes_transferred * 8) / (total_time_sec * 1e9) if total_time_sec > 0 else 0

        results.append(BandwidthResult(
            size_mb=actual_size_mb,
            serialize_time_ms=avg_serialize,
            deserialize_time_ms=avg_deserialize,
            transfer_time_ms=None,
            throughput_gbps=throughput_gbps,
        ))

        console.print(f"  {actual_size_mb:.1f} MB: {throughput_gbps:.2f} Gbps")

    return results


async def benchmark_network_transfer(
    host: str,
    port: int,
    sizes_mb: List[float],
    iterations: int = 3,
) -> List[BandwidthResult]:
    """Benchmark network tensor transfer throughput.

    Args:
        host: Remote host to send to.
        port: Remote port.
        sizes_mb: List of tensor sizes in MB.
        iterations: Number of iterations per size.

    Returns:
        List of BandwidthResult.
    """
    from macfleet.comm.transport import TensorTransport
    from macfleet.utils.tensor_utils import MessageType

    console.print(f"[bold blue]Benchmarking Network Transfer to {host}:{port}[/bold blue]")

    transport = TensorTransport()

    try:
        conn_key = await transport.connect(host, port)
        console.print(f"[green]Connected to {host}:{port}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to connect: {e}[/red]")
        return []

    results = []

    for size_mb in sizes_mb:
        numel = int((size_mb * 1024 * 1024) / 4)  # float32
        tensor = torch.randn(numel)
        actual_size_mb = (tensor.numel() * 4) / (1024 * 1024)

        transfer_times = []

        # Warmup
        await transport.send_tensor(tensor, conn_key, MessageType.TENSOR_GRADIENT)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            await transport.send_tensor(tensor, conn_key, MessageType.TENSOR_GRADIENT)
            transfer_times.append((time.perf_counter() - start) * 1000)

        avg_transfer = sum(transfer_times) / len(transfer_times)
        throughput_gbps = (actual_size_mb * 1024 * 1024 * 8) / (avg_transfer / 1000 * 1e9)

        results.append(BandwidthResult(
            size_mb=actual_size_mb,
            serialize_time_ms=0,
            deserialize_time_ms=0,
            transfer_time_ms=avg_transfer,
            throughput_gbps=throughput_gbps,
        ))

        console.print(f"  {actual_size_mb:.1f} MB: {avg_transfer:.1f} ms ({throughput_gbps:.2f} Gbps)")

    await transport.disconnect(conn_key)
    return results


async def run_server(port: int):
    """Run a benchmark server that receives tensors.

    Args:
        port: Port to listen on.
    """
    from macfleet.comm.transport import TensorTransport

    console.print(f"[bold blue]Starting Benchmark Server on port {port}[/bold blue]")

    received_count = 0

    async def on_receive(tensor, msg_type, addr):
        nonlocal received_count
        received_count += 1
        if received_count % 10 == 0:
            console.print(f"  Received {received_count} tensors, last shape: {tensor.shape}")

    transport = TensorTransport(host="0.0.0.0", port=port)
    await transport.start_server(on_receive)

    console.print(f"[green]Server listening on 0.0.0.0:{port}[/green]")
    console.print("Press Ctrl+C to stop")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await transport.stop_server()


def print_results(results: List[BandwidthResult]):
    """Print benchmark results as a table."""
    table = Table(title="Bandwidth Benchmark Results")
    table.add_column("Size (MB)", justify="right", style="cyan")
    table.add_column("Serialize (ms)", justify="right")
    table.add_column("Deserialize (ms)", justify="right")
    table.add_column("Transfer (ms)", justify="right")
    table.add_column("Throughput (Gbps)", justify="right", style="green")

    for r in results:
        transfer = f"{r.transfer_time_ms:.1f}" if r.transfer_time_ms else "-"
        table.add_row(
            f"{r.size_mb:.1f}",
            f"{r.serialize_time_ms:.2f}",
            f"{r.deserialize_time_ms:.2f}",
            transfer,
            f"{r.throughput_gbps:.2f}",
        )

    console.print(table)

    # Summary
    if results:
        avg_throughput = sum(r.throughput_gbps for r in results) / len(results)
        max_throughput = max(r.throughput_gbps for r in results)
        console.print()
        console.print(f"[bold]Average throughput: {avg_throughput:.2f} Gbps[/bold]")
        console.print(f"[bold]Peak throughput: {max_throughput:.2f} Gbps[/bold]")


def main():
    parser = argparse.ArgumentParser(description="MacFleet Bandwidth Benchmark")
    parser.add_argument("--server", action="store_true", help="Run as server")
    parser.add_argument("--client", type=str, default=None, help="Connect to server")
    parser.add_argument("--port", type=int, default=50099, help="Port for network test")
    parser.add_argument("--sizes", type=str, default="1,10,50,100,200,500",
                       help="Comma-separated sizes in MB")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per size")
    args = parser.parse_args()

    sizes = [float(s) for s in args.sizes.split(",")]

    if args.server:
        asyncio.run(run_server(args.port))
    elif args.client:
        results = asyncio.run(benchmark_network_transfer(
            args.client, args.port, sizes, args.iterations
        ))
        print_results(results)
    else:
        # Local serialization benchmark
        results = benchmark_serialization(sizes, args.iterations)
        print_results(results)


if __name__ == "__main__":
    main()
