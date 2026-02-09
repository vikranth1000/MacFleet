#!/usr/bin/env python3
"""AllReduce latency benchmark for MacFleet.

Measures AllReduce operation latency at various tensor sizes,
with and without gradient compression.

Usage:
    # Local benchmark (simulated with loopback):
    python benchmarks/allreduce_bench.py

    # Distributed benchmark (requires cluster):
    python benchmarks/allreduce_bench.py --master 10.0.0.1 --rank 0
    python benchmarks/allreduce_bench.py --master 10.0.0.1 --rank 1
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
class AllReduceResult:
    """Result of an AllReduce benchmark."""
    size_mb: float
    num_params: int
    latency_ms: float
    throughput_gbps: float
    compression: str
    compression_ratio: float


async def benchmark_allreduce_loopback(
    sizes_mb: List[float],
    iterations: int = 10,
    compression: str = "none",
    topk_ratio: float = 0.1,
) -> List[AllReduceResult]:
    """Benchmark AllReduce with loopback simulation.

    Simulates a 2-node AllReduce on a single machine by
    running both sides asynchronously.

    Args:
        sizes_mb: List of tensor sizes in MB.
        iterations: Iterations per size.
        compression: Compression type ("none", "topk", "fp16", "topk_fp16").
        topk_ratio: Top-K ratio if using compression.

    Returns:
        List of AllReduceResult.
    """
    from macfleet.comm.transport import TensorTransport
    from macfleet.comm.collectives import CollectiveGroup, AllReduce
    from macfleet.compression.pipeline import create_pipeline

    console.print(f"[bold blue]Benchmarking AllReduce (Loopback)[/bold blue]")
    console.print(f"Compression: {compression}")
    console.print()

    # Create two transports (simulating two nodes)
    port0 = 50100
    port1 = 50101

    transport0 = TensorTransport("127.0.0.1", port0)
    transport1 = TensorTransport("127.0.0.1", port1)

    # Start servers
    received0 = []
    received1 = []

    async def on_recv0(tensor, msg_type, addr):
        received0.append(tensor)

    async def on_recv1(tensor, msg_type, addr):
        received1.append(tensor)

    await transport0.start_server(on_recv0)
    await transport1.start_server(on_recv1)

    # Create collective groups
    group0 = CollectiveGroup(rank=0, world_size=2, transport=transport0)
    group1 = CollectiveGroup(rank=1, world_size=2, transport=transport1)

    # Connect peers
    await group0.connect_to_peer(1, "127.0.0.1", port1)
    await group1.connect_to_peer(0, "127.0.0.1", port0)

    # Create compression pipeline
    pipeline = create_pipeline(compression, topk_ratio) if compression != "none" else None

    results = []

    for size_mb in sizes_mb:
        numel = int((size_mb * 1024 * 1024) / 4)  # float32
        tensor0 = torch.randn(numel)
        tensor1 = torch.randn(numel)
        actual_size_mb = (numel * 4) / (1024 * 1024)

        # Create AllReduce operations
        if pipeline:
            allreduce0 = AllReduce(
                group0,
                compress_fn=lambda t: pipeline.compress(t),
                decompress_fn=lambda i, v, n, d: pipeline.decompress(
                    type('', (), {'indices': i, 'values': v, 'original_numel': n, 'original_dtype': d, 'is_sparse': True})()
                ),
            )
            allreduce1 = AllReduce(
                group1,
                compress_fn=lambda t: pipeline.compress(t),
                decompress_fn=lambda i, v, n, d: pipeline.decompress(
                    type('', (), {'indices': i, 'values': v, 'original_numel': n, 'original_dtype': d, 'is_sparse': True})()
                ),
            )
        else:
            allreduce0 = AllReduce(group0)
            allreduce1 = AllReduce(group1)

        latencies = []

        # Warmup
        received0.clear()
        received1.clear()

        # Benchmark
        for _ in range(iterations):
            received0.clear()
            received1.clear()

            start = time.perf_counter()

            # Run both sides concurrently
            result0, result1 = await asyncio.gather(
                allreduce0(tensor0.clone()),
                allreduce1(tensor1.clone()),
            )

            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)

        # Calculate throughput
        bytes_transferred = actual_size_mb * 1024 * 1024 * 2  # Both directions
        throughput_gbps = (bytes_transferred * 8) / (avg_latency / 1000 * 1e9)

        # Compression ratio
        if pipeline:
            comp_ratio = pipeline.theoretical_ratio
        else:
            comp_ratio = 1.0

        results.append(AllReduceResult(
            size_mb=actual_size_mb,
            num_params=numel,
            latency_ms=avg_latency,
            throughput_gbps=throughput_gbps,
            compression=compression,
            compression_ratio=comp_ratio,
        ))

        console.print(f"  {actual_size_mb:.1f} MB: {avg_latency:.2f} ms ({throughput_gbps:.2f} Gbps)")

    # Cleanup
    await group0.disconnect_all()
    await group1.disconnect_all()
    await transport0.stop_server()
    await transport1.stop_server()

    return results


def benchmark_compression_overhead(
    sizes_mb: List[float],
    iterations: int = 10,
) -> dict:
    """Benchmark compression overhead separately.

    Args:
        sizes_mb: List of sizes to test.
        iterations: Iterations per size.

    Returns:
        Dictionary with timing breakdowns.
    """
    from macfleet.compression.pipeline import create_pipeline

    console.print("[bold blue]Benchmarking Compression Overhead[/bold blue]")
    console.print()

    results = {"none": [], "topk": [], "fp16": [], "topk_fp16": []}

    for compression in ["none", "topk", "fp16", "topk_fp16"]:
        pipeline = create_pipeline(compression, 0.1)

        for size_mb in sizes_mb:
            numel = int((size_mb * 1024 * 1024) / 4)
            tensor = torch.randn(numel)

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                compressed = pipeline.compress(tensor)
                _ = pipeline.decompress(compressed)
                times.append((time.perf_counter() - start) * 1000)

            avg_time = sum(times) / len(times)
            results[compression].append((size_mb, avg_time))

            console.print(f"  {compression:10} {size_mb:5.1f} MB: {avg_time:.2f} ms")

    return results


def print_results(results: List[AllReduceResult]):
    """Print benchmark results as a table."""
    table = Table(title="AllReduce Benchmark Results")
    table.add_column("Size (MB)", justify="right", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Throughput (Gbps)", justify="right", style="green")
    table.add_column("Compression", style="yellow")
    table.add_column("Ratio", justify="right")

    for r in results:
        table.add_row(
            f"{r.size_mb:.1f}",
            f"{r.num_params:,}",
            f"{r.latency_ms:.2f}",
            f"{r.throughput_gbps:.2f}",
            r.compression,
            f"{r.compression_ratio:.1%}",
        )

    console.print(table)


def print_model_estimates():
    """Print estimated AllReduce times for common models."""
    console.print()
    console.print("[bold]Estimated AllReduce Times for Common Models[/bold]")
    console.print("(Assuming 4 GB/s Thunderbolt bandwidth)")
    console.print()

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Size (FP32)", justify="right")
    table.add_column("Uncompressed", justify="right")
    table.add_column("TopK+FP16", justify="right", style="green")

    models = [
        ("ResNet-50", 25_000_000),
        ("GPT-2 (124M)", 124_000_000),
        ("ViT-Base", 86_000_000),
        ("GPT-2 (355M)", 355_000_000),
    ]

    bandwidth_gbps = 32  # Thunderbolt 4
    bytes_per_sec = bandwidth_gbps * 1e9 / 8

    for name, params in models:
        size_bytes = params * 4  # FP32
        size_mb = size_bytes / (1024 * 1024)

        # Uncompressed time (both directions)
        uncomp_time = (size_bytes * 2) / bytes_per_sec * 1000

        # TopK (10%) + FP16: indices (4B) + values (2B) = 6B per kept value
        # Keep 10% = 0.1 * params values
        compressed_size = int(params * 0.1) * 6
        comp_time = (compressed_size * 2) / bytes_per_sec * 1000

        table.add_row(
            name,
            f"{params:,}",
            f"{size_mb:.0f} MB",
            f"{uncomp_time:.1f} ms",
            f"{comp_time:.1f} ms",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="MacFleet AllReduce Benchmark")
    parser.add_argument("--sizes", type=str, default="1,10,25,50,100",
                       help="Comma-separated sizes in MB")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per size")
    parser.add_argument("--compression", type=str, default="none",
                       choices=["none", "topk", "fp16", "topk_fp16"],
                       help="Compression type")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare all compression types")
    parser.add_argument("--overhead", action="store_true",
                       help="Benchmark compression overhead only")
    args = parser.parse_args()

    sizes = [float(s) for s in args.sizes.split(",")]

    if args.overhead:
        benchmark_compression_overhead(sizes, args.iterations)
    elif args.compare_all:
        all_results = []
        for comp in ["none", "topk_fp16"]:
            results = asyncio.run(benchmark_allreduce_loopback(
                sizes, args.iterations, comp
            ))
            all_results.extend(results)
        print_results(all_results)
    else:
        results = asyncio.run(benchmark_allreduce_loopback(
            sizes, args.iterations, args.compression
        ))
        print_results(results)

    print_model_estimates()


if __name__ == "__main__":
    main()
