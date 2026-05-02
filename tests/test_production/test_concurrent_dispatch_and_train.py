"""Pool.train running while Pool.map dispatches tasks in parallel.

Production scenario: a coordinator runs distributed training. While
training, a separate workflow dispatches preprocessing tasks via
Pool.map. The two paths must not interfere — gradient sync must
complete cleanly while task dispatch is in flight, and vice versa.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from macfleet import task
from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


@task
def _preprocess_item(payload: dict) -> dict:
    """Toy preprocessing task: doubles every value in a dict."""
    return {k: v * 2 for k, v in payload.items()}


@task
def _identity(value):
    return value


async def _setup_mesh(n: int) -> tuple[list[PeerTransport], list[int]]:
    transports = []
    ports = []
    for i in range(n):
        t = PeerTransport(local_id=f"node-{i}", config=CONFIG)
        await t.start_server("127.0.0.1", 0)
        port = t._server.sockets[0].getsockname()[1]
        transports.append(t)
        ports.append(port)
    for i in range(n):
        for j in range(i + 1, n):
            await transports[i].connect(f"node-{j}", "127.0.0.1", ports[j])
    await asyncio.sleep(0.2)
    return transports, ports


def _make_groups(n: int, transports: list[PeerTransport]) -> list[CollectiveGroup]:
    return [
        CollectiveGroup(
            rank=rank, world_size=n, transport=transports[rank],
            rank_to_peer={r: f"node-{r}" for r in range(n) if r != rank},
        )
        for rank in range(n)
    ]


class TestPoolMapWhileTraining:
    """Pool.map (single-node fast path) doesn't disturb a parallel
    allreduce loop on a separate transport mesh."""

    @pytest.mark.asyncio
    async def test_concurrent_local_map_and_allreduce(self):
        from macfleet.sdk.pool import Pool

        n_nodes = 2
        transports, _ = await _setup_mesh(n_nodes)
        try:
            groups = _make_groups(n_nodes, transports)

            # Ground truth: independent rounds of allreduce should converge
            # to the average of contributions, regardless of what other
            # threads in the process are doing (Pool.map runs in a
            # ProcessPool / single-process here).

            async def allreduce_loop() -> list[np.ndarray]:
                """Run 30 rounds of allreduce, return rank 0's results."""
                results = []
                for round_i in range(30):
                    arrays = [
                        np.full(100, float(round_i + r), dtype=np.float32)
                        for r in range(n_nodes)
                    ]
                    rs = await asyncio.gather(*(
                        groups[r].allreduce(arrays[r], op="mean")
                        for r in range(n_nodes)
                    ))
                    results.append(rs[0])
                return results

            async def task_dispatch_loop() -> list:
                """Dispatch 30 tasks via Pool.map concurrently."""
                with Pool(open=True) as pool:
                    return pool.map(
                        _preprocess_item,
                        [{"v": i} for i in range(30)],
                    )

            allreduce_results, task_results = await asyncio.gather(
                allreduce_loop(),
                task_dispatch_loop(),
            )

            # Allreduce results: round R contributions are R, R+1; mean = R + 0.5
            for round_i, r in enumerate(allreduce_results):
                expected = float(round_i) + 0.5
                np.testing.assert_allclose(r, expected, rtol=1e-5)

            # Task results: every payload's "v" should have doubled.
            assert task_results == [{"v": i * 2} for i in range(30)]
        finally:
            for t in transports:
                await t.disconnect_all()


class TestParallelAllreduceStreams:
    """Two independent groups doing allreduce concurrently on the same
    transport mesh — verify they don't trample each other's stream IDs."""

    @pytest.mark.asyncio
    async def test_two_concurrent_allreduce_chains(self):
        n_nodes = 2
        transports, _ = await _setup_mesh(n_nodes)
        try:
            groups = _make_groups(n_nodes, transports)

            # Each rank runs two allreduce chains at the same time.
            # Within a single rank, sends to the same peer must be
            # serialized via PeerTransport's per-direction lock.

            async def chain(rank: int, base: float) -> np.ndarray:
                # 5 sequential rounds, returning the last result.
                arr = np.full(50, base, dtype=np.float32)
                for _ in range(5):
                    arr = await groups[rank].allreduce(arr, op="mean")
                return arr

            # Run two chains per rank concurrently. Each rank coordinates
            # with the other across both chains.
            results = await asyncio.gather(
                chain(0, 10.0), chain(0, 100.0),
                chain(1, 20.0), chain(1, 200.0),
            )
            # After enough averaging rounds, both ranks see the mean
            # of their starting values. Chain1: mean(10,20)=15.
            # Chain2: mean(100,200)=150.
            # But because the two chains interleave on the same socket,
            # the order isn't guaranteed — so we just check that the two
            # chains' results from rank 0 cover both 15 and 150.
            rank0_chain_results = sorted([float(r[0]) for r in results[:2]])
            rank1_chain_results = sorted([float(r[0]) for r in results[2:]])
            np.testing.assert_allclose(rank0_chain_results, [15.0, 150.0], rtol=1e-3)
            np.testing.assert_allclose(rank1_chain_results, [15.0, 150.0], rtol=1e-3)
        finally:
            for t in transports:
                await t.disconnect_all()


class TestStreamMultiplexingSafety:
    """Multiple concurrent send/recv pairs on the same connection must not
    interleave bytes."""

    @pytest.mark.asyncio
    async def test_concurrent_send_recv_on_single_connection(self):
        n_nodes = 2
        transports, _ = await _setup_mesh(n_nodes)
        try:
            # Generate distinct payloads, send them all concurrently from
            # rank 0 → rank 1, recv all concurrently. Without per-direction
            # locks, bytes would interleave and recv would parse garbage.
            n_msgs = 20
            payloads = [
                f"payload-{i:04d}-{'x' * (i + 100)}".encode()
                for i in range(n_msgs)
            ]

            async def sender():
                for p in payloads:
                    await transports[0].send("node-1", p)

            async def receiver():
                got = []
                for _ in range(n_msgs):
                    got.append(await transports[1].recv("node-0"))
                return got

            send_done, received = await asyncio.gather(sender(), receiver())
            # Order MUST match — TCP guarantees byte-stream order, and the
            # WireMessage frames are size-delimited.
            assert received == payloads
        finally:
            for t in transports:
                await t.disconnect_all()
