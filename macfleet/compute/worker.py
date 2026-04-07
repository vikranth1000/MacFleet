"""Worker-side task executor.

Receives tasks from the coordinator, executes them in a process pool
for isolation, and sends results back.

    worker = TaskWorker(transport, "coordinator-id")
    await worker.start()
    # ... worker runs until stopped ...
    await worker.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cloudpickle

from macfleet.comm.protocol import MessageType
from macfleet.comm.transport import PeerTransport
from macfleet.compute.models import TaskResult, TaskSpec

logger = logging.getLogger(__name__)


def _execute_task(fn_bytes: bytes, args_bytes: bytes, kwargs_bytes: bytes) -> bytes:
    """Execute a serialized task in a worker process.

    This function runs in a separate process via ProcessPoolExecutor.
    It deserializes the function and arguments, calls the function,
    and returns the serialized result.

    Returns cloudpickle-serialized result bytes.
    Raises on failure (caught by the caller).
    """
    fn = cloudpickle.loads(fn_bytes)
    args = cloudpickle.loads(args_bytes)
    kwargs = cloudpickle.loads(kwargs_bytes)
    result = fn(*args, **kwargs)
    return cloudpickle.dumps(result)


class TaskWorker:
    """Receives tasks from coordinator, executes them, sends results.

    Uses a ProcessPoolExecutor for isolation — a crashing task won't
    take down the worker process or the event loop.
    """

    def __init__(
        self,
        transport: PeerTransport,
        coordinator_peer_id: str,
        max_workers: Optional[int] = None,
    ):
        self._transport = transport
        self._coordinator = coordinator_peer_id
        self._max_workers = max_workers or min(os.cpu_count() or 1, 4)
        self._executor: Optional[ProcessPoolExecutor] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start listening for tasks from the coordinator."""
        self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_tasks())
        logger.info(
            "TaskWorker started (max_workers=%d, coordinator=%s)",
            self._max_workers, self._coordinator,
        )

    async def stop(self) -> None:
        """Stop the worker and shut down the process pool."""
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def _listen_tasks(self) -> None:
        """Listen for TASK messages from the coordinator."""
        conn = self._transport.get_connection(self._coordinator)
        if conn is None:
            logger.error("No connection to coordinator %s", self._coordinator)
            return

        while self._running:
            try:
                msg = await conn.recv_message(
                    timeout=self._transport.config.recv_timeout_sec,
                )
                if msg.msg_type == MessageType.TASK:
                    spec = TaskSpec.unpack(msg.payload)
                    asyncio.create_task(self._execute_and_reply(spec))
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            except (ConnectionError, OSError) as e:
                logger.warning(
                    "Lost connection to coordinator %s: %s",
                    self._coordinator, e,
                )
                return

    async def _execute_and_reply(self, spec: TaskSpec) -> None:
        """Execute a task in the process pool and send the result back."""
        loop = asyncio.get_event_loop()
        try:
            result_bytes = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    _execute_task,
                    spec.fn_bytes,
                    spec.args_bytes,
                    spec.kwargs_bytes,
                ),
                timeout=spec.timeout_sec,
            )
            result = TaskResult(
                task_id=spec.task_id,
                ok=True,
                value_bytes=result_bytes,
            )
        except asyncio.TimeoutError:
            result = TaskResult(
                task_id=spec.task_id,
                ok=False,
                error=f"Task timed out after {spec.timeout_sec}s",
            )
        except Exception:
            result = TaskResult.failure(spec.task_id, None)

        try:
            await self._transport.send(
                self._coordinator,
                result.pack(),
                msg_type=MessageType.RESULT,
            )
        except (ConnectionError, OSError) as e:
            logger.error(
                "Failed to send result for task %s: %s",
                spec.task_id[:8], e,
            )
