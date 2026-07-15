"""Bounded asynchronous scheduler with priority aging."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


class QueueFullError(RuntimeError):
    """Raised when the bounded waiting queue is full."""


class QueueTimeoutError(TimeoutError):
    """Raised when a request waits longer than the configured limit."""


@dataclass
class _Ticket:
    sequence: int
    group: str
    priority: int
    enqueued_at: float
    future: asyncio.Future[float]


class SchedulerLease:
    def __init__(
        self, scheduler: "PriorityScheduler", group: str, queue_ms: float
    ) -> None:
        self.scheduler = scheduler
        self.group = group
        self.queue_ms = queue_ms
        self._released = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self.scheduler.release(self.group)


class PriorityScheduler:
    """Limits concurrency and selects queued work by priority with starvation protection."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue_size: int,
        queue_timeout_seconds: float,
        aging_interval_seconds: float,
        starvation_timeout_seconds: float = 15.0,
        strategy: str = "priority_aging",
    ) -> None:
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self.queue_timeout_seconds = queue_timeout_seconds
        self.aging_interval_seconds = aging_interval_seconds
        self.starvation_timeout_seconds = starvation_timeout_seconds
        self.strategy = strategy
        self._lock = asyncio.Lock()
        self._pending: list[_Ticket] = []
        self._active = 0
        self._sequence = 0
        self._dispatched = 0
        self._completed = 0
        self._queue_time_total_ms = 0.0
        self._dispatched_by_group: dict[str, int] = {}
        self._completed_by_group: dict[str, int] = {}
        self._queue_ms_by_group: dict[str, float] = {}

    async def acquire(self, group: str, priority: int) -> SchedulerLease:
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        future: asyncio.Future[float] = loop.create_future()
        async with self._lock:
            if len(self._pending) >= self.max_queue_size:
                raise QueueFullError("Priority gateway queue is full")
            ticket = _Ticket(
                sequence=self._sequence,
                group=group,
                priority=priority,
                enqueued_at=now,
                future=future,
            )
            self._sequence += 1
            self._pending.append(ticket)
            self._dispatch_locked(now)

        try:
            queue_ms = await asyncio.wait_for(
                asyncio.shield(future), timeout=self.queue_timeout_seconds
            )
            return SchedulerLease(self, group, queue_ms)
        except TimeoutError as error:
            async with self._lock:
                if ticket in self._pending:
                    self._pending.remove(ticket)
                elif future.done() and not future.cancelled():
                    self._active -= 1
                    self._dispatch_locked(time.perf_counter())
            raise QueueTimeoutError(
                "Timed out while waiting for an LLM slot"
            ) from error
        except asyncio.CancelledError:
            async with self._lock:
                if ticket in self._pending:
                    self._pending.remove(ticket)
                elif future.done() and not future.cancelled():
                    self._active -= 1
                    self._dispatch_locked(time.perf_counter())
            raise

    async def release(self, group: str | None = None) -> None:
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError("Scheduler slot released more than once")
            self._active -= 1
            self._completed += 1
            if group is not None:
                self._completed_by_group[group] = (
                    self._completed_by_group.get(group, 0) + 1
                )
            self._dispatch_locked(time.perf_counter())

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            queued_by_group: dict[str, int] = {}
            for ticket in self._pending:
                queued_by_group[ticket.group] = queued_by_group.get(ticket.group, 0) + 1
            mean_queue_ms = (
                self._queue_time_total_ms / self._dispatched
                if self._dispatched
                else 0.0
            )
            group_names = set(queued_by_group) | set(self._dispatched_by_group)
            groups = {}
            for group in sorted(group_names):
                dispatched = self._dispatched_by_group.get(group, 0)
                groups[group] = {
                    "queued": queued_by_group.get(group, 0),
                    "dispatched": dispatched,
                    "completed": self._completed_by_group.get(group, 0),
                    "mean_queue_ms": (
                        round(self._queue_ms_by_group.get(group, 0.0) / dispatched, 3)
                        if dispatched
                        else 0.0
                    ),
                }
            return {
                "strategy": self.strategy,
                "active": self._active,
                "max_concurrency": self.max_concurrency,
                "queued": len(self._pending),
                "max_queue_size": self.max_queue_size,
                "queued_by_group": queued_by_group,
                "completed": self._completed,
                "dispatched": self._dispatched,
                "mean_dispatched_queue_ms": round(mean_queue_ms, 3),
                "groups": groups,
            }

    def _dispatch_locked(self, now: float) -> None:
        while self._active < self.max_concurrency and self._pending:
            index = min(
                range(len(self._pending)),
                key=lambda item: self._selection_key(self._pending[item], now),
            )
            ticket = self._pending.pop(index)
            queue_ms = max(0.0, (now - ticket.enqueued_at) * 1000)
            self._active += 1
            self._dispatched += 1
            self._queue_time_total_ms += queue_ms
            self._dispatched_by_group[ticket.group] = (
                self._dispatched_by_group.get(ticket.group, 0) + 1
            )
            self._queue_ms_by_group[ticket.group] = (
                self._queue_ms_by_group.get(ticket.group, 0.0) + queue_ms
            )
            if not ticket.future.done():
                ticket.future.set_result(queue_ms)

    def _selection_key(self, ticket: _Ticket, now: float) -> tuple[float, float, int]:
        if self.strategy == "fifo":
            return 0.0, 0.0, ticket.sequence
        waited = max(0.0, now - ticket.enqueued_at)
        if waited >= self.starvation_timeout_seconds:
            return 0.0, 0.0, ticket.sequence
        aging_steps = waited / self.aging_interval_seconds
        effective_priority = ticket.priority - aging_steps
        return 1.0, effective_priority, ticket.sequence
