"""Low-overhead stage timing for asynchronous CPU and CUDA learners."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch


class StageProfiler:
    """Accumulate stage timings and synchronize CUDA only when metrics are read."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._cpu_seconds: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)
        self._cuda_events: dict[
            str,
            list[tuple[torch.cuda.Event, torch.cuda.Event]],
        ] = defaultdict(list)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not stage:
            raise ValueError("Profiler stage names must not be empty.")
        self._calls[stage] += 1
        if self.device.type == "cuda" and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                self._cuda_events[stage].append((start, end))
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_seconds[stage] += time.perf_counter() - started

    def drain_metrics(self, *, prefix: str = "profile/") -> dict[str, int | float]:
        """Resolve pending events once and clear the current logging interval."""

        if self._cuda_events:
            torch.cuda.synchronize(self.device)
        seconds = dict(self._cpu_seconds)
        for stage, events in self._cuda_events.items():
            seconds[stage] = seconds.get(stage, 0.0) + sum(
                start.elapsed_time(end) / 1_000.0 for start, end in events
            )
        total = sum(seconds.values())
        metrics: dict[str, int | float] = {}
        for stage in sorted(set(seconds) | set(self._calls)):
            duration = seconds.get(stage, 0.0)
            metrics[f"{prefix}{stage}_seconds"] = duration
            metrics[f"{prefix}{stage}_calls"] = self._calls.get(stage, 0)
            metrics[f"{prefix}{stage}_fraction"] = duration / total if total > 0.0 else 0.0
        metrics[f"{prefix}timed_seconds"] = total
        self._cpu_seconds.clear()
        self._calls.clear()
        self._cuda_events.clear()
        return metrics
