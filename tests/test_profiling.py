"""Tests for deferred learner stage profiling."""

from isaaclab_tasks.manager_based.locomotion.velocity.config.fastwmr.algorithm.utils import (
    StageProfiler,
)


def test_cpu_stage_profiler_drains_interval_metrics() -> None:
    profiler = StageProfiler("cpu")

    with profiler.measure("sample"):
        sum(range(10))
    with profiler.measure("sample"):
        sum(range(20))

    metrics = profiler.drain_metrics()
    assert metrics["profile/sample_calls"] == 2
    assert metrics["profile/sample_seconds"] >= 0.0
    assert metrics["profile/sample_fraction"] == 1.0
    assert profiler.drain_metrics() == {"profile/timed_seconds": 0}
