"""Versioned checkpoint save and resume support for FastSAC and FastWMR."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from ..buffers import EstimatorRolloutCache
from ..utils.normalization import RunningObservationNormalizer
from .estimator_update import EstimatorUpdater
from .fastwmr_agent import (
    FastSACReplayUpdateLoop,
    FastWMRSequenceUpdateLoop,
    FastWMRV2UpdateLoop,
)
from .rollout_worker import FastWMREstimatorRuntime
from .sac_update import SACUpdater


CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = (1, CHECKPOINT_FORMAT_VERSION)


class TrainingMode(str, Enum):
    FASTSAC = "fastsac"
    FASTWMR = "fastwmr"


@dataclass(frozen=True)
class CheckpointCounters:
    """Persisted learner counters; rollout-only state is deliberately absent."""

    environment_steps: int
    gradient_steps: int
    agent_updates: int
    estimator_version: int
    estimator_updates: int = 0
    estimator_triggers: int = 0
    control_estimator_version: int = 0
    sac_updates_since_estimator: int = 0


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Metadata restored alongside model and optimizer state."""

    path: Path
    mode: TrainingMode
    counters: CheckpointCounters
    config: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointMetadata:
    """Checkpoint fields needed to construct evaluation-only modules."""

    path: Path
    mode: TrainingMode
    counters: CheckpointCounters
    config: Mapping[str, Any]
    architecture: Mapping[str, Any]
    has_normalizer: bool


def save_training_checkpoint(
    path: str | Path,
    *,
    mode: TrainingMode | str,
    sac_updater: SACUpdater,
    update_loop: FastSACReplayUpdateLoop,
    normalizer: RunningObservationNormalizer | None,
    estimator_updater: EstimatorUpdater | None = None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save trainable state without replay or recurrent runtime state."""

    path = Path(path).expanduser().resolve()
    resolved_mode = TrainingMode(mode)
    _validate_components(
        resolved_mode,
        update_loop,
        estimator_updater=estimator_updater,
        runtime=None,
        rollout_cache=None,
        loading=False,
    )
    agent_updates = 0
    estimator_version = 0
    estimator_state = None
    control_estimator_state = None
    estimator_optimizer_state = None
    learner_state: Mapping[str, int] | None = None
    estimator_updates = 0
    estimator_triggers = 0
    control_estimator_version = 0
    sac_updates_since_estimator = 0
    if estimator_updater is not None:
        estimator_version = estimator_updater.version
        estimator_state = estimator_updater.estimator.state_dict()
        estimator_optimizer_state = estimator_updater.optimizer.state_dict()
        if isinstance(update_loop, FastWMRSequenceUpdateLoop) and update_loop.agent is not None:
            agent_updates = update_loop.agent.update_steps
            estimator_updates = update_loop.sequence_feature_processor.updates
            estimator_triggers = update_loop.sequence_feature_processor.updates
            control_estimator_version = estimator_version
        elif isinstance(update_loop, FastWMRV2UpdateLoop):
            controller = update_loop.estimator_controller
            control_estimator_state = controller.ema_estimator.control_estimator.state_dict()
            learner_state = update_loop.state_dict()
            estimator_updates = controller.estimator_updates
            estimator_triggers = controller.estimator_triggers
            control_estimator_version = controller.control_estimator_version
            sac_updates_since_estimator = update_loop.sac_updates_since_estimator

    counters = CheckpointCounters(
        environment_steps=update_loop.environment_steps,
        gradient_steps=update_loop.gradient_steps,
        agent_updates=agent_updates,
        estimator_version=estimator_version,
        estimator_updates=estimator_updates,
        estimator_triggers=estimator_triggers,
        control_estimator_version=control_estimator_version,
        sac_updates_since_estimator=sac_updates_since_estimator,
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "mode": resolved_mode.value,
        "learner_kind": (
            "v2"
            if isinstance(update_loop, FastWMRV2UpdateLoop)
            else "v1" if resolved_mode is TrainingMode.FASTWMR else "fastsac"
        ),
        "models": {
            "actor": sac_updater.actor.state_dict(),
            "critic": sac_updater.critic.state_dict(),
            "target_critic": sac_updater.target_critic.state_dict(),
            "temperature": sac_updater.temperature.state_dict(),
            "estimator": estimator_state,
            "control_estimator": control_estimator_state,
        },
        "optimizers": {
            "actor": sac_updater.actor_optimizer.state_dict(),
            "critic": sac_updater.critic_optimizer.state_dict(),
            "temperature": sac_updater.temperature_optimizer.state_dict(),
            "estimator": estimator_optimizer_state,
        },
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
        "normalizer_training": normalizer.training if normalizer is not None else None,
        "normalizer_freeze_iteration": update_loop.normalizer_freeze_iteration,
        "learner_state": dict(learner_state) if learner_state is not None else None,
        "counters": asdict(counters),
        "config": _plain_data(config or {}),
        "architecture": {
            "actor_input_dim": sac_updater.actor.input_dim,
            "action_dim": sac_updater.actor.action_dim,
            "critic_type": type(sac_updater.critic).__name__,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_training_checkpoint(
    path: str | Path,
    *,
    mode: TrainingMode | str,
    sac_updater: SACUpdater,
    update_loop: FastSACReplayUpdateLoop,
    normalizer: RunningObservationNormalizer | None,
    estimator_updater: EstimatorUpdater | None = None,
    runtime: FastWMREstimatorRuntime | None = None,
    rollout_cache: EstimatorRolloutCache | None = None,
    map_location: torch.device | str | None = None,
) -> CheckpointLoadResult:
    """Restore persistent state and restart all ephemeral rollout state."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    resolved_mode = TrainingMode(mode)
    _validate_components(
        resolved_mode,
        update_loop,
        estimator_updater=estimator_updater,
        runtime=runtime,
        rollout_cache=rollout_cache,
        loading=True,
    )
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}; "
            f"expected one of {SUPPORTED_CHECKPOINT_FORMAT_VERSIONS}."
        )
    checkpoint_mode = TrainingMode(payload.get("mode"))
    if checkpoint_mode is not resolved_mode:
        raise ValueError(
            f"Checkpoint mode is {checkpoint_mode.value!r}, expected {resolved_mode.value!r}."
        )
    if resolved_mode is TrainingMode.FASTWMR:
        checkpoint_learner = payload.get("learner_kind", "v1")
        expected_learner = "v2" if isinstance(update_loop, FastWMRV2UpdateLoop) else "v1"
        if checkpoint_learner != expected_learner:
            raise ValueError(
                f"Checkpoint learner is {checkpoint_learner!r}, expected {expected_learner!r}."
            )

    architecture = _require_mapping(payload, "architecture")
    expected_architecture = {
        "actor_input_dim": sac_updater.actor.input_dim,
        "action_dim": sac_updater.actor.action_dim,
        "critic_type": type(sac_updater.critic).__name__,
    }
    if dict(architecture) != expected_architecture:
        raise ValueError(
            f"Checkpoint architecture {dict(architecture)} does not match {expected_architecture}."
        )

    models = _require_mapping(payload, "models")
    optimizers = _require_mapping(payload, "optimizers")
    normalizer_state = payload.get("normalizer")
    if (normalizer_state is None) != (normalizer is None):
        raise ValueError("Checkpoint and runtime observation-normalizer settings do not match.")
    if resolved_mode is TrainingMode.FASTWMR:
        if models.get("estimator") is None or optimizers.get("estimator") is None:
            raise ValueError("FastWMR checkpoint is missing estimator state.")
    elif models.get("estimator") is not None or optimizers.get("estimator") is not None:
        raise ValueError("FastSAC checkpoint unexpectedly contains estimator state.")

    counter_values = _require_mapping(payload, "counters")
    counters = CheckpointCounters(
        environment_steps=int(counter_values["environment_steps"]),
        gradient_steps=int(counter_values["gradient_steps"]),
        agent_updates=int(counter_values["agent_updates"]),
        estimator_version=int(counter_values["estimator_version"]),
        estimator_updates=int(
            counter_values.get("estimator_updates", counter_values["agent_updates"])
        ),
        estimator_triggers=int(
            counter_values.get("estimator_triggers", counter_values["agent_updates"])
        ),
        control_estimator_version=int(
            counter_values.get("control_estimator_version", counter_values["estimator_version"])
        ),
        sac_updates_since_estimator=int(
            counter_values.get("sac_updates_since_estimator", 0)
        ),
    )
    if min(asdict(counters).values()) < 0:
        raise ValueError("Checkpoint counters must be non-negative.")

    sac_updater.actor.load_state_dict(models["actor"])
    sac_updater.critic.load_state_dict(models["critic"])
    sac_updater.target_critic.load_state_dict(models["target_critic"])
    sac_updater.temperature.load_state_dict(models["temperature"])
    sac_updater.actor_optimizer.load_state_dict(optimizers["actor"])
    sac_updater.critic_optimizer.load_state_dict(optimizers["critic"])
    sac_updater.temperature_optimizer.load_state_dict(optimizers["temperature"])

    if normalizer is not None:
        normalizer.load_state_dict(normalizer_state)
        normalizer.train(bool(payload.get("normalizer_training", True)))

    if "normalizer_freeze_iteration" in payload:
        saved_freeze_iteration = payload["normalizer_freeze_iteration"]
        if saved_freeze_iteration is not None:
            saved_freeze_iteration = int(saved_freeze_iteration)
            if saved_freeze_iteration < 0:
                raise ValueError(
                    "Checkpoint normalizer freeze iteration must be non-negative."
                )
        requested_freeze_iteration = update_loop.normalizer_freeze_iteration
        if (
            requested_freeze_iteration is not None
            and requested_freeze_iteration != saved_freeze_iteration
        ):
            raise ValueError(
                "Checkpoint normalizer freeze iteration does not match the runtime."
            )
        update_loop.normalizer_freeze_iteration = saved_freeze_iteration

    update_loop.environment_steps = counters.environment_steps
    update_loop.gradient_steps = counters.gradient_steps
    update_loop.replay.reset()

    if resolved_mode is TrainingMode.FASTWMR:
        assert estimator_updater is not None
        assert runtime is not None
        assert rollout_cache is not None
        estimator_updater.estimator.load_state_dict(models["estimator"])
        estimator_updater.optimizer.load_state_dict(optimizers["estimator"])
        estimator_updater.version = counters.estimator_version
        rollout_cache.clear()
        if isinstance(update_loop, FastWMRV2UpdateLoop):
            control_estimator_state = models.get("control_estimator")
            learner_state = payload.get("learner_state")
            if control_estimator_state is None or not isinstance(learner_state, Mapping):
                raise ValueError("FastWMR v2 checkpoint is missing EMA or scheduler state.")
            controller = update_loop.estimator_controller
            controller.ema_estimator.control_estimator.load_state_dict(
                control_estimator_state
            )
            update_loop.load_state_dict(dict(learner_state))
            runtime.restart(estimator_version=counters.control_estimator_version)
        else:
            assert isinstance(update_loop, FastWMRSequenceUpdateLoop)
            runtime.restart(estimator_version=counters.estimator_version)
            update_loop.last_agent_updates = ()
            assert update_loop.agent is not None
            processor = update_loop.sequence_feature_processor
            processor.updates = counters.agent_updates
            processor.last_estimator_update = None
            processor.last_runtime_rebuild = None
            update_loop.agent.update_steps = counters.agent_updates
            update_loop.agent.last_update = None
    config = _require_mapping(payload, "config")
    return CheckpointLoadResult(
        path=path,
        mode=resolved_mode,
        counters=counters,
        config=dict(config),
    )


def inspect_training_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = "cpu",
) -> CheckpointMetadata:
    """Read trusted checkpoint metadata before constructing evaluation networks."""

    resolved_path, payload = _load_payload(path, map_location=map_location)
    return CheckpointMetadata(
        path=resolved_path,
        mode=TrainingMode(payload.get("mode")),
        counters=_checkpoint_counters(payload),
        config=dict(_require_mapping(payload, "config")),
        architecture=dict(_require_mapping(payload, "architecture")),
        has_normalizer=payload.get("normalizer") is not None,
    )


def load_policy_checkpoint(
    path: str | Path,
    *,
    mode: TrainingMode | str,
    actor: torch.nn.Module,
    estimator: torch.nn.Module | None = None,
    normalizer: RunningObservationNormalizer | None = None,
    map_location: torch.device | str | None = None,
) -> CheckpointLoadResult:
    """Load only deployment modules, leaving optimizer and replay state behind."""

    resolved_path, payload = _load_payload(path, map_location=map_location)
    resolved_mode = TrainingMode(mode)
    checkpoint_mode = TrainingMode(payload.get("mode"))
    if checkpoint_mode is not resolved_mode:
        raise ValueError(
            f"Checkpoint mode is {checkpoint_mode.value!r}, expected {resolved_mode.value!r}."
        )
    architecture = _require_mapping(payload, "architecture")
    expected_actor = {
        "actor_input_dim": getattr(actor, "input_dim", None),
        "action_dim": getattr(actor, "action_dim", None),
    }
    for name, expected in expected_actor.items():
        if architecture.get(name) != expected:
            raise ValueError(
                f"Checkpoint {name}={architecture.get(name)!r} does not match {expected!r}."
            )

    models = _require_mapping(payload, "models")
    estimator_state = models.get("control_estimator")
    if estimator_state is None:
        estimator_state = models.get("estimator")
    if resolved_mode is TrainingMode.FASTWMR:
        if estimator is None or estimator_state is None:
            raise ValueError("FastWMR policy loading requires estimator checkpoint state.")
    elif estimator is not None or estimator_state is not None:
        raise ValueError("FastSAC policy loading must not receive estimator state.")
    normalizer_state = payload.get("normalizer")
    if (normalizer_state is None) != (normalizer is None):
        raise ValueError("Checkpoint and evaluation normalizer settings do not match.")

    actor.load_state_dict(models["actor"])
    actor.eval()
    if estimator is not None:
        estimator.load_state_dict(estimator_state)
        estimator.eval()
    if normalizer is not None:
        normalizer.load_state_dict(normalizer_state)
        normalizer.eval()
    return CheckpointLoadResult(
        path=resolved_path,
        mode=resolved_mode,
        counters=_checkpoint_counters(payload),
        config=dict(_require_mapping(payload, "config")),
    )


def write_config_snapshot(path: str | Path, config: Mapping[str, Any]) -> Path:
    """Atomically write a human-readable JSON snapshot beside checkpoints."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_plain_data(config), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _validate_components(
    mode: TrainingMode,
    update_loop: FastSACReplayUpdateLoop,
    *,
    estimator_updater: EstimatorUpdater | None,
    runtime: FastWMREstimatorRuntime | None,
    rollout_cache: EstimatorRolloutCache | None,
    loading: bool,
) -> None:
    if mode is TrainingMode.FASTWMR:
        if not isinstance(
            update_loop,
            (FastWMRSequenceUpdateLoop, FastWMRV2UpdateLoop),
        ):
            raise ValueError("FastWMR checkpoints require a FastWMR update loop.")
        if isinstance(update_loop, FastWMRSequenceUpdateLoop) and update_loop.agent is None:
            raise ValueError("FastWMR v1 checkpoints require an integrated update loop.")
        if estimator_updater is None:
            raise ValueError("FastWMR checkpoints require an estimator updater.")
        if loading and (runtime is None or rollout_cache is None):
            raise ValueError("FastWMR resume requires runtime and rollout cache objects.")
    elif estimator_updater is not None or runtime is not None or rollout_cache is not None:
        raise ValueError("FastSAC checkpoints must not receive FastWMR-only components.")


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint field {key!r} must be a mapping.")
    return value


def _load_payload(
    path: str | Path,
    *,
    map_location: torch.device | str | None,
) -> tuple[Path, dict[str, Any]]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")
    payload = torch.load(resolved_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}; "
            f"expected one of {SUPPORTED_CHECKPOINT_FORMAT_VERSIONS}."
        )
    return resolved_path, payload


def _checkpoint_counters(payload: Mapping[str, Any]) -> CheckpointCounters:
    values = _require_mapping(payload, "counters")
    counters = CheckpointCounters(
        environment_steps=int(values["environment_steps"]),
        gradient_steps=int(values["gradient_steps"]),
        agent_updates=int(values["agent_updates"]),
        estimator_version=int(values["estimator_version"]),
        estimator_updates=int(values.get("estimator_updates", values["agent_updates"])),
        estimator_triggers=int(values.get("estimator_triggers", values["agent_updates"])),
        control_estimator_version=int(
            values.get("control_estimator_version", values["estimator_version"])
        ),
        sac_updates_since_estimator=int(values.get("sac_updates_since_estimator", 0)),
    )
    if min(asdict(counters).values()) < 0:
        raise ValueError("Checkpoint counters must be non-negative.")
    return counters


def _plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_data(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
