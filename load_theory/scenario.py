"""Model-independent scenario configuration loading."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .core import Model, Parallel, Precision, optimizer_state_bytes_per_param


@dataclass(frozen=True)
class InferenceScenario:
    prefill_batch_size: int
    prefill_length: int
    decode_batch_size: int
    decode_length: int
    context_length: int


@dataclass(frozen=True)
class TrainingScenario:
    global_batch_size: int
    seq_len: int
    gradient_checkpointing: bool


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    optimizer: str
    parallel: Parallel
    precision: Precision
    inference: InferenceScenario
    training: TrainingScenario
    expert_precision_bytes: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"inference", "training", "both"}:
            raise ValueError("scenario.mode must be inference, training or both")
        optimizer_state_bytes_per_param(self.optimizer)
        self.parallel.validate()

    def apply_expert_precision(self, model: Model) -> Model:
        if self.expert_precision_bytes is None:
            return model
        return replace(
            model,
            mlp=replace(model.mlp, expert_precision_bytes=self.expert_precision_bytes),
        )


def _element_bytes(value: Any, field_name: str) -> float:
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, dict):
        result = float(value["bytes_per_element"])
    else:
        raise ValueError(f"precision.{field_name} must be a number or an object with bytes_per_element")
    if result <= 0:
        raise ValueError(f"precision.{field_name}.bytes_per_element must be positive")
    return result


def _positive_int(value: int, field_name: str) -> int:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    precision_raw = raw["precision"]
    parallel_raw = raw["parallel"]
    inference_raw = raw["inference"]
    training_raw = raw["training"]
    optimizer = raw.get("optimizer", "adamw")
    scenario = Scenario(
        name=raw["name"],
        mode=raw.get("mode", "both"),
        optimizer=optimizer,
        parallel=Parallel(
            tp=_positive_int(parallel_raw["tp"], "parallel.tp"),
            ep=_positive_int(parallel_raw["ep"], "parallel.ep"),
            pp=_positive_int(parallel_raw["pp"], "parallel.pp"),
            dp=_positive_int(parallel_raw["dp"], "parallel.dp"),
        ),
        precision=Precision(
            weight=_element_bytes(precision_raw["weight"], "weight"),
            activation=_element_bytes(precision_raw["activation"], "activation"),
            kv_cache=_element_bytes(precision_raw["kv_cache"], "kv_cache"),
            gradient=_element_bytes(precision_raw["gradient"], "gradient"),
        ),
        inference=InferenceScenario(
            prefill_batch_size=_positive_int(inference_raw["prefill_batch_size"], "inference.prefill_batch_size"),
            prefill_length=_positive_int(inference_raw["prefill_length"], "inference.prefill_length"),
            decode_batch_size=_positive_int(inference_raw["decode_batch_size"], "inference.decode_batch_size"),
            decode_length=_positive_int(inference_raw["decode_length"], "inference.decode_length"),
            context_length=_positive_int(inference_raw["context_length"], "inference.context_length"),
        ),
        training=TrainingScenario(
            global_batch_size=_positive_int(training_raw["global_batch_size"], "training.global_batch_size"),
            seq_len=_positive_int(training_raw["seq_len"], "training.seq_len"),
            gradient_checkpointing=training_raw["gradient_checkpointing"],
        ),
        expert_precision_bytes=(
            _element_bytes(precision_raw["expert_weight"], "expert_weight")
            if "expert_weight" in precision_raw
            else None
        ),
    )
    return scenario
