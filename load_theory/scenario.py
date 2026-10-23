"""Model-independent scenario configuration loading."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .core import (
    CacheSpec,
    Model,
    Parallel,
    Precision,
    optimizer_state_bytes_per_param,
)


@dataclass(frozen=True)
class InferenceScenario:
    prefill_batch_size: int
    prefill_length: int
    decode_batch_size: int
    decode_length: int
    context_length: int
    cache: CacheSpec = field(default_factory=CacheSpec)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache.hit_rate


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


def _ratio(value: Any, field_name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return result


def cache_spec_from_dict(raw: dict, prefill_batch_size: int) -> CacheSpec:
    source = str(raw.get("source", "hbm"))
    load_required = raw.get("load_required", source != "hbm")
    return CacheSpec(
        hit_rate=_ratio(raw.get("hit_rate", 0.0), "cache.hit_rate"),
        source=source,
        load_required=load_required,
        entry_count=(
            _positive_int(raw["entry_count"], "cache.entry_count")
            if "entry_count" in raw
            else prefill_batch_size
        ),
        block_size=_positive_int(raw.get("block_size", 1), "cache.block_size"),
        compression_ratio=float(raw.get("compression_ratio", 1.0)),
        cached_length=(
            max(0, int(raw["cached_length"]))
            if "cached_length" in raw
            else None
        ),
    )


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    precision_raw = raw["precision"]
    parallel_raw = raw["parallel"]
    inference_raw = raw["inference"]
    training_raw = raw["training"]
    optimizer = raw.get("optimizer", "adamw")
    prefill_batch_size = _positive_int(
        inference_raw["prefill_batch_size"],
        "inference.prefill_batch_size",
    )
    cache_raw = dict(inference_raw.get("cache", {}))
    if "cache_hit_rate" in inference_raw:
        cache_raw.setdefault("hit_rate", inference_raw["cache_hit_rate"])
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
            prefill_batch_size=prefill_batch_size,
            prefill_length=_positive_int(inference_raw["prefill_length"], "inference.prefill_length"),
            decode_batch_size=_positive_int(inference_raw["decode_batch_size"], "inference.decode_batch_size"),
            decode_length=_positive_int(inference_raw["decode_length"], "inference.decode_length"),
            context_length=_positive_int(inference_raw["context_length"], "inference.context_length"),
            cache=cache_spec_from_dict(cache_raw, prefill_batch_size),
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
