"""Loader for Agent task scenario JSON files."""

import json
from pathlib import Path

from ..core import Parallel, Precision
from ..scenario import cache_spec_from_dict
from .schema import AgentTurn, AgentWorkload, CACHE_LOAD_SCOPES


def _non_negative_int(value: int, field_name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _positive_int(value: int, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def load_agent_scenario(
    path: str | Path,
    parallel: Parallel,
    precision: Precision,
) -> AgentWorkload:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    batch_size = _positive_int(raw["batch_size"], "agent.batch_size")
    default_cache_raw = dict(raw.get("cache", {}))
    default_cache = cache_spec_from_dict(default_cache_raw, batch_size)
    cache_load_scope = raw.get("cache_load_scope", "per_round")
    if cache_load_scope not in CACHE_LOAD_SCOPES:
        raise ValueError(f"agent.cache_load_scope must be one of {sorted(CACHE_LOAD_SCOPES)}")

    turns = []
    for index, turn_raw in enumerate(raw["turns"]):
        turn_cache_raw = {**default_cache_raw, **dict(turn_raw.get("cache", {}))}
        turn_cache = cache_spec_from_dict(turn_cache_raw, batch_size)
        cached_length = turn_raw.get("cached_length")
        if cached_length is not None:
            cached_length = _non_negative_int(cached_length, "turn.cached_length")
        turns.append(
            AgentTurn(
                name=str(turn_raw.get("name", f"turn_{index + 1}")),
                new_input_length=_positive_int(turn_raw["new_input_length"], "turn.new_input_length"),
                output_length=_positive_int(turn_raw["output_length"], "turn.output_length"),
                context_before_length=_non_negative_int(
                    turn_raw.get("context_before_length", 0),
                    "turn.context_before_length",
                ),
                cached_length=cached_length,
                cache=turn_cache,
                depends_on=tuple(str(item) for item in turn_raw.get("depends_on", [])),
            )
        )

    return AgentWorkload(
        name=raw["name"],
        batch_size=batch_size,
        turns=tuple(turns),
        parallel=parallel,
        precision=precision,
        cache=default_cache,
        cache_load_scope=cache_load_scope,
    )
