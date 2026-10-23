"""Data classes for multi-round Agent inference workloads."""

from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Optional

from ..core import CacheSpec, Parallel, Precision


CACHE_LOAD_SCOPES = frozenset({"per_round", "once", "none"})


@dataclass(frozen=True)
class AgentTurn:
    """One model interaction in an Agent task."""

    name: str
    new_input_length: int
    output_length: int
    context_before_length: int = 0
    cached_length: Optional[int] = None
    cache: Optional[CacheSpec] = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Agent turn name must not be empty")
        if self.new_input_length <= 0:
            raise ValueError("Agent turn new_input_length must be positive")
        if self.output_length <= 0:
            raise ValueError("Agent turn output_length must be positive")
        if self.context_before_length < 0:
            raise ValueError("Agent turn context_before_length must be non-negative")
        if self.name in self.depends_on:
            raise ValueError("Agent turn must not depend on itself")

    @property
    def prompt_length(self) -> int:
        return self.context_before_length + self.new_input_length


@dataclass(frozen=True)
class AgentWorkload:
    """A complete Agent task represented as serial model interactions."""

    name: str
    batch_size: int
    turns: tuple[AgentTurn, ...]
    parallel: Parallel = field(default_factory=Parallel)
    precision: Precision = field(default_factory=Precision)
    cache: CacheSpec = field(default_factory=CacheSpec)
    cache_load_scope: str = "per_round"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("Agent batch_size must be positive")
        if not self.turns:
            raise ValueError("Agent workload must contain at least one turn")
        if self.cache_load_scope not in CACHE_LOAD_SCOPES:
            raise ValueError(
                f"cache_load_scope must be one of {sorted(CACHE_LOAD_SCOPES)}, "
                f"got {self.cache_load_scope!r}"
            )
        turn_names = {turn.name for turn in self.turns}
        if len(turn_names) != len(self.turns):
            raise ValueError("Agent turn names must be unique")
        unknown_dependencies = {
            dependency
            for turn in self.turns
            for dependency in turn.depends_on
            if dependency not in turn_names
        }
        if unknown_dependencies:
            raise ValueError(
                f"Agent turn dependencies not found: {sorted(unknown_dependencies)}"
            )
        dependency_graph = {
            turn.name: set(turn.depends_on) for turn in self.turns
        }
        try:
            TopologicalSorter(dependency_graph).prepare()
        except CycleError as error:
            raise ValueError("Agent turn dependencies must form a DAG") from error

    @property
    def new_input_tokens(self) -> int:
        return self.batch_size * sum(turn.new_input_length for turn in self.turns)

    @property
    def output_tokens(self) -> int:
        return self.batch_size * sum(turn.output_length for turn in self.turns)
