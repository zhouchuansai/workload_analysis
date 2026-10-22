"""Core data classes for hardware-independent load modeling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttentionKind(str, Enum):
    MHA = "mha"
    GQA = "gqa"
    MLA = "mla"


class MLPKind(str, Enum):
    DENSE = "dense"
    MOE = "moe"


@dataclass(frozen=True)
class Precision:
    """Element size in bytes for different model states."""

    weight: int = 2
    activation: int = 2
    kv_cache: int = 2
    gradient: int = 2


def optimizer_state_bytes_per_param(optimizer: str) -> int:
    """Return optimizer state bytes per parameter.

    AdamW uses FP32 master weight, momentum, and variance by default:
    4 + 4 + 4 = 12 bytes per parameter. The optimizer is selected by name in
    the scenario configuration; users do not configure master-weight precision.
    """
    normalized = optimizer.strip().lower()
    if normalized != "adamw":
        raise ValueError(f"unsupported optimizer {optimizer!r}; supported: adamw")
    return 12


@dataclass(frozen=True)
class Parallel:
    """Logical parallel degree used for byte-volume modeling."""

    tp: int = 1
    ep: int = 1
    pp: int = 1
    dp: int = 1

    def validate(self) -> None:
        if min(self.tp, self.ep, self.pp, self.dp) < 1:
            raise ValueError("parallel degrees must be positive")
        if self.tp & (self.tp - 1) != 0:
            raise ValueError("tp must be a power of two in this first version")


@dataclass(frozen=True)
class AttentionSpec:
    kind: AttentionKind
    num_heads: int
    head_dim: int
    num_kv_heads: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    rope_dim: Optional[int] = None
    o_lora_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == AttentionKind.MLA:
            if None in (self.q_lora_rank, self.kv_lora_rank, self.o_lora_rank):
                raise ValueError("MLA requires q_lora_rank, kv_lora_rank and o_lora_rank")
        if self.num_kv_heads is None:
            object.__setattr__(self, "num_kv_heads", self.num_heads)


@dataclass(frozen=True)
class MLPSpec:
    kind: MLPKind
    intermediate_size: int
    num_experts: int = 1
    topk: int = 0
    num_shared_experts: int = 0
    expert_precision_bytes: float = 2.0

    @property
    def active_experts(self) -> int:
        if self.kind == MLPKind.DENSE:
            return 1
        return self.topk + self.num_shared_experts


@dataclass(frozen=True)
class Model:
    """A concrete model. Future modules can be added through extra_modules."""

    name: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    attention: AttentionSpec
    mlp: MLPSpec
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 32768
    extra_modules: tuple = ()


@dataclass(frozen=True)
class InferenceWorkload:
    prefill_batch_size: int
    prefill_length: int
    decode_batch_size: int
    decode_length: int
    context_length: int
    parallel: Parallel = field(default_factory=Parallel)
    precision: Precision = field(default_factory=Precision)


@dataclass(frozen=True)
class TrainingWorkload:
    global_batch_size: int
    seq_len: int
    optimizer: str = "adamw"
    gradient_checkpointing: bool = True
    parallel: Parallel = field(default_factory=Parallel)
    precision: Precision = field(default_factory=Precision)


@dataclass(frozen=True)
class FuzzyModel:
    """Approximate model used when a concrete config is unavailable.

    active_params is the parameter count touched by one token. For a Dense model
    it is usually close to total_params; for MoE it is top-k plus shared experts.
    """

    name: str
    total_params: int
    active_params: int
    num_layers: int
    vocab_size: int
    kv_bytes_per_token_per_layer: int
    vector_flops_per_token_per_layer: int = 0
    attention_flops_per_token_per_layer: int = 0
    communication_bytes_per_token: int = 0


@dataclass
class Demand:
    """All demand quantities are non-negative byte or operation counts."""

    cube_flops: float = 0.0
    vector_flops: float = 0.0
    weight_bytes: float = 0.0
    active_weight_bytes: float = 0.0
    kv_cache_bytes: float = 0.0
    activation_bytes: float = 0.0
    communication_bytes: float = 0.0
    optimizer_bytes: float = 0.0
    gradient_bytes: float = 0.0

    @property
    def total_flops(self) -> float:
        return self.cube_flops + self.vector_flops

    @property
    def data_movement_bytes(self) -> float:
        """Theoretical byte movement, not resident HBM capacity."""
        return (
            self.weight_bytes
            + self.kv_cache_bytes
            + self.activation_bytes
            + self.communication_bytes
        )

    @property
    def total_memory_bytes(self) -> float:
        """Backward-compatible alias for data_movement_bytes."""
        return self.data_movement_bytes

    @property
    def compute_memory_ratio(self) -> float:
        return self.total_flops / self.data_movement_bytes if self.data_movement_bytes else 0.0

    @property
    def compute_capacity_ratio(self) -> float:
        return self.total_flops / self.weight_bytes if self.weight_bytes else 0.0

    @property
    def compute_capacity_ratio_active(self) -> float:
        return self.total_flops / self.active_weight_bytes if self.active_weight_bytes else 0.0

    @property
    def cube_vector_ratio(self) -> float:
        return self.cube_flops / self.vector_flops if self.vector_flops else float("inf")

    def __add__(self, other: "Demand") -> "Demand":
        return Demand(*(getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__))
