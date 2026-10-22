"""Theoretical load equations for Dense, MoE, GQA and MLA models."""

from dataclasses import asdict, dataclass
from typing import Iterable

from .core import (
    AttentionKind,
    Demand,
    FuzzyModel,
    InferenceWorkload,
    MLPKind,
    Model,
    TrainingWorkload,
    optimizer_state_bytes_per_param,
)


@dataclass(frozen=True)
class ScenarioReport:
    name: str
    description: str
    total: Demand
    per_token: Demand
    metrics: dict

    def as_dict(self) -> dict:
        total = _demand_as_json(self.total)
        per_token = _demand_as_json(self.per_token)
        return {
            "name": self.name,
            "description": self.description,
            "total": total,
            "per_token": per_token,
            "metrics": self.metrics,
        }


def _demand_as_json(demand: Demand) -> dict:
    result = {
        "cube_gflops": demand.cube_flops / 1e9,
        "vector_gflops": demand.vector_flops / 1e9,
        "total_gflops": demand.total_flops / 1e9,
        "weight_bytes": demand.weight_bytes,
        "active_weight_bytes": demand.active_weight_bytes,
        "kv_cache_bytes": demand.kv_cache_bytes,
        "activation_bytes": demand.activation_bytes,
        "communication_bytes": demand.communication_bytes,
        "optimizer_bytes": demand.optimizer_bytes,
        "gradient_bytes": demand.gradient_bytes,
        "total_memory_bytes": demand.total_memory_bytes,
    }
    return result


def json_field_notes() -> dict:
    return {
        "name": "场景名称：prefill、decode 或 training_step",
        "description": "场景对应的计算批次说明",
        "total": "整个场景批次的聚合需求；通信量与访存量单位为 byte，计算量单位为 GFLOPs",
        "per_token": "按单 token 归一化后的需求；通信量与访存量单位为 byte，计算量单位为 GFLOPs",
        "metrics": "核心派生指标；比值均为无量纲或 FLOPs/byte",
        "cube_gflops": "Cube 单元理论计算需求，10^9 FLOPs；主要覆盖 GEMM/QK^T/AV 等矩阵乘",
        "vector_gflops": "Vector 单元理论计算需求，10^9 元素级操作数；主要覆盖 Norm/激活/Softmax 相关操作",
        "total_gflops": "Cube GFLOPs + Vector GFLOPs",
        "weight_bytes": "模型全部权重存储/读放量，单位 byte；MoE 为全部 expert 权重",
        "active_weight_bytes": "单 token 实际激活的权重读放量，单位 byte；MoE 为 top-k expert + shared expert + router",
        "kv_cache_bytes": "KV cache 读/写量，单位 byte；依赖上下文长度",
        "activation_bytes": "激活值访存量估计，单位 byte",
        "communication_bytes": "TP/EP/PP/DP 理论通信字节数，单位 byte",
        "optimizer_bytes": "训练 optimizer state 大小，单位 byte；AdamW 默认 FP32 master weight + m/v，共 12 bytes/param",
        "gradient_bytes": "训练梯度存储/同步相关大小，单位 byte",
        "total_memory_bytes": "weight + KV + activation + communication 的访存流量合计，不含 optimizer/gradient 状态",
        "total_gflops_metric": "核心指标中的总计算需求，单位 GFLOPs",
        "cube_vector_ratio": "Cube 计算需求 / Vector 计算需求；越大表示越依赖矩阵算力",
        "compute_memory_ratio_flops_per_byte": "计算访存比 = total FLOPs / total memory bytes",
        "compute_capacity_ratio_flops_per_byte": "算容比 = total FLOPs / 全部权重大小，单位 FLOPs/byte",
        "compute_capacity_ratio_active_flops_per_byte": "激活算容比 = total FLOPs / 单 token 激活权重大小，单位 FLOPs/byte",
        "warnings": "当前近似建模未覆盖的模型组件或公式假设提示",
        "scenario": "本次计算使用的 scenario 配置名称",
        "optimizer": "训练 optimizer 名称；当前默认 AdamW",
    }


def _attention_weights(model: Model) -> int:
    spec = model.attention
    hidden = model.hidden_size
    if spec.kind == AttentionKind.MLA:
        rope = spec.rope_dim or 0
        q = hidden * spec.q_lora_rank + spec.q_lora_rank * spec.num_heads * spec.head_dim
        kv = hidden * (spec.kv_lora_rank + rope) + 2 * spec.kv_lora_rank * spec.num_heads * spec.head_dim
        output = spec.o_lora_rank * hidden
        return q + kv + output
    return spec.num_heads * spec.head_dim + 2 * spec.num_kv_heads * spec.head_dim + spec.num_heads * spec.head_dim


def _mlp_weights(model: Model) -> tuple[int, int]:
    hidden = model.hidden_size
    intermediate = model.mlp.intermediate_size
    if model.mlp.kind == MLPKind.DENSE:
        params = 3 * hidden * intermediate
        return params, params
    routed = model.mlp.num_experts * 3 * hidden * intermediate
    shared = model.mlp.num_shared_experts * 3 * hidden * intermediate
    router = model.mlp.num_experts * hidden
    active = model.mlp.topk * 3 * hidden * intermediate + shared + router
    return routed + shared + router, active


def model_weight_params(model: Model) -> tuple[int, int]:
    attention = _attention_weights(model)
    mlp_total, mlp_active = _mlp_weights(model)
    norm = 4 * model.hidden_size
    per_layer = attention + norm
    embedding = 0 if model.tie_word_embeddings else model.vocab_size * model.hidden_size
    total = model.num_layers * (per_layer + mlp_total) + embedding
    active = model.num_layers * (per_layer + mlp_active) + embedding
    for module in model.extra_modules:
        if hasattr(module, "weight_params"):
            module_total, module_active = module.weight_params()
            total += module_total
            active += module_active
    return total, active


def model_weight_bytes(model: Model, precision) -> tuple[float, float]:
    """Return total and active weight bytes, allowing expert-only FP4 storage."""
    attention = _attention_weights(model)
    norm = 4 * model.hidden_size
    per_layer_non_expert = attention + norm
    embedding = 0 if model.tie_word_embeddings else model.vocab_size * model.hidden_size
    if model.mlp.kind == MLPKind.DENSE:
        expert_total = expert_active = 0.0
        non_expert_mlp = 3 * model.hidden_size * model.mlp.intermediate_size
        per_layer_non_expert += non_expert_mlp
    else:
        expert_total = model.mlp.num_experts * 3 * model.hidden_size * model.mlp.intermediate_size * model.mlp.expert_precision_bytes
        expert_active = model.mlp.topk * 3 * model.hidden_size * model.mlp.intermediate_size * model.mlp.expert_precision_bytes
        per_layer_non_expert += (
            model.mlp.num_experts * model.hidden_size
            + model.mlp.num_shared_experts * 3 * model.hidden_size * model.mlp.intermediate_size
        )
    total = model.num_layers * (per_layer_non_expert * precision.weight + expert_total) + embedding * precision.weight
    active = model.num_layers * (per_layer_non_expert * precision.weight + expert_active) + embedding * precision.weight
    return total, active


def kv_bytes_per_token_per_layer(model: Model, kv_precision: int) -> int:
    spec = model.attention
    if spec.kind == AttentionKind.MLA:
        kv_dim = spec.kv_lora_rank + (spec.rope_dim or 0)
        return 2 * kv_dim * kv_precision
    return 2 * spec.num_kv_heads * spec.head_dim * kv_precision


def _attention_gemm_params_per_token(model: Model) -> int:
    spec = model.attention
    if spec.kind == AttentionKind.MLA:
        return (
            model.hidden_size * spec.q_lora_rank
            + spec.q_lora_rank * spec.num_heads * spec.head_dim
            + model.hidden_size * (spec.kv_lora_rank + (spec.rope_dim or 0))
            + 2 * spec.kv_lora_rank * spec.num_heads * spec.head_dim
            + spec.o_lora_rank * model.hidden_size
        )
    q = model.hidden_size * spec.num_heads * spec.head_dim
    kv = 2 * model.hidden_size * spec.num_kv_heads * spec.head_dim
    out = spec.num_heads * spec.head_dim * model.hidden_size
    return q + kv + out


def _mlp_gemm_params_per_token(model: Model) -> int:
    hidden = model.hidden_size
    intermediate = model.mlp.intermediate_size
    if model.mlp.kind == MLPKind.DENSE:
        return 3 * hidden * intermediate
    active = model.mlp.active_experts * 3 * hidden * intermediate
    router = model.hidden_size * model.mlp.num_experts
    return active + router


def _vector_flops_per_token(model: Model) -> int:
    hidden = model.hidden_size
    layernorm = 8 * hidden
    if model.mlp.kind == MLPKind.DENSE:
        mlp = 3 * model.mlp.intermediate_size
    else:
        mlp = model.mlp.active_experts * model.mlp.intermediate_size
    return model.num_layers * (layernorm + mlp) + 8 * model.vocab_size


def _communication_bytes_per_token(
    model: Model,
    parallel,
    precision: int,
    training: bool,
) -> float:
    hidden = model.hidden_size
    factor = 2 * (parallel.tp - 1) / parallel.tp if parallel.tp > 1 else 0.0
    # One all-reduce after attention and one after MLP in each layer.
    tp_bytes = model.num_layers * 4 * hidden * precision * factor
    if model.mlp.kind == MLPKind.MOE and parallel.ep > 1:
        routed = model.mlp.topk
        tp_bytes += model.num_layers * 4 * routed * hidden * precision
    if parallel.pp > 1:
        tp_bytes += (parallel.pp - 1) * hidden * precision
    return tp_bytes


def _base_demand(
    model: Model,
    context_length: int,
    precision,
) -> Demand:
    total_weight, active_weight = model_weight_bytes(model, precision)
    cube_params = _attention_gemm_params_per_token(model) + _mlp_gemm_params_per_token(model)
    cube_flops = model.num_layers * 2 * cube_params
    if not model.tie_word_embeddings:
        cube_flops += 2 * model.hidden_size * model.vocab_size
    attention_context_flops = model.num_layers * 4 * context_length * model.attention.num_heads * model.attention.head_dim
    activation = model.num_layers * 4 * model.hidden_size * precision.activation
    kv = kv_bytes_per_token_per_layer(model, precision.kv_cache) * model.num_layers * context_length
    return Demand(
        cube_flops=cube_flops + attention_context_flops,
        vector_flops=_vector_flops_per_token(model),
        weight_bytes=total_weight,
        active_weight_bytes=active_weight,
        kv_cache_bytes=kv,
        activation_bytes=activation,
    )


def _demand_metrics(demand: Demand) -> dict:
    return {
        "total_gflops": demand.total_flops / 1e9,
        "cube_gflops": demand.cube_flops / 1e9,
        "vector_gflops": demand.vector_flops / 1e9,
        "cube_vector_ratio": demand.cube_vector_ratio,
        "total_memory_bytes": demand.total_memory_bytes,
        "compute_memory_ratio_flops_per_byte": demand.compute_memory_ratio,
        "compute_capacity_ratio_flops_per_byte": demand.compute_capacity_ratio,
        "compute_capacity_ratio_active_flops_per_byte": demand.compute_capacity_ratio_active,
    }


def analyze_inference(model: Model, workload: InferenceWorkload) -> dict:
    workload.parallel.validate()
    prefill = _base_demand(model, workload.prefill_length, workload.precision)
    decode = _base_demand(model, workload.context_length, workload.precision)
    prefill.communication_bytes = _communication_bytes_per_token(
        model, workload.parallel, workload.precision.activation, False
    )
    decode.communication_bytes = prefill.communication_bytes
    prefill_tokens = workload.prefill_batch_size * workload.prefill_length
    decode_step_tokens = workload.decode_batch_size
    prefill_total = Demand(**{key: value * prefill_tokens for key, value in asdict(prefill).items()})
    decode_total = Demand(**{key: value * decode_step_tokens for key, value in asdict(decode).items()})

    reports = [
        ScenarioReport(
            "prefill",
            "one full prefill batch",
            prefill_total,
            prefill,
            _demand_metrics(prefill),
        ),
        ScenarioReport(
            "decode",
            "one autoregressive decode step",
            decode_total,
            decode,
            _demand_metrics(decode),
        ),
    ]
    return {
        "model": model.name,
        "scenarios": [report.as_dict() for report in reports],
        "field_notes": json_field_notes(),
    }


def analyze_training(model: Model, workload: TrainingWorkload) -> dict:
    workload.parallel.validate()
    forward = _base_demand(model, workload.seq_len, workload.precision)
    # Backward and optimizer update add roughly 2x forward FLOPs.
    train = Demand(
        cube_flops=forward.cube_flops * 3,
        vector_flops=forward.vector_flops * 3,
        weight_bytes=forward.weight_bytes,
        active_weight_bytes=forward.active_weight_bytes,
        kv_cache_bytes=forward.kv_cache_bytes,
        activation_bytes=forward.activation_bytes,
    )
    total_params, _ = model_weight_params(model)
    train.gradient_bytes = total_params * workload.precision.gradient
    train.optimizer_bytes = total_params * optimizer_state_bytes_per_param(workload.optimizer)
    train.communication_bytes = _communication_bytes_per_token(
        model, workload.parallel, workload.precision.activation, True
    )
    if workload.parallel.dp > 1:
        factor = 2 * (workload.parallel.dp - 1) / workload.parallel.dp
        train.communication_bytes += train.gradient_bytes * factor
    if workload.gradient_checkpointing:
        train.activation_bytes *= model.num_layers**0.5 / model.num_layers
    tokens = workload.global_batch_size * workload.seq_len
    total = Demand(**{key: value * tokens for key, value in asdict(train).items()})
    return {
        "model": model.name,
        "scenarios": [
            ScenarioReport(
                "training_step",
                "one optimizer step with forward, backward and update",
                total,
                train,
                _demand_metrics(train),
            ).as_dict()
        ],
        "field_notes": json_field_notes(),
    }


def analyze_fuzzy(
    model: FuzzyModel,
    context_length: int,
    precision_weight: int,
    precision_kv: int,
    parallel,
) -> dict:
    parallel.validate()
    gemm_flops = 2 * model.active_params
    attention_flops = model.attention_flops_per_token_per_layer * context_length * model.num_layers
    vector_flops = model.vector_flops_per_token_per_layer * model.num_layers
    demand = Demand(
        cube_flops=gemm_flops + attention_flops,
        vector_flops=vector_flops,
        weight_bytes=model.total_params * precision_weight,
        active_weight_bytes=model.active_params * precision_weight,
        kv_cache_bytes=model.kv_bytes_per_token_per_layer * model.num_layers * context_length * precision_kv,
        communication_bytes=model.communication_bytes_per_token,
    )
    return {
        "model": model.name,
        "mode": "fuzzy",
        "scenarios": [
            ScenarioReport(
                "approximate_forward",
                "hardware-independent forward estimate for a fuzzy model",
                demand,
                demand,
                _demand_metrics(demand),
            ).as_dict()
        ],
        "field_notes": json_field_notes(),
    }


def summarize(reports: Iterable[dict]) -> dict:
    return {"reports": list(reports)}
