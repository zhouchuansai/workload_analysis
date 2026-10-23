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
    per_first_token: Demand | None = None
    cache: dict | None = None

    def as_dict(self) -> dict:
        result = {
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
        }
        if self.name == "prefill":
            result["total"] = _demand_as_json(self.total)
            if self.cache is not None:
                result["cache"] = self.cache
        elif self.name == "decode":
            result["per_output_token"] = _demand_as_json(self.per_token)
        else:
            result["per_token"] = _demand_as_json(self.per_token)
        return result


def _memory_value(byte_count: float) -> dict:
    use_tb = byte_count >= 1e12
    divisor = 1e12 if use_tb else 1e9
    unit = "TB" if use_tb else "GB"
    return {"value": byte_count / divisor, "unit": unit}


def _demand_as_json(demand: Demand) -> dict:
    result = {
        "cube_gflops": demand.cube_flops / 1e9,
        "vector_gflops": demand.vector_flops / 1e9,
        "total_gflops": demand.total_flops / 1e9,
        "weight": _memory_value(demand.weight_bytes),
        "active_weight": _memory_value(demand.active_weight_bytes),
        "kv_cache": _memory_value(demand.kv_cache_bytes),
        "activation": _memory_value(demand.activation_bytes),
        "communication": _memory_value(demand.communication_bytes),
        "optimizer_state": _memory_value(demand.optimizer_bytes),
        "gradient_state": _memory_value(demand.gradient_bytes),
        "cache_load": _memory_value(demand.cache_load_bytes),
        "data_movement": _memory_value(demand.data_movement_bytes),
    }
    return result


def json_field_notes() -> dict:
    return {
        "name": "场景名称：prefill、decode 或 training_step",
        "description": "场景对应的计算批次说明",
        "total": "prefill 场景从处理整段 prompt 到产出 first token 的端到端需求",
        "per_output_token": "decode 场景平均输出单个 token 的需求；上下文长度按 decode_length 求平均",
        "per_token": "training 场景按训练 token 归一化后的需求",
        "prefill_total_only": "prefill 场景仅输出端到端 total 口径，不输出 prompt token 或 first token 归一化字段",
        "cache": "prefill prompt 前缀 KV cache 命中信息，包括命中率、来源、加载策略、命中长度、需计算长度和 cache_load",
        "metrics": "核心派生指标；比值均为无量纲或 FLOPs/byte",
        "cube_gflops": "Cube 单元理论计算需求，10^9 FLOPs；主要覆盖 GEMM/QK^T/AV 等矩阵乘",
        "vector_gflops": "Vector 单元理论计算需求，10^9 元素级操作数；主要覆盖 Norm/激活/Softmax 相关操作",
        "total_gflops": "Cube GFLOPs + Vector GFLOPs",
        "weight": "理论权重读放量；prefill 为一次 forward 的权重访问上界，decode 为激活权重读放量；不是从 host/remote 装载到 HBM 的 cold-load；对象包含 value 和 unit",
        "active_weight": "单 token 实际激活的权重大小；对象包含 value 和 unit，低于 1TB 用 GB，达到 1TB 用 TB",
        "kv_cache": "KV cache 访存量；decode 为读放量，prefill 为 causal 读放量 + 全量写入量；对象包含 value 和 unit",
        "activation": "激活值访存量估计；对象包含 value 和 unit",
        "communication": "TP/EP/PP/DP 理论通信字节数；对象包含 value 和 unit",
        "optimizer_state": "训练 optimizer state 大小；AdamW 默认 FP32 master weight + m/v，共 12 bytes/param；对象包含 value 和 unit",
        "gradient_state": "训练梯度状态大小；对象包含 value 和 unit",
        "cache_load": "从 host/peer/remote/disk 等位置搬入 HBM 的 prompt KV cache 字节数；HBM 常驻时为 0；对象包含 value 和 unit",
        "data_movement": "weight + KV + activation + communication + cache_load 的理论数据搬移量，不是 HBM 常驻容量；对象包含 value 和 unit",
        "total_gflops_metric": "核心指标中的总计算需求，单位 GFLOPs",
        "cube_vector_ratio": "Cube 计算需求 / Vector 计算需求；越大表示越依赖矩阵算力",
        "compute_memory_ratio_flops_per_byte": "计算访存比 = total FLOPs / 理论数据搬移 bytes",
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
    return model.num_layers * (layernorm + mlp)


def _lm_head_vector_flops(model: Model) -> int:
    return 8 * model.vocab_size


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
    include_lm_head: bool = True,
) -> Demand:
    total_weight, active_weight = model_weight_bytes(model, precision)
    cube_params = _attention_gemm_params_per_token(model) + _mlp_gemm_params_per_token(model)
    cube_flops = model.num_layers * 2 * cube_params
    if include_lm_head and not model.tie_word_embeddings:
        cube_flops += 2 * model.hidden_size * model.vocab_size
    attention_context_flops = model.num_layers * 4 * context_length * model.attention.num_heads * model.attention.head_dim
    activation = model.num_layers * 4 * model.hidden_size * precision.activation
    kv = kv_bytes_per_token_per_layer(model, precision.kv_cache) * model.num_layers * context_length
    return Demand(
        cube_flops=cube_flops + attention_context_flops,
        vector_flops=_vector_flops_per_token(model)
        + (_lm_head_vector_flops(model) if include_lm_head else 0),
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
        "data_movement": _memory_value(demand.data_movement_bytes),
        "compute_memory_ratio_flops_per_byte": demand.compute_memory_ratio,
        "compute_capacity_ratio_flops_per_byte": demand.compute_capacity_ratio,
        "compute_capacity_ratio_active_flops_per_byte": demand.compute_capacity_ratio_active,
    }


def inference_demands(model: Model, workload: InferenceWorkload) -> tuple[Demand, Demand, dict]:
    """Return prefill total, decode per output token, and cache metadata."""
    workload.parallel.validate()
    prefill_per_prompt_token = _base_demand(
        model, 0, workload.precision, include_lm_head=False
    )
    average_decode_context_length = (
        workload.context_length + (workload.decode_length - 1) / 2
    )
    decode = _base_demand(model, average_decode_context_length, workload.precision)
    decode.weight_bytes = decode.active_weight_bytes
    prefill_per_prompt_token.communication_bytes = _communication_bytes_per_token(
        model, workload.parallel, workload.precision.activation, False
    )
    decode.communication_bytes = prefill_per_prompt_token.communication_bytes
    prompt_length = workload.prefill_length
    batch_size = workload.prefill_batch_size
    raw_cached_prompt_length = (
        workload.cache.cached_length
        if workload.cache.cached_length is not None
        else int((prompt_length - 1) * workload.cache.hit_rate)
    )
    if raw_cached_prompt_length > prompt_length - 1:
        raise ValueError("cached_length must be less than or equal to prompt_length - 1")
    block_size = workload.cache.block_size
    cached_prompt_length = raw_cached_prompt_length // block_size * block_size
    computed_prompt_length = prompt_length - cached_prompt_length
    computed_prompt_tokens = batch_size * computed_prompt_length
    decode_step_tokens = workload.decode_batch_size

    heads = model.attention.num_heads
    head_dim = model.attention.head_dim
    causal_context_positions = (
        computed_prompt_length * cached_prompt_length
        + computed_prompt_length * (computed_prompt_length + 1) / 2
    )
    causal_attention_flops = (
        batch_size
        * 4
        * model.num_layers
        * heads
        * head_dim
        * causal_context_positions
    )
    lm_head_cube_flops = 0 if model.tie_word_embeddings else 2 * model.hidden_size * model.vocab_size
    kv_bytes_per_layer = kv_bytes_per_token_per_layer(model, workload.precision.kv_cache)
    kv_write_bytes = batch_size * computed_prompt_length * model.num_layers * kv_bytes_per_layer
    kv_read_bytes = (
        batch_size
        * model.num_layers
        * kv_bytes_per_layer
        * causal_context_positions
    )
    cache_entry_count = workload.cache.entry_count
    if cache_entry_count is None:
        cache_entry_count = batch_size
    cache_load_bytes = 0.0
    if workload.cache.load_required:
        cache_load_bytes = (
            cache_entry_count
            * cached_prompt_length
            * model.num_layers
            * kv_bytes_per_layer
            * workload.cache.compression_ratio
        )

    prefill_total = Demand(
        **{
            key: value * computed_prompt_tokens
            for key, value in asdict(prefill_per_prompt_token).items()
        }
    )
    prefill_total.cube_flops += causal_attention_flops + batch_size * lm_head_cube_flops
    prefill_total.vector_flops += batch_size * _lm_head_vector_flops(model)
    prefill_total.kv_cache_bytes = kv_write_bytes + kv_read_bytes
    prefill_total.cache_load_bytes = cache_load_bytes
    prefill_total.weight_bytes = prefill_per_prompt_token.weight_bytes
    prefill_total.active_weight_bytes = prefill_per_prompt_token.active_weight_bytes

    prefill_per_first_token = Demand(
        **{key: value / batch_size for key, value in asdict(prefill_total).items()}
    )
    prefill_per_first_token.weight_bytes = prefill_total.weight_bytes
    prefill_per_first_token.active_weight_bytes = prefill_total.active_weight_bytes

    decode_total = Demand(
        **{
            key: value * decode_step_tokens
            for key, value in asdict(decode).items()
            if key not in ("weight_bytes", "active_weight_bytes")
        }
    )
    decode_total.weight_bytes = decode.weight_bytes * max(1, decode_step_tokens)
    decode_total.active_weight_bytes = decode.active_weight_bytes * max(
        1, decode_step_tokens
    )

    cache_metadata = {
        "hit_rate": workload.cache.hit_rate,
        "source": workload.cache.source,
        "load_required": workload.cache.load_required,
        "entry_count": cache_entry_count,
        "block_size": workload.cache.block_size,
        "compression_ratio": workload.cache.compression_ratio,
        "cached_prompt_length_per_sequence": cached_prompt_length,
        "computed_prompt_length_per_sequence": computed_prompt_length,
        "cached_prompt_tokens": batch_size * cached_prompt_length,
        "computed_prompt_tokens": computed_prompt_tokens,
        "cache_load": _memory_value(cache_load_bytes),
    }
    return prefill_total, decode, cache_metadata


def analyze_inference(model: Model, workload: InferenceWorkload) -> dict:
    prefill_total, decode, cache_metadata = inference_demands(model, workload)
    reports = [
        ScenarioReport(
            "prefill",
            "end-to-end prompt processing through first output token for one batch",
            prefill_total,
            prefill_total,
            _demand_metrics(prefill_total),
            None,
            cache_metadata,
        ),
        ScenarioReport(
            "decode",
            "average demand per output token across decode_length autoregressive steps",
            Demand(**{key: value * workload.decode_batch_size for key, value in asdict(decode).items()}),
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
