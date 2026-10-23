"""End-to-end theoretical load analysis for serial Agent interactions."""

from dataclasses import asdict, replace

from ..core import Demand, InferenceWorkload
from ..modeling import (
    _demand_as_json,
    _demand_metrics,
    _memory_value,
    inference_demands,
    model_weight_bytes,
)
from .schema import AgentWorkload


def _decode_total(demand: Demand, batch_size: int, output_length: int) -> Demand:
    sequence_tokens = batch_size * output_length
    result = Demand(
        **{
            key: value * sequence_tokens
            for key, value in asdict(demand).items()
            if key not in ("weight_bytes", "active_weight_bytes")
        }
    )
    result.weight_bytes = demand.weight_bytes * output_length
    result.active_weight_bytes = demand.active_weight_bytes * output_length
    return result


def _without_cache_load(demand: Demand) -> Demand:
    result = Demand(**asdict(demand))
    result.cache_load_bytes = 0.0
    return result


def analyze_agent(model, workload: AgentWorkload) -> dict:
    workload.parallel.validate()
    task_total = Demand()
    round_reports = []
    cache_load_charged = False
    prefill_cube_gflops = 0.0
    decode_cube_gflops = 0.0
    computed_prompt_tokens = 0

    for index, turn in enumerate(workload.turns):
        cache = turn.cache if turn.cache is not None else workload.cache
        if turn.cached_length is not None:
            cache = replace(cache, cached_length=turn.cached_length)
        if turn.cached_length is not None and turn.cached_length > turn.prompt_length - 1:
            raise ValueError(
                f"turn {turn.name!r} cached_length exceeds prompt_length - 1"
            )
        inference_workload = InferenceWorkload(
            prefill_batch_size=workload.batch_size,
            prefill_length=turn.prompt_length,
            decode_batch_size=workload.batch_size,
            decode_length=turn.output_length,
            context_length=turn.prompt_length,
            parallel=workload.parallel,
            precision=workload.precision,
            cache=cache,
        )
        prefill_total, decode_per_output_token, cache_metadata = inference_demands(
            model, inference_workload
        )
        output_tokens = workload.batch_size * turn.output_length
        decode_total = _decode_total(
            decode_per_output_token, workload.batch_size, turn.output_length
        )
        prefill_cube_gflops += prefill_total.cube_flops / 1e9
        decode_cube_gflops += decode_total.cube_flops / 1e9
        computed_prompt_tokens += cache_metadata["computed_prompt_tokens"]
        round_total = prefill_total + decode_total

        if workload.cache_load_scope == "none":
            round_total = _without_cache_load(round_total)
        elif workload.cache_load_scope == "once":
            if cache_load_charged:
                round_total = _without_cache_load(round_total)
            elif round_total.cache_load_bytes > 0:
                cache_load_charged = True

        task_total = task_total + round_total
        round_reports.append(
            {
                "index": index,
                "name": turn.name,
                "context_before_length": turn.context_before_length,
                "new_input_length": turn.new_input_length,
                "prompt_length": turn.prompt_length,
                "output_length": turn.output_length,
                "output_tokens": output_tokens,
                "cache": cache_metadata,
                "prefill_total": _demand_as_json(prefill_total),
                "decode_per_output_token": _demand_as_json(decode_per_output_token),
                "decode_total": _demand_as_json(decode_total),
                "total": _demand_as_json(round_total),
            }
        )

    round_count = len(workload.turns)
    metrics = _demand_metrics(task_total)
    metrics.update(
        {
            "round_count": round_count,
            "new_input_tokens": workload.new_input_tokens,
            "output_tokens": workload.output_tokens,
            "gflops_per_round": task_total.total_flops / 1e9 / round_count,
            "gflops_per_output_token": task_total.total_flops / 1e9 / workload.output_tokens,
            "data_movement_per_round": _memory_value(
                task_total.data_movement_bytes / round_count
            ),
            "data_movement_per_output_token": _memory_value(
                task_total.data_movement_bytes / workload.output_tokens
            ),
            "prefill_cube_gflops_per_computed_prompt_token": (
                prefill_cube_gflops / computed_prompt_tokens
            ),
            "prefill_cube_gflops_per_first_token": (
                prefill_cube_gflops / (workload.batch_size * round_count)
            ),
            "decode_cube_gflops_per_output_token": (
                decode_cube_gflops / workload.output_tokens
            ),
        }
    )

    total_weight_bytes, active_weight_bytes = model_weight_bytes(
        model, workload.precision
    )
    dag = _build_agent_dag(workload, round_reports)
    return {
        "model": model.name,
        "agent_scenario": workload.name,
        "scope": {
            "batch_size": workload.batch_size,
            "round_count": round_count,
            "new_input_tokens": workload.new_input_tokens,
            "output_tokens": workload.output_tokens,
            "cache_load_scope": workload.cache_load_scope,
            "weight_access_scope": "active_weight_per_decode_step",
        },
        "total": _demand_as_json(task_total),
        "capacity": {
            "weight_hbm": _memory_value(total_weight_bytes),
            "active_weight": _memory_value(active_weight_bytes),
        },
        "metrics": metrics,
        "rounds": round_reports,
        "dag": dag,
        "field_notes": {
            "agent_scenario": "Agent 任务名称；一个任务由多轮串行模型交互组成",
            "scope": "Agent 任务级别统计范围，包括 batch、轮数、输入 token 数、输出 token 数和 cache 计费范围",
            "total": "整个 Agent 任务端到端总需求，等于所有轮次 prefill 总需求与 decode 总需求之和",
            "metrics": "Agent 任务级核心指标；计算需求单位为 GFLOPs，数据搬移按 GB/TB 输出",
            "rounds": "每轮模型交互的明细；每轮包含当前完整 prompt 的 prefill 和 output_length 个 decode token",
            "prompt_length": "本轮 context_before_length + new_input_length",
            "prefill_total": "本轮从处理当前 prompt 到产出第一个输出 token 的端到端需求",
            "decode_per_output_token": "本轮平均输出单个 token 的需求，上下文按 output_length 求平均",
            "decode_total": "本轮 decode_per_output_token × batch_size × output_length",
            "round_total": "本轮 prefill_total + decode_total",
            "cache_load_scope": "cache 搬移计费方式：per_round 表示每轮计费，once 表示整个任务只计一次，none 表示不计入",
            "weight_access_scope": "decode 权重读放口径：按每个 decode step 读取一次激活权重，同一 batch 内多条序列共享，不按序列重复计费",
            "weight_load_semantics": "weight 字段表示 HBM 到计算单元的理论读放，不是每个 Agent 轮次从 host/remote 重新装载一份权重",
            "capacity": "模型常驻容量；weight_hbm 是全量权重 HBM 容量，active_weight 是单 token 激活权重容量下限，不随轮次累计",
            "prefill_cube_gflops_per_computed_prompt_token": "prefill 阶段平均每个实际参与计算的 prompt token 的 Cube 计算量；不是输出 token 数",
            "prefill_cube_gflops_per_first_token": "prefill 阶段平均每次产出 first token 的 Cube 计算量；等于每轮 prefill 的端到端计算需求",
            "decode_cube_gflops_per_output_token": "decode 阶段平均每个输出 token 的 Cube 计算量",
            "dag": "Agent 任务的有向无环图；节点包含 prefill、decode、source 和 sink，边表示上下文传递和 first token 依赖",
        },
    }


def _build_agent_dag(workload: AgentWorkload, round_reports: list[dict]) -> dict:
    nodes = [
        {
            "id": "agent_start",
            "type": "source",
            "label": "Agent start",
        }
    ]
    edges = []
    turn_ids = {}

    for index, turn in enumerate(workload.turns):
        prefill_id = f"turn_{index + 1}_prefill"
        decode_id = f"turn_{index + 1}_decode"
        turn_ids[turn.name] = decode_id
        round_report = round_reports[index]

        nodes.extend(
            [
                {
                    "id": prefill_id,
                    "type": "prefill",
                    "turn_index": index,
                    "turn_name": turn.name,
                    "prompt_length": turn.prompt_length,
                    "new_input_length": turn.new_input_length,
                    "computed_prompt_length": round_report["cache"][
                        "computed_prompt_length_per_sequence"
                    ],
                    "cache_hit_rate": round_report["cache"]["hit_rate"],
                    "demand": round_report["prefill_total"],
                },
                {
                    "id": decode_id,
                    "type": "decode",
                    "turn_index": index,
                    "turn_name": turn.name,
                    "output_length": turn.output_length,
                    "output_tokens": round_report["output_tokens"],
                    "demand": round_report["decode_total"],
                },
            ]
        )
        edges.append(
            {
                "from": prefill_id,
                "to": decode_id,
                "kind": "first_token",
            }
        )

        if index == 0:
            edges.append(
                {
                    "from": "agent_start",
                    "to": prefill_id,
                    "kind": "task_start",
                }
            )
        elif turn.depends_on:
            for dependency in turn.depends_on:
                edges.append(
                    {
                        "from": turn_ids[dependency],
                        "to": prefill_id,
                        "kind": "context",
                    }
                )
        else:
            edges.append(
                {
                    "from": f"turn_{index}_decode",
                    "to": prefill_id,
                    "kind": "context",
                }
            )

    nodes.append(
        {
            "id": "agent_end",
            "type": "sink",
            "label": "Agent end",
        }
    )
    edges.append(
        {
            "from": f"turn_{len(workload.turns)}_decode",
            "to": "agent_end",
            "kind": "task_end",
        }
    )

    return {
        "execution_order": [turn.name for turn in workload.turns],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
