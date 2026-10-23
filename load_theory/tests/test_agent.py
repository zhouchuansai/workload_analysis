from dataclasses import replace
from pathlib import Path

from load_theory.agent import analyze_agent, load_agent_scenario
from load_theory.agent.schema import AgentTurn, AgentWorkload
from load_theory.core import (
    AttentionKind,
    AttentionSpec,
    CacheSpec,
    MLPKind,
    MLPSpec,
    Model,
    Parallel,
    Precision,
)
from load_theory.modeling import model_weight_bytes


ROOT = Path(__file__).resolve().parents[2]


def dense_model() -> Model:
    return Model(
        name="agent-dense-test",
        hidden_size=64,
        num_layers=4,
        vocab_size=128,
        attention=AttentionSpec(
            kind=AttentionKind.GQA,
            num_heads=4,
            head_dim=16,
            num_kv_heads=2,
        ),
        mlp=MLPSpec(kind=MLPKind.DENSE, intermediate_size=128),
    )


def agent_workload(cache_load_scope: str = "per_round") -> AgentWorkload:
    precision = Precision(weight=2, activation=2, kv_cache=2)
    cache = CacheSpec(
        source="remote_node",
        load_required=True,
        entry_count=1,
        block_size=1,
        compression_ratio=1.0,
    )
    turns = tuple(
        AgentTurn(
            name=f"turn_{index}",
            new_input_length=1,
            output_length=1,
            context_before_length=16,
            cached_length=16,
        )
        for index in range(2)
    )
    return AgentWorkload(
        name="two-round-agent",
        batch_size=2,
        turns=turns,
        parallel=Parallel(),
        precision=precision,
        cache=cache,
        cache_load_scope=cache_load_scope,
    )


def test_agent_scenario_config_loads_multi_round_task():
    workload = load_agent_scenario(
        ROOT / "configs" / "agent_scenario.json",
        Parallel(tp=2),
        Precision(weight=1, activation=1, kv_cache=2),
    )
    assert workload.name == "tool_agent_4_round"
    assert len(workload.turns) == 4
    assert workload.new_input_tokens == 2048 + 512 + 1024 + 768
    assert workload.output_tokens == 256 + 128 + 192 + 160
    assert workload.cache_load_scope == "once"
    assert workload.cache.source == "remote_node"
    assert workload.turns[1].prompt_length == 2816
    assert workload.turns[1].cached_length == 2304


def test_agent_total_equals_round_totals():
    report = analyze_agent(dense_model(), agent_workload())
    round_total = sum(item["total"]["total_gflops"] for item in report["rounds"])
    assert report["scope"]["round_count"] == 2
    assert report["scope"]["new_input_tokens"] == 4
    assert report["scope"]["output_tokens"] == 4
    assert report["total"]["total_gflops"] == round_total
    assert report["metrics"]["gflops_per_output_token"] > 0
    assert report["field_notes"]["cache_load_scope"]


def test_agent_cache_load_scope_controls_cache_movement():
    per_round = analyze_agent(dense_model(), agent_workload("per_round"))
    once = analyze_agent(dense_model(), agent_workload("once"))
    none = analyze_agent(dense_model(), agent_workload("none"))

    per_round_round_sum = sum(
        item["total"]["cache_load"]["value"] for item in per_round["rounds"]
    )
    once_round_sum = sum(
        item["total"]["cache_load"]["value"] for item in once["rounds"]
    )
    none_round_sum = sum(
        item["total"]["cache_load"]["value"] for item in none["rounds"]
    )

    assert per_round_round_sum > 0
    assert per_round_round_sum == 2 * once_round_sum
    assert none_round_sum == 0
    assert none["total"]["cache_load"]["value"] == 0


def test_agent_cached_context_reduces_prefill_compute():
    model = dense_model()
    workload = agent_workload("none")
    cached = analyze_agent(model, workload)
    uncached = analyze_agent(model, replace(workload, turns=tuple(
        AgentTurn(
            name=turn.name,
            new_input_length=turn.new_input_length,
            output_length=turn.output_length,
            context_before_length=turn.context_before_length,
            cached_length=0,
        )
        for turn in workload.turns
    )))
    assert cached["rounds"][0]["prefill_total"]["total_gflops"] < uncached["rounds"][0]["prefill_total"]["total_gflops"]


def test_agent_decode_weight_reads_are_shared_by_batch():
    model = dense_model()
    workload = agent_workload("none")
    workload = replace(
        workload,
        turns=(workload.turns[0],),
    )
    report = analyze_agent(model, workload)
    total_weight, active_weight = model_weight_bytes(model, workload.precision)
    expected_weight_bytes = total_weight + active_weight * workload.turns[0].output_length
    assert report["scope"]["weight_access_scope"] == "active_weight_per_decode_step"
    assert (
        report["rounds"][0]["total"]["weight"]["value"] * 1e9
        == expected_weight_bytes
    )


def test_agent_report_contains_capacity_prefill_metrics_and_dag():
    report = analyze_agent(dense_model(), agent_workload())

    assert report["capacity"]["weight_hbm"]["value"] > 0
    assert report["capacity"]["active_weight"]["value"] > 0
    assert report["metrics"]["prefill_cube_gflops_per_computed_prompt_token"] > 0
    assert report["metrics"]["prefill_cube_gflops_per_first_token"] > 0
    assert report["metrics"]["decode_cube_gflops_per_output_token"] > 0
    assert report["dag"]["node_count"] == 2 * len(report["rounds"]) + 2
    assert report["dag"]["edge_count"] == 2 * len(report["rounds"]) + 1
    assert report["dag"]["nodes"][0]["type"] == "source"
    assert report["dag"]["nodes"][-1]["type"] == "sink"


def test_agent_explicit_dependencies_create_dag_edges():
    workload = agent_workload()
    first = workload.turns[0]
    second = workload.turns[1]
    turns = (
        first,
        second,
        AgentTurn(
            name="fan_in",
            new_input_length=1,
            output_length=1,
            context_before_length=16,
            cached_length=16,
            depends_on=(first.name, second.name),
        ),
    )
    report = analyze_agent(dense_model(), replace(workload, turns=turns))
    fan_in_edges = [
        edge
        for edge in report["dag"]["edges"]
        if edge["to"] == "turn_3_prefill"
    ]
    assert {edge["from"] for edge in fan_in_edges} == {
        "turn_1_decode",
        "turn_2_decode",
    }
