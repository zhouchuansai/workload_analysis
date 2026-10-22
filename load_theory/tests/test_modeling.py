import pathlib

from load_theory.adapters import load_deepseek_config
from load_theory.core import (
    AttentionKind,
    AttentionSpec,
    InferenceWorkload,
    MLPSpec,
    MLPKind,
    Model,
    Parallel,
    Precision,
    TrainingWorkload,
)
from load_theory.modeling import analyze_inference, analyze_training, model_weight_params
from load_theory.core import optimizer_state_bytes_per_param
from load_theory.scenario import load_scenario


def dense_model() -> Model:
    return Model(
        name="dense-test",
        hidden_size=64,
        num_layers=4,
        vocab_size=128,
        attention=AttentionSpec(kind=AttentionKind.GQA, num_heads=4, head_dim=16, num_kv_heads=2),
        mlp=MLPSpec(kind=MLPKind.DENSE, intermediate_size=128),
    )


def test_dense_weights_and_forward():
    model = dense_model()
    total, active = model_weight_params(model)
    assert total == active
    report = analyze_inference(
        model,
        InferenceWorkload(1, 16, 1, 4, 32, Parallel(tp=2), Precision(weight=2, kv_cache=2)),
    )
    decode = report["scenarios"][1]["per_token"]
    assert decode["active_weight_bytes"] == total * 2
    assert decode["cube_gflops"] > decode["vector_gflops"]


def test_moe_active_weight_is_smaller_than_total():
    model = Model(
        name="moe-test",
        hidden_size=64,
        num_layers=2,
        vocab_size=128,
        attention=AttentionSpec(kind=AttentionKind.GQA, num_heads=4, head_dim=16, num_kv_heads=2),
        mlp=MLPSpec(
            kind=MLPKind.MOE,
            intermediate_size=128,
            num_experts=16,
            topk=2,
            num_shared_experts=1,
            expert_precision_bytes=0.5,
        ),
    )
    report = analyze_inference(
        model,
        InferenceWorkload(1, 16, 1, 4, 32, Parallel(tp=1, ep=4), Precision(weight=2, kv_cache=2)),
    )
    decode = report["scenarios"][1]["per_token"]
    assert decode["active_weight_bytes"] < decode["weight_bytes"]
    assert decode["communication_bytes"] > 0


def test_training_has_optimizer_and_gradient_bytes():
    report = analyze_training(
        dense_model(),
        TrainingWorkload(8, 128, parallel=Parallel(dp=2), precision=Precision(weight=2)),
    )
    per_token = report["scenarios"][0]["per_token"]
    assert per_token["gradient_bytes"] > 0
    assert per_token["optimizer_bytes"] > 0
    assert report["scenarios"][0]["metrics"]["total_gflops"] > 0
    assert "field_notes" in report


def test_deepseek_adapter_loads_local_config():
    import pathlib

    config = (
        pathlib.Path(__file__).resolve().parents[2]
        / "models"
        / "deepseek-v4.1"
        / "config.json"
    )
    model, warnings = load_deepseek_config(config)
    assert model.attention.kind == AttentionKind.MLA
    assert model.mlp.kind == MLPKind.MOE
    assert model.mlp.expert_precision_bytes == 0.5
    assert warnings


def test_scenario_config_is_loaded_and_applied():
    config = (
        pathlib.Path(__file__).resolve().parents[2]
        / "configs"
        / "scenario.json"
    )
    scenario = load_scenario(config)
    assert scenario.name == "deepseek_v41_default"
    assert scenario.mode == "both"
    assert scenario.optimizer == "adamw"
    assert scenario.precision.weight == 1
    assert scenario.precision.activation == 1
    assert scenario.precision.kv_cache == 2
    assert scenario.precision.gradient == 2
    assert scenario.parallel == Parallel(tp=8, ep=64, pp=8, dp=16)
    assert scenario.expert_precision_bytes == 0.5
    assert optimizer_state_bytes_per_param("adamw") == 12

    model, _ = load_deepseek_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "models"
        / "deepseek-v4.1"
        / "config.json"
    )
    updated = scenario.apply_expert_precision(model)
    assert updated.mlp.expert_precision_bytes == 0.5
