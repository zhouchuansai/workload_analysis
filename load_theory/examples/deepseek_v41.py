"""Run the local DeepSeek V4.1 example and save a JSON report."""

from pathlib import Path

from ..adapters import load_deepseek_config
from ..core import InferenceWorkload, TrainingWorkload
from ..modeling import analyze_inference, analyze_training
from ..report import print_report, save_json
from ..scenario import load_scenario

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "models" / "deepseek-v4.1" / "config.json"
SCENARIO = ROOT / "configs" / "scenario.json"


def main() -> None:
    model, warnings = load_deepseek_config(CONFIG)
    scenario = load_scenario(SCENARIO)
    model = scenario.apply_expert_precision(model)
    print(f"Loaded {model.name}")
    print(f"Scenario: {scenario.name}")
    print("\n".join(f"Note: {item}" for item in warnings))
    reports = [
        analyze_inference(
            model,
            InferenceWorkload(
                prefill_batch_size=scenario.inference.prefill_batch_size,
                prefill_length=scenario.inference.prefill_length,
                decode_batch_size=scenario.inference.decode_batch_size,
                decode_length=scenario.inference.decode_length,
                context_length=scenario.inference.context_length,
                parallel=scenario.parallel,
                precision=scenario.precision,
            ),
        ),
        analyze_training(
            model,
            TrainingWorkload(
                global_batch_size=scenario.training.global_batch_size,
                seq_len=scenario.training.seq_len,
                optimizer=scenario.optimizer,
                gradient_checkpointing=scenario.training.gradient_checkpointing,
                parallel=scenario.parallel,
                precision=scenario.precision,
            ),
        ),
    ]
    for report in reports:
        report["scenario"] = scenario.name
        report["optimizer"] = scenario.optimizer
    for report in reports:
        print()
        print_report(report)
    save_json(reports, ROOT / "reports" / "deepseek_v41_load_report.json")


if __name__ == "__main__":
    main()
