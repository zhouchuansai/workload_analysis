"""Run the DeepSeek multi-round Agent example and save a JSON report."""

from pathlib import Path

from ..adapters import load_deepseek_config
from ..report import save_json
from ..scenario import load_scenario
from . import analyze_agent, load_agent_scenario, print_agent_report


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "models" / "deepseek-v4.1" / "config.json"
SCENARIO_CONFIG = ROOT / "configs" / "scenario.json"
AGENT_CONFIG = ROOT / "configs" / "agent_scenario.json"


def main() -> None:
    model, warnings = load_deepseek_config(MODEL_CONFIG)
    scenario = load_scenario(SCENARIO_CONFIG)
    model = scenario.apply_expert_precision(model)
    workload = load_agent_scenario(
        AGENT_CONFIG,
        scenario.parallel,
        scenario.precision,
    )
    report = analyze_agent(model, workload)
    report["scenario"] = scenario.name
    report["optimizer"] = scenario.optimizer
    report["warnings"] = warnings
    print_agent_report(report)
    save_json([report], ROOT / "reports" / "deepseek_agent_load_report.json")


if __name__ == "__main__":
    main()
