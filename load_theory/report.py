"""Human-readable and machine-readable report helpers."""

import json
from pathlib import Path


METRICS = [
    "total_gflops",
    "cube_gflops",
    "vector_gflops",
    "cube_vector_ratio",
    "compute_memory_ratio_flops_per_byte",
    "compute_capacity_ratio_flops_per_byte",
    "compute_capacity_ratio_active_flops_per_byte",
]


def _format_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "value" in value and "unit" in value:
        return f"{value['value']:.6g} {value['unit']}"
    return f"{value:.6g}"


def print_report(report: dict) -> None:
    print(f"Model: {report['model']}")
    for scenario in report["scenarios"]:
        print(f"\n[{scenario['name']}] {scenario['description']}")
        for metric in METRICS:
            value = scenario["metrics"][metric]
            print(f"  {metric:44s} {_format_value(value)}")
        if "total" in scenario:
            print("  end-to-end total demand:")
            for key, value in scenario["total"].items():
                print(f"    {key:28s} {_format_value(value)}")
            if "cache" in scenario:
                print("  cache:")
                for key, value in scenario["cache"].items():
                    print(f"    {key:28s} {_format_value(value)}")
        elif "per_output_token" in scenario:
            print("  per-output-token demand:")
            for key, value in scenario["per_output_token"].items():
                print(f"    {key:28s} {_format_value(value)}")
        else:
            print("  per-token demand:")
            for key, value in scenario["per_token"].items():
                print(f"    {key:28s} {_format_value(value)}")


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
