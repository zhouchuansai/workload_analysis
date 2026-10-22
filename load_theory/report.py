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


def print_report(report: dict) -> None:
    print(f"Model: {report['model']}")
    for scenario in report["scenarios"]:
        print(f"\n[{scenario['name']}] {scenario['description']}")
        for metric in METRICS:
            value = scenario["metrics"][metric]
            print(f"  {metric:44s} {value:.6g}")
        print("  per-token demand:")
        for key, value in scenario["per_token"].items():
            print(f"    {key:28s} {value:.6g}")


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
