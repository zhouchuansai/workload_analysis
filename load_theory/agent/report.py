"""Console rendering for Agent end-to-end reports."""

from ..report import _format_value


def print_agent_report(report: dict) -> None:
    print(f"Model: {report['model']}")
    print(f"Agent scenario: {report['agent_scenario']}")
    print("Task scope:")
    for key, value in report["scope"].items():
        print(f"  {key:24s} {value}")
    print("End-to-end metrics:")
    for key, value in report["metrics"].items():
        print(f"  {key:44s} {_format_value(value)}")
    print("End-to-end total demand:")
    for key, value in report["total"].items():
        print(f"  {key:28s} {_format_value(value)}")
    for round_report in report["rounds"]:
        print(
            f"\n[round {round_report['index'] + 1}: {round_report['name']}] "
            f"prompt={round_report['prompt_length']} output={round_report['output_length']}"
        )
        for key, value in round_report["total"].items():
            print(f"  {key:28s} {_format_value(value)}")
