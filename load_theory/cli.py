"""Command-line interface for the load theory framework."""

import argparse
from dataclasses import replace
from pathlib import Path

from .adapters import load_deepseek_config
from .core import (
    CACHE_SOURCES,
    CacheSpec,
    InferenceWorkload,
    Parallel,
    Precision,
    TrainingWorkload,
)
from .modeling import analyze_inference, analyze_training
from .scenario import load_scenario
from .report import print_report, save_json
from .agent import analyze_agent, load_agent_scenario, print_agent_report


def _parallel(value: str) -> Parallel:
    parts = {item.split("=")[0]: int(item.split("=")[1]) for item in value.split(",")}
    return Parallel(**{key: parts.get(key, 1) for key in ("tp", "ep", "pp", "dp")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Model load theory calculator for Ascend-style Cube/Vector systems")
    parser.add_argument("config", help="DeepSeek-style config.json")
    parser.add_argument("--scenario", type=Path, help="scenario.json containing workload, parallel and precision")
    parser.add_argument("--mode", choices=("inference", "training", "agent", "both"))
    parser.add_argument(
        "--agent-scenario",
        type=Path,
        help="agent scenario JSON containing serial multi-round interactions",
    )
    parser.add_argument("--prefill-batch", type=int)
    parser.add_argument("--prefill-length", type=int)
    parser.add_argument("--decode-batch", type=int)
    parser.add_argument("--decode-length", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--cache-hit-rate", type=float)
    parser.add_argument("--cache-source", choices=sorted(CACHE_SOURCES))
    parser.add_argument(
        "--cache-load-required",
        action=argparse.BooleanOptionalAction,
        help="load cached KV into HBM; use --no-cache-load-required to disable",
    )
    parser.add_argument("--cache-entry-count", type=int)
    parser.add_argument("--cache-block-size", type=int)
    parser.add_argument("--cache-compression-ratio", type=float)
    parser.add_argument("--train-batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--parallel", type=_parallel)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    model, warnings = load_deepseek_config(args.config)
    scenario = load_scenario(args.scenario) if args.scenario else None
    if scenario is not None:
        model = scenario.apply_expert_precision(model)
        print(f"Scenario: {scenario.name}")
        print(f"Optimizer: {scenario.optimizer}")
    print(f"Loaded: {model.name}")
    for warning in warnings:
        print(f"Note: {warning}")

    if scenario is not None:
        precision = scenario.precision
        parallel = scenario.parallel
        mode = scenario.mode
        inference = scenario.inference
        training = scenario.training
        optimizer = scenario.optimizer
    else:
        precision = Precision(weight=1, activation=1, kv_cache=2, gradient=2)
        parallel = Parallel()
        mode = "both"
        optimizer = "adamw"
        from .scenario import InferenceScenario, TrainingScenario

        inference = InferenceScenario(8, 4096, 128, 512, 8192)
        training = TrainingScenario(512, 8192, True)

    mode = args.mode or mode
    parallel = args.parallel or parallel
    inference_values = (
        args.prefill_batch if args.prefill_batch is not None else inference.prefill_batch_size,
        args.prefill_length if args.prefill_length is not None else inference.prefill_length,
        args.decode_batch if args.decode_batch is not None else inference.decode_batch_size,
        args.decode_length if args.decode_length is not None else inference.decode_length,
        args.context_length if args.context_length is not None else inference.context_length,
    )
    cache = inference.cache
    if args.cache_hit_rate is not None:
        cache = replace(cache, hit_rate=args.cache_hit_rate)
    if args.cache_source is not None:
        cache = replace(cache, source=args.cache_source)
        if args.cache_load_required is None:
            cache = replace(cache, load_required=args.cache_source != "hbm")
    if args.cache_load_required is not None:
        cache = replace(cache, load_required=args.cache_load_required)
    if args.cache_entry_count is not None:
        cache = replace(cache, entry_count=args.cache_entry_count)
    if args.cache_block_size is not None:
        cache = replace(cache, block_size=args.cache_block_size)
    if args.cache_compression_ratio is not None:
        cache = replace(cache, compression_ratio=args.cache_compression_ratio)
    training = (
        args.train_batch if args.train_batch is not None else training.global_batch_size,
        args.seq_len if args.seq_len is not None else training.seq_len,
        training.gradient_checkpointing,
    )

    reports = []
    if mode == "agent":
        if args.agent_scenario is None:
            parser.error("--agent-scenario is required when --mode agent")
        agent_workload = load_agent_scenario(args.agent_scenario, parallel, precision)
        report = analyze_agent(model, agent_workload)
        report["scenario"] = scenario.name if scenario else "cli_default"
        report["optimizer"] = optimizer
        report["warnings"] = warnings
        reports.append(report)
    elif mode in ("inference", "both"):
        workload = InferenceWorkload(
            prefill_batch_size=inference_values[0],
            prefill_length=inference_values[1],
            decode_batch_size=inference_values[2],
            decode_length=inference_values[3],
            context_length=inference_values[4],
            parallel=parallel,
            precision=precision,
            cache=cache,
        )
        report = analyze_inference(model, workload)
        report["scenario"] = scenario.name if scenario else "cli_default"
        report["optimizer"] = optimizer
        report["warnings"] = warnings
        reports.append(report)
    if mode in ("training", "both"):
        workload = TrainingWorkload(
            global_batch_size=training[0],
            seq_len=training[1],
            optimizer=optimizer,
            gradient_checkpointing=training[2],
            parallel=parallel,
            precision=precision,
        )
        report = analyze_training(model, workload)
        report["scenario"] = scenario.name if scenario else "cli_default"
        report["optimizer"] = optimizer
        report["warnings"] = warnings
        reports.append(report)
    for report in reports:
        print()
        if mode == "agent":
            print_agent_report(report)
        else:
            print_report(report)
    if args.json:
        save_json(reports, args.json)


if __name__ == "__main__":
    main()
