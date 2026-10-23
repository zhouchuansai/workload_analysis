"""Convert a Claude Code JSONL trace into an Agent workload scenario."""

import argparse
import json
from pathlib import Path


DEFAULT_TRACE = (
    Path.home()
    / ".claude/projects/-Users-zhouchuansai-Desktop-code-agents-workloads/"
    / "9d9bf09c-d8ad-428e-9e06-1cd2c60ebbf6.jsonl"
)


def _int_usage(value, field_name: str) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError(f"usage.{field_name} must be non-negative")
    return result


def convert_trace(
    trace_path: Path,
    name: str,
    batch_size: int,
) -> tuple[dict, dict]:
    turns = []
    skipped = {
        "non_assistant": 0,
        "sidechain": 0,
        "missing_usage": 0,
        "non_positive_input": 0,
        "non_positive_output": 0,
    }
    input_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    output_tokens = 0

    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {trace_path}:{line_number}: {error}"
                ) from error

            if event.get("type") != "assistant":
                skipped["non_assistant"] += 1
                continue
            if event.get("isSidechain"):
                skipped["sidechain"] += 1
                continue

            message = event.get("message") or {}
            usage = message.get("usage")
            if not isinstance(usage, dict):
                skipped["missing_usage"] += 1
                continue

            non_cached_input = _int_usage(
                usage.get("input_tokens"), "input_tokens"
            )
            cache_creation = _int_usage(
                usage.get("cache_creation_input_tokens"),
                "cache_creation_input_tokens",
            )
            cache_read = _int_usage(
                usage.get("cache_read_input_tokens"),
                "cache_read_input_tokens",
            )
            output = _int_usage(usage.get("output_tokens"), "output_tokens")

            new_input_length = non_cached_input + cache_creation
            if new_input_length <= 0:
                skipped["non_positive_input"] += 1
                continue
            if output <= 0:
                skipped["non_positive_output"] += 1
                continue

            input_tokens += non_cached_input
            cache_creation_tokens += cache_creation
            cache_read_tokens += cache_read
            output_tokens += output

            prompt_length = cache_read + new_input_length
            if prompt_length <= cache_read:
                raise ValueError(
                    f"invalid prompt length at {trace_path}:{line_number}"
                )

            turns.append(
                {
                    "name": f"model_call_{len(turns) + 1:04d}",
                    "new_input_length": new_input_length,
                    "output_length": output,
                    "context_before_length": cache_read,
                    "cached_length": cache_read,
                }
            )

    if not turns:
        raise ValueError(f"no valid assistant model calls found in {trace_path}")

    metadata = {
        "source_trace": str(trace_path),
        "model_call_count": len(turns),
        "input_tokens": input_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "skipped_events": skipped,
        "conversion": {
            "new_input_length": "input_tokens + cache_creation_input_tokens",
            "context_before_length": "cache_read_input_tokens",
            "cached_length": "cache_read_input_tokens",
            "output_length": "output_tokens",
        },
    }

    workload = {
        "name": name,
        "batch_size": batch_size,
        "cache_load_scope": "none",
        "cache": {
            "hit_rate": 0.0,
            "source": "hbm",
            "load_required": False,
            "entry_count": batch_size,
            "block_size": 1,
            "compression_ratio": 1.0,
        },
        "turns": turns,
    }
    return workload, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("claude_trace_agent.json"),
    )
    parser.add_argument("--name", default="claude_code_trace")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    workload, metadata = convert_trace(args.input, args.name, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(workload['turns'])} model calls to {args.output}")
    print(f"Wrote conversion metadata to {metadata_path}")
    print(
        "Tokens: "
        f"input={metadata['input_tokens']}, "
        f"cache_creation={metadata['cache_creation_tokens']}, "
        f"cache_read={metadata['cache_read_tokens']}, "
        f"output={metadata['output_tokens']}"
    )


if __name__ == "__main__":
    main()
