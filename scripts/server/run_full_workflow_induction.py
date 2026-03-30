#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from procmem2skills.inducer.workflow_export import export_grouped_workflows_json
from procmem2skills.recorder.jsonl import load_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch workflow induction over trajectories and export one JSON file grouped by task. "
            "Errors are discarded; success and failure attempts are retained."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input trajectories JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output grouped workflows JSON path.")
    parser.add_argument(
        "--terminal-like-max-events-per-segment",
        type=int,
        default=6,
        help="Maximum events per segment for terminal-like traces.",
    )
    parser.add_argument(
        "--induction-mode",
        choices=["rule", "llm", "hybrid"],
        default="hybrid",
        help="Workflow induction mode.",
    )
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model for workflow induction.")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base URL for chat completions.")
    parser.add_argument("--llm-api-key", type=str, default=None, help="Optional API key override for workflow induction.")
    parser.add_argument("--llm-timeout-sec", type=int, default=120, help="LLM request timeout seconds.")
    parser.add_argument("--llm-max-retries", type=int, default=1, help="LLM retry count on malformed outputs.")
    parser.add_argument(
        "--llm-non-strict",
        action="store_true",
        help="Relax strict checks (coverage/schema) for LLM outputs.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Write grouped workflow checkpoint every N retained attempts (0 disables).",
    )
    parser.add_argument(
        "--collection-target-k",
        type=int,
        default=0,
        help=(
            "Apply k-balanced task collection policy after induction: "
            "first evaluate first k attempts; all-success/all-failure stop at k, mixed continues to k success + k failure."
        ),
    )
    parser.add_argument(
        "--collection-metadata-output",
        type=Path,
        default=None,
        help="Optional output path for collection-policy metadata JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectories = load_trajectories(args.input)
    collection_target_k = max(0, int(args.collection_target_k))
    collection_metadata_output = args.collection_metadata_output
    if collection_target_k > 0 and collection_metadata_output is None:
        collection_metadata_output = args.output.with_suffix(".collection-metadata.json")
    summary = export_grouped_workflows_json(
        trajectories,
        args.output,
        terminal_like_max_events_per_segment=max(1, int(args.terminal_like_max_events_per_segment)),
        induction_mode=args.induction_mode,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout_sec=max(1, int(args.llm_timeout_sec)),
        llm_max_retries=max(0, int(args.llm_max_retries)),
        llm_strict=not bool(args.llm_non_strict),
        checkpoint_every=max(0, int(args.checkpoint_every)),
        collection_target_k=collection_target_k,
        collection_metadata_output_path=collection_metadata_output,
    )
    payload = {
        "input": str(args.input),
        "output": str(args.output),
        "collection_target_k": collection_target_k,
        "collection_metadata_output": str(collection_metadata_output) if collection_metadata_output else None,
        **summary,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
