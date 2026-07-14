# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CLI helpers for Triton comm tuning entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from comm_tuning import artifacts, engine, selection
from comm_tuning.adapter import CommKernelTuningAdapter


def run_tuning_cli(adapter: CommKernelTuningAdapter) -> None:
    parser = argparse.ArgumentParser(description=f"Tune {adapter.name}.")
    parser.add_argument("--mode", choices=("parent", "child", "select"), required=True)
    parser.add_argument("--work-item-json", default=None)
    parser.add_argument("--rank-output-path", default=None)
    parser.add_argument("--run-name", default="local")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--base-master-port", type=int, default=31000)
    parser.add_argument("--master-port-stride", type=int, default=100)
    parser.add_argument(
        "--candidate-port-mode",
        choices=("dynamic", "deterministic"),
        default="dynamic",
    )
    parser.add_argument("--candidate-timeout-sec", type=int, default=180)
    parser.add_argument("--work-item-timeout-sec", type=int, default=None)
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--pg-timeout-sec", type=int, default=120)
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--torchx",
        action="store_true",
        help="Accepted for compatibility with ops/mast.py.",
    )
    adapter.add_cli_args(parser)
    args = parser.parse_args()
    adapter.configure(args)

    if args.mode == "child":
        if args.work_item_json is None:
            raise ValueError("--work-item-json is required in child mode")
        if args.rank_output_path is None:
            raise ValueError("--rank-output-path is required in child mode")
        engine.run_child(args, adapter)
        return

    if args.mode == "parent":
        engine.run_parent(args, adapter, entrypoint_path=sys.argv[0])
        return

    if args.input_dir is None:
        raise ValueError("--input-dir is required in select mode")
    if args.output_dir is None:
        raise ValueError("--output-dir is required in select mode")
    output_dir = Path(args.output_dir)
    records = selection.load_result_records(args.input_dir)
    best_configs = selection.select_best_configs(
        adapter=adapter,
        records=records,
    )
    best_path = artifacts.write_best_configs_json(output_dir, best_configs)
    generated_path = artifacts.write_generated_python(
        adapter=adapter,
        output_dir=output_dir,
        best_configs=best_configs,
    )
    print(f"wrote {best_path}")
    print(f"wrote {generated_path}")
