#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Auto-tune sweep orchestrator for Triton alltoallv_dynamic.

Launches the existing benchmark binary via buck2 run with batches of
custom configs, parses the sweep table output, and tracks the best
config per message size across all batches.

This script runs LOCALLY (not via buck2 run) — it orchestrates
benchmark runs by invoking buck2 as a subprocess.

Usage:
    cd fbsource
    python3 fbcode/comms/torchcomms/triton/fb/tests/autotune_sweep.py
"""

import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


BUCK_CMD_PREFIX = [
    "buck2",
    "run",
    "@fbcode//mode/opt",
    "-c",
    "nccl.enable_scuba=false",
    "-c",
    "comms.envs=NCCL_DEBUG=WARN;NCCL_GIN_ENABLE=1;NCCL_GIN_TYPE=-1;"
    "NCCL_COMM_EVENT_LOGGING=;RUN_DEVICE_API_TEST=true;"
    "TEST_BACKEND=ncclx;TEST_FILTER=",
    "-c",
    "fbcode.enable_gpu_sections=true",
    "-c",
    "fbcode.platform010_cuda_version=12.8",
    "-c",
    "fbcode.nvcc_arch=h100a",
    "-c",
    "hpc_comms.use_ncclx=stable",
    "fbcode//comms/torchcomms/triton/fb/tests:benchmark_device_alltoallv_dynamic",
    "--",
]


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.0f}GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.0f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}KB"
    return f"{size_bytes}B"


def parse_sweep_output(raw: str) -> Dict[int, List[Tuple[str, float]]]:
    """Parse the sweep table from rank-0 stdout lines.
    Returns {msg_size_bytes: [(config_name, latency_us), ...]}.
    """
    # Extract rank-0 stdout
    lines = []
    for line in raw.split("\n"):
        if "[1,0]<stdout>:" in line:
            lines.append(line.split("[1,0]<stdout>:")[1])

    results: Dict[int, List[Tuple[str, float]]] = {}
    header_cols: List[str] = []

    for line in lines:
        line = line.strip()

        # Header row
        if line.startswith("Size") and "|" in line:
            header_cols = [p.strip() for p in line.split("|")][1:]
            continue

        if not header_cols or "|" not in line:
            continue

        # Data row: "  1.0MB |  37.12 | ..."
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        size_str = parts[0]
        try:
            if "KB" in size_str:
                msg_size = int(float(size_str.replace("KB", "")) * 1024)
            elif "MB" in size_str:
                msg_size = int(float(size_str.replace("MB", "")) * 1024 * 1024)
            elif "GB" in size_str:
                msg_size = int(float(size_str.replace("GB", "")) * 1024**3)
            else:
                continue
        except ValueError:
            continue

        row = []
        for i, val in enumerate(parts[1:]):
            if i >= len(header_cols):
                break
            name = header_cols[i]
            try:
                row.append((name, float(val)))
            except ValueError:
                pass
        if row:
            results[msg_size] = row

    return results


def run_sweep(
    min_size: int, max_size: int, custom_configs: str, iters: int = 10, warmup: int = 3
) -> str:
    """Run one benchmark sweep and return the combined stdout+stderr."""
    cmd = list(BUCK_CMD_PREFIX) + [
        "--sweep-warps",
        f"--min-size={min_size}",
        f"--max-size={max_size}",
        f"--iters={iters}",
        f"--warmup={warmup}",
    ]
    if custom_configs:
        cmd.append(f"--custom-configs={custom_configs}")

    print(f"  CMD: ...{' '.join(cmd[-5:])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    return r.stdout + "\n" + r.stderr


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-tune sweep orchestrator")
    parser.add_argument(
        "--mode",
        choices=["full", "sweep"],
        default="full",
        help=(
            "full: test 120 custom configs in batches (10i/3w). "
            "sweep: run default sweep only with 100i/10w, no custom configs."
        ),
    )
    args = parser.parse_args()

    if args.mode == "sweep":
        return run_sweep_only()
    else:
        return run_full_sweep()


def run_sweep_only() -> int:
    """Run the default sweep (no custom configs) with 100 iters / 10 warmup."""
    size_ranges = [
        (1024, 16 * 1024 * 1024, 100, 10),
        (32 * 1024 * 1024, 256 * 1024 * 1024, 20, 10),
        (512 * 1024 * 1024, 1024 * 1024 * 1024, 10, 5),
    ]

    print("Sweep-only mode: default configs (Gw4-32, GB2-32, GB16c64, Auto)")
    print(
        f"Ranges: {[(format_size(a), format_size(b), f'{i}i/{w}w') for a, b, i, w in size_ranges]}"
    )
    print(flush=True)

    best: Dict[int, Tuple[str, float]] = {}

    for ri, (lo, hi, iters, warmup) in enumerate(size_ranges):
        print(f"\n{'=' * 80}")
        print(
            f"Range {ri + 1}/{len(size_ranges)}: {format_size(lo)} – {format_size(hi)} ({iters}i/{warmup}w)"
        )
        print(f"{'=' * 80}", flush=True)

        try:
            raw = run_sweep(lo, hi, "", iters=iters, warmup=warmup)
            results = parse_sweep_output(raw)

            for ms, entries in results.items():
                for name, lat in entries:
                    if name in ("Auto", "AutoCfg", "Best", "NCCLgr"):
                        continue
                    if ms not in best or lat < best[ms][1]:
                        old = best.get(ms)
                        best[ms] = (name, lat)
                        if old and (old[1] - lat) / old[1] > 0.01:
                            print(
                                f"  ↑ {format_size(ms):>8s}: {name:>14s} "
                                f"{lat:.2f}µs (was {old[0]} {old[1]:.2f}µs)",
                                flush=True,
                            )
        except subprocess.TimeoutExpired:
            print("  ✗ timed out", flush=True)
        except Exception as e:
            print(f"  ✗ {e}", flush=True)

        # Report
        print(f"\n{'=' * 80}")
        print(f"Best after {format_size(lo)}–{format_size(hi)}")
        print(f"{'=' * 80}")
        print(f"{'Size':>10s} | {'Config':>14s} | {'Latency':>10s}")
        print(f"{'-' * 10}-+-{'-' * 14}-+-{'-' * 10}")
        for ms in sorted(best):
            c, l = best[ms]
            print(f"{format_size(ms):>10s} | {c:>14s} | {l:>10.2f}")
        print(f"{'=' * 80}\n", flush=True)

    # Final
    print("\n" + "=" * 80)
    print("FINAL AUTO-TUNE TABLE (sweep-only, no custom configs)")
    print("=" * 80)
    for ms in sorted(best):
        c, l = best[ms]
        print(f"{format_size(ms):>10s} | {c:>14s} | {l:>10.2f}")
    print("=" * 80)
    return 0


def run_full_sweep() -> int:
    """Run the full sweep with custom configs."""
    # Config search space (BxWxC_kb strings for --custom-configs)
    configs: List[str] = []
    for b in [1, 2, 4, 8, 16, 32]:
        for w in [4, 8, 16, 32]:
            for c in [32, 64, 128, 256, 512]:
                # Skip configs the default sweep already tests
                if b == 1 and c == 64 and w in (4, 8, 16, 32):
                    continue
                if w == 16 and c == 64 and b in (2, 4, 8, 16, 32):
                    continue
                if b == 16 and w == 16 and c == 64:
                    continue
                configs.append(f"{b}x{w}x{c}")

    batch_size = 8
    batches = [configs[i : i + batch_size] for i in range(0, len(configs), batch_size)]

    size_ranges = [
        (1024, 16 * 1024 * 1024, 10, 3),  # 1KB-16MB: 10 iters, 3 warmup
        (32 * 1024 * 1024, 256 * 1024 * 1024, 10, 3),  # 32MB-256MB: 10 iters, 3 warmup
        (512 * 1024 * 1024, 1024 * 1024 * 1024, 10, 3),  # 512MB-1GB: 10 iters, 3 warmup
    ]

    print(f"Auto-tune: {len(configs)} custom configs in {len(batches)} batches")
    print(
        f"Ranges: {[(format_size(a), format_size(b), f'{i}i/{w}w') for a, b, i, w in size_ranges]}"
    )
    print(f"Total benchmark runs: {len(batches) * len(size_ranges)}")
    print(flush=True)

    best: Dict[int, Tuple[str, float]] = {}

    for ri, (lo, hi, iters, warmup) in enumerate(size_ranges):
        print(f"\n{'=' * 80}")
        print(
            f"Range {ri + 1}/{len(size_ranges)}: {format_size(lo)} – {format_size(hi)} ({iters} iters, {warmup} warmup)"
        )
        print(f"{'=' * 80}", flush=True)

        for bi, batch in enumerate(batches):
            cc = ",".join(batch)
            print(f"\nBatch {bi + 1}/{len(batches)}: {cc}", flush=True)

            try:
                raw = run_sweep(lo, hi, cc)
                results = parse_sweep_output(raw)

                for ms, entries in results.items():
                    for name, lat in entries:
                        if name in ("Auto", "AutoCfg", "Best", "NCCLgr"):
                            continue
                        if ms not in best or lat < best[ms][1]:
                            old = best.get(ms)
                            best[ms] = (name, lat)
                            if old and (old[1] - lat) / old[1] > 0.01:
                                print(
                                    f"  ↑ {format_size(ms):>8s}: {name:>14s} "
                                    f"{lat:.2f}µs (was {old[0]} {old[1]:.2f}µs)",
                                    flush=True,
                                )
            except subprocess.TimeoutExpired:
                print(f"  ✗ timed out", flush=True)
            except Exception as e:
                print(f"  ✗ {e}", flush=True)

        # Progress report
        print(f"\n{'=' * 80}")
        print(f"Best after {format_size(lo)}–{format_size(hi)}")
        print(f"{'=' * 80}")
        print(f"{'Size':>10s} | {'Config':>14s} | {'Latency':>10s}")
        print(f"{'-' * 10}-+-{'-' * 14}-+-{'-' * 10}")
        for ms in sorted(best):
            c, l = best[ms]
            print(f"{format_size(ms):>10s} | {c:>14s} | {l:>10.2f}")
        print(f"{'=' * 80}\n", flush=True)

    # Final
    print("\n" + "=" * 80)
    print("FINAL AUTO-TUNE TABLE")
    print("=" * 80)
    for ms in sorted(best):
        c, l = best[ms]
        print(f"{format_size(ms):>10s} | {c:>14s} | {l:>10.2f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
