#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Generate comparison tables from uniflow CSV results and ib_write_bw baseline.
#
# Usage:
#   python3 generate_comparison.py <results_dir> [--ib-baseline <path>]
#
# Examples:
#   # Use ib_write_bw baseline from a results.log in the same directory
#   python3 generate_comparison.py ~/me/benchmarks/rdma/gb200/inter-host-per-nic-buf
#
#   # Use ib_write_bw baseline from a different directory
#   python3 generate_comparison.py ~/me/benchmarks/rdma/gb200/inter-host-per-nic-buf \
#       --ib-baseline ~/me/benchmarks/rdma/gb200/inter-host/chunk_512KB/results.log
#
# Expects directory structure:
#   <results_dir>/
#     chunk_512KB/results.csv
#     chunk_1MB/results.csv
#     chunk_2MB/results.csv
#     chunk_4MB/results.csv
#
# Outputs results.log (comparison table) next to each results.csv.

import argparse
import csv
import os
import re
import sys


def format_size(n):
    if n >= 1 << 30:
        return f"{n >> 30} GB"
    if n >= 1 << 20:
        return f"{n >> 20} MB"
    if n >= 1 << 10:
        return f"{n >> 10} KB"
    return f"{n} B"


def fmt(v, d=2):
    return f"{v:.{d}f}" if v is not None else "N/A"


def parse_uniflow_csv(path):
    results = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                size = int(row["size_bytes"])
                results[size] = {
                    "bw": float(row["bw_gbps"]),
                    "avg_us": float(row.get("lat_avg_us", 0)),
                    "p50_us": float(row.get("lat_p50_us", 0)),
                    "p99_us": float(row.get("lat_p99_us", 0)),
                }
            except (ValueError, KeyError):
                continue
    return results


_SIZE_MULT = {"B": 1, "KB": 1024, "MB": 1 << 20, "GB": 1 << 30}


def _parse_comparison_table(content):
    """Parse ib_write_bw column from a '| Size | uniflow | ib_write_bw |' table."""
    results = {}
    in_table = False
    for line in content.splitlines():
        if line.startswith("| Size"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 3:
            continue
        parts = cols[0].strip().split()
        if len(parts) != 2:
            continue
        try:
            size = int(float(parts[0]) * _SIZE_MULT.get(parts[1], 1))
            results[size] = float(cols[2])
        except ValueError:
            pass
    return results


def _parse_perftest_output(content):
    """Parse raw ib_write_bw output (size in col 0, BW average in col 3)."""
    results = {}
    unit_is_mb = "BW average[MB/sec]" in content
    for line in content.splitlines():
        if not re.match(r"\s*\d+\s+", line):
            continue
        cols = line.split()
        if len(cols) < 4:
            continue
        try:
            size = int(cols[0])
            raw = float(cols[3])
            results[size] = raw / 1024.0 if unit_is_mb else raw / 8.0
        except ValueError:
            pass
    return results


def parse_ib_from_log(path):
    """Extract ib_write_bw bandwidth from a log file.

    Tries comparison table format first, falls back to raw perftest output.
    """
    with open(path) as f:
        content = f.read()
    return _parse_comparison_table(content) or _parse_perftest_output(content)


def generate_table(uniflow, ib_results, sizes):
    lines = []
    lines.append(
        f"| {'Size':<11} | {'uniflow':>9}"
        f" | {'ib_write_bw':>11} | {'Gap':>6}"
        f" | {'Avg (us)':>10} | {'P99 (us)':>10} |"
    )
    lines.append(f"|{'-' * 13}|{'-' * 11}|{'-' * 13}|{'-' * 8}|{'-' * 12}|{'-' * 12}|")

    for size in sizes:
        ur = uniflow.get(size, {})
        uf = ur.get("bw")
        ib = ib_results.get(size)

        if uf is not None and ib is not None and ib > 0:
            gap_pct = (ib - uf) / ib * 100
            gap_s = "*" if gap_pct > 50 else f"{gap_pct:.1f}%"
        else:
            gap_s = "N/A"

        lines.append(
            f"| {format_size(size):<11} | {fmt(uf):>9}"
            f" | {fmt(ib):>11} | {gap_s:>6}"
            f" | {fmt(ur.get('avg_us'), 1):>10}"
            f" | {fmt(ur.get('p99_us'), 1):>10} |"
        )

    lines.append("")
    lines.append("  Gap = (ib - uniflow) / ib. Negative = uniflow faster (dual-NIC).")
    lines.append("  BW in GB/s. Latency from uniflow.")
    lines.append("")
    return "\n".join(lines)


def find_ib_baseline(results_dir):
    """Search for an existing results.log with ib_write_bw data."""
    for chunk in ["chunk_512KB", "chunk_1MB", "chunk_2MB", "chunk_4MB"]:
        path = os.path.join(results_dir, chunk, "results.log")
        if os.path.exists(path):
            ib = parse_ib_from_log(path)
            if ib:
                return path, ib
    # Try parent directory (e.g., inter-host/ has single-NIC baselines)
    parent = os.path.dirname(results_dir.rstrip("/"))
    for chunk in ["chunk_512KB", "chunk_1MB", "chunk_2MB", "chunk_4MB"]:
        path = os.path.join(parent, chunk, "results.log")
        if os.path.exists(path):
            ib = parse_ib_from_log(path)
            if ib:
                return path, ib
    return None, {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison tables from uniflow CSV results"
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing chunk_*/results.csv files",
    )
    parser.add_argument(
        "--ib-baseline",
        default="",
        help="Path to results.log with ib_write_bw baseline"
        " (default: auto-detect from results_dir or parent)",
    )
    args = parser.parse_args()

    if args.ib_baseline:
        ib_path = args.ib_baseline
        ib_results = parse_ib_from_log(ib_path)
    else:
        ib_path, ib_results = find_ib_baseline(args.results_dir)

    if ib_results:
        print(f"Using ib_write_bw baseline from: {ib_path}")
    else:
        print("WARNING: No ib_write_bw baseline found. Gap column will show N/A.")

    chunk_dirs = sorted(
        d
        for d in os.listdir(args.results_dir)
        if d.startswith("chunk_")
        and os.path.isfile(os.path.join(args.results_dir, d, "results.csv"))
    )

    if not chunk_dirs:
        sys.exit(f"ERROR: No chunk_*/results.csv found in {args.results_dir}")

    for chunk_dir in chunk_dirs:
        csv_path = os.path.join(args.results_dir, chunk_dir, "results.csv")
        out_path = os.path.join(args.results_dir, chunk_dir, "results.log")

        uniflow = parse_uniflow_csv(csv_path)
        if not uniflow:
            print(f"  {chunk_dir}: empty CSV, skipping")
            continue

        sizes = sorted(uniflow.keys())
        table = generate_table(uniflow, ib_results, sizes)

        with open(out_path, "w") as f:
            f.write(table + "\n")

        peak_bw = max(r["bw"] for r in uniflow.values())
        print(
            f"  {chunk_dir}: {len(sizes)} sizes, peak {peak_bw:.1f} GB/s → {out_path}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
