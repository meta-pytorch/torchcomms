#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Launch uniflow benchmarks using torchrun (local, single-node).
#
# Usage:
#   Add --bench-target fbcode//neteng/network_ai/uniflow_extensions/benchmarks:uniflow_bench for extensions.
#   bash run_benchmark.sh [--nproc N] [-- BENCH_ARGS...]
#
# Examples:
#   bash run_benchmark.sh --nproc 2
#   bash run_benchmark.sh --nproc 2 -- --benchmark bandwidth --iterations 50
#   bash run_benchmark.sh --nproc 8 -- --format csv --output results.csv

set -euo pipefail

BENCH_TARGET="fbcode//comms/uniflow/benchmarks:uniflow_bench"

NPROC=2
BENCH_ARGS=()

# Parse wrapper args (before --), pass the rest to the benchmark binary.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-target) BENCH_TARGET="$2"; shift 2 ;;
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --)
      shift
      BENCH_ARGS=("$@")
      break
      ;;
    *)
      BENCH_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "=== Uniflow Benchmark Launcher ==="
echo "  Ranks: ${NPROC}"
echo "  Extra args: ${BENCH_ARGS[*]:-<none>}"
echo ""

# Build the benchmark binary.
echo "Building uniflow_bench..."
BENCH_BIN=$(buck2 build @fbcode//mode/opt -c hpc_comms.use_ncclx=stable "${BENCH_TARGET}" --show-full-output 2>/dev/null | awk '{print $2}')

if [[ -z "${BENCH_BIN}" || ! -x "${BENCH_BIN}" ]]; then
  echo "ERROR: Failed to build uniflow_bench or binary not found at: ${BENCH_BIN:-<empty>}"
  exit 1
fi

echo "Binary: ${BENCH_BIN}"
echo ""

# Launch with torchrun.
exec torchrun \
  --standalone \
  --nproc-per-node="${NPROC}" \
  --no-python \
  "${BENCH_BIN}" \
  ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
