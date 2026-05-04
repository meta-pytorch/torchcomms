#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Launch uniflow benchmarks directly (no torchrun dependency).
# Spawns N processes with env vars set manually.
#
# Usage:
#   bash run_benchmark_direct.sh [--nproc N] [-- BENCH_ARGS...]
#
# Examples:
#   bash run_benchmark_direct.sh --nproc 2
#   bash run_benchmark_direct.sh --nproc 2 -- --benchmark bandwidth

set -euo pipefail

NPROC=2
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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

echo "=== Uniflow Benchmark Launcher (Direct) ==="
echo "  Ranks: ${NPROC}"
echo "  Extra args: ${BENCH_ARGS[*]:-<none>}"
echo ""

# Build the benchmark binary.
echo "Building uniflow_bench..."
BENCH_BIN=$(buck build @fbcode//mode/opt  fbcode//comms/uniflow/benchmarks:uniflow_bench --show-full-output 2>/dev/null | awk '{print $2}')

if [[ -z "${BENCH_BIN}" || ! -x "${BENCH_BIN}" ]]; then
  echo "ERROR: Failed to build uniflow_bench or binary not found at: ${BENCH_BIN:-<empty>}"
  exit 1
fi

echo "Binary: ${BENCH_BIN}"
echo ""

PIDS=()

for ((i = 0; i < NPROC; i++)); do
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT=29500 \
  RANK=$i \
  WORLD_SIZE=$NPROC \
  LOCAL_RANK=$i \
    "${BENCH_BIN}" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} &
  PIDS+=($!)
done

echo "Launched ${NPROC} processes: ${PIDS[*]}"
echo ""

# Wait for all and collect exit codes.
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "Process ${pid} failed"
    EXIT_CODE=1
  fi
done

exit ${EXIT_CODE}
