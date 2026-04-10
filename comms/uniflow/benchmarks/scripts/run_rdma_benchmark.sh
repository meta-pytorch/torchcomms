#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Run RDMA benchmarks with 2 ranks.
#
# Supports two modes:
#   1. Intra-host (single machine): Both ranks run on the same host using the
#      same NIC in loopback mode. This exercises the full RDMA QP data path
#      (memory registration, work request posting, CQ polling, DMA) without
#      requiring network fabric. Auto-discover defaults to same-NIC loopback.
#
#   2. Inter-host (two machines): Run this script on each machine separately,
#      pointing MASTER_ADDR to the rank 0 machine. Each machine uses its own
#      NIC on the shared fabric.
#
# Usage:
#   bash run_rdma_benchmark.sh [OPTIONS]
#
# Options:
#   --benchmark <name>   rdma_bandwidth (default: rdma_bandwidth)
#   --nic0 <dev>         RDMA device for rank 0 (default: auto-discover)
#   --nic1 <dev>         RDMA device for rank 1 (default: same as nic0 for loopback)
#   --iterations <n>     Iterations per size (default: 20)
#   --warmup <n>         Warmup iterations (default: 5)
#   --min-size <bytes>   Minimum message size (default: 1)
#   --max-size <bytes>   Maximum message size (default: 1073741824)
#   --direction <dir>    put | get | both (default: both)
#   --                   Pass remaining args directly to uniflow_bench
#
# Intra-host examples (single machine, same-NIC loopback):
#
#   # Default: auto-discovers first NIC, uses it for both ranks (loopback).
#   bash run_rdma_benchmark.sh
#
#   # Explicit NIC selection, same NIC for loopback:
#   bash run_rdma_benchmark.sh --nic0 mlx5_0 --nic1 mlx5_0
#
#   # Two NICs on the same subnet (if available on your host):
#   bash run_rdma_benchmark.sh --nic0 mlx5_0 --nic1 mlx5_3
#
#   # Quick bandwidth test with fewer iterations and smaller sizes:
#   bash run_rdma_benchmark.sh --iterations 5 --warmup 2 --max-size 1048576
#
# Inter-host examples (two machines on the same RDMA fabric):
#
#   # On machine A (rank 0, the master):
#   MASTER_ADDR=machineA MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 \
#     bash run_rdma_benchmark.sh --nic0 mlx5_0
#
#   # On machine B (rank 1):
#   MASTER_ADDR=machineA MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 \
#     bash run_rdma_benchmark.sh --nic0 mlx5_0
#
#   Note: For inter-host, set RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR env vars
#   and run the script on each machine. The script detects these env vars and
#   launches only one rank per invocation (instead of two).

set -euo pipefail

BENCHMARK="rdma_bandwidth"
NIC0=""
NIC1=""
ITERATIONS=20
WARMUP=5
MIN_SIZE=1
MAX_SIZE=1073741824
DIRECTION="both"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)   BENCHMARK="$2"; shift 2 ;;
    --nic0)        NIC0="$2"; shift 2 ;;
    --nic1)        NIC1="$2"; shift 2 ;;
    --iterations)  ITERATIONS="$2"; shift 2 ;;
    --warmup)      WARMUP="$2"; shift 2 ;;
    --min-size)    MIN_SIZE="$2"; shift 2 ;;
    --max-size)    MAX_SIZE="$2"; shift 2 ;;
    --direction)   DIRECTION="$2"; shift 2 ;;
    --)            shift; EXTRA_ARGS=("$@"); break ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Auto-discover NICs if not specified.
# Default: use the same NIC for both ranks (loopback). This always works on
# any host with at least one RDMA NIC. Using two different NICs requires them
# to be on the same L2 network/subnet, which is not guaranteed intra-host
# (e.g., mlx5_0 on beth3 and mlx5_1 on eth0 are on different subnets).
if [[ -z "${NIC0}" ]]; then
  mapfile -t NICS < <(ibstat -l 2>/dev/null || ls /sys/class/infiniband/ 2>/dev/null || true)
  if [[ ${#NICS[@]} -lt 1 ]]; then
    echo "ERROR: No RDMA NICs found."
    echo "Specify NICs manually with --nic0 (and optionally --nic1)"
    exit 1
  fi
  NIC0="${NICS[0]}"
fi
# Default NIC1 to same as NIC0 for loopback.
NIC1="${NIC1:-${NIC0}}"

# Build.
echo "Building uniflow_bench..."
BENCH_BIN=$(buck build @fbcode//mode/opt fbcode//comms/uniflow/benchmarks:uniflow_bench --show-full-output 2>/dev/null | awk '{print $2}')

if [[ -z "${BENCH_BIN}" || ! -x "${BENCH_BIN}" ]]; then
  echo "ERROR: Failed to build uniflow_bench"
  exit 1
fi

# Detect inter-host mode: if RANK and WORLD_SIZE are already set in the
# environment, the user is launching one rank per machine. Run a single
# rank and exit.
if [[ -n "${RANK:-}" && -n "${WORLD_SIZE:-}" ]]; then
  NIC="${NIC0}"
  echo "=== Uniflow RDMA Benchmark (inter-host, rank ${RANK}) ==="
  echo "  Benchmark:  ${BENCHMARK}"
  echo "  NIC:        ${NIC}"
  echo "  Iterations: ${ITERATIONS} (warmup: ${WARMUP})"
  echo "  Size range: ${MIN_SIZE} - ${MAX_SIZE}"
  echo "  Direction:  ${DIRECTION}"
  echo "  Binary:     ${BENCH_BIN}"
  echo ""

  exec "${BENCH_BIN}" \
    --benchmark "${BENCHMARK}" \
    --rdma-devices "${NIC}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    --min-size "${MIN_SIZE}" \
    --max-size "${MAX_SIZE}" \
    --direction "${DIRECTION}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi

# Intra-host mode: launch both ranks on this machine.
echo "=== Uniflow RDMA Benchmark (intra-host) ==="
echo "  Benchmark:  ${BENCHMARK}"
echo "  NIC rank 0: ${NIC0}"
echo "  NIC rank 1: ${NIC1}"
echo "  Iterations: ${ITERATIONS} (warmup: ${WARMUP})"
echo "  Size range: ${MIN_SIZE} - ${MAX_SIZE}"
echo "  Direction:  ${DIRECTION}"
echo "  Binary:     ${BENCH_BIN}"
echo ""

MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
RANK=0 \
WORLD_SIZE=2 \
LOCAL_RANK=0 \
  "${BENCH_BIN}" \
    --benchmark "${BENCHMARK}" \
    --rdma-devices "${NIC0}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    --min-size "${MIN_SIZE}" \
    --max-size "${MAX_SIZE}" \
    --direction "${DIRECTION}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
PID0=$!

MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
RANK=1 \
WORLD_SIZE=2 \
LOCAL_RANK=1 \
  "${BENCH_BIN}" \
    --benchmark "${BENCHMARK}" \
    --rdma-devices "${NIC1}" \
    --iterations "${ITERATIONS}" \
    --warmup "${WARMUP}" \
    --min-size "${MIN_SIZE}" \
    --max-size "${MAX_SIZE}" \
    --direction "${DIRECTION}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
PID1=$!

echo "Launched rank 0 (pid ${PID0}, ${NIC0}) and rank 1 (pid ${PID1}, ${NIC1})"
echo ""

EXIT_CODE=0
if ! wait "${PID0}"; then
  echo "Rank 0 (pid ${PID0}) failed"
  EXIT_CODE=1
fi
if ! wait "${PID1}"; then
  echo "Rank 1 (pid ${PID1}) failed"
  EXIT_CODE=1
fi

exit ${EXIT_CODE}
