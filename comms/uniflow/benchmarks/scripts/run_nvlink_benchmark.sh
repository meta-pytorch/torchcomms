#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Run NVLink bandwidth benchmarks with 2 ranks.
#
# Supports three modes:
#   1. Intra-host local (default): Both ranks run on the local machine using
#      GPU 0 (LOCAL_RANK=0) and GPU 1 (LOCAL_RANK=1). Requires at least 2
#      NVLink-connected GPUs (H100 or B200).
#
#   2. Intra-host remote (--hosts host): Cross-compile for aarch64, copy the
#      binary to a single GB200 host, and run both ranks there using GPU 0
#      and GPU 1.
#
#   3. Inter-host (--hosts host0,host1): Cross-compile for aarch64, copy the
#      binary to both GB200 hosts via suscp, and run via sush2. Requires
#      MNNVL fabric between the hosts.
#
# Usage:
#   bash run_nvlink_benchmark.sh [OPTIONS]
#
# Options:
#   --benchmark <name>   nvlink_bandwidth (default: nvlink_bandwidth)
#   --hosts <h0[,h1]>   One or two comma-separated hostnames for remote mode
#   --gpu <type>         GPU type: h100 or b200 (default: b200, used with --hosts)
#   --iterations <n>     Iterations per size (default: 20)
#   --warmup <n>         Warmup iterations (default: 5)
#   --min-size <bytes>   Minimum message size (default: 1)
#   --max-size <bytes>   Maximum message size (default: 1073741824)
#   --direction <dir>    put | get | both (default: both)
#   --                   Pass remaining args directly to uniflow_bench
#
# Intra-host examples (single machine, 2 GPUs):
#
#   bash run_nvlink_benchmark.sh --iterations 5 --warmup 2 --max-size 67108864
#
# Intra-host remote GB200 (single remote host, 2 GPUs):
#
#   bash run_nvlink_benchmark.sh \
#     --hosts rtptest2356.nha6 \
#     --iterations 5 --warmup 2 --max-size 67108864 --direction put
#
# Inter-host examples (two GB200 machines in same MNNVL domain):
#
#   bash run_nvlink_benchmark.sh \
#     --hosts rtptest2356.nha6,rtptest2357.nha6 \
#     --iterations 5 --warmup 2 --max-size 67108864 --direction put

set -euo pipefail

BENCHMARK="nvlink_bandwidth"
HOSTS=""
GPU="b200"
ITERATIONS=20
WARMUP=5
LOOP_COUNT=1
BIDIRECTIONAL=""
MIN_SIZE=1
MAX_SIZE=1073741824
DIRECTION="both"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)   BENCHMARK="$2"; shift 2 ;;
    --hosts)       HOSTS="$2"; shift 2 ;;
    --gpu)         GPU="$2"; shift 2 ;;
    --iterations)  ITERATIONS="$2"; shift 2 ;;
    --warmup)      WARMUP="$2"; shift 2 ;;
    --loop-count)  LOOP_COUNT="$2"; shift 2 ;;
    --bidirectional) BIDIRECTIONAL="--bidirectional"; shift ;;
    --min-size)    MIN_SIZE="$2"; shift 2 ;;
    --max-size)    MAX_SIZE="$2"; shift 2 ;;
    --direction)   DIRECTION="$2"; shift 2 ;;
    --)            shift; EXTRA_ARGS=("$@"); break ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

BENCH_ARGS=(
  --benchmark "${BENCHMARK}"
  --iterations "${ITERATIONS}"
  --warmup "${WARMUP}"
  --loop-count "${LOOP_COUNT}"
  ${BIDIRECTIONAL:+"${BIDIRECTIONAL}"}
  --min-size "${MIN_SIZE}"
  --max-size "${MAX_SIZE}"
  --direction "${DIRECTION}"
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
)

# ---------------------------------------------------------------------------
# Remote mode: --hosts host0[,host1]
# ---------------------------------------------------------------------------
if [[ -n "${HOSTS}" ]]; then
  IFS=',' read -r HOST0 HOST1 <<< "${HOSTS}"
  if [[ -z "${HOST0}" ]]; then
    echo "ERROR: --hosts requires at least one hostname"
    exit 1
  fi

  # Determine if single-host (intra) or two-host (inter) mode.
  if [[ -z "${HOST1}" ]]; then
    MODE="intra-host-remote"
  else
    MODE="inter-host"
  fi

  # Build for the target GPU architecture.
  GPU_LOWER=$(echo "${GPU}" | tr '[:upper:]' '[:lower:]')
  if [[ "${GPU_LOWER}" == "b200" ]]; then
    echo "Building uniflow_bench for aarch64 (B200)..."
    BENCH_BIN=$(buck2 build @fbcode//mode/opt \
      -c fbcode.arch=aarch64 \
      -c fbcode.enable_gpu_sections=true \
      -c fbcode.nvcc_arch=b200 \
      -c fbcode.platform010_cuda_version=12.8 \
      fbcode//comms/uniflow/benchmarks:uniflow_bench \
      --show-full-output 2>/dev/null | awk '{print $2}')
  elif [[ "${GPU_LOWER}" == "h100" ]]; then
    echo "Building uniflow_bench for x86_64 (H100)..."
    BENCH_BIN=$(buck2 build @fbcode//mode/opt \
      fbcode//comms/uniflow/benchmarks:uniflow_bench \
      --show-full-output 2>/dev/null | awk '{print $2}')
  else
    echo "ERROR: --gpu must be h100 or b200 (got: ${GPU})"
    exit 1
  fi

  if [[ -z "${BENCH_BIN}" ]]; then
    echo "ERROR: Failed to build uniflow_bench for --gpu ${GPU}"
    echo "Try running the build manually to see errors."
    exit 1
  fi

  REMOTE_DIR="/tmp/uniflow_bench_$$"
  REMOTE_BIN="${REMOTE_DIR}/uniflow_bench"

  echo "Copying binary to ${HOST0}${HOST1:+ and ${HOST1}}..."
  sush2 --reason 'create benchmark directory' "root@${HOST0}" "mkdir -p ${REMOTE_DIR}" &
  if [[ -n "${HOST1}" ]]; then
    sush2 --reason 'create benchmark directory' "root@${HOST1}" "mkdir -p ${REMOTE_DIR}" &
  fi
  wait

  suscp --reason 'copy NVLink benchmark binary' "${BENCH_BIN}" "root@${HOST0}:${REMOTE_BIN}" &
  if [[ -n "${HOST1}" ]]; then
    suscp --reason 'copy NVLink benchmark binary' "${BENCH_BIN}" "root@${HOST1}:${REMOTE_BIN}" &
  fi
  wait

  # Resolve HOST0's IP address for MASTER_ADDR (TcpController requires numeric IP).
  MASTER_IP=$(sush2 --reason 'resolve IP' "root@${HOST0}" "hostname -i | awk '{print \$1}'" 2>/dev/null)
  if [[ -z "${MASTER_IP}" ]]; then
    echo "ERROR: Could not resolve IP for ${HOST0}"
    exit 1
  fi

  echo ""
  if [[ "${MODE}" == "intra-host-remote" ]]; then
    echo "=== Uniflow NVLink Benchmark (intra-host remote) ==="
    echo "  Benchmark:  ${BENCHMARK}"
    echo "  Host:       ${HOST0} (${MASTER_IP})"
    echo "  GPU rank 0: cuda:0"
    echo "  GPU rank 1: cuda:1"
  else
    echo "=== Uniflow NVLink Benchmark (inter-host) ==="
    echo "  Benchmark:  ${BENCHMARK}"
    echo "  Host rank 0: ${HOST0} (${MASTER_IP})"
    echo "  Host rank 1: ${HOST1}"
  fi
  echo "  Iterations: ${ITERATIONS} (warmup: ${WARMUP})"
  echo "  Size range: ${MIN_SIZE} - ${MAX_SIZE}"
  echo "  Direction:  ${DIRECTION}"
  echo ""

  BENCH_CMD_ARGS=$(printf " %s" "${BENCH_ARGS[@]}")

  # Determine rank 1's host and LOCAL_RANK.
  if [[ "${MODE}" == "intra-host-remote" ]]; then
    RANK1_HOST="${HOST0}"
    RANK1_LOCAL_RANK=1
  else
    RANK1_HOST="${HOST1}"
    RANK1_LOCAL_RANK=0
  fi

  # Launch rank 1 first (it connects to rank 0).
  sush2 --reason 'run NVLink benchmark rank 1' "root@${RANK1_HOST}" \
    "MASTER_ADDR=${MASTER_IP} MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 LOCAL_RANK=${RANK1_LOCAL_RANK} ${REMOTE_BIN}${BENCH_CMD_ARGS}" \
    > /dev/null 2>&1 &
  PID1=$!

  # Small delay to let rank 0 start its server first.
  sleep 1

  # Launch rank 0 (its output is what we display).
  sush2 --reason 'run NVLink benchmark rank 0' "root@${HOST0}" \
    "MASTER_ADDR=${MASTER_IP} MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 ${REMOTE_BIN}${BENCH_CMD_ARGS}"
  EXIT_CODE=$?

  # Wait for rank 1 to finish.
  wait "${PID1}" 2>/dev/null || true

  # Cleanup.
  sush2 --reason 'cleanup benchmark binary' "root@${HOST0}" "rm -rf ${REMOTE_DIR}" &
  if [[ -n "${HOST1}" ]]; then
    sush2 --reason 'cleanup benchmark binary' "root@${HOST1}" "rm -rf ${REMOTE_DIR}" &
  fi
  wait

  exit ${EXIT_CODE}
fi

# ---------------------------------------------------------------------------
# Intra-host mode (default)
# ---------------------------------------------------------------------------
echo "Building uniflow_bench..."
BENCH_BIN=$(buck build @fbcode//mode/opt fbcode//comms/uniflow/benchmarks:uniflow_bench --show-full-output 2>/dev/null | awk '{print $2}')

if [[ -z "${BENCH_BIN}" || ! -x "${BENCH_BIN}" ]]; then
  echo "ERROR: Failed to build uniflow_bench"
  exit 1
fi

echo "=== Uniflow NVLink Benchmark (intra-host) ==="
echo "  Benchmark:  ${BENCHMARK}"
echo "  GPU rank 0: cuda:0"
echo "  GPU rank 1: cuda:1"
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
  "${BENCH_BIN}" "${BENCH_ARGS[@]}" &
PID0=$!

MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
RANK=1 \
WORLD_SIZE=2 \
LOCAL_RANK=1 \
  "${BENCH_BIN}" "${BENCH_ARGS[@]}" &
PID1=$!

echo "Launched rank 0 (pid ${PID0}, cuda:0) and rank 1 (pid ${PID1}, cuda:1)"
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
