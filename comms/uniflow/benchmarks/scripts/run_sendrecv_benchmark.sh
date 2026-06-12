#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Benchmark uniflow copy-based send/recv and compare with NCCL sendrecv.
# Runs both benchmarks under the same framework and prints a side-by-side
# comparison table.
#
# Usage:
#   bash run_sendrecv_benchmark.sh [OPTIONS]
#
# Options:
#   --nproc N           Number of ranks (default: 2)
#   --gpu0 ID           GPU for rank 0 (default: 0)
#   --gpu1 ID           GPU for rank 1 (default: 1)
#   --cpu               Use CPU (host) memory instead of GPU memory
#   --nic0 NAME         RDMA device for rank 0 (default: auto)
#   --nic1 NAME         RDMA device for rank 1 (default: auto)
#   --min-size BYTES    Minimum message size (default: 1048576)
#   --max-size BYTES    Maximum message size (default: 1073741824)
#   --iterations N      Timed iterations per size (default: 100)
#   --warmup N          Warmup iterations (default: 20)
#   --chunk-size BYTES  Slab/RDMA chunk size (default: 2097152)
#   --pipeline-depth N  Staging pipeline depth (default: 4)
#   --topology TYPE     fanout|fanin (default: fanout)
#   --skip-nccl         Skip NCCL baseline
#   --skip-uniflow      Skip uniflow send/recv
#   --nccl-net          Force NCCL through network (disable P2P/SHM)
#   --slab-size BYTES   Staging slab size (default: chunk-size)
#   --slab-num N        Number of staging slabs (default: pipeline-depth)
#
# Examples:
#   # GPU memory comparison:
#   bash run_sendrecv_benchmark.sh --nic0 mlx5_0 --nic1 mlx5_3 --nccl-net
#
#   # CPU memory comparison:
#   bash run_sendrecv_benchmark.sh --cpu --nic0 mlx5_0 --nic1 mlx5_3 --nccl-net
#
#   # Tune chunk size and pipeline depth:
#   bash run_sendrecv_benchmark.sh --chunk-size 4194304 --pipeline-depth 8

set -euo pipefail

# Defaults
NPROC=2
GPU0=0
GPU1=1
USE_CPU=false
NIC0=""
NIC1=""
MIN_SIZE=1048576
MAX_SIZE=1073741824
ITERATIONS=100
WARMUP=20
CHUNK_SIZE=2097152
PIPELINE_DEPTH=4
SLAB_SIZE=""
SLAB_NUM=""
TOPOLOGY="fanout"
SKIP_NCCL=false
SKIP_UNIFLOW=false
NCCL_NET=false
NO_GDR=false
PORT=29500

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc)        NPROC="$2"; shift 2 ;;
    --gpu0)         GPU0="$2"; shift 2 ;;
    --gpu1)         GPU1="$2"; shift 2 ;;
    --cpu)          USE_CPU=true; shift ;;
    --nic0)         NIC0="$2"; shift 2 ;;
    --nic1)         NIC1="$2"; shift 2 ;;
    --min-size)     MIN_SIZE="$2"; shift 2 ;;
    --max-size)     MAX_SIZE="$2"; shift 2 ;;
    --iterations)   ITERATIONS="$2"; shift 2 ;;
    --warmup)       WARMUP="$2"; shift 2 ;;
    --chunk-size)   CHUNK_SIZE="$2"; shift 2 ;;
    --pipeline-depth) PIPELINE_DEPTH="$2"; shift 2 ;;
    --slab-size)    SLAB_SIZE="$2"; shift 2 ;;
    --slab-num)     SLAB_NUM="$2"; shift 2 ;;
    --topology)     TOPOLOGY="$2"; shift 2 ;;
    --skip-nccl)    SKIP_NCCL=true; shift ;;
    --skip-uniflow) SKIP_UNIFLOW=true; shift ;;
    --nccl-net)     NCCL_NET=true; shift ;;
    --no-gdr)       NO_GDR=true; shift ;;
    --port)         PORT="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Build
echo "=== Building uniflow_bench (opt mode) ==="
BENCH_BIN=$(buck build @fbcode//mode/opt fbcode//comms/uniflow/benchmarks:uniflow_bench \
  --show-full-output 2>/dev/null | awk '{print $2}')

if [[ -z "${BENCH_BIN}" || ! -x "${BENCH_BIN}" ]]; then
  echo "ERROR: build failed"
  exit 1
fi
echo "Binary: ${BENCH_BIN}"
echo ""

# Memory mode
MEM_LABEL="GPU"
GPU0_ARG="--cuda-device ${GPU0}"
GPU1_ARG="--cuda-device ${GPU1}"
if [[ "${USE_CPU}" == "true" ]]; then
  MEM_LABEL="CPU"
  GPU0_ARG=""
  GPU1_ARG=""
fi

COMMON_ARGS="--min-size ${MIN_SIZE} --max-size ${MAX_SIZE} --iterations ${ITERATIONS} --warmup ${WARMUP}"

# NIC args
NIC0_ARGS=""
NIC1_ARGS=""
if [[ -n "${NIC0}" ]]; then
  NIC0_ARGS="--rdma-devices ${NIC0} --num-nics 1"
fi
if [[ -n "${NIC1}" ]]; then
  NIC1_ARGS="--rdma-devices ${NIC1} --num-nics 1"
fi

run_benchmark() {
  local LABEL="$1"
  local R0_ARGS="$2"
  local R1_ARGS="$3"
  local R0_ENV="${4:-}"
  local R1_ENV="${5:-}"

  echo "=== ${LABEL} ==="
  echo ""

  local LPORT=$((PORT + RANDOM % 100))

  env MASTER_ADDR=127.0.0.1 MASTER_PORT=${LPORT} RANK=1 WORLD_SIZE=${NPROC} LOCAL_RANK=1 \
    SPDLOG_LEVEL=err ${R1_ENV} \
    "${BENCH_BIN}" ${R1_ARGS} &
  local PID1=$!

  sleep 0.5

  env MASTER_ADDR=127.0.0.1 MASTER_PORT=${LPORT} RANK=0 WORLD_SIZE=${NPROC} LOCAL_RANK=0 \
    SPDLOG_LEVEL=warn ${R0_ENV} \
    "${BENCH_BIN}" ${R0_ARGS}

  wait ${PID1} 2>/dev/null || true
  echo ""
}

# --- Uniflow send/recv ---
if [[ "${SKIP_UNIFLOW}" != "true" ]]; then
  SLAB_ARGS=""
  SLAB_LABEL=""
  if [[ -n "${SLAB_SIZE}" ]]; then
    SLAB_ARGS="${SLAB_ARGS} --slab-size ${SLAB_SIZE}"
    SLAB_LABEL=", slabSize=$(( SLAB_SIZE / 1024 ))K"
  fi
  if [[ -n "${SLAB_NUM}" ]]; then
    SLAB_ARGS="${SLAB_ARGS} --slab-num ${SLAB_NUM}"
    SLAB_LABEL="${SLAB_LABEL}, slabNum=${SLAB_NUM}"
  fi
  SENDRECV_ARGS="--benchmark sendrecv_bandwidth --topology ${TOPOLOGY} ${COMMON_ARGS} --chunk-size ${CHUNK_SIZE} --pipeline-depth ${PIPELINE_DEPTH}${SLAB_ARGS}"
  run_benchmark \
    "Uniflow send/recv (${TOPOLOGY}, ${MEM_LABEL}, chunk=$(( CHUNK_SIZE / 1024 ))K, depth=${PIPELINE_DEPTH}${SLAB_LABEL})" \
    "${GPU0_ARG} ${NIC0_ARGS} ${SENDRECV_ARGS}" \
    "${GPU1_ARG} ${NIC1_ARGS} ${SENDRECV_ARGS}"
fi

# --- NCCL sendrecv baseline ---
if [[ "${SKIP_NCCL}" != "true" ]]; then
  NCCL_ARGS="--benchmark nccl_sendrecv ${COMMON_ARGS} --topology ${TOPOLOGY}"
  NCCL_ENV=""
  NCCL_LABEL="NCCL sendrecv (${MEM_LABEL})"
  if [[ "${NCCL_NET}" == "true" ]]; then
    NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1"
    NCCL_LABEL="NCCL sendrecv (${MEM_LABEL}, network only)"
  fi
  if [[ "${NO_GDR}" == "true" ]]; then
    NCCL_ENV="${NCCL_ENV} NCCL_NET_GDR_LEVEL=0"
    NCCL_LABEL="${NCCL_LABEL}, no-GDR"
  fi
  run_benchmark \
    "${NCCL_LABEL}" \
    "${GPU0_ARG} ${NCCL_ARGS}" \
    "${GPU1_ARG} ${NCCL_ARGS}" \
    "${NCCL_ENV}" \
    "${NCCL_ENV}"
fi

echo "=== Done ==="
