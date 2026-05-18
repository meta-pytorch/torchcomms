#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# End-to-end RDMA benchmark: runs uniflow + ib_write_bw across multiple
# chunk sizes and generates comparison tables.
#
# Usage:
#   bash rdma_benchmark.sh --host0 <host> --host1 <host> [OPTIONS]
#
# Options:
#   --host0 <host>       Rank 0 host (required)
#   --host1 <host>       Rank 1 host (required)
#   --output-dir <dir>   Results directory (default: ~/me/benchmarks/rdma/<gpu>/<mode>)
#   --iterations <n>     Iterations per size (default: 500)
#   --warmup <n>         Warmup iterations (default: 10)
#   --min-size <n>       Min message size (default: 1)
#   --max-size <n>       Max message size (default: 1073741824)
#   --chunks <list>      Comma-separated chunk sizes in bytes (default: 524288,1048576,2097152,4194304)
#   --batch-size <n>     Requests per transport call (default: 1)
#   --tx-depth <n>       Outstanding transport calls before waiting (default: 128)
#   --num-nics <n>       Cap number of NICs to use, 0 = all (default: 0)
#   --rebuild            Force rebuild of binaries
#   --force-copy         Force re-copy binaries to remote hosts
#   --skip-ib            Skip ib_write_bw (uniflow only)
#   --skip-uniflow       Skip uniflow (ib_write_bw only)
#
# Example:
#   bash rdma_benchmark.sh --host0 rtptest2356.nha6 --host1 rtptest2357.nha6
#   bash rdma_benchmark.sh --host0 rtptest2356.nha6 --host1 rtptest2357.nha6 --skip-ib
#   bash rdma_benchmark.sh --host0 rtptest2356.nha6 --host1 rtptest2357.nha6 --rebuild --force-copy

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_rdma_bandwidth.py"
COMPARISON_SCRIPT="${SCRIPT_DIR}/generate_comparison.py"

HOST0=""
HOST1=""
OUTPUT_DIR=""
ITERATIONS=500
WARMUP=10
MIN_SIZE=1
MAX_SIZE=1073741824
CHUNKS="524288,1048576,2097152,4194304"
BATCH_SIZE=1
TX_DEPTH=128
NUM_NICS=0
REBUILD=0
FORCE_COPY=0
SKIP_IB=0
SKIP_UNIFLOW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host0)        HOST0="$2"; shift 2 ;;
    --host1)        HOST1="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --iterations)   ITERATIONS="$2"; shift 2 ;;
    --warmup)       WARMUP="$2"; shift 2 ;;
    --min-size)     MIN_SIZE="$2"; shift 2 ;;
    --max-size)     MAX_SIZE="$2"; shift 2 ;;
    --chunks)       CHUNKS="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --tx-depth)     TX_DEPTH="$2"; shift 2 ;;
    --num-nics)     NUM_NICS="$2"; shift 2 ;;
    --rebuild)      REBUILD=1; shift ;;
    --force-copy)   FORCE_COPY=1; shift ;;
    --skip-ib)      SKIP_IB=1; shift ;;
    --skip-uniflow) SKIP_UNIFLOW=1; shift ;;
    *)              echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "${HOST0}" || -z "${HOST1}" ]]; then
  echo "Usage: $0 --host0 <host> --host1 <host> [OPTIONS]"
  exit 1
fi

# Parse chunk sizes
IFS=',' read -ra CHUNK_SIZES <<< "${CHUNKS}"
CHUNK_LABELS=()
for cs in "${CHUNK_SIZES[@]}"; do
  if [[ ${cs} -ge 1048576 ]]; then
    CHUNK_LABELS+=("$((cs / 1048576))MB")
  else
    CHUNK_LABELS+=("$((cs / 1024))KB")
  fi
done

# Determine mode
if [[ "${HOST0}" == "${HOST1}" ]]; then
  MODE="intra-host"
else
  MODE="inter-host"
fi

# Default output directory
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${HOME}/me/benchmarks/rdma/${MODE}"
fi

ts() { date "+%H:%M:%S"; }

echo "============================================================"
echo "  RDMA Benchmark Suite"
echo "============================================================"
echo "  Host 0:     ${HOST0}"
echo "  Host 1:     ${HOST1}"
echo "  Mode:       ${MODE}"
echo "  Chunks:     ${CHUNK_LABELS[*]}"
echo "  Iterations: ${ITERATIONS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  TX depth:   ${TX_DEPTH}"
[[ ${NUM_NICS} -gt 0 ]] && echo "  Num NICs:   ${NUM_NICS}"
echo "  Sizes:      ${MIN_SIZE} - ${MAX_SIZE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "============================================================"
echo

COMMON_ARGS=(
  --hosts "${HOST0},${HOST1}"
  --iterations "${ITERATIONS}"
  --warmup "${WARMUP}"
  --min-size "${MIN_SIZE}"
  --max-size "${MAX_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --tx-depth "${TX_DEPTH}"
  --num-nics "${NUM_NICS}"
)

TOTAL=${#CHUNK_SIZES[@]}
SWEEP_START=$(date +%s)

# ── Phase 1: Uniflow ────────────────────────────────────────

if [[ ${SKIP_UNIFLOW} -eq 0 ]]; then
  echo "[$(ts)] Phase 1: Uniflow benchmarks"
  echo

  for i in "${!CHUNK_SIZES[@]}"; do
    cs="${CHUNK_SIZES[$i]}"
    label="${CHUNK_LABELS[$i]}"
    chunk_dir="${OUTPUT_DIR}/chunk_${label}"
    mkdir -p "${chunk_dir}"

    echo "[$(ts)] [$(( i + 1 ))/${TOTAL}] uniflow chunk_${label}..."

    # --rebuild and --force-copy only on first chunk
    EXTRA=()
    if [[ ${i} -eq 0 ]]; then
      [[ ${REBUILD} -eq 1 ]] && EXTRA+=(--rebuild)
      [[ ${FORCE_COPY} -eq 1 ]] && EXTRA+=(--force-copy)
    fi

    # Retry up to 3 times per chunk
    for attempt in 1 2 3; do
      python3 "${COMPARE_SCRIPT}" \
        --chunk-size "${cs}" \
        --tool uniflow \
        --save-csv "${chunk_dir}/results.csv" \
        "${COMMON_ARGS[@]}" \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        2>&1 | tee "${chunk_dir}/uniflow.log"
      rc=${PIPESTATUS[0]}

      if [[ ${rc} -eq 0 ]]; then
        echo "[$(ts)] uniflow chunk_${label}: OK"
        break
      fi

      if [[ ${attempt} -lt 3 ]]; then
        echo "[$(ts)] uniflow chunk_${label}: FAILED (attempt ${attempt}/3), retrying in 30s..."
        sleep 30
      else
        echo "[$(ts)] uniflow chunk_${label}: FAILED after 3 attempts"
      fi
    done
    echo
  done
fi

# ── Phase 2: ib_write_bw ───────────────────────────────────

if [[ ${SKIP_IB} -eq 0 ]]; then
  echo "[$(ts)] Phase 2: ib_write_bw baseline (single run, chunk-size independent)"
  echo "[$(ts)] Waiting 30s for SSH cooldown..."
  sleep 30
  echo

  ib_log="${OUTPUT_DIR}/ib_baseline.log"

  for attempt in 1 2 3; do
    python3 "${COMPARE_SCRIPT}" \
      --chunk-size "${CHUNK_SIZES[0]}" \
      --tool ib \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "${ib_log}"
    rc=${PIPESTATUS[0]}

    if [[ ${rc} -eq 0 ]]; then
      echo "[$(ts)] ib_write_bw: OK"
      break
    fi

    if [[ ${attempt} -lt 3 ]]; then
      echo "[$(ts)] ib_write_bw: FAILED (attempt ${attempt}/3), retrying in 30s..."
      sleep 30
    else
      echo "[$(ts)] ib_write_bw: FAILED after 3 attempts"
    fi
  done
  echo
fi

# ── Phase 3: Comparison tables ──────────────────────────────

echo "[$(ts)] Phase 3: Generating comparison tables..."

# Find the ib_write_bw baseline
IB_BASELINE="${OUTPUT_DIR}/ib_baseline.log"
if [[ ! -f "${IB_BASELINE}" ]]; then
  # Fall back to results.log from previous runs
  IB_BASELINE=""
  for label in "${CHUNK_LABELS[@]}"; do
    candidate="${OUTPUT_DIR}/chunk_${label}/results.log"
    if [[ -f "${candidate}" ]] && grep -q "ib_write_bw" "${candidate}" 2>/dev/null; then
      IB_BASELINE="${candidate}"
      break
    fi
  done
fi

if [[ -n "${IB_BASELINE}" ]]; then
  python3 "${COMPARISON_SCRIPT}" "${OUTPUT_DIR}" --ib-baseline "${IB_BASELINE}"
else
  python3 "${COMPARISON_SCRIPT}" "${OUTPUT_DIR}"
fi

# ── Summary ─────────────────────────────────────────────────

SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))

echo
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
printf "  Total time: %dh %dm %ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "  Results:     ${OUTPUT_DIR}"
echo
[[ -f "${OUTPUT_DIR}/ib_baseline.log" ]] && echo "  ib baseline: ib_baseline.log"
echo "  Files per chunk:"
for label in "${CHUNK_LABELS[@]}"; do
  dir="${OUTPUT_DIR}/chunk_${label}"
  echo -n "    chunk_${label}: "
  files=()
  [[ -f "${dir}/results.csv" ]] && files+=("results.csv")
  [[ -f "${dir}/results.log" ]] && files+=("results.log")
  [[ -f "${dir}/uniflow.log" ]] && files+=("uniflow.log")
  echo "${files[*]}"
done

# Print the best chunk's comparison table
BEST_CHUNK=""
BEST_BW=0
for label in "${CHUNK_LABELS[@]}"; do
  csv="${OUTPUT_DIR}/chunk_${label}/results.csv"
  if [[ -f "${csv}" ]]; then
    bw=$(tail -1 "${csv}" | cut -d, -f9)
    if [[ -n "${bw}" ]] && python3 -c "exit(0 if float('${bw}') > float('${BEST_BW}') else 1)" 2>/dev/null; then
      BEST_BW="${bw}"
      BEST_CHUNK="${label}"
    fi
  fi
done

if [[ -n "${BEST_CHUNK}" && -f "${OUTPUT_DIR}/chunk_${BEST_CHUNK}/results.log" ]]; then
  echo
  echo "  Best chunk: ${BEST_CHUNK} (peak ${BEST_BW} GB/s)"
  echo
  cat "${OUTPUT_DIR}/chunk_${BEST_CHUNK}/results.log"
fi

echo "============================================================"
