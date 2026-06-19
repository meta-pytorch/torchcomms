#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -euo pipefail

WORLD_SIZES="1,2,3,4,5,6,7,8"
TOPOLOGIES="NVL_ONLY,IB_ONLY"
ALGORITHMS="nccl_tree,nccl_tree_simple,ctree"
ARCH="auto"
MIN_BYTES="4"
MAX_BYTES="1G"
FACTOR="2"
ITERS="20"
WARMUP="5"
BUCK_MODE="${BUCK_MODE:-@fbcode//mode/opt}"
BUCK_TARGET="${BUCK_TARGET:-fbsource//third-party/nccl-tests/master:nccl_allreduce_perf}"
MPI_BIN="${MPI_BIN:-/usr/local/fbcode/platform010/bin/mpirun}"

usage() {
  cat <<'EOF'
Usage: benchmark_ctree_allreduce.sh [options]

Options:
  --world-sizes CSV   World sizes to run locally (default: 1,2,3,4,5,6,7,8)
  --topologies CSV    NVL_ONLY,IB_ONLY, or both (default: NVL_ONLY,IB_ONLY)
  --algorithms CSV    nccl,nccl_tree,nccl_tree_simple,ctree,cthierarchical_ring (default: nccl_tree,nccl_tree_simple,ctree)
                      (despite the script name, it is multi-algorithm)
  --arch NAME         auto,h100,gb200,gb300 (default: auto)
  --min-bytes VALUE   nccl_allreduce_perf -b value (default: 4)
  --max-bytes VALUE   nccl_allreduce_perf -e value (default: 1G)
  --factor VALUE      nccl_allreduce_perf -f value (default: 2)
  --iters VALUE       nccl_allreduce_perf -n value (default: 20)
  --warmup VALUE      nccl_allreduce_perf -w value (default: 5)
  --buck-mode VALUE   Buck mode to build nccl_allreduce_perf (default: @fbcode//mode/opt)
  --buck-target VALUE Buck target for nccl_allreduce_perf (default: third-party nccl-tests)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world-sizes) WORLD_SIZES="$2"; shift 2 ;;
    --topologies) TOPOLOGIES="$2"; shift 2 ;;
    --algorithms) ALGORITHMS="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --min-bytes) MIN_BYTES="$2"; shift 2 ;;
    --max-bytes) MAX_BYTES="$2"; shift 2 ;;
    --factor) FACTOR="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --buck-mode) BUCK_MODE="$2"; shift 2 ;;
    --buck-target) BUCK_TARGET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -x "$MPI_BIN" ]]; then
  if command -v mpirun >/dev/null 2>&1; then
    MPI_BIN="$(command -v mpirun)"
  else
    echo "mpirun not found; set MPI_BIN to the MPI launcher path" >&2
    exit 2
  fi
fi

gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi --query-gpu=name --format=csv,noheader | python3 -c \
    'import sys; print(sum(1 for line in sys.stdin if line.strip()))'
}

detect_arch() {
  if [[ "$ARCH" != "auto" ]]; then
    echo "$ARCH"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "h100"
    return
  fi
  nvidia-smi --query-gpu=name --format=csv,noheader | python3 -c '
import sys
names = " ".join(line.lower() for line in sys.stdin)
if "gb300" in names:
    print("gb300")
elif "gb200" in names or "b200" in names:
    print("gb200")
else:
    print("h100")
'
}

nvcc_arch_for_build() {
  local arch="$1"
  if [[ "$arch" == "gb200" || "$arch" == "gb300" ]]; then
    echo "b200"
  else
    echo "h100"
  fi
}

build_perf_binary() {
  local arch="$1"
  local json_file
  json_file="$(mktemp)"
  local -a build_flags=(
    "$BUCK_MODE"
    -c fbcode.enable_gpu_sections=true
    -c hpc_comms.use_ncclx=stable
    -c "fbcode.nvcc_arch=$(nvcc_arch_for_build "$arch")"
  )
  if [[ "$arch" == "gb200" || "$arch" == "gb300" ]]; then
    build_flags+=(-c fbcode.platform010-aarch64_clang=17)
  fi
  if [[ -n "${USE_MCCL_ADAPTER:-}" ]]; then
    # Link the MCCL adapter so the binary exercises the LOCAL comms/ctran
    # AllReduce code (e.g. cthierarchical_ring and the measurement-only
    # NCCL_CTRAN_ALLREDUCE_RING_FORCE_NBLOCKS override) instead of the pinned
    # use_ncclx=stable build. Required when sweeping uncommitted kernel changes.
    build_flags+=(-c hpc_comms.nccl_testonly_override=nccl_adapter -c nccl.enable_profapi=True)
  fi
  if ! buck2 build "${build_flags[@]}" \
    "$BUCK_TARGET" --show-full-json-output >"$json_file"; then
    rm -f "$json_file"
    return 1
  fi

  local perf_bin
  if ! perf_bin="$(python3 - "$json_file" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

for outputs in data.values():
    if isinstance(outputs, dict):
        for value in outputs.values():
            if isinstance(value, str) and value.endswith("nccl_allreduce_perf"):
                print(value)
                sys.exit(0)
    if isinstance(outputs, str) and outputs.endswith("nccl_allreduce_perf"):
        print(outputs)
        sys.exit(0)

raise SystemExit("could not locate nccl_allreduce_perf in buck output")
PY
)"; then
    rm -f "$json_file"
    return 1
  fi

  rm -f "$json_file"
  printf '%s\n' "$perf_bin"
}

read_csv_values() {
  local csv="$1"
  local -n parsed_values="$2"
  local -a raw_values
  IFS=',' read -r -a raw_values <<<"$csv"
  parsed_values=()
  local value
  for value in "${raw_values[@]}"; do
    value="${value//[[:space:]]/}"
    if [[ -n "$value" ]]; then
      parsed_values+=("$value")
    fi
  done
}

array_contains() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    if [[ "$value" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

validate_csv_values() {
  local field_name="$1"
  local -n values="$2"
  shift 2
  local -a allowed_values=("$@")
  local supported
  local IFS=,
  supported="${allowed_values[*]}"
  local value
  for value in "${values[@]}"; do
    if ! array_contains "$value" "${allowed_values[@]}"; then
      echo "Unsupported ${field_name}: ${value}. Supported values: ${supported}" >&2
      exit 2
    fi
  done
}

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}

run_one() {
  local perf_bin="$1"
  local arch="$2"
  local topology="$3"
  local algorithm="$4"
  local np="$5"
  local min_bytes="$6"
  local max_bytes="$7"
  local -a envs=()

  if [[ "$topology" == "IB_ONLY" ]]; then
    envs+=("NCCL_MNNVL_ENABLE=0" "NCCL_P2P_DISABLE=1")
    envs+=("NCCL_COMM_STATE_DEBUG_TOPO=nolocal" "NCCL_IGNORE_TOPO_LOAD_FAILURE=1")
    if [[ "$arch" == "gb200" || "$arch" == "gb300" ]]; then
      envs+=("NCCL_CTRAN_IB_DEVICES_PER_RANK=2")
    fi
  fi

  envs+=("NCCL_DEBUG=${NCCL_DEBUG:-WARN}")

  if [[ "$algorithm" == "nccl_tree" || "$algorithm" == "nccl_tree_simple" ]]; then
    envs+=("NCCL_ALGO=Tree")
    if [[ "$algorithm" == "nccl_tree_simple" ]]; then
      envs+=("NCCL_PROTO=Simple")
    fi
  fi

  # ctree and cthierarchical_ring share the same CTran selection env; the
  # algorithm name is the NCCL_ALLREDUCE_ALGO CVAR token for both.
  if [[ "$algorithm" == "ctree" || "$algorithm" == "cthierarchical_ring" ]]; then
    envs+=("NCCL_CTRAN_ENABLE=1")
    envs+=("NCCL_ALLREDUCE_ALGO=${algorithm}")
    envs+=("NCCL_CTRAN_USE_PIPES=1")
    envs+=("NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=${NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE:-33554432}")
    envs+=("NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1")
  fi

  # Forward optional ring-tuning knobs (only when set in the caller's env) so a
  # parameter sweep can vary block count / pipeline / QP depth without further
  # script edits. Applies to cthierarchical_ring only.
  if [[ "$algorithm" == "cthierarchical_ring" ]]; then
    local knob
    for knob in NCCL_CTRAN_ALLREDUCE_RING_FORCE_NBLOCKS NCCL_CTRAN_MAX_NBLOCKS \
      NCCL_CTRAN_IBGDA_SENDRECV_PIPELINE_DEPTH NCCL_CTRAN_IBGDA_QP_DEPTH; do
      if [[ -n "${!knob:-}" ]]; then
        envs+=("${knob}=${!knob}")
      fi
    done
  fi

  local -a mpi_envs=()
  for env_kv in "${envs[@]}"; do
    mpi_envs+=("-x" "$env_kv")
  done

  echo
  if [[ "$min_bytes" == "$max_bytes" ]]; then
    echo "== ${algorithm} ${topology} np=${np} arch=${arch} size=${min_bytes} =="
  else
    echo "== ${algorithm} ${topology} np=${np} arch=${arch} size=${min_bytes}..${max_bytes} =="
  fi

  # cthierarchical_ring is fp32/sum-only; its support gate does NOT check
  # dtype/op, so a non-fp32/non-sum run passes the gate then hard-fails inside
  # ctranAllReduceHierarchicalRing. Pin -d float -o sum (-c 1 = explicit check).
  # Leave ctree/nccl* on binary defaults to preserve current behavior.
  local -a dtype_args=()
  if [[ "$algorithm" == "cthierarchical_ring" ]]; then
    dtype_args+=("-d" "float" "-o" "sum" "-c" "1")
  fi

  local -a cmd=(
    env "${envs[@]}" "$MPI_BIN" --allow-run-as-root --bind-to none
    --mca btl "^openib" -np "$np" -host "localhost:${np}"
    "${mpi_envs[@]}"
    "$perf_bin" -b "$min_bytes" -e "$max_bytes" -f "$FACTOR" -g 1
    -n "$ITERS" -w "$WARMUP" ${dtype_args[@]+"${dtype_args[@]}"}
  )

  print_command "${cmd[@]}"
  local start_ns
  start_ns="$(date +%s%N)"
  "${cmd[@]}"
  local end_ns
  end_ns="$(date +%s%N)"
  local elapsed_ms
  elapsed_ms=$(((end_ns - start_ns) / 1000000))
  printf 'elapsed_seconds topology=%s algorithm=%s np=%s seconds=%s.%03d\n' \
    "$topology" \
    "$algorithm" \
    "$np" \
    "$((elapsed_ms / 1000))" \
    "$((elapsed_ms % 1000))"
}

main() {
  local -a topology_values
  read_csv_values "$TOPOLOGIES" topology_values
  validate_csv_values "topology" topology_values NVL_ONLY IB_ONLY

  local -a algorithm_values
  read_csv_values "$ALGORITHMS" algorithm_values
  validate_csv_values "algorithm" algorithm_values nccl nccl_tree nccl_tree_simple ctree cthierarchical_ring

  local arch
  arch="$(detect_arch)"
  local ngpus
  ngpus="$(gpu_count)"
  local perf_bin
  perf_bin="$(build_perf_binary "$arch")"

  local -a sizes
  read_csv_values "$WORLD_SIZES" sizes
  for np in "${sizes[@]}"; do
    if [[ "$ngpus" -gt 0 && "$np" -gt "$ngpus" ]]; then
      echo "skip np=${np}: host has ${ngpus} GPUs"
      continue
    fi
    for topology in "${topology_values[@]}"; do
      for algorithm in "${algorithm_values[@]}"; do
        run_one "$perf_bin" "$arch" "$topology" "$algorithm" "$np" \
          "$MIN_BYTES" "$MAX_BYTES"
      done
    done
  done
}

main "$@"
