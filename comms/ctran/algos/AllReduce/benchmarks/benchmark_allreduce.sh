#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBSOURCE_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
CURRENT_USER="${USER:-$(id -un)}"

WORLD_SIZES="1,2,3,4,5,6,7,8"
TOPOLOGIES="NVL_ONLY,IB_ONLY"
VARIANTS="ctree,nccl_tree_simple"
ARCH="auto"
ARCHS="h100,gb300"
NODES="2,4,8,16,32"
MIN_BYTES="1"
MAX_BYTES="2G"
FACTOR="2"
ITERS="100"
WARMUP="5"
DTYPE="${DTYPE:-commFloat32}"
OP="${OP:-sum}"
LAUNCHER="local"
PPN="1"
TENANT="msl_tbd_iris"
DP="networkai_mast_job_identity"
CLUSTER="MastGenAICluster"
MAST_TIMEOUT="3600"
MAST_PRIORITY="${MAST_PRIORITY:-}"
H100_HW="${H100_HW:-grandteton_80g_roce}"
H100_LOCALITY="${H100_LOCALITY:-region;pci}"
GB300_LOCALITY="${GB300_LOCALITY:-region;lco}"
BHA="32"
GB300_CTREE_MAX_NBLOCKS="${GB300_CTREE_MAX_NBLOCKS:-8}"
GB300_CTREE_IB_MAX_GROUPS="${GB300_CTREE_IB_MAX_GROUPS:-16}"
GB300_VPPN="${GB300_VPPN:-}"
DRY_RUN="0"
SKIP_SMOKE="0"
REPORT="0"
REPORT_OUTPUT=""
REPORT_PREFS="${REPORT_PREFS:-}"
REPORT_FORCE_DOWNLOAD="0"
JOBS_MANIFEST="${JOBS_MANIFEST:-/tmp/${CURRENT_USER}/allreduce_benchmark_jobs.tsv}"
SMOKE_MIN_BYTES="1"
SMOKE_MAX_BYTES="1M"
SMOKE_ITERS="5"
SMOKE_WARMUP="1"
BUCK_MODE="${BUCK_MODE:-@fbcode//mode/opt}"
BUCK_TARGET="${BUCK_TARGET:-fbsource//third-party/nccl-tests/master:nccl_allreduce_perf}"
MPI_BIN="${MPI_BIN:-/usr/local/fbcode/platform010/bin/mpirun}"
NCCL_TEST_SCRIPT="${NCCL_TEST_SCRIPT:-${FBSOURCE_ROOT}/fbcode/hpc_comms/tests/nccl-tests-suite/scripts/run_nccl_tests.sh}"
NCCL_REPORT_SCRIPT="${NCCL_REPORT_SCRIPT:-${FBSOURCE_ROOT}/fbcode/hpc_comms/tests/nccl-tests-suite/scripts/generate_comparison_report.py}"
H100_IMG="${H100_IMG:-}"
GB300_IMG="${GB300_IMG:-}"
MAST_IMG="${MAST_IMG:-}"
H100_NVCC_CONFIG="${H100_NVCC_CONFIG:--c fbcode.nvcc_arch=h100a -c fbcode.platform010_cuda_version=13.0 -m cuda13}"
GB300_NVCC_CONFIG="${GB300_NVCC_CONFIG:--c fbcode.nvcc_arch=b300a_native -c fbcode.platform010_cuda_version=13.0 -m cuda13}"
REMOTE_HOST=""
REMOTE_DIR=""
REMOTE_KEEP=0
REMOTE_EXTRA_ENV=""
REMOTE_MPI_BIN="${REMOTE_MPI_BIN:-/usr/local/fbcode/bin/mpirun}"
LOG_DIR=""
SUSH_BIN="${SUSH_BIN:-sush}"
SUSCP_BIN="${SUSCP_BIN:-suscp}"
REMOTE_PERF_BIN=""
BLOCK_CAPS=""
SUMMARY_CSV=""
CONTINUE_ON_ERROR=0
SPLIT_SIZES=0
RUN_TIMEOUT_SECONDS=""
RUN_RETRIES=0

usage() {
  cat <<'EOF'
Usage: benchmark_allreduce.sh [options]

Options:
  --launcher local|mast  Run local mpirun benchmarks or submit MAST jobs (default: local)
  --world-sizes CSV      World sizes to run locally (default: 1,2,3,4,5,6,7,8)
  --topologies CSV       NVL_ONLY,IB_ONLY, or both (default: NVL_ONLY,IB_ONLY)
  --variants CSV         nccl,nccl_tree,nccl_tree_simple,nccl_ring_simple,ctree,ctring (default: ctree,nccl_tree_simple)
  --algorithms CSV       Deprecated alias for --variants
  --arch NAME            Local build architecture: auto,h100,gb200,gb300 (default: auto)
  --archs CSV            MAST architectures to launch: h100,gb300; platform fbpkg builds run in parallel (default: h100,gb300)
  --nodes CSV            MAST node counts to launch (default: 2,4,8,16,32)
  --ppn N                MAST processes/GPUs per node (default: 1)
  --tenant NAME          MAST entitlement/rmAttribution (default: msl_tbd_iris)
  --dp NAME              MAST hpcIdentity (default: networkai_mast_job_identity)
  --cluster NAME         MAST cluster UUID/name (default: MastGenAICluster)
  --priority NAME        Optional MAST priority, e.g. HIGH
  --h100-hw NAME         H100 MAST hardware target (default: grandteton_80g_roce)
  --h100-locality VAL    H100 locality constraints (default: region;pci)
  --gb300-locality VAL   GB300 locality constraints (default: region;lco)
  --bha N                GB300 max BHA domain multiple; capped at node count (default: 32)
  --min-bytes VALUE      nccl_allreduce_perf -b value (default: 1)
  --max-bytes VALUE      nccl_allreduce_perf -e value (default: 2G)
  --factor VALUE         nccl_allreduce_perf -f value (default: 2)
  --iters VALUE          nccl_allreduce_perf -n value (default: 100)
  --warmup VALUE         nccl_allreduce_perf -w value (default: 5)
  --dtype VALUE          nccl_allreduce_perf -d value; commFloat32 maps to float (default: commFloat32)
  --op VALUE             Reduction op used for report parsing (default: sum)
  --dry-run              Print MAST commands without building/submitting jobs
  --skip-smoke           Skip the 2-node smoke launches before the full MAST sweep
  --jobs-manifest PATH   TSV manifest for launched MAST jobs (default: /tmp/$USER/allreduce_benchmark_jobs.tsv)
  --report               Generate an HTML report from completed jobs in --jobs-manifest, then exit
  --report-output PATH   Optional HTML report output path; defaults next to --jobs-manifest
  --report-prefs PATH    JSON report preferences path; defaults next to --jobs-manifest
  --report-html PATH     Deprecated alias for --report --report-output PATH
  --force-download       Re-download MAST logs while generating --report
  --img VALUE            Reuse one fbpkg image for all MAST architectures
  --h100-img VALUE       Reuse an H100 fbpkg image
  --gb300-img VALUE      Reuse a GB300 fbpkg image
  --buck-mode VALUE      Buck mode to build nccl_allreduce_perf (default: @fbcode//mode/opt)
  --buck-target VALUE    Buck target for nccl_allreduce_perf (default: third-party nccl-tests)
  --remote-host HOST  Copy the binary to HOST and run there with sush
  --remote-dir DIR    Remote working directory (default: /tmp/$USER/ctree_allreduce_TIMESTAMP)
  --remote-keep       Keep --remote-dir after the run
  --remote-extra-env  Space-separated KEY=VALUE envs added to every remote run
  --remote-mpi-bin    MPI launcher path on the remote host (default: /usr/local/fbcode/bin/mpirun)
  --log-dir DIR       Directory for raw per-run logs
  --block-caps CSV    CTREE-only NCCL_CTRAN_MAX_NBLOCKS values to sweep
  --summary-csv PATH  Append parsed nccl-tests result rows to PATH
  --split-sizes       Run each generated size as a separate benchmark command
  --run-timeout SEC   Wrap each benchmark command in timeout --foreground SEC
  --retries N         Retry each failed benchmark command N times (default: 0)
  --continue-on-error Continue after a benchmark command exits non-zero
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --world-sizes) WORLD_SIZES="$2"; shift 2 ;;
    --topologies) TOPOLOGIES="$2"; shift 2 ;;
    --variants|--algorithms) VARIANTS="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --archs) ARCHS="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --ppn) PPN="$2"; shift 2 ;;
    --tenant) TENANT="$2"; shift 2 ;;
    --dp) DP="$2"; shift 2 ;;
    --cluster) CLUSTER="$2"; shift 2 ;;
    --priority) MAST_PRIORITY="$2"; shift 2 ;;
    --h100-hw) H100_HW="$2"; shift 2 ;;
    --h100-locality) H100_LOCALITY="$2"; shift 2 ;;
    --gb300-locality) GB300_LOCALITY="$2"; shift 2 ;;
    --bha) BHA="$2"; shift 2 ;;
    --min-bytes) MIN_BYTES="$2"; shift 2 ;;
    --max-bytes) MAX_BYTES="$2"; shift 2 ;;
    --factor) FACTOR="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --op) OP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --skip-smoke) SKIP_SMOKE="1"; shift ;;
    --jobs-manifest) JOBS_MANIFEST="$2"; shift 2 ;;
    --report) REPORT="1"; shift ;;
    --report-output) REPORT_OUTPUT="$2"; shift 2 ;;
    --report-prefs) REPORT_PREFS="$2"; shift 2 ;;
    --report-html) REPORT="1"; REPORT_OUTPUT="$2"; shift 2 ;;
    --force-download) REPORT_FORCE_DOWNLOAD="1"; shift ;;
    --img) MAST_IMG="$2"; shift 2 ;;
    --h100-img) H100_IMG="$2"; shift 2 ;;
    --gb300-img) GB300_IMG="$2"; shift 2 ;;
    --buck-mode) BUCK_MODE="$2"; shift 2 ;;
    --buck-target) BUCK_TARGET="$2"; shift 2 ;;
    --remote-host) REMOTE_HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --remote-keep) REMOTE_KEEP=1; shift ;;
    --remote-extra-env) REMOTE_EXTRA_ENV="$2"; shift 2 ;;
    --remote-mpi-bin) REMOTE_MPI_BIN="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --block-caps) BLOCK_CAPS="$2"; shift 2 ;;
    --summary-csv) SUMMARY_CSV="$2"; shift 2 ;;
    --split-sizes) SPLIT_SIZES=1; shift ;;
    --run-timeout) RUN_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --retries) RUN_RETRIES="$2"; shift 2 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "$RUN_TIMEOUT_SECONDS" &&
      (! "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ || "$RUN_TIMEOUT_SECONDS" -lt 1) ]]; then
  echo "Unsupported run timeout: ${RUN_TIMEOUT_SECONDS}. Expected positive integer seconds." >&2
  exit 2
fi
if [[ ! "$RUN_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "Unsupported retries: ${RUN_RETRIES}. Expected non-negative integer." >&2
  exit 2
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
  if [[ "$arch" == "gb300" ]]; then
    echo "b300a_native"
  elif [[ "$arch" == "gb200" ]]; then
    echo "b200a"
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
    local cuda_version="12.8"
    local cuda_mode="cuda128"
    if [[ "$arch" == "gb300" ]]; then
      cuda_version="13.0"
      cuda_mode="ovr_config//third-party/cuda/constraints:13.0"
    fi
    build_flags+=(
      -c fbcode.arch=aarch64
      -c "fbcode.platform010_cuda_version=${cuda_version}"
      -c fbcode.platform010-aarch64_clang=17
      -m "$cuda_mode"
    )
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

prefer_h100_first() {
  local -n values="$1"
  local -a ordered_values=()
  local value
  if array_contains h100 "${values[@]}"; then
    ordered_values+=(h100)
  fi
  for value in "${values[@]}"; do
    if [[ "$value" != "h100" ]]; then
      ordered_values+=("$value")
    fi
  done
  values=("${ordered_values[@]}")
}

require_executable() {
  local path="$1"
  local description="$2"
  if [[ ! -x "$path" ]]; then
    echo "${description} not found or not executable: ${path}" >&2
    exit 2
  fi
}

normalize_dtype() {
  case "$DTYPE" in
    commFloat32|float32) DTYPE="float" ;;
  esac
}

default_report_prefs() {
  if [[ "$JOBS_MANIFEST" == *.tsv ]]; then
    printf '%s\n' "${JOBS_MANIFEST%.tsv}.report.json"
  else
    printf '%s\n' "${JOBS_MANIFEST}.report.json"
  fi
}

validate_positive_int_values() {
  local field_name="$1"
  local -n values="$2"
  local value
  for value in "${values[@]}"; do
    if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
      echo "Unsupported ${field_name}: ${value}. Expected positive integers." >&2
      exit 2
    fi
  done
}

expand_size_values() {
  local min_bytes="$1"
  local max_bytes="$2"
  local factor="$3"
  python3 - "$min_bytes" "$max_bytes" "$factor" <<'PY'
import re
import sys


UNITS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
}


def parse_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([A-Za-z]*)", value.strip())
    if not match:
        raise SystemExit(f"invalid size: {value}")
    number = int(match.group(1))
    suffix = match.group(2).upper()
    if suffix not in UNITS:
        raise SystemExit(f"unsupported size suffix: {value}")
    return number * UNITS[suffix]


def format_size(value: int) -> str:
    for suffix, unit in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if value >= unit and value % unit == 0:
            return f"{value // unit}{suffix}"
    return str(value)


start = parse_size(sys.argv[1])
end = parse_size(sys.argv[2])
try:
    factor = float(sys.argv[3])
except ValueError as exc:
    raise SystemExit(f"invalid factor: {sys.argv[3]}") from exc

if start < 1 or end < start:
    raise SystemExit("expected 1 <= min-bytes <= max-bytes")
if factor <= 1.0 and start != end:
    raise SystemExit("split size expansion requires factor > 1")

size = start
while size <= end:
    print(format_size(size))
    if size == end:
        break
    next_size = int(size * factor)
    if next_size <= size:
        next_size = size + 1
    size = next_size
PY
}

read_env_values() {
  local env_string="$1"
  local -n parsed_envs="$2"
  local -a raw_envs
  read -r -a raw_envs <<<"$env_string"
  parsed_envs=()
  local env_kv
  for env_kv in "${raw_envs[@]}"; do
    if [[ -n "$env_kv" ]]; then
      if [[ "$env_kv" != *=* ]]; then
        echo "Invalid env override '${env_kv}'; expected KEY=VALUE" >&2
        exit 2
      fi
      parsed_envs+=("$env_kv")
    fi
  done
}

remote_target() {
  if [[ "$REMOTE_HOST" == *@* ]]; then
    printf '%s\n' "$REMOTE_HOST"
  else
    printf 'root@%s\n' "$REMOTE_HOST"
  fi
}

shell_join() {
  printf '%q ' "$@"
}

default_remote_dir() {
  printf '/tmp/%s/ctree_allreduce_%s\n' \
    "${USER:-ctree}" \
    "$(date +%Y%m%d_%H%M%S)"
}

setup_remote_binary() {
  local perf_bin="$1"
  local target
  target="$(remote_target)"
  if [[ -z "$REMOTE_DIR" ]]; then
    REMOTE_DIR="$(default_remote_dir)"
  fi
  REMOTE_PERF_BIN="${REMOTE_DIR}/nccl_allreduce_perf"

  local -a mkdir_cmd=(
    "$SUSH_BIN" --reason "prepare CTREE AllReduce benchmark directory"
    "$target" "mkdir -p $(printf '%q' "$REMOTE_DIR")"
  )
  print_command "${mkdir_cmd[@]}"
  "${mkdir_cmd[@]}"

  local -a copy_cmd=(
    "$SUSCP_BIN" --reason "copy CTREE AllReduce benchmark binary"
    "$perf_bin" "${target}:${REMOTE_PERF_BIN}"
  )
  print_command "${copy_cmd[@]}"
  "${copy_cmd[@]}"

  local -a chmod_cmd=(
    "$SUSH_BIN" --reason "chmod CTREE AllReduce benchmark binary"
    "$target" "chmod +x $(printf '%q' "$REMOTE_PERF_BIN")"
  )
  print_command "${chmod_cmd[@]}"
  "${chmod_cmd[@]}"
}

cleanup_remote_binary() {
  if [[ -n "$REMOTE_HOST" && "$REMOTE_KEEP" -eq 0 && -n "$REMOTE_DIR" ]]; then
    local target
    target="$(remote_target)"
    "$SUSH_BIN" --reason "cleanup CTREE AllReduce benchmark directory" \
      "$target" "rm -rf $(printf '%q' "$REMOTE_DIR")" || true
  fi
}

print_command() {
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
}

variant_uses_ctran() {
  case "$1" in
    ctree|ctring) return 0 ;;
    *) return 1 ;;
  esac
}

variant_envs() {
  local variant="$1"
  case "$variant" in
    nccl)
      ;;
    nccl_tree)
      printf '%s\n' "NCCL_ALLREDUCE_ALGO=orig" "NCCL_ALGO=allreduce:tree"
      ;;
    nccl_tree_simple)
      printf '%s\n' \
        "NCCL_ALLREDUCE_ALGO=orig" \
        "NCCL_ALGO=allreduce:tree" \
        "NCCL_PROTO=Simple" \
        "TORCH_NCCL_BCAST_UNIQUEID=1"
      ;;
    nccl_ring_simple)
      printf '%s\n' \
        "NCCL_ALLREDUCE_ALGO=orig" \
        "NCCL_ALGO=allreduce:ring" \
        "NCCL_PROTO=Simple" \
        "TORCH_NCCL_BCAST_UNIQUEID=1"
      ;;
    ctree)
      printf '%s\n' \
        "NCCL_CTRAN_USE_PIPES=1" \
        "NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE=${NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE:-33554432}" \
        "NCCL_CTRAN_IBGDA_SENDRECV_ENABLE=1"
      ;;
    ctring)
      ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

append_summary_csv() {
  local log_file="$1"
  local topology="$2"
  local algorithm="$3"
  local np="$4"
  local block_cap="$5"
  local exit_code="$6"

  if [[ -z "$SUMMARY_CSV" || -z "$log_file" ]]; then
    return
  fi

  mkdir -p "$(dirname "$SUMMARY_CSV")"
  if [[ ! -e "$SUMMARY_CSV" ]]; then
    printf '%s\n' \
      "algorithm,topology,np,block_cap,size_bytes,oop_time_us,oop_busbw,oop_wrong,ip_time_us,ip_busbw,ip_wrong,exit_code" \
      >"$SUMMARY_CSV"
  fi

  awk \
    -v algorithm="$algorithm" \
    -v topology="$topology" \
    -v np="$np" \
    -v block_cap="$block_cap" \
    -v exit_code="$exit_code" \
    '($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && NF >= 13) {
       printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
         algorithm, topology, np, block_cap, $1, $6, $8, $9, $10, $12, $13, exit_code
     }' "$log_file" >>"$SUMMARY_CSV"
}

run_one() {
  local perf_bin="$1"
  local arch="$2"
  local topology="$3"
  local variant="$4"
  local np="$5"
  local min_bytes="$6"
  local max_bytes="$7"
  local block_cap="$8"
  local -a envs=()
  local -a variant_env_values=()
  local -a extra_envs=()
  read_env_values "$REMOTE_EXTRA_ENV" extra_envs

  if [[ "$topology" == "IB_ONLY" ]]; then
    envs+=("NCCL_MNNVL_ENABLE=0" "NCCL_P2P_DISABLE=1")
    envs+=("NCCL_COMM_STATE_DEBUG_TOPO=nolocal" "NCCL_IGNORE_TOPO_LOAD_FAILURE=1")
    if [[ "$arch" == "gb200" || "$arch" == "gb300" ]]; then
      envs+=("NCCL_CTRAN_IB_DEVICES_PER_RANK=2")
    fi
  fi

  envs+=("NCCL_DEBUG=${NCCL_DEBUG:-WARN}")
  mapfile -t variant_env_values < <(variant_envs "$variant")
  envs+=("${variant_env_values[@]}")
  if [[ -n "${NCCL_DEBUG_FILE:-}" ]]; then
    envs+=("NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}")
  else
    envs+=("NCCL_DEBUG_FILE=/dev/stderr")
  fi

  if variant_uses_ctran "$variant"; then
    envs+=("NCCL_CTRAN_ENABLE=1")
    envs+=("NCCL_ALLREDUCE_ALGO=${variant}")
  fi

  envs+=("${extra_envs[@]}")

  if [[ "$variant" == "ctree" && -n "$block_cap" ]]; then
    envs+=("NCCL_CTRAN_MAX_NBLOCKS=${block_cap}")
  fi

  local -a mpi_envs=()
  local env_kv
  for env_kv in "${envs[@]}"; do
    mpi_envs+=("-x" "$env_kv")
  done

  echo
  local block_desc=""
  if [[ -n "$block_cap" ]]; then
    block_desc=" blocks=${block_cap}"
  fi
  if [[ "$min_bytes" == "$max_bytes" ]]; then
    echo "== ${variant} ${topology} np=${np} arch=${arch}${block_desc} size=${min_bytes} =="
  else
    echo "== ${variant} ${topology} np=${np} arch=${arch}${block_desc} size=${min_bytes}..${max_bytes} =="
  fi

  local -a cmd=(
    env "${envs[@]}" "$MPI_BIN" --allow-run-as-root --bind-to none
    --mca btl "^openib" -np "$np" -host "localhost:${np}"
    "${mpi_envs[@]}"
    "$perf_bin" -b "$min_bytes" -e "$max_bytes" -f "$FACTOR" -g 1
    -n "$ITERS" -w "$WARMUP" -d "$DTYPE"
  )
  if [[ -n "$RUN_TIMEOUT_SECONDS" ]]; then
    cmd=(timeout --foreground "$RUN_TIMEOUT_SECONDS" "${cmd[@]}")
  fi

  local -a run_cmd=("${cmd[@]}")
  if [[ -n "$REMOTE_HOST" ]]; then
    local remote_cmd
    remote_cmd="$(shell_join "${cmd[@]}")"
    run_cmd=(
      "$SUSH_BIN" --reason "run CTREE AllReduce benchmark"
      "$(remote_target)" "$remote_cmd"
    )
  fi

  local log_file=""
  if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    local log_block_suffix=""
    if [[ -n "$block_cap" ]]; then
      log_block_suffix="_blocks${block_cap}"
    fi
    log_file="${LOG_DIR}/${variant}_${topology}_np${np}${log_block_suffix}_${min_bytes}_${max_bytes}.log"
  fi

  print_command "${run_cmd[@]}"
  local summary_log_file="$log_file"
  local attempt=0
  local run_status=0
  while true; do
    local attempt_log_file="$log_file"
    if [[ -n "$log_file" && "$attempt" -gt 0 ]]; then
      attempt_log_file="${log_file}.retry${attempt}"
    fi
    summary_log_file="$attempt_log_file"
    if [[ "$attempt" -gt 0 ]]; then
      printf 'retry topology=%s variant=%s np=%s attempt=%s/%s\n' \
        "$topology" "$variant" "$np" "$attempt" "$RUN_RETRIES"
      print_command "${run_cmd[@]}"
    fi

    local start_ns
    start_ns="$(date +%s%N)"
    if [[ -n "$attempt_log_file" ]]; then
      set +e
      "${run_cmd[@]}" 2>&1 | tee "$attempt_log_file"
      run_status="${PIPESTATUS[0]}"
      set -e
    else
      set +e
      "${run_cmd[@]}"
      run_status="$?"
      set -e
    fi
    local end_ns
    end_ns="$(date +%s%N)"
    local elapsed_ms
    elapsed_ms=$(((end_ns - start_ns) / 1000000))
    printf 'elapsed_seconds topology=%s variant=%s np=%s attempt=%s seconds=%s.%03d exit_code=%s\n' \
      "$topology" \
      "$variant" \
      "$np" \
      "$attempt" \
      "$((elapsed_ms / 1000))" \
      "$((elapsed_ms % 1000))" \
      "$run_status"

    if [[ "$run_status" -eq 0 || "$attempt" -ge "$RUN_RETRIES" ]]; then
      break
    fi
    attempt=$((attempt + 1))
  done
  append_summary_csv \
    "$summary_log_file" "$topology" "$variant" "$np" "$block_cap" "$run_status"
  if [[ "$run_status" -ne 0 && "$CONTINUE_ON_ERROR" -eq 0 ]]; then
    return "$run_status"
  fi
}

run_range() {
  local perf_bin="$1"
  local arch="$2"
  local topology="$3"
  local variant="$4"
  local np="$5"
  local block_cap="$6"
  local -n split_size_values_ref="$7"

  if [[ "$SPLIT_SIZES" -eq 1 ]]; then
    local size
    for size in "${split_size_values_ref[@]}"; do
      run_one "$perf_bin" "$arch" "$topology" "$variant" "$np" \
        "$size" "$size" "$block_cap"
    done
  else
    run_one "$perf_bin" "$arch" "$topology" "$variant" "$np" \
      "$MIN_BYTES" "$MAX_BYTES" "$block_cap"
  fi
}

run_local() {
  if [[ -z "$REMOTE_HOST" && ! -x "$MPI_BIN" ]]; then
    if command -v mpirun >/dev/null 2>&1; then
      MPI_BIN="$(command -v mpirun)"
    else
      echo "mpirun not found; set MPI_BIN to the MPI launcher path" >&2
      exit 2
    fi
  fi

  local -a topology_values
  read_csv_values "$TOPOLOGIES" topology_values
  validate_csv_values "topology" topology_values NVL_ONLY IB_ONLY

  local -a variant_values
  read_csv_values "$VARIANTS" variant_values
  validate_csv_values "variant" variant_values nccl nccl_tree nccl_tree_simple nccl_ring_simple ctree ctring

  local -a block_cap_values=()
  if [[ -n "$BLOCK_CAPS" ]]; then
    read_csv_values "$BLOCK_CAPS" block_cap_values
    validate_positive_int_values "block cap" block_cap_values
  fi

  local -a split_size_values=()
  if [[ "$SPLIT_SIZES" -eq 1 ]]; then
    # shellcheck disable=SC2034
    mapfile -t split_size_values < <(
      expand_size_values "$MIN_BYTES" "$MAX_BYTES" "$FACTOR")
  fi

  local arch
  arch="$(detect_arch)"
  local ngpus
  ngpus="$(gpu_count)"
  local perf_bin
  perf_bin="$(build_perf_binary "$arch")"
  if [[ -n "$REMOTE_HOST" ]]; then
    MPI_BIN="$REMOTE_MPI_BIN"
    if [[ -z "$LOG_DIR" ]]; then
      LOG_DIR="/tmp/${USER:-ctree}/ctree-gb300-$(date +%Y%m%d_%H%M%S)"
    fi
    setup_remote_binary "$perf_bin"
    perf_bin="$REMOTE_PERF_BIN"
    trap cleanup_remote_binary EXIT
    echo "raw logs: ${LOG_DIR}"
  fi

  local -a sizes
  read_csv_values "$WORLD_SIZES" sizes
  local np topology variant
  for np in "${sizes[@]}"; do
    if [[ -z "$REMOTE_HOST" && "$ngpus" -gt 0 && "$np" -gt "$ngpus" ]]; then
      echo "skip np=${np}: host has ${ngpus} GPUs"
      continue
    fi
    for topology in "${topology_values[@]}"; do
      for variant in "${variant_values[@]}"; do
        if [[ "$variant" == "ctree" && "${#block_cap_values[@]}" -gt 0 ]]; then
          for block_cap in "${block_cap_values[@]}"; do
            run_range "$perf_bin" "$arch" "$topology" "$variant" "$np" \
              "$block_cap" split_size_values
          done
        else
          run_range "$perf_bin" "$arch" "$topology" "$variant" "$np" \
            "" split_size_values
        fi
      done
    done
  done
}

join_by_space() {
  local IFS=' '
  printf '%s' "$*"
}

arch_envs() {
  local arch="$1"
  case "$arch" in
    h100)
      printf '%s\n' \
        "NCCL_IB_ADAPTIVE_ROUTING=1" \
        "NCCL_IB_QPS_PER_CONNECTION=16" \
        "NCCL_IB_SPLIT_DATA_ON_QPS=0" \
        "NCCL_MIN_NCHANNELS=4" \
        "NCCL_NET_OVERHEAD=2750"
      ;;
    gb300)
      printf '%s\n' \
        "NCCL_MAX_P2P_NCHANNELS=64" \
        "NCCL_LL128_BUFFSIZE=-2" \
        "NCCL_IB_ADAPTIVE_ROUTING=1" \
        "NCCL_IB_QPS_PER_CONNECTION=4" \
        "NCCL_CTRAN_ALLREDUCE_RING_BIDIR_AG_MAX_SIZE=0" \
        "NCCL_MNNVL_ENABLE=1" \
        "NCCL_MNNVL_DETERMINISTIC_COLLECTIVE_ENABLE=true"
      ;;
    *)
      echo "Unsupported MAST arch: ${arch}" >&2
      exit 2
      ;;
  esac
}

mast_nvcc_config() {
  local arch="$1"
  case "$arch" in
    h100) printf '%s\n' "$H100_NVCC_CONFIG" ;;
    gb300) printf '%s\n' "$GB300_NVCC_CONFIG" ;;
    *) echo "Unsupported MAST arch: ${arch}" >&2; exit 2 ;;
  esac
}

mast_fbpkg_target() {
  local arch="$1"
  case "$arch" in
    h100) printf '%s\n' "nccl_tests_suite" ;;
    gb300) printf '%s\n' "nccl_tests_suite_gb300" ;;
    *) echo "Unsupported MAST arch: ${arch}" >&2; exit 2 ;;
  esac
}

prebuild_mast_arch_image() {
  local arch="$1"
  local current_img
  current_img="$(image_for_arch "$arch")"
  if [[ -n "$current_img" ]]; then
    echo "Reusing provided ${arch} fbpkg: ${current_img}"
    return 0
  fi

  local nvcc_config
  nvcc_config="$(mast_nvcc_config "$arch")"
  local fbpkg_target
  fbpkg_target="$(mast_fbpkg_target "$arch")"
  local -a cmd=(
    env
    "NVCC_CONFIG=${nvcc_config}"
    "FBPKG_TARGET=${fbpkg_target}"
    "$NCCL_TEST_SCRIPT"
    --collective allreduce
    --min-nodes 1
    --max-nodes 1
    --ppn "$PPN"
    --build-only
    --nccl-debug INFO
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  echo "== mast prebuild arch=${arch} target=${fbpkg_target} =="
  print_command "${cmd[@]}"
  "${cmd[@]}"
}

prebuild_mast_images() {
  local -a arch_values=("$@")
  local log_dir="/tmp/${CURRENT_USER}"
  mkdir -p "$log_dir"

  local -a build_arches=()
  local -a build_logs=()
  local -a build_pids=()
  local arch
  for arch in "${arch_values[@]}"; do
    if [[ -n "$(image_for_arch "$arch")" ]]; then
      echo "Reusing provided ${arch} fbpkg: $(image_for_arch "$arch")"
      continue
    fi

    local log_file="${log_dir}/allreduce_${arch}_prebuild.$$.log"
    echo "Starting ${arch} fbpkg prebuild in background; log: ${log_file}"
    (
      prebuild_mast_arch_image "$arch"
    ) >"$log_file" 2>&1 &
    build_arches+=("$arch")
    build_logs+=("$log_file")
    build_pids+=("$!")
  done

  if [[ "${#build_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  local failed=0
  local i
  for i in "${!build_pids[@]}"; do
    if ! wait "${build_pids[$i]}"; then
      failed=1
    fi
  done

  for i in "${!build_logs[@]}"; do
    echo
    echo "== ${build_arches[$i]} prebuild output =="
    cat "${build_logs[$i]}"
  done

  if [[ "$failed" != "0" ]]; then
    echo "One or more platform fbpkg prebuilds failed." >&2
    return 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    for arch in "${build_arches[@]}"; do
      set_image_for_arch "$arch" "<dry-run>"
    done
    return 0
  fi

  for i in "${!build_arches[@]}"; do
    arch="${build_arches[$i]}"
    local fbpkg_target
    fbpkg_target="$(mast_fbpkg_target "$arch")"
    local created_img
    created_img="$(extract_created_img "${build_logs[$i]}" "$fbpkg_target")"
    if [[ -z "$created_img" ]]; then
      echo "Failed to extract ${arch} fbpkg image from ${build_logs[$i]}" >&2
      return 1
    fi
    set_image_for_arch "$arch" "$created_img"
    echo "Reusing ${created_img} for ${arch} launches."
  done
}

write_gb300_bha_config() {
  local multiple="$1"
  local dir="/tmp/${CURRENT_USER}"
  mkdir -p "$dir"
  local config="${dir}/allreduce_gb300_bha${multiple}_config.json"
  cat >"$config" <<EOF
{
  "trainer": [
    {
      "domain": {
        "multiple": ${multiple},
        "quorumBuffer": 0,
        "maxGroupCount": null
      }
    }
  ]
}
EOF
  printf '%s\n' "$config"
}

mast_launcher_args() {
  local arch="$1"
  local nodes="$2"
  local priority_args=""
  if [[ -n "$MAST_PRIORITY" ]]; then
    priority_args=" --priority ${MAST_PRIORITY}"
  fi
  case "$arch" in
    h100)
      local h100_args="--cluster ${CLUSTER} --entitlement ${TENANT} --dp ${DP} --hw ${H100_HW}"
      if [[ -n "$H100_LOCALITY" ]]; then
        h100_args+=" --localityConstraints '${H100_LOCALITY}'"
      fi
      printf '%s\n' "${h100_args} --timeout ${MAST_TIMEOUT}${priority_args}"
      ;;
    gb300)
      local effective_bha="$BHA"
      if (( nodes < effective_bha )); then
        effective_bha="$nodes"
      fi
      local bha_config
      bha_config="$(write_gb300_bha_config "$effective_bha")"
      local gb300_args="--cluster ${CLUSTER} --entitlement ${TENANT} --dp ${DP} --hw gb300"
      if [[ -n "$GB300_LOCALITY" ]]; then
        gb300_args+=" --localityConstraints '${GB300_LOCALITY}'"
      fi
      printf '%s\n' "${gb300_args} --timeout ${MAST_TIMEOUT} --bha-config-map ${bha_config}${priority_args}"
      ;;
    *)
      echo "Unsupported MAST arch: ${arch}" >&2
      exit 2
      ;;
  esac
}

image_for_arch() {
  local arch="$1"
  if [[ -n "$MAST_IMG" ]]; then
    printf '%s\n' "$MAST_IMG"
    return
  fi
  case "$arch" in
    h100) printf '%s\n' "$H100_IMG" ;;
    gb300) printf '%s\n' "$GB300_IMG" ;;
    *) echo "Unsupported MAST arch: ${arch}" >&2; exit 2 ;;
  esac
}

set_image_for_arch() {
  local arch="$1"
  local img="$2"
  case "$arch" in
    h100) H100_IMG="$img" ;;
    gb300) GB300_IMG="$img" ;;
    *) echo "Unsupported MAST arch: ${arch}" >&2; exit 2 ;;
  esac
}

extract_created_img() {
  local log_file="$1"
  local target="$2"
  python3 - "$log_file" "$target" <<'PY'
import re
import sys

pattern = re.compile(rf"Created fbpkg:\s+({re.escape(sys.argv[2])}:[a-f0-9]{{32}})")
created = None
with open(sys.argv[1]) as f:
    for line in f:
        match = pattern.search(line)
        if match:
            created = match.group(1)

if created:
    print(created)
PY
}

init_manifest() {
  mkdir -p "$(dirname "$JOBS_MANIFEST")"
  printf 'phase\tarch\tcollective\tnodes\tppn\tvariant\tjob_name\tlog_path\timg\tuse_ctran\talgo\tnccl_args\n' >"$JOBS_MANIFEST"
}

write_report_preferences() {
  mkdir -p "$(dirname "$REPORT_PREFS")"
  python3 - "$REPORT_PREFS" "$REQUESTED_DTYPE" "$DTYPE" "$OP" "$VARIANTS" <<'PY'
import json
import sys

path, requested_dtype, nccl_dtype, op, variants_csv = sys.argv[1:6]
variants = [variant.strip() for variant in variants_csv.split(",") if variant.strip()]
variant_set = set(variants)
comparisons = []
if {"nccl_tree_simple", "ctree"} <= variant_set:
    comparisons.append({"baseline": "nccl_tree_simple", "variants": ["ctree"]})
if {"nccl_ring_simple", "ctring"} <= variant_set:
    comparisons.append({"baseline": "nccl_ring_simple", "variants": ["ctring"]})
if not comparisons and len(variants) >= 2:
    baseline = variants[0]
    comparisons.append(
        {"baseline": baseline, "variants": [variant for variant in variants[1:] if variant != baseline]}
    )

prefs = {
    "title": "AllReduce Benchmark Comparison",
    "collective": "allreduce",
    "dtype": requested_dtype,
    "nccl_dtype": nccl_dtype,
    "op": op,
    "primary_metric": "latency",
    "ratio": "baseline_latency / variant_latency",
    "columns": [
        "latency_us",
        "busbw_gbps",
        "baseline_channels",
        "variant_channels",
        "variant_blocks",
        "nonzero_error",
    ],
    "comparisons": comparisons,
}
with open(path, "w") as f:
    json.dump(prefs, f, indent=2, sort_keys=True)
    f.write("\n")
PY
  echo "Wrote report preferences: ${REPORT_PREFS}"
}

generate_report() {
  if [[ ! -f "$NCCL_REPORT_SCRIPT" ]]; then
    echo "nccl-test-suite comparison report generator not found: ${NCCL_REPORT_SCRIPT}" >&2
    exit 2
  fi
  if [[ ! -f "$JOBS_MANIFEST" ]]; then
    echo "jobs manifest not found: ${JOBS_MANIFEST}" >&2
    exit 2
  fi
  write_report_preferences
  local -a cmd=(
    python3 "$NCCL_REPORT_SCRIPT"
    --jobs-manifest "$JOBS_MANIFEST"
    --report-prefs "$REPORT_PREFS"
  )
  if [[ -n "$REPORT_OUTPUT" ]]; then
    cmd+=(--output "$REPORT_OUTPUT")
  fi
  if [[ "$REPORT_FORCE_DOWNLOAD" == "1" ]]; then
    cmd+=(--force-download)
  fi
  print_command "${cmd[@]}"
  "${cmd[@]}"
}

run_mast_job() {
  local phase="$1"
  local arch="$2"
  local nodes="$3"
  local variant="$4"
  local min_bytes="$5"
  local max_bytes="$6"
  local iters="$7"
  local warmup="$8"

  local -a nccl_envs=()
  local -a variant_env_values=()
  mapfile -t nccl_envs < <(arch_envs "$arch")
  mapfile -t variant_env_values < <(variant_envs "$variant")
  nccl_envs+=("${variant_env_values[@]}")
  if [[ "$arch" == "gb300" && "$variant" == "ctree" ]]; then
    nccl_envs+=("NCCL_CTRAN_MAX_NBLOCKS=${GB300_CTREE_MAX_NBLOCKS}")
    nccl_envs+=("NCCL_CTRAN_IB_MAX_GROUPS=${GB300_CTREE_IB_MAX_GROUPS}")
  fi
  if [[ "$arch" == "gb300" ]]; then
    local gb300_vppn="$((nodes * PPN))"
    if [[ -n "$GB300_VPPN" ]]; then
      gb300_vppn="$GB300_VPPN"
    fi
    nccl_envs+=("VPPN=${gb300_vppn}")
  fi

  local launcher_args
  launcher_args="$(mast_launcher_args "$arch" "$nodes")"
  local nccl_args="-b ${min_bytes} -e ${max_bytes} -f ${FACTOR} -g 1 -n ${iters} -w ${warmup} -d ${DTYPE}"
  local nvcc_config
  nvcc_config="$(mast_nvcc_config "$arch")"
  local fbpkg_target
  fbpkg_target="$(mast_fbpkg_target "$arch")"
  local current_img
  current_img="$(image_for_arch "$arch")"
  local prefix="allreduce-${arch}-${phase}-${variant}"
  local output_file
  output_file="$(mktemp)"

  local -a cmd=(
    env
    "NCCL_ENVS=$(join_by_space "${nccl_envs[@]}")"
    "NVCC_CONFIG=${nvcc_config}"
    "FBPKG_TARGET=${fbpkg_target}"
    "$NCCL_TEST_SCRIPT"
    --collective allreduce
    --min-nodes "$nodes"
    --max-nodes "$nodes"
    --ppn "$PPN"
    --nccl-args "$nccl_args"
    --prefix "$prefix"
    --launcher-args "$launcher_args"
    --jobs-manifest "$JOBS_MANIFEST"
    --manifest-phase "$phase"
    --manifest-arch "$arch"
    --manifest-variant "$variant"
    --nccl-debug INFO
  )

  if variant_uses_ctran "$variant"; then
    cmd+=(--use-ctran --algo "$variant")
  else
    cmd+=(--no-ctran)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  if [[ -n "$current_img" ]]; then
    cmd+=(--img "$current_img")
  fi

  echo
  echo "== mast ${phase} arch=${arch} nodes=${nodes} ppn=${PPN} variant=${variant} size=${min_bytes}..${max_bytes} =="
  print_command "${cmd[@]}"
  if ! "${cmd[@]}" 2>&1 | tee "$output_file"; then
    rm -f "$output_file"
    return 1
  fi

  local created_img
  created_img="$(extract_created_img "$output_file" "$fbpkg_target")"
  if [[ -n "$created_img" && -z "$current_img" ]]; then
    set_image_for_arch "$arch" "$created_img"
    echo "Reusing ${created_img} for later ${arch} launches."
  fi

  rm -f "$output_file"
}

run_mast() {
  require_executable "$NCCL_TEST_SCRIPT" "nccl-test-suite launcher"

  local -a arch_values
  read_csv_values "$ARCHS" arch_values
  validate_csv_values "MAST arch" arch_values h100 gb300
  prefer_h100_first arch_values

  local -a node_values
  read_csv_values "$NODES" node_values

  local -a variant_values
  read_csv_values "$VARIANTS" variant_values
  validate_csv_values "MAST variant" variant_values nccl_tree_simple nccl_ring_simple ctree ctring

  init_manifest
  write_report_preferences
  prebuild_mast_images "${arch_values[@]}"

  local arch variant nodes
  if [[ "$SKIP_SMOKE" != "1" ]]; then
    for arch in "${arch_values[@]}"; do
      for variant in "${variant_values[@]}"; do
        run_mast_job smoke "$arch" 2 "$variant" \
          "$SMOKE_MIN_BYTES" "$SMOKE_MAX_BYTES" "$SMOKE_ITERS" "$SMOKE_WARMUP"
      done
    done
  fi

  for arch in "${arch_values[@]}"; do
    for nodes in "${node_values[@]}"; do
      for variant in "${variant_values[@]}"; do
        run_mast_job full "$arch" "$nodes" "$variant" \
          "$MIN_BYTES" "$MAX_BYTES" "$ITERS" "$WARMUP"
      done
    done
  done

  echo
  echo "Wrote MAST job manifest: ${JOBS_MANIFEST}"
  echo "Wrote report preferences: ${REPORT_PREFS}"
}

main() {
  REQUESTED_DTYPE="$DTYPE"
  normalize_dtype
  if [[ -z "$REPORT_PREFS" ]]; then
    REPORT_PREFS="$(default_report_prefs)"
  fi

  if [[ "$REPORT" == "1" ]]; then
    generate_report
    return
  fi

  case "$LAUNCHER" in
    local) run_local ;;
    mast) run_mast ;;
    *) echo "Unsupported launcher: ${LAUNCHER}" >&2; exit 2 ;;
  esac
}

main "$@"
