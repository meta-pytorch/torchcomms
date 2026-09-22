#!/usr/bin/env bash
# Quiet by default; use -v/--verbose to see output from the tests.

set -e
set -u
( set -o pipefail ) 2>/dev/null || true

# Try to infer AMD SMI root if not set
: "${PAPI_AMDSMI_ROOT:=${PAPI_ROCM_ROOT:-/opt/rocm-6.4.0}}"

banner() { printf "Running: \033[36m%s\033[0m %s\n" "$1" "${2-}"; }
sep()    { printf "%s\n\n" "-------------------------------------"; }

VERBOSE=0

usage() {
  cat <<EOF
Usage: $0 [-v]

  -v, --verbose   Show output from tests
EOF
}

for arg in "$@"; do
  case "$arg" in
    TESTS_QUIET) ;;                 # ignore literal token
    -v|--verbose) VERBOSE=1 ;;
    -h|--help)    usage; exit 0 ;;
    --*) echo "Unknown option: $arg"; usage; exit 2 ;;
    *) ;;                           # ignore stray args
  esac
done

run_test() {
  # $1 = binary, $2.. = extra args
  local bin="$1"; shift || true

  if [[ ! -x "./$bin" ]]; then
    echo "SKIP: missing ./$bin"; sep; return 0
  fi

  if [[ $VERBOSE -eq 1 ]]; then
    banner "./$bin" "$*"
    "./$bin" "$@" || true
  else
    banner "./$bin TESTS_QUIET" "$*"
    TESTS_QUIET=TESTS_QUIET "./$bin" TESTS_QUIET "$@" || true
  fi
  sep
}

# ---------------------------
# Run the five tests
# ---------------------------

run_test amdsmi_hello
run_test amdsmi_basics
run_test amdsmi_gemm
run_test amdsmi_energy_monotonic
run_test amdsmi_ctx_conflict

# Done

