#!/bin/bash
# Compatibility wrapper for the Python online accuracy/performance runner.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}

if [[ -n "${FBPKG_ROOT:-}" && -x "${FBPKG_ROOT}/penv.par" ]]; then
  PYTHON="${FBPKG_ROOT}/penv.par"
fi

exec "$PYTHON" "$SCRIPT_DIR/disagg_online_test.py" "$@"
