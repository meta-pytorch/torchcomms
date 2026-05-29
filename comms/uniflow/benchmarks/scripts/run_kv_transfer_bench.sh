#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Run the UniFlow KV-cache transfer benchmark.
#
# Requires 2 CUDA GPUs.  Builds the benchmark via buck and runs it.
# All arguments are forwarded to the benchmark script.
#
# Examples:
#   # Run with defaults
#   bash run_kv_transfer_bench.sh
#
#   # Custom KV layout matching Llama-3.1-8B (GQA: 8 KV heads, 128 dim)
#   bash run_kv_transfer_bench.sh -- --num-blocks 512 --block-size 128
#
#   # Write CSV output
#   bash run_kv_transfer_bench.sh -- --csv /tmp/uniflow_kv_bench.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "Building and running KV-cache transfer benchmark..."
echo "  Requires: 2 CUDA GPUs"
echo ""

cd "${REPO_ROOT}"
buck test //comms/uniflow/benchmarks/py:kv_transfer_bench "$@"
