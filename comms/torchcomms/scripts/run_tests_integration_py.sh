#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

SKIP_TESTS=""

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --skips, -s         Comma-separated list of tests to skip"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --skips|-s)
            SKIP_TESTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

run_tests () {
    # Convert comma-separated skip tests to array
    IFS=',' read -ra SKIP_TESTS_ARRAY <<< "$SKIP_TESTS"
    for file in *Test.py; do
        skip=false
        for skip_file in "${SKIP_TESTS_ARRAY[@]}"; do
            if [[ "$file" == "$skip_file" ]]; then
                skip=true
                break
            fi
        done
        if ! $skip; then
            torchrun --nnodes 1 --nproc_per_node 4 "$file" --verbose
        fi
    done
}

# NCCL
export TEST_BACKEND=nccl
run_tests

# NCCLX
export TEST_BACKEND=ncclx
run_tests

# Gloo with CPU
export TEST_BACKEND=gloo
export TEST_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
run_tests
unset TEST_DEVICE
unset CUDA_VISIBLE_DEVICES

# Gloo with CUDA
export TEST_BACKEND=gloo
export TEST_DEVICE=cuda
run_tests
unset TEST_DEVICE
