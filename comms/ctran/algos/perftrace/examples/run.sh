#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This script runs PerfTrace examples and generates a combined trace.
#
# Usage:
#   ./run.sh                              # Run local multi-threaded example
#   ./run.sh dist                         # Run distributed MPI example on localhost
#   ./run.sh dist host1,host2             # Run on specified hosts
#   ./run.sh dist host1,host2 /tmp/trace  # Run with custom trace directory

set -e

MODE="${1:-local}"
HOSTS="${2:-rtptest013.dkl2}"
TRACE_DIR="${3:-/tmp/perftrace_test}"

pushd "${HOME}"/fbsource > /dev/null

if [[ "${MODE}" == "dist" ]]; then
    echo "=== Running distributed PerfTrace example ==="
    echo "Hosts: ${HOSTS}"
    echo "Trace directory: ${TRACE_DIR}"
    echo ""

    # -c comms.tmpdir allows launcher to create a temporary directory on each host,
    # download the trace to the local machine after run and cleanup the temporary
    # directory on each host. Best practice is to use with comms_gpu_cpp_distributed_unittest.

    buck2 run @fbcode//mode/opt \
        -c comms.hosts="${HOSTS}" \
        -c comms.envs="NCCL_CTRAN_ENABLE_PERFTRACE=1;NCCL_CTRAN_PERFTRACE_DIR=${TRACE_DIR};NCCL_DEBUG=INFO;" \
        -c comms.tmpdir="${TRACE_DIR}" \
        fbcode//comms/ctran/algos/perftrace/examples:perf_trace_dist_hello_world
else
    # Clean up any existing trace files for local mode
    rm -rf "${TRACE_DIR}"
    mkdir -p "${TRACE_DIR}"

    echo "=== Running local PerfTrace HelloWorld test ==="
    NCCL_CTRAN_PERFTRACE_DIR="${TRACE_DIR}" \
        buck2 run @fbcode//mode/opt fbcode//comms/ctran/algos/perftrace/examples:perf_trace_hello_world
fi

echo ""
echo "=== Trace files generated ==="
ls -la "${TRACE_DIR}"

echo ""
echo "=== Combining traces and generating Perfetto URL ==="
python3 fbcode/comms/testinfra/combined_trace.py --local_dir "${TRACE_DIR}"

popd > /dev/null
