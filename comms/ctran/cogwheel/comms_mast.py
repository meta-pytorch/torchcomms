# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Shared TorchX MAST job configuration for comms distributed tests.

Used by both the cogwheel test (comms_dist_test.py) and the manual
launcher (manual_launch.py).
"""

from typing import Dict

from torchx.specs import AppDef, CfgVal, named_resources, Role

COMMS_FBPKG_NAME: str = "hpc_comms.tests.latest"
COMMS_ENTITLEMENT: str = "infra_genai_ncclx"
JOB_TIMEOUT_SEC: int = 3600

# Test binaries included in the comms test fbpkg.
# Keep in sync with _COMMS_TESTS in fbcode/comms/def_mast.bzl.
COMMS_TEST_BINARIES: list[str] = [
    "allgather_test_2x1",
    "allgather_test_2x8",
    "ctran_dist_rma_2x4",
    "fast_init_test_2x8",
    "multi_stream_test_2x8",
]

# Inline launch script for MAST tasks.
# Runs all test binaries sequentially. For each binary, launches
# LOCAL_SIZE processes (one per GPU) and waits for completion.
RUN_SCRIPT: str = r"""
set -euo pipefail
FBPKG_NAME="${MAST_APPLICATION_PACKAGES%%:*}"
IFS="," read -r -a hosts <<< "$MAST_HPC_TASK_GROUP_HOSTNAMES"
NUM_HOSTS=${#hosts[@]}
export MASTER_ADDR=${MASTER_ADDR:-${hosts[0]}}
export MASTER_PORT=${MASTER_PORT:-29500}
IDX=$TW_TASK_ID

for binary_name in $TEST_BINARIES; do
    binary="/packages/$FBPKG_NAME/$binary_name"
    if [ ! -f "$binary" ]; then
        echo "=== SKIP $binary_name (not found) ==="
        continue
    fi
    local_size=$(echo "$binary_name" | grep -oP '\d+x\K\d+' | head -1)
    local_size=${local_size:-8}
    export LOCAL_SIZE=$local_size
    export WORLD_SIZE=$(( NUM_HOSTS * local_size ))
    echo "=== Running $binary_name (LOCAL_SIZE=$local_size, WORLD_SIZE=$WORLD_SIZE, MASTER_PORT=$MASTER_PORT) ==="
    for i in $(seq 0 $((local_size - 1))); do
        LOCAL_RANK=$i GLOBAL_RANK=$((IDX * local_size + i)) "$binary" &
    done
    wait
    echo "=== Done $binary_name ==="
done
echo "=== All tests completed ==="
""".strip()

# TorchX scheduler config for MAST.
MAST_CFG: Dict[str, CfgVal] = {
    "rmAttribution": COMMS_ENTITLEMENT,
    "hpcIdentity": "networkai_mast_job_identity",
    "hpcClusterUuid": "MastProdCluster",
    "hpcJobOncall": "networkai_comms",
    "smcTier": "mast.api.write",
    "runningTimeoutSec": JOB_TIMEOUT_SEC,
    "useStrictName": True,
}


def build_app(
    fbpkg_id: str,
    job_name: str,
    test_binaries: list[str] | None = None,
) -> AppDef:
    """Build a TorchX AppDef that runs test binaries sequentially on 2 H100 nodes.

    Args:
        fbpkg_id: fbpkg identifier (e.g. "hpc_comms.tests.latest:<hash>")
        job_name: MAST job name
        test_binaries: list of binary names to run. Defaults to COMMS_TEST_BINARIES.
    """
    if test_binaries is None:
        test_binaries = COMMS_TEST_BINARIES

    resource = named_resources["grandteton"]

    role = Role(
        name="trainer",
        image=fbpkg_id,
        entrypoint="/bin/bash",
        args=["-c", RUN_SCRIPT],
        num_replicas=2,
        resource=resource,
        env={
            "TEST_BINARIES": " ".join(test_binaries),
            "NCCL_DEBUG": "WARN",
        },
        port_map={"tcp": 29500},
        max_retries=0,
    )

    return AppDef(
        name=job_name,
        roles=[role],
        metadata={
            "rmAttribution": COMMS_ENTITLEMENT,
            "hpcIdentity": "networkai_mast_job_identity",
            "hpcClusterUuid": "MastProdCluster",
            "hpcJobOncall": "networkai_comms",
        },
    )
