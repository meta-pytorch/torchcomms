# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""
MAST launcher for TorchComm Device API Iterated Tests.

Submits TorchComm device API iterated (soak) tests to MAST using TorchX's
fb_dist_hpc. Tests exercise put, signal, barrier, combined ops, multi-window,
and lifecycle patterns repeatedly to catch race conditions and resource leaks.

The C++ test binaries use TorchCommTestWrapper for distributed coordination,
which reads environment variables set by MAST:
- MASTER_ADDR, MASTER_PORT: TCP store connection info
- RANK, WORLD_SIZE: Global process info
- LOCAL_RANK, LOCAL_SIZE: Per-node process info

Available tests:
- ncclx_iterated: NCCLx device API iterated tests (20 tests)
- pipes_iterated: Pipes device API iterated tests (13 tests)

Usage:
  # First build the fbpkg for your target hardware:
  fbpkg build fbcode//comms/torchcomms/tests/integration/cpp:torchcomm_device_test_launcher_h100

  # Then launch the test with the built image:
  buck2 run @mode/opt -c hpc_comms.use_ncclx=stable \\
    //comms/torchcomms/tests/integration/cpp:torchcomm_device_test_launcher_bin -- \\
    --img torchcomm_device_test_launcher_h100:<hash> \\
    --entitlement <your_entitlement> \\
    --test ncclx_iterated \\
    --hw h100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from torchx.components.fb.dist import hpc as fb_dist_hpc  # pyre-ignore[21]
from torchx.runner import get_runner  # pyre-ignore[21]
from torchx.specs import CfgVal, named_resources  # pyre-ignore[21]

logger: logging.Logger = logging.getLogger(__name__)

# Registry of available tests: maps --test argument to binary resource name.
# These must match the resource keys in the python_binary BUCK target.
TESTS: dict[str, str] = {
    "ncclx_iterated": "ncclx_iterated_test",
    "pipes_iterated": "pipes_iterated_test",
}

TCPSTORE_ENV_VARS: list[str] = [
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_SIZE",
]

MSL_TIER_MAPPING: dict[str, str] = {
    "PROD": "msl_scheduler.api.write.prod",
    "STAGING": "msl_scheduler.api.write.staging",
    "RC": "msl_scheduler.api.write.rc",
}

SUPPORTED_HARDWARE: list[str] = ["h100", "gb200", "gb300"]

MAX_SCHEDULE_RETRIES: int = 3
RETRY_DELAY_SECONDS: int = 5


def verify_tcpstore_env_vars() -> bool:
    """Verify that all environment variables required by TcpStoreBootstrap are set."""
    all_present = True
    logger.info("=== TcpStoreBootstrap Environment Variables ===")

    for var in TCPSTORE_ENV_VARS:
        value = os.getenv(var)
        if value is not None:
            logger.info("  %s = %s", var, value)
        else:
            logger.warning("  %s = NOT SET", var)
            all_present = False

    return all_present


def get_test_binary(test_name: str) -> Path:
    """Get the path to a test binary from the PAR package."""
    if test_name not in TESTS:
        raise ValueError(f"Unknown test: {test_name}. Available: {list(TESTS.keys())}")

    par_path = os.getenv("FB_PAR_RUNTIME_FILES")
    if par_path is None:
        par_path = str(Path.cwd())

    binary_name = TESTS[test_name]
    return (
        Path(par_path)
        / "comms"
        / "torchcomms"
        / "tests"
        / "integration"
        / "cpp"
        / binary_name
    )


def run_test_binary(args: argparse.Namespace, test_name: str) -> int:
    """Run a single test binary. Returns the process return code."""
    binary_path = get_test_binary(test_name)
    logger.info("Test binary: %s", binary_path)

    if not binary_path.exists():
        raise FileNotFoundError(f"Test binary not found: {binary_path}")

    cmd = [str(binary_path)]

    # Pass through gtest filter if specified
    if args.gtest_filter:
        cmd.append(f"--gtest_filter={args.gtest_filter}")

    logger.info("Executing: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    return result.returncode


def local_launcher(args: argparse.Namespace) -> None:
    """
    Run the test locally on MAST hosts.
    Called when the launcher detects it's running on a MAST host.
    """
    logger.info("Running locally on MAST host")
    logger.info("Selected test: %s", args.test)

    verify_tcpstore_env_vars()

    tests = list(TESTS.keys()) if args.test == "all" else [args.test]
    failed: list[str] = []

    for test_name in tests:
        logger.info("=== Running %s ===", test_name)
        rc = run_test_binary(args, test_name)
        if rc != 0:
            logger.error("%s failed with return code %d", test_name, rc)
            failed.append(test_name)
        else:
            logger.info("%s passed", test_name)

    if failed:
        logger.error("Failed test suites: %s", ", ".join(failed))
        sys.exit(1)


def schedule_with_retry(
    runner: Any,
    dryrun_info: Any,
    max_retries: int = MAX_SCHEDULE_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
) -> Any:
    """Schedule a job with retry logic for transient failures."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info("Scheduling job (attempt %d/%d)", attempt + 1, max_retries)
            app_handle = runner.schedule(dryrun_info)
            logger.info("Job scheduled successfully")
            return app_handle
        except Exception as e:
            last_exception = e
            logger.warning(
                "Scheduling attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                str(e),
            )
            if attempt < max_retries - 1:
                logger.info("Retrying in %d seconds...", retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2

    logger.error("All %d scheduling attempts failed", max_retries)
    assert last_exception is not None
    raise last_exception


def mast_launcher(args: argparse.Namespace) -> None:
    """
    Submit the test job to MAST via TorchX.
    If already running on MAST, runs locally.
    """
    if os.getenv("MAST_HPC_JOB_NAME") is not None:
        logger.info("Detected MAST environment, running locally")
        os.chdir("/logs")
        local_launcher(args)
        return

    img = args.img
    logger.info("Using fbpkg image: %s", img)

    tier = MSL_TIER_MAPPING.get(args.tier, args.tier)
    if not tier.startswith("msl_scheduler."):
        logger.error("Unrecognized MSL tier: %s", args.tier)
        sys.exit(1)

    rsrc = named_resources[args.hw]
    # Use full GPU count for j so nproc_per_node matches res.gpu, avoiding
    # fractional resource lookups that may not be registered. The test binary
    # reads WORLD_SIZE from env and works with any rank count.
    ppn = rsrc.gpu

    fbpkg_name = img.split(":")[0]
    workspace_dir = f"/packages/{fbpkg_name}"

    env: dict[str, str] = {
        "LOGLEVEL": "DEBUG",
        "NCCL_DEBUG": "WARN",
        "NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET": "1",
        "NCCL_GIN_ENABLE": "1",
        "NCCL_GIN_TYPE": "-1",
        "RUN_DEVICE_API_TEST": "true",
        "RUN_DEVICE_ITERATED_TEST": "true",
        "NUM_ITERATIONS": str(args.iterations),
        "TEST_BACKEND": "ncclx",
        "WORKSPACE_DIR": workspace_dir,
    }

    # Add Pipes-specific env vars
    if args.test == "pipes_iterated":
        env["RUN_PIPES_DEVICE_API_TEST"] = "true"
        env["NCCL_CTRAN_USE_PIPES"] = "1"
        env["NCCL_CTRAN_ENABLE"] = "true"
        env["NCCL_CUMEM_ENABLE"] = "0"

    # P2P disable
    if args.p2p_disabled:
        env["NCCL_P2P_DISABLE"] = "1"

    if args.envs:
        for e in args.envs.split():
            if "=" in e:
                key, val = e.split("=", 1)
                env[key] = val

    if args.job_name:
        job_name = args.job_name
    else:
        job_name = f"torchcomm-{args.test}-n{args.nnode}-{uuid.uuid4().hex[:6]}"

    logger.info("Job name: %s", job_name)
    logger.info("Test: %s", args.test)
    logger.info("Iterations: %d", args.iterations)
    logger.info(
        "Hardware: %s, Nodes: %d, GPUs per host: %d, Processes per node: %d",
        args.hw,
        args.nnode,
        rsrc.gpu,
        ppn,
    )

    args_list = sys.argv[1:] if len(sys.argv) > 1 else []
    module_name = (
        "comms.torchcomms.tests.integration.cpp.torchcomm_device_test_launcher"
    )

    app = fb_dist_hpc(
        *args_list,
        m=module_name,
        img=img,
        name=job_name,
        h=args.hw,
        j=f"{args.nnode}x{ppn}",
        env=env,
        max_retries=1,
        metadata={
            "rmAttribution": args.entitlement,
            "hpcIdentity": args.dp,
            "hpcClusterUuid": args.cluster,
            "hpcJobOncall": args.oncall,
            "smcTier": tier,
        },
    )

    app.roles[0].image = f"{app.roles[0].image};torchx_python:stable"
    app.roles[0].entrypoint = "/packages/torchx_python/python"
    app.roles[0].env["PENV_PAR"] = f"/packages/{fbpkg_name}/penv.par"

    cfg: dict[str, CfgVal] = {
        "rmAttribution": args.entitlement,
        "hpcIdentity": args.dp,
        "hpcClusterUuid": args.cluster,
        "hpcJobOncall": args.oncall,
        "smcTier": tier,
        "runningTimeoutSec": args.timeout,
        "useStrictName": True,
        "enableGracefulPreemption": True,
        "enableAiEnvelope": False,
        "enableAiEnvelopeAppController": False,
        "runtimeAppMetadataJson": json.dumps(app.metadata),
        "localityConstraints": ["REGION", args.region],
    }

    runner = get_runner()
    dryrun_info = runner.dryrun(app=app, scheduler="msl_conda", cfg=cfg)
    logger.debug("Job to be submitted: %s", dryrun_info)

    app_handle = schedule_with_retry(runner, dryrun_info)

    status = runner.status(app_handle)
    assert status, f"Failed to get job status for {app_handle}"

    mast_url = status.ui_url
    logger.info("Started job, see: %s", mast_url)


def init_argparse() -> argparse.ArgumentParser:
    """Create argument parser for the launcher."""
    parser = argparse.ArgumentParser(
        description="TorchComm Device API Iterated Test MAST Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help=(
            "Fbpkg image to use (e.g., torchcomm_device_test_launcher_h100:abc123). "
            "Build with: fbpkg build fbcode//comms/torchcomms/tests/integration/cpp:torchcomm_device_test_launcher_<hw>"
        ),
    )
    parser.add_argument(
        "--entitlement",
        type=str,
        required=True,
        help="Resource entitlement for MAST job",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="ncclx_iterated",
        choices=list(TESTS.keys()) + ["all"],
        help="Which test suite to run ('all' runs both sequentially)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per test",
    )
    parser.add_argument(
        "--p2p-disabled",
        action="store_true",
        default=False,
        help="Disable P2P (forces GIN/RDMA path for NCCLx)",
    )
    parser.add_argument(
        "--gtest-filter",
        type=str,
        default=None,
        help="Gtest filter pattern (e.g., '*PutThread*')",
    )
    parser.add_argument(
        "--nnode",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument(
        "--hw",
        type=str,
        default="h100",
        choices=SUPPORTED_HARDWARE,
        help="Hardware type for MAST scheduling",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="MastGenAICluster",
        help="HPC cluster UUID",
    )
    parser.add_argument(
        "--dp",
        type=str,
        default="networkai_mast_job_identity",
        help="HPC identity (data project/ACL)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="PROD",
        choices=[
            "PROD",
            "STAGING",
            "RC",
            "msl_scheduler.api.write.prod",
            "msl_scheduler.api.write.staging",
            "msl_scheduler.api.write.rc",
        ],
        help="MSL scheduler environment tier",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Job timeout in seconds",
    )
    parser.add_argument(
        "--oncall",
        type=str,
        default="ncclx",
        help="Oncall for MAST job",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Custom job name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=None,
        help="Space-separated environment variables (e.g., 'VAR1=val1 VAR2=val2')",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="eag",
        help="MSL region for locality constraints (e.g., 'eag', 'pci')",
    )

    return parser


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    logger.info("TorchComm Device Test Launcher started with args: %s", sys.argv)

    parser = init_argparse()
    args = parser.parse_args()

    mast_launcher(args)


if __name__ == "__main__":
    main()
