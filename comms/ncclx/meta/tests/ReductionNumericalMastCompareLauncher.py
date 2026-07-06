#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# pyre-strict

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comms.ncclx.meta.tests.ReductionNumericalVersionCompare import (
    compare_outputs,
    is_numerical_test_failure,
    parse_actual_outputs,
    print_numerical_failure_summary,
)
from torchx.components.fb.dist import hpc as fb_dist_hpc  # pyre-ignore[21]
from torchx.runner import get_runner  # pyre-ignore[21]
from torchx.specs import CfgVal, named_resources  # pyre-ignore[21]

logger: logging.Logger = logging.getLogger(__name__)

MSL_TIER_MAPPING: dict[str, str] = {
    "PROD": "msl_scheduler.api.write.prod",
    "STAGING": "msl_scheduler.api.write.staging",
    "RC": "msl_scheduler.api.write.rc",
}

SUPPORTED_HARDWARE: list[str] = ["gb200", "gb300", "gb300_dsf", "gcp_gb300"]
MAX_SCHEDULE_RETRIES: int = 3
RETRY_DELAY_SECONDS: int = 5


@dataclass(frozen=True)
class Case:
    collective: str
    datatype: str
    old_binary: str
    new_binary: str
    gtest_filter: str


# Float32 and BF16 coverage for each reduction collective. Float32 excludes the
# CtranDirect algorithm, which is BF16-only in this matrix. ReduceScatter uses the
# Ring algorithm because Pat/CtranDirect are skipped on some fixed topologies.
DEFAULT_CASES: tuple[Case, ...] = (
    Case(
        "AllReduce",
        "Float32",
        "allreduce_numerical_test_v2_29",
        "allreduce_numerical_test_v2_30",
        "*Float32*:-*CtranDirect*",
    ),
    Case(
        "AllReduce",
        "Bfloat16",
        "allreduce_numerical_test_v2_29",
        "allreduce_numerical_test_v2_30",
        "*Bfloat16*",
    ),
    Case(
        "ReduceScatter",
        "Float32",
        "reducescatter_numerical_test_v2_29",
        "reducescatter_numerical_test_v2_30",
        "*Ring_Float32*",
    ),
    Case(
        "ReduceScatter",
        "Bfloat16",
        "reducescatter_numerical_test_v2_29",
        "reducescatter_numerical_test_v2_30",
        "*Ring_Bfloat16*",
    ),
    Case(
        "Reduce",
        "Float32",
        "reduce_numerical_test_v2_29",
        "reduce_numerical_test_v2_30",
        "*Float32*:-*CtranDirect*",
    ),
    Case(
        "Reduce",
        "Bfloat16",
        "reduce_numerical_test_v2_29",
        "reduce_numerical_test_v2_30",
        "*Bfloat16*",
    ),
)


def parse_case(values: list[str]) -> Case:
    if len(values) != 5:
        raise ValueError(
            "--case expects COLLECTIVE DATATYPE OLD_BINARY NEW_BINARY GTEST_FILTER"
        )
    return Case(values[0], values[1], values[2], values[3], values[4])


def resource_dir() -> Path:
    par_path = os.getenv("FB_PAR_RUNTIME_FILES")
    if par_path is None:
        par_path = str(Path.cwd())
    return Path(par_path) / "comms" / "ncclx" / "meta" / "tests"


def resource_path(resource_name: str) -> Path:
    path = resource_dir() / resource_name
    if not path.exists():
        raise FileNotFoundError(f"Packaged resource not found: {path}")
    return path


def run_binary(
    binary_name: str,
    gtest_filter: str,
    run_index: int,
    args: argparse.Namespace,
) -> dict[str, str]:
    binary_path = resource_path(binary_name)
    cmd = [str(binary_path), f"--gtest_filter={gtest_filter}"]
    run_env = os.environ.copy()

    rank = os.environ["RANK"]
    world_size = os.environ["WORLD_SIZE"]
    local_rank = os.environ["LOCAL_RANK"]
    local_size = os.environ.get("LOCAL_SIZE") or os.environ["LOCAL_WORLD_SIZE"]
    base_port = int(os.environ.get("MASTER_PORT", "29500"))
    job_name = os.environ.get("MAST_HPC_JOB_NAME", "mast")

    run_env.update(
        {
            "GLOBAL_RANK": rank,
            "RANK": rank,
            "WORLD_SIZE": world_size,
            "LOCAL_RANK": local_rank,
            "LOCAL_SIZE": local_size,
            "MASTER_PORT": str(base_port + (run_index + 1) * 100),
            "COMMS_TEST_RUN_ID": f"{job_name}-{run_index:04d}"[:31],
            "COMMS_TEST_TCPSTORE_TIMEOUT_SECONDS": str(args.tcpstore_timeout_seconds),
            "REDUCTION_NUMERICAL_PRINT_ACTUAL": "1",
        }
    )

    logger.info(
        "Running rank %s/%s binary=%s filter=%s master_port=%s",
        rank,
        world_size,
        binary_path,
        gtest_filter,
        run_env["MASTER_PORT"],
    )
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, env=run_env
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)

    outputs = parse_actual_outputs(result.stdout)
    if result.returncode != 0:
        if args.allow_numerical_test_failures and is_numerical_test_failure(
            result.returncode, result.stdout, outputs
        ):
            print(
                "FP64_REFERENCE_STATUS "
                f"binary={binary_name} rank={rank} status=FAIL "
                f"returncode={result.returncode} structured_outputs={len(outputs)}"
            )
            print_numerical_failure_summary(result.stdout)
            return outputs
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if not outputs and int(rank) == 0:
        # Rank 0 is the Reduce root and participates in every collective, so it
        # must always emit structured output. Non-root ranks legitimately emit
        # nothing for root-only collectives (Reduce), so an empty result there
        # is expected and compared as a no-op.
        raise RuntimeError(f"{binary_name} produced no structured numerical outputs")
    print(
        "FP64_REFERENCE_STATUS "
        f"binary={binary_name} rank={rank} status=PASS "
        f"structured_outputs={len(outputs)}"
    )
    return outputs


def local_launcher(args: argparse.Namespace) -> None:
    rank = os.environ["RANK"]
    world_size = os.environ["WORLD_SIZE"]
    logger.info("Running MAST local rank %s/%s", rank, world_size)

    cases = (
        [parse_case(case) for case in args.case] if args.case else list(DEFAULT_CASES)
    )
    all_passed = True
    run_index = 0
    for case in cases:
        old_outputs = run_binary(case.old_binary, case.gtest_filter, run_index, args)
        run_index += 1
        new_outputs = run_binary(case.new_binary, case.gtest_filter, run_index, args)
        run_index += 1
        comparison_passed = compare_outputs(
            args.old_version,
            args.new_version,
            old_outputs,
            new_outputs,
        )
        print(
            "RELEASE_COMPARE_STATUS "
            f"collective={case.collective} datatype={case.datatype} rank={rank} "
            f"old_version={args.old_version} new_version={args.new_version} "
            f"status={'PASS' if comparison_passed else 'FAIL'}"
        )
        all_passed = comparison_passed and all_passed

    if not all_passed:
        sys.exit(1)


def schedule_with_retry(
    runner: Any,
    dryrun_info: Any,
    max_retries: int = MAX_SCHEDULE_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
) -> Any:
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info("Scheduling job attempt %d/%d", attempt + 1, max_retries)
            return runner.schedule(dryrun_info)
        except Exception as error:
            last_exception = error
            logger.warning("Scheduling attempt failed: %s", error)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    assert last_exception is not None
    raise last_exception


def mast_launcher(args: argparse.Namespace) -> None:
    if os.getenv("MAST_HPC_JOB_NAME") is not None:
        os.chdir("/logs")
        local_launcher(args)
        return

    tier = MSL_TIER_MAPPING.get(args.tier, args.tier)
    if not tier.startswith("msl_scheduler."):
        raise ValueError(f"Unrecognized MSL tier: {args.tier}")

    rsrc = named_resources[args.hw]
    ppn = rsrc.gpu
    fbpkg_name = args.img.split(":")[0]
    job_name = (
        args.job_name
        if args.job_name
        else f"reduction-numerics-{args.hw}-n{args.nnode}-{uuid.uuid4().hex[:6]}"
    )

    env: dict[str, str] = {
        "LOGLEVEL": "DEBUG",
        "NCCL_DEBUG": args.nccl_debug,
        "WORKSPACE_DIR": f"/packages/{fbpkg_name}",
    }
    if args.hw in {"gb300", "gb300_dsf", "gcp_gb300"}:
        env.update(
            {
                "NCCL_IGNORE_TOPO_LOAD_FAILURE": "1",
                "NCCL_SHM_DISABLE": "0",
            }
        )
    if args.envs:
        for item in args.envs.split():
            if "=" in item:
                key, value = item.split("=", 1)
                env[key] = value

    logger.info(
        "Submitting job=%s hw=%s nodes=%d ppn=%d cases=%s",
        job_name,
        args.hw,
        args.nnode,
        ppn,
        args.case if args.case else "default-bf16",
    )

    app = fb_dist_hpc(
        *sys.argv[1:],
        m="comms.ncclx.meta.tests.ReductionNumericalMastCompareLauncher",
        img=args.img,
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
    app_handle = schedule_with_retry(runner, dryrun_info)
    status = runner.status(app_handle)
    assert status is not None, f"Failed to get job status for {app_handle}"
    logger.info("Started job, see: %s", status.ui_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAST launcher for NCCL reduction numerical version comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--img", required=True)
    parser.add_argument("--entitlement", required=True)
    parser.add_argument("--hw", choices=SUPPORTED_HARDWARE, required=True)
    parser.add_argument("--nnode", type=int, default=2)
    parser.add_argument("--old-version", default="2.29")
    parser.add_argument("--new-version", default="2.30")
    parser.add_argument("--allow-numerical-test-failures", action="store_true")
    parser.add_argument("--tcpstore-timeout-seconds", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cluster", default="MastGenAICluster")
    parser.add_argument("--dp", default="networkai_mast_job_identity")
    parser.add_argument("--tier", default="PROD", choices=list(MSL_TIER_MAPPING.keys()))
    parser.add_argument("--oncall", default="ncclx")
    parser.add_argument("--job-name")
    parser.add_argument("--region", default="nao")
    parser.add_argument("--nccl-debug", default="WARN")
    parser.add_argument("--envs")
    parser.add_argument(
        "--case",
        action="append",
        nargs=5,
        metavar=("COLLECTIVE", "DATATYPE", "OLD_BINARY", "NEW_BINARY", "GTEST_FILTER"),
        help=(
            "Case to compare. Defaults to Float32 and BF16 AllReduce, "
            "ReduceScatter Ring, and Reduce cases packaged by this target."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    mast_launcher(parse_args())


if __name__ == "__main__":
    main()
