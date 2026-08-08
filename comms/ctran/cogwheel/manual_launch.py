# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Manual launcher to test comms binaries on MAST via TorchX.

Usage:
  # Run all test binaries sequentially:
  buck2 run //comms/ctran/cogwheel:manual_launch -- \
    --img hpc_comms.tests.latest:<hash>

  # Run a specific test binary:
  buck2 run //comms/ctran/cogwheel:manual_launch -- \
    --img hpc_comms.tests.latest:<hash> \
    --test-binary fast_init_test_2x8
"""

import argparse
import logging
import sys
import time

from comms.ctran.cogwheel.comms_mast import build_app, COMMS_TEST_BINARIES, MAST_CFG
from torchx.runner import get_runner
from torchx.specs import AppState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual comms test MAST launcher")
    parser.add_argument(
        "--img",
        required=True,
        help="fbpkg image (e.g. hpc_comms.tests.latest:<hash>)",
    )
    parser.add_argument(
        "--test-binary",
        nargs="+",
        default=None,
        help="test binary name(s) to run (default: all)",
    )
    parser.add_argument("--job-name", default=None, help="custom job name")
    parser.add_argument("--wait", action="store_true", help="wait for job to complete")
    parser.add_argument(
        "--timeout", type=int, default=3600, help="job timeout in seconds"
    )
    args = parser.parse_args()

    test_binaries = args.test_binary or COMMS_TEST_BINARIES
    job_name = args.job_name or "comms-dist-test-manual"
    app = build_app(args.img, job_name, test_binaries)

    runner = get_runner()
    logger.info("Submitting job: %s", job_name)
    logger.info("Image: %s", args.img)
    logger.info("Test binaries: %s", " ".join(test_binaries))

    dryrun_info = runner.dryrun(app=app, scheduler="mast", cfg=MAST_CFG)
    handle = runner.schedule(dryrun_info)

    status = runner.status(handle)
    if status and status.ui_url:
        logger.info("MAST job URL: %s", status.ui_url)
    logger.info("App handle: %s", handle)

    if not args.wait:
        logger.info("Job submitted. Use --wait to poll until completion.")
        return

    logger.info("Waiting for job to complete (timeout=%ds)...", args.timeout)
    elapsed = 0
    poll_interval = 30
    while elapsed < args.timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        status = runner.status(handle)
        if status is None:
            logger.error("Job no longer exists")
            sys.exit(1)
        state = status.state
        logger.info("Status: %s (%ds/%ds)", state.name, elapsed, args.timeout)
        if state in (AppState.SUCCEEDED, AppState.FAILED, AppState.CANCELLED):
            if state == AppState.SUCCEEDED:
                logger.info("Job SUCCEEDED")
            else:
                logger.error("Job %s", state.name)
                sys.exit(1)
            return

    logger.error("Job timed out after %ds", args.timeout)
    sys.exit(1)


if __name__ == "__main__":
    main()
