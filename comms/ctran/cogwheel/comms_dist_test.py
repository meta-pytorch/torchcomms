# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import asyncio
import logging

from comms.ctran.cogwheel.comms_mast import (
    build_app,
    COMMS_FBPKG_NAME,
    JOB_TIMEOUT_SEC,
    MAST_CFG,
)
from hpc_comms.cclops.cogwheel.common import BenchmarkCogwheelTest
from torchx.runner import get_runner
from torchx.specs import AppState
from windtunnel.cogwheel.test import cogwheel_test

logger: logging.Logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 60


class CommsDistTest(BenchmarkCogwheelTest):
    """Cogwheel test for comms distributed tests via Conveyor.

    Builds the comms test fbpkg via Conveyor and schedules a single MAST
    job on 2 H100 nodes that runs all test binaries sequentially.
    """

    _app_handle: str | None = None

    def setUp(self) -> None:
        super().setUp()
        self._app_handle = None
        self._runner = get_runner()

    def tearDown(self) -> None:
        if self._app_handle is not None:
            try:
                self._runner.cancel(self._app_handle)
                logger.info("Cancelled MAST job: %s", self._app_handle)
            except Exception as e:
                logger.error("Failed to cancel MAST job %s: %s", self._app_handle, e)
        super().tearDown()

    @cogwheel_test
    async def test_comms_dist(self) -> None:
        """Build comms test fbpkg and run all tests sequentially in one MAST job."""
        fbpkg_version = self.get_fbpkg_version(COMMS_FBPKG_NAME)
        fbpkg_id = f"{COMMS_FBPKG_NAME}:{fbpkg_version}"
        logger.info("Comms test fbpkg: %s", fbpkg_id)
        self.add_custom_link(
            "comms_test_fbpkg",
            f"https://www.internalfb.com/intern/fbpkg/?name={COMMS_FBPKG_NAME}&version={fbpkg_version}",
        )

        version_prefix = fbpkg_version[:12]
        job_name = f"comms-dist-test-{version_prefix}"
        app = build_app(fbpkg_id, job_name)

        logger.info("Scheduling MAST job: %s", job_name)
        dryrun_info = self._runner.dryrun(app=app, scheduler="mast", cfg=MAST_CFG)
        self._app_handle = self._runner.schedule(dryrun_info)

        status = self._runner.status(self._app_handle)
        if status and status.ui_url:
            self.add_custom_link("mast_job", status.ui_url)
        logger.info("Scheduled MAST job: %s (handle=%s)", job_name, self._app_handle)

        # Poll until completion.
        elapsed = 0
        while elapsed < JOB_TIMEOUT_SEC:
            await asyncio.sleep(POLL_INTERVAL_SEC)
            elapsed += POLL_INTERVAL_SEC
            status = self._runner.status(self._app_handle)
            if status is None:
                self.fail(f"MAST job no longer exists ({self._app_handle})")
            state = status.state
            logger.info(
                "MAST job status: %s (%ds/%ds)",
                state.name,
                elapsed,
                JOB_TIMEOUT_SEC,
            )
            if state == AppState.SUCCEEDED:
                self._app_handle = None
                return
            if state in (AppState.FAILED, AppState.CANCELLED):
                self.fail(f"MAST job {state.name}: {self._app_handle}")

        self.fail(f"MAST job did not complete within {JOB_TIMEOUT_SEC}s")


def main() -> None:
    CommsDistTest().main()


if __name__ == "__main__":
    main()
