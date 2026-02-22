#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import signal
import subprocess
import sys


class FatalStateTestMixin:
    """Mixin for tests that verify process-fatal scenarios via subprocess.

    Provides helpers to re-invoke the current test binary with a sentinel
    env var, using FileStore-based bootstrap to avoid TCPStore port races.
    """

    def make_subprocess_env(self, sentinel_var: str) -> dict:
        """Build env for subprocess with FileStore bootstrap.

        Sets sentinel_var="1", configures TORCHCOMM_STORE_PATH for FileStore,
        and offsets MASTER_PORT to avoid collision with parent process.
        """
        env = os.environ.copy()
        env[sentinel_var] = "1"
        # Deterministic path: all parent ranks share MASTER_PORT, so their
        # subprocesses will agree on the same FileStore file.
        parent_port = os.environ.get("MASTER_PORT", "29500")
        store_path = f"/tmp/torchcomm_test_store_{sentinel_var}_{parent_port}"
        # Remove stale store file from previous runs. All parent ranks
        # attempt this; ENOENT on later ranks is harmless.
        try:
            os.remove(store_path)
        except FileNotFoundError:
            pass
        # TORCHCOMM_STORE_PATH makes StoreManager create a FileStore instead
        # of a TCPStore. MASTER_ADDR/MASTER_PORT must still be set because
        # the NCCLX bootstrap's createStore() guards on their presence before
        # calling StoreManager — but they won't be used for actual store
        # creation when TORCHCOMM_STORE_PATH is set.
        env["TORCHCOMM_STORE_PATH"] = store_path
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(int(parent_port) + 1000)
        return env

    def run_subprocess(
        self, env: dict, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        """Re-invoke current test binary with modified env.

        Calls subprocess.run([sys.executable, sys.argv[0]], ...).
        Fails the test on TimeoutExpired.
        """
        try:
            return subprocess.run(
                [sys.executable, sys.argv[0]],
                env=env,
                timeout=timeout,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as e:
            raise AssertionError(
                f"Subprocess timed out after {timeout}s.\n"
                f"stdout: {(e.stdout or b'').decode(errors='replace')}\n"
                f"stderr: {(e.stderr or b'').decode(errors='replace')}"
            )

    def assert_subprocess_aborted(
        self,
        result: subprocess.CompletedProcess,
        expected_stderr: str | None = None,
    ) -> None:
        """Assert subprocess was killed by SIGABRT, optionally check stderr."""
        # pyre-ignore[16]: unittest.TestCase.assertEqual
        self.assertEqual(
            result.returncode,
            -signal.SIGABRT,
            f"Expected SIGABRT (-{signal.SIGABRT}), "
            f"got {result.returncode}.\n"
            f"stderr: {result.stderr.decode(errors='replace')}",
        )
        if expected_stderr:
            # pyre-ignore[16]: unittest.TestCase.assertIn
            self.assertIn(
                expected_stderr,
                result.stderr.decode(errors="replace"),
                "Expected message not found in subprocess stderr.",
            )

    def assert_subprocess_failed(self, result: subprocess.CompletedProcess) -> None:
        """Assert subprocess exited with non-zero (for asymmetric rank scenarios)."""
        # pyre-ignore[16]: unittest.TestCase.assertNotEqual
        self.assertNotEqual(
            result.returncode,
            0,
            "Expected subprocess to fail (peers are dead).\n"
            f"stderr: {result.stderr.decode(errors='replace')}",
        )
