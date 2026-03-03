#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import signal
import subprocess
import sys
import time


class FatalStateTestMixin:
    """Mixin for tests that verify process-fatal scenarios via subprocess.

    Provides helpers to re-invoke the current test binary with a sentinel
    env var, using FileStore-based bootstrap to avoid TCPStore port races.
    """

    def make_subprocess_env(self, sentinel_var: str, rank: int | None = None) -> dict:
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
        if rank is not None:
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
        return env

    def run_subprocesses(
        self, sentinel_var: str, timeout: int = 120
    ) -> list[subprocess.CompletedProcess]:
        """Spawn world_size children in parallel and collect results.

        Each child gets RANK=i via make_subprocess_env(sentinel_var, rank=i).
        On timeout, kills all remaining processes and fails the test.
        """
        world_size = int(os.environ.get("WORLD_SIZE", "8"))
        procs: list[subprocess.Popen] = []
        for i in range(world_size):
            env = self.make_subprocess_env(sentinel_var, rank=i)
            proc = subprocess.Popen(
                [sys.executable, sys.argv[0]],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(proc)

        deadline = time.monotonic() + timeout
        results: list[subprocess.CompletedProcess | None] = [
            None for _ in range(world_size)
        ]
        try:
            for i, proc in enumerate(procs):
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    stdout, stderr = proc.communicate(timeout=remaining)
                    results[i] = subprocess.CompletedProcess(
                        proc.args, proc.returncode, stdout, stderr
                    )
                except subprocess.TimeoutExpired:
                    raise AssertionError(
                        f"Subprocess rank {i} timed out after {timeout}s."
                    )
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            for i, proc in enumerate(procs):
                if results[i] is None:
                    try:
                        stdout, stderr = proc.communicate(timeout=5)
                    except (subprocess.TimeoutExpired, ValueError):
                        stdout, stderr = b"", b""
                    results[i] = subprocess.CompletedProcess(
                        proc.args, proc.returncode or -9, stdout, stderr
                    )

        return results  # type: ignore[return-value]

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

    def assert_subprocess_succeeded(self, result: subprocess.CompletedProcess) -> None:
        """Assert subprocess exited successfully (returncode 0)."""
        # pyre-ignore[16]: unittest.TestCase.assertEqual
        self.assertEqual(
            result.returncode,
            0,
            f"Expected success (returncode 0), got {result.returncode}.\n"
            f"stderr: {result.stderr.decode(errors='replace')}",
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
