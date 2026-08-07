# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Host-only unit tests for the MAST launcher's pure arg-parsing + access-knob guard."""

import argparse
import unittest

from comms.dsl.tests.mast_launch import (
    _ncu_bench_args,
    _parse_envs,
    build_app_and_cfg,
    launch,
)


class ParseEnvsTest(unittest.TestCase):
    def test_parses_pairs(self) -> None:
        self.assertEqual(_parse_envs("A=1;B=2"), {"A": "1", "B": "2"})

    def test_skips_malformed_and_empty(self) -> None:
        # An entry without '=' is skipped (logged), not fatal; blanks are ignored.
        self.assertEqual(_parse_envs("A=1;bad;;B=2"), {"A": "1", "B": "2"})
        self.assertEqual(_parse_envs(""), {})
        self.assertEqual(_parse_envs(None), {})

    def test_value_may_contain_equals(self) -> None:
        self.assertEqual(_parse_envs("K=a=b"), {"K": "a=b"})


class SubmitAccessKnobGuardTest(unittest.TestCase):
    def test_submit_without_access_knobs_exits(self) -> None:
        # --submit with no entitlement/identity/oncall must fail fast (before any spec
        # assembly or capacity use). Only the fields the guard reads are needed here.
        args = argparse.Namespace(submit=True, entitlement=None, dp=None, oncall=None)
        with self.assertRaises(SystemExit):
            launch(args)


class NcuBenchArgsTest(unittest.TestCase):
    def test_empty_when_ncu_off(self) -> None:
        self.assertEqual(_ncu_bench_args(argparse.Namespace(ncu=False)), [])

    def test_injects_flags(self) -> None:
        out = _ncu_bench_args(
            argparse.Namespace(
                ncu=True,
                ncu_metrics="launch__x",
                ncu_launch_count=2,
                ncu_kernel_regex="",
            )
        )
        adjacent = set(zip(out, out[1:]))
        self.assertIn("--ncu", out)
        self.assertIn("--ncu-driver-shim", out)
        self.assertIn(("--ncu-metrics", "launch__x"), adjacent)
        self.assertIn(("--ncu-launch-count", "2"), adjacent)
        self.assertNotIn("--ncu-kernel-regex", out)  # omitted when empty

    def test_kernel_regex_forwarded(self) -> None:
        out = _ncu_bench_args(
            argparse.Namespace(
                ncu=True,
                ncu_metrics="m",
                ncu_launch_count=1,
                ncu_kernel_regex="_a2a_kernel",
            )
        )
        self.assertIn(("--ncu-kernel-regex", "_a2a_kernel"), set(zip(out, out[1:])))


class NcuDeliveryGuardTest(unittest.TestCase):
    def test_fbpkg_plus_ncu_fails_loud(self) -> None:
        # --ncu needs the conda base-image swap; combining it with --delivery fbpkg must raise
        # rather than silently run an un-profiled job.
        with self.assertRaises(SystemExit):
            build_app_and_cfg(argparse.Namespace(ncu=True, delivery="fbpkg"))
