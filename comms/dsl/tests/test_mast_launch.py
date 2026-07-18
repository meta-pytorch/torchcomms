# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Host-only unit tests for the MAST launcher's pure arg-parsing + access-knob guard."""

import argparse
import unittest

from comms.dsl.tests.mast_launch import _parse_envs, launch


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
