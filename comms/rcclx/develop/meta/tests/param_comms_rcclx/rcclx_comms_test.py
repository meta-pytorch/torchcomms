# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import sys
import unittest

from param_bench.train.comms.pt.fb.launcher import main as test_launcher


class RcclxCommsTest(unittest.TestCase):
    def test_rcclx_all_reduce(self) -> None:
        args = [
            "launcher.py",
            "--launcher",
            "local",
            "--nnode",
            "1",
            "--ppn",
            "2",
            "--z",
            "1",
            "--b",
            "8",
            "--e",
            "64",
            "--collective",
            "all_reduce",
            "--nw-stack",
            "pytorch-torchcomms",
            "--backend",
            "rcclx",
        ]
        sys.argv = args
        test_launcher()
