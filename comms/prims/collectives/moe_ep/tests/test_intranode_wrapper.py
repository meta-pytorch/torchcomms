#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""
unittest wrapper around `test_intranode.test_loop`. Dispatched to
an 8-GPU host by `comms_py_unittest(... num_gpus = 8)`. Validates intranode
dispatch / combine end-to-end against the BF16 + FP8 test matrix.

Hardware: 1 node with 8 GPUs. Equivalent CLI entry point:

    python test_intranode.py --num-processes 8 --num-tokens 4096 \\
        --hidden 7168 --num-topk 8 --num-experts 256

`buck2 test @fbcode//mode/opt-amd-gpu //comms/prims/collectives/moe_ep/tests:test_intranode_wrapper`
"""

from __future__ import annotations

import argparse
import sys
import unittest

# Install the import shims BEFORE the test file's first
# `import deep_ep` / `import deep_ep_cpp` line runs.
import comms.prims.collectives.moe_ep._cpp as _moe_ep_cpp  # noqa: E402
import comms.prims.collectives.moe_ep.moe_ep as _moe_ep  # noqa: E402
import torch

sys.modules["deep_ep"] = _moe_ep
sys.modules["deep_ep_cpp"] = _moe_ep_cpp

from comms.prims.collectives.moe_ep.tests.test_intranode import test_loop  # noqa: E402


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 8,
    "test_intranode requires an 8-GPU host",
)
class Intranode8GpuTest(unittest.TestCase):
    """End-to-end smoke + perf test for intranode dispatch / combine."""

    def test_main(self) -> None:
        """Drives `test_loop` for 8 ranks. Each child rank exercises:
        - get_dispatch_layout
        - notify_dispatch + intranode_dispatch (BF16 / FP8 / with+without topk,
          async + sync, previous_event chaining)
        - intranode_combine (sum-without-weights, sum-with-weights)
        - cached-mode dispatch (handle reuse from prior dispatch)
        """
        # Match test_intranode.py defaults.
        args = argparse.Namespace(
            num_processes=8,
            num_tokens=4096,
            hidden=7168,
            num_topk=8,
            num_experts=256,
            allow_mnnvl=False,
            use_fabric=False,
        )
        torch.multiprocessing.spawn(test_loop, args=(8, args), nprocs=8)


if __name__ == "__main__":
    unittest.main()
