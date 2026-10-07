#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import tempfile
import unittest
from collections import defaultdict

import torch
from torch.profiler import profile
from torchcomms import new_comm, ReduceOp

TENSOR_COUNT: int = 8192


def get_profiler_mccl_meta(prof):
    """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
    We will need to test metadata obtained from profiler here"""
    tf = tempfile.NamedTemporaryFile(mode="w+t", suffix=".json", delete=False)
    tf.close()
    trace_file = tf.name

    prof.export_chrome_trace(trace_file)
    with open(trace_file) as f:
        events = json.load(f)["traceEvents"]
    print(f"Trace saved to {trace_file}")

    os.remove(trace_file)

    return [e for e in events if e.get("name") == "record_param_comms"]


class ProfilerTest(unittest.TestCase):
    """Test class to verify tracing output  across all operations in TorchComm."""

    def setUp(self) -> None:
        os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
        """Set up test environment before each test."""
        device = torch.device("cuda")

        self.torchcomm = new_comm("mccl", device, name="comms_test_name")

    def tearDown(self):
        """Clean up after each test."""
        # Reset the TorchComm object to ensure proper cleanup
        if hasattr(self, "torchcomm") and self.torchcomm:
            self.torchcomm = None

    def _sanity_check_profiler_mccl_meta(self, mccl_meta_events):
        """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
        We test for basic fields in this profiler event that correspond to the nccl communication
        collectives"""
        per_coll_meta = defaultdict(list)
        for e in mccl_meta_events:
            args = e.get("args", {})
            collname = args.get("Collective name", "")
            self.assertNotEqual(collname, "")
            self.assertNotEqual(args.get("dtype", ""), "")

            per_coll_meta[collname].append(args)

            self.assertEqual(args["Process Group Name"], "comms_test_name")
            self.assertNotEqual(args["Process Group Ranks"], "")

            self.assertGreaterEqual(args.get("In msg nelems", -1), 0)
            self.assertGreaterEqual(args.get("Out msg nelems", -1), 0)
            self.assertGreaterEqual(args.get("Group size", -1), 0)

        return per_coll_meta

    def test_all_tests(self) -> None:
        with profile() as prof:
            # Set rank and size information
            self.rank = self.torchcomm.get_rank()
            self.num_ranks = self.torchcomm.get_size()

            # Calculate device index as rank % num_devices
            device_index = self.rank % torch.cuda.device_count()
            self.device = torch.device(f"cuda:{device_index}")

            options = {"dtype": torch.float, "device": self.device}

            # Prepare multiple tensors for all the operations to be tested.
            send_tensor = torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
            recv_tensor = torch.zeros(TENSOR_COUNT, **options)
            tensors_all_ranks = []
            for _ in range(self.num_ranks):
                tensors_all_ranks.append(torch.zeros(TENSOR_COUNT, **options))
            input_tensors = []
            for _ in range(self.num_ranks):
                input_tensors.append(
                    torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
                )
            send_rank = (self.rank + 1) % self.num_ranks
            recv_rank = (self.rank + self.num_ranks - 1) % self.num_ranks

            # Execute all operations to validate profiler output.
            if self.rank % 2 == 0:
                # Even ranks: send first, then receive
                self.torchcomm.send(send_tensor, send_rank, False)
                self.torchcomm.recv(recv_tensor, recv_rank, False)
            else:
                # Odd ranks: receive first, then send
                self.torchcomm.recv(recv_tensor, recv_rank, False)
                self.torchcomm.send(send_tensor, send_rank, False)

            self.torchcomm.all_reduce(send_tensor, ReduceOp.SUM, False)

            self.torchcomm.all_gather(tensors_all_ranks, send_tensor, False)

            self.torchcomm.all_to_all(tensors_all_ranks, input_tensors, False)

            self.torchcomm.broadcast(send_tensor, 0, False)

            work = self.torchcomm.barrier(False)
            work.wait()
            work = None

            rank_ = self.torchcomm.get_rank()
            self.torchcomm.finalize()

        # Synchronize the CUDA stream before finishing
        torch.cuda.current_stream(self.device).synchronize()

        if rank_ == 0:
            mccl_meta_events = get_profiler_mccl_meta(prof)
            self.assertGreater(len(mccl_meta_events), 0)

            mccl_meta = self._sanity_check_profiler_mccl_meta(mccl_meta_events)
            self.assertEqual(len(mccl_meta["send"]), 1)
            self.assertEqual(len(mccl_meta["recv"]), 1)
            self.assertEqual(len(mccl_meta["all_reduce"]), 1)
            self.assertEqual(len(mccl_meta["all_to_all"]), 1)
            self.assertEqual(len(mccl_meta["all_gather"]), 1)
            self.assertEqual(len(mccl_meta["broadcast"]), 1)
            self.assertEqual(len(mccl_meta["barrier"]), 1)
            self.assertEqual(len(mccl_meta["wait"]), 1)


if __name__ == "__main__":
    unittest.main()
