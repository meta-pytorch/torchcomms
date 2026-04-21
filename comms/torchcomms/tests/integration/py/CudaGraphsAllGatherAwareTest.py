#!/usr/bin/env python3
# pyre-unsafe
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for cudagraph-aware AllGather auto-conversion to AGP.

When NCCL_ALLGATHER_ALGO=ctgraph, a regular all_gather_single() call during
CUDA graph capture is transparently converted to the persistent window-based
AGP algorithm (SM-free CE+IB). These tests verify correct data after graph
replay.

To collect Kineto traces, set TORCH_PROFILE_DIR to an output directory:
  TORCH_PROFILE_DIR=/tmp/traces buck2 run ...
Traces are written as <test_name>_rank<N>.json per rank.
"""

import contextlib
import os
import unittest

import torch
import torchcomms  # noqa: F401 — side-effect import registers ncclx backend

# pyre-fixme[21]: Could not find name `ProfilerActivity` in `torch.profiler`.
from torch.profiler import profile, ProfilerActivity
from torchcomms.tests.helpers.py.cuda_graph_test_helpers import (
    _wait,
    CudaGraphTestBase,
    GraphTestBuilder,
    skip_unless_ncclx,
)
from torchcomms.tests.integration.py.TorchCommTestHelpers import get_rank_and_size


class TestAllGatherCudaGraphAware(CudaGraphTestBase):
    """Tests that regular all_gather_single auto-converts to AGP under
    CUDA graph capture when NCCL_ALLGATHER_ALGO=ctgraph."""

    NUM_REPLAYS = 3
    ELEM_COUNT = 1024

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["NCCL_CTRAN_ALLOW_CUDA_GRAPH"] = "1"
        os.environ["NCCL_ALLGATHER_ALGO"] = "ctgraph"
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "COLL")
        super().setUpClass()

    @skip_unless_ncclx
    def test_allgather_cudagraph_aware(self) -> None:
        """Capture a regular all_gather_single into a CUDA graph and verify
        correct results on replay. The ctgraph algo transparently converts
        it to window-based AGP.

        Set TORCH_PROFILE_DIR to collect Kineto traces (handled by
        GraphTestBuilder internally).
        """

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            rank = b.comms[0].get_rank()
            size = b.comms[0].get_size()
            sendbuf = (
                torch.ones(self.ELEM_COUNT, dtype=torch.float32, device=self.device)
                * rank
            )
            recvbuf = torch.zeros(
                self.ELEM_COUNT * size, dtype=torch.float32, device=self.device
            )
            return [sendbuf, recvbuf]

        def capture(b: GraphTestBuilder) -> None:
            _wait(b.comms[0].all_gather_single(b.inputs[1], b.inputs[0], async_op=True))

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            sendbuf = b.inputs[0].clone()
            recvbuf = torch.cat(
                [
                    torch.ones(self.ELEM_COUNT, dtype=torch.float32, device=self.device)
                    * r
                    for r in range(size)
                ]
            )
            return [sendbuf, recvbuf]

        GraphTestBuilder(self).add_capture(capture).run_serial(
            inputs=make_inputs,
            expected=make_expected,
        )

    @skip_unless_ncclx
    def test_allgather_cudagraph_aware_changing_data(self) -> None:
        """Verify graph replay picks up modified sendbuf data each iteration,
        confirming the auto-converted AGP re-reads from the same address.

        Set TORCH_PROFILE_DIR to collect Kineto traces.
        """
        rank, _ = get_rank_and_size()
        profile_dir = os.environ.get("TORCH_PROFILE_DIR")

        with self.create_comms(1) as comms:
            comm = comms[0]
            size = comm.get_size()
            count = self.ELEM_COUNT

            sendbuf = torch.zeros(count, dtype=torch.float32, device=self.device)
            recvbuf = torch.zeros(count * size, dtype=torch.float32, device=self.device)

            # Wrap everything in the profiler so the trace shows:
            #   1. Eager warmup allgathers (regular NCCL kernel path)
            #   2. Graph capture (auto-converts to AGP)
            #   3. Graph replays (SM-free CE+host-node replay)
            profile_ctx = (
                # pyre-fixme[16]: Module `torch.profiler` has no attribute `ProfilerActivity`.
                profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                if profile_dir
                else contextlib.nullcontext()
            )
            with profile_ctx as prof:
                # Eager warmup: run a few regular allgathers before capture
                # so the profile shows the contrast with AGP graph replay.
                NUM_WARMUP = 3
                for _ in range(NUM_WARMUP):
                    sendbuf.fill_(float(rank))
                    _wait(comm.all_gather_single(recvbuf, sendbuf, async_op=True))
                torch.cuda.synchronize()

                # Graph capture: allgather auto-converts to AGP here
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _wait(comm.all_gather_single(recvbuf, sendbuf, async_op=True))

                comm.barrier(False)

                # Graph replay
                for replay in range(self.NUM_REPLAYS):
                    val = float(rank * 100 + replay)
                    sendbuf.fill_(val)
                    recvbuf.zero_()
                    torch.cuda.synchronize()
                    comm.barrier(False)

                    graph.replay()
                    torch.cuda.synchronize()
                    comm.barrier(False)

                    for r in range(size):
                        expected_val = float(r * 100 + replay)
                        chunk = recvbuf[r * count : (r + 1) * count]
                        expected = torch.full(
                            (count,),
                            expected_val,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        torch.testing.assert_close(
                            chunk,
                            expected,
                            rtol=1e-5,
                            atol=1e-5,
                            msg=(
                                f"Replay {replay}: rank {rank} expected {expected_val} "
                                f"from rank {r}, got {chunk[:4].tolist()}"
                            ),
                        )

            if profile_dir and prof:
                os.makedirs(profile_dir, exist_ok=True)
                trace_path = os.path.join(
                    profile_dir,
                    f"test_allgather_cudagraph_aware_changing_data_rank{rank}.json",
                )
                prof.export_chrome_trace(trace_path)

            graph.reset()
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
