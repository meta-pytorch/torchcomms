#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
import torchcomms
from torchcomms.tests.helpers.py.cuda_graph_test_helpers import (
    _wait,
    create_capture,
    CudaGraphTestBase,
    GraphTestBuilder,
    skip_unless_ncclx,
)


class TestGraphLifecycle(CudaGraphTestBase):
    """Tests graph creation, destruction, and recreation with the same comm."""

    @skip_unless_ncclx
    def test_graph_destroy_and_recreate(self) -> None:
        """Destroy a comm-containing CUDA graph and recreate it with the same comm."""
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):

                def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
                    return [torch.ones(10, 10, device=self.device)]

                def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
                    return [b.inputs[0] * b.comms[0].get_size()]

                def assert_graph(b: GraphTestBuilder) -> None:
                    info = b.graph_infos[0]
                    ar_kernels = info.kernels_with_name("AllReduce")
                    self.assertEqual(len(ar_kernels), 1)

                def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                    _wait(
                        b.comms[0].all_reduce(
                            b.inputs[0], torchcomms.ReduceOp.SUM, async_op=_async
                        )
                    )

                for _ in range(2):
                    GraphTestBuilder(self).with_comms(1).add_capture(
                        capture
                    ).run_serial(
                        inputs=make_inputs,
                        expected=make_expected,
                        graph_assertions=assert_graph,
                    )

    @skip_unless_ncclx
    def test_graph_recreate_with_different_body(self) -> None:
        """Destroy graph with one comm (simple allreduce), then recreate with
        a different, more complex graph body (allreduce → sum → allgather)
        using two comms. Tests that comms can participate in graphs with
        different topologies across their lifetime."""

        # Cycle 1: simple allreduce with comm0 only
        def capture1(b: GraphTestBuilder) -> None:
            _wait(
                b.comms[0].all_reduce(
                    b.inputs[0], torchcomms.ReduceOp.SUM, async_op=False
                )
            )

        # Cycle 2: complex capture — allreduce(comm0) → sum → allgather(comm1)
        def make_inputs2(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

        def make_expected2(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device) * size,
                torch.tensor([100.0 * size], device=self.device),
                torch.full((size,), 100.0 * size, device=self.device),
            ]

        with self.create_comms(2) as comms:
            (
                GraphTestBuilder(self)
                .with_existing_comms([comms[0]])
                .add_capture(capture1)
                .run_serial(
                    inputs=lambda b: [torch.ones(10, 10, device=self.device)],
                    expected=lambda b: [b.inputs[0] * b.comms[0].get_size()],
                )
            )
            (
                GraphTestBuilder(self)
                .with_existing_comms(list(comms))
                .add_capture(create_capture(0, 1, 2, comm0_idx=0, comm1_idx=1))
                .run_serial(
                    inputs=make_inputs2,
                    expected=make_expected2,
                )
            )

    @skip_unless_ncclx
    def test_graph_replay_concurrent_with_graph_capture(self) -> None:
        """Graph1 replay on comm0 runs concurrently with graph2 capture on
        comm1. Tests that graph capture doesn't interfere with ongoing graph
        replays on a different comm, and that both graphs produce correct
        results afterward."""
        with self.create_comms(4) as comms:
            size = comms[0].get_size()

            # --- Graph 1: capture with comm0/comm1 ---
            g1_inputs: list[torch.Tensor] = [
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]
            g1_originals: list[torch.Tensor] = [t.clone() for t in g1_inputs]
            g1_expected: list[torch.Tensor] = [
                torch.ones(10, 10, device=self.device) * size,
                torch.tensor([100.0 * size], device=self.device),
                torch.full((size,), 100.0 * size, device=self.device),
            ]

            graph1 = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph1):
                stream0 = torch.cuda.Stream()
                stream1 = torch.cuda.Stream()
                initial = torch.cuda.current_stream()

                stream0.wait_stream(initial)
                with torch.cuda.stream(stream0):
                    comms[0].all_reduce(
                        g1_inputs[0], torchcomms.ReduceOp.SUM, async_op=False
                    )
                    torch.sum(
                        g1_inputs[0].flatten(),
                        dim=0,
                        keepdim=True,
                        out=g1_inputs[1],
                    )
                stream1.wait_stream(stream0)
                with torch.cuda.stream(stream1):
                    _wait(
                        comms[1].all_gather_single(
                            g1_inputs[2], g1_inputs[1], async_op=True
                        )
                    )
                initial.wait_stream(stream1)
            graph1.instantiate()

            # --- Concurrently: replay graph1 while capturing graph2 ---
            g2_inputs: list[torch.Tensor] = [
                torch.ones(10, 10, device=self.device) * 2,
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

            graph2 = None
            try:
                replay_stream = torch.cuda.Stream()

                # Launch graph1 replay on a side stream
                with torch.cuda.stream(replay_stream):
                    graph1.replay()

                # Concurrently capture graph2 on the default stream with
                # different comms while graph1 replays on the GPU
                graph2 = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph2):
                    stream0 = torch.cuda.Stream()
                    stream1 = torch.cuda.Stream()
                    initial = torch.cuda.current_stream()

                    stream0.wait_stream(initial)
                    with torch.cuda.stream(stream0):
                        comms[2].all_reduce(
                            g2_inputs[0], torchcomms.ReduceOp.SUM, async_op=False
                        )
                        torch.sum(
                            g2_inputs[0].flatten(),
                            dim=0,
                            keepdim=True,
                            out=g2_inputs[1],
                        )
                    stream1.wait_stream(stream0)
                    with torch.cuda.stream(stream1):
                        _wait(
                            comms[3].all_gather_single(
                                g2_inputs[2], g2_inputs[1], async_op=True
                            )
                        )
                    initial.wait_stream(stream1)
                graph2.instantiate()

                # Wait for graph1 replay to complete
                torch.cuda.synchronize()

                # Verify graph1 replay result
                for inp, exp in zip(g1_inputs, g1_expected):
                    torch.testing.assert_close(inp, exp)

                # Replay graph2 and verify
                g2_originals: list[torch.Tensor] = [t.clone() for t in g2_inputs]
                g2_expected: list[torch.Tensor] = [
                    torch.ones(10, 10, device=self.device) * 2 * size,
                    torch.tensor([200.0 * size], device=self.device),
                    torch.full((size,), 200.0 * size, device=self.device),
                ]
                for _ in range(self.NUM_REPLAYS):
                    for inp, orig in zip(g2_inputs, g2_originals):
                        inp.copy_(orig)
                    graph2.replay()
                    torch.cuda.synchronize()
                    for inp, exp in zip(g2_inputs, g2_expected):
                        torch.testing.assert_close(inp, exp)

                # Also re-replay graph1 to verify it still works
                for inp, orig in zip(g1_inputs, g1_originals):
                    inp.copy_(orig)
                graph1.replay()
                torch.cuda.synchronize()
                for inp, exp in zip(g1_inputs, g1_expected):
                    torch.testing.assert_close(inp, exp)
            finally:
                graph1.reset()
                if graph2 is not None:
                    graph2.reset()


if __name__ == "__main__":
    unittest.main()
