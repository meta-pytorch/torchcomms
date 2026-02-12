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
    PipelineStep,
    skip_unless_ncclx,
    TENSORS_PER_CAPTURE,
)


class TestMultipleGraphs(CudaGraphTestBase):
    """Tests capturing multiple CUDA graphs separately, replayed serially,
    concurrently, or with inter-graph dependencies.

    Each graph uses complex_capture: allreduce(sync) → sum → allgather(async)
    with intra-graph stream dependencies, rather than a trivial single
    collective (which is already covered by TestSingleGraph).
    """

    def _make_inputs(self, b: GraphTestBuilder) -> list[torch.Tensor]:
        """Create 3 tensors per graph: input (10x10), intermediate (1,), output (size,)."""
        size = b.comms[0].get_size()
        tensors: list[torch.Tensor] = []
        for i in range(self.NUM_GRAPHS):
            tensors.extend(
                [
                    torch.ones(10, 10, device=self.device) * (i + 1),
                    torch.zeros(1, device=self.device),
                    torch.zeros(size, device=self.device),
                ]
            )
        return tensors

    def _make_expected(self, b: GraphTestBuilder) -> list[torch.Tensor]:
        """Expected results after complex_capture per graph.

        allreduce scales input by world_size, sum reduces the 10x10 to a
        scalar, allgather replicates the scalar across ranks.
        """
        size = b.comms[0].get_size()
        tensors: list[torch.Tensor] = []
        for i in range(self.NUM_GRAPHS):
            ar_value = float((i + 1) * size)
            sum_value = 100.0 * (i + 1) * size
            tensors.extend(
                [
                    torch.ones(10, 10, device=self.device) * ar_value,
                    torch.tensor([sum_value], device=self.device),
                    torch.full((size,), sum_value, device=self.device),
                ]
            )
        return tensors

    def _assert_complex_graphs(self, b: GraphTestBuilder) -> None:
        """Assert each graph has the expected kernel structure from complex_capture."""
        self.assertEqual(len(b.graph_infos), self.NUM_GRAPHS)
        for info in b.graph_infos:
            ar_kernels = info.kernels_with_name("AllReduce")
            ag_kernels = info.kernels_with_name("AllGather")
            reduce_kernels = info.kernels_with_name("reduce_kernel")
            self.assertEqual(len(ar_kernels), 1)
            self.assertEqual(len(ag_kernels), 1)
            self.assertEqual(len(reduce_kernels), 1)
            self.assertTrue(
                info.has_path(ar_kernels[0].id, reduce_kernels[0].id),
                "AllReduce must precede reduce (sum)",
            )
            self.assertTrue(
                info.has_path(reduce_kernels[0].id, ag_kernels[0].id),
                "reduce (sum) must precede AllGather",
            )
            self.assertGreater(len(info.nodes_of_type("EVENT_WAIT")), 0)
            self.assertGreater(len(info.nodes_of_type("EVENT_RECORD")), 0)
            self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)

    @skip_unless_ncclx
    def test_multiple_graphs_serial(self) -> None:
        """Multiple complex CUDA graphs replayed serially, shared comms."""
        builder = GraphTestBuilder(self).with_comms(2)
        for i in range(self.NUM_GRAPHS):
            base = i * TENSORS_PER_CAPTURE
            builder.add_capture(
                create_capture(base, base + 1, base + 2, comm0_idx=0, comm1_idx=1)
            )
        builder.run_serial(
            inputs=self._make_inputs,
            expected=self._make_expected,
            graph_assertions=self._assert_complex_graphs,
        )

    @skip_unless_ncclx
    def test_multiple_graphs_serial_different_comms(self) -> None:
        """Multiple complex CUDA graphs with per-graph comm pairs, replayed serially."""
        builder = GraphTestBuilder(self).with_comms(2 * self.NUM_GRAPHS)
        for i in range(self.NUM_GRAPHS):
            base = i * TENSORS_PER_CAPTURE
            builder.add_capture(
                create_capture(
                    base,
                    base + 1,
                    base + 2,
                    comm0_idx=2 * i,
                    comm1_idx=2 * i + 1,
                )
            )
        builder.run_serial(
            inputs=self._make_inputs,
            expected=self._make_expected,
            graph_assertions=self._assert_complex_graphs,
        )

    @skip_unless_ncclx
    def test_multiple_graphs_concurrent(self) -> None:
        """Multiple complex CUDA graphs replayed concurrently, shared comms."""
        builder = GraphTestBuilder(self).with_comms(2).with_streams(self.NUM_GRAPHS)
        for i in range(self.NUM_GRAPHS):
            base = i * TENSORS_PER_CAPTURE
            builder.add_capture(
                create_capture(base, base + 1, base + 2, comm0_idx=0, comm1_idx=1),
                stream=i,
            )
        builder.run_concurrent(
            inputs=self._make_inputs,
            expected=self._make_expected,
            graph_assertions=self._assert_complex_graphs,
        )

    @skip_unless_ncclx
    def test_multiple_graphs_concurrent_different_comms(self) -> None:
        """Multiple complex CUDA graphs with per-graph comm pairs, replayed concurrently."""
        builder = (
            GraphTestBuilder(self)
            .with_comms(2 * self.NUM_GRAPHS)
            .with_streams(self.NUM_GRAPHS)
        )
        for i in range(self.NUM_GRAPHS):
            base = i * TENSORS_PER_CAPTURE
            builder.add_capture(
                create_capture(
                    base,
                    base + 1,
                    base + 2,
                    comm0_idx=2 * i,
                    comm1_idx=2 * i + 1,
                ),
                stream=i,
            )
        builder.run_concurrent(
            inputs=self._make_inputs,
            expected=self._make_expected,
            graph_assertions=self._assert_complex_graphs,
        )

    @skip_unless_ncclx
    def test_multiple_graphs_with_dependency(self) -> None:
        """Two complex CUDA graphs where graph1's allreduce input is copied
        from graph0's allreduce output, creating an inter-graph dependency."""
        num_graphs: int = 2

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            tensors: list[torch.Tensor] = []
            for i in range(num_graphs):
                tensors.extend(
                    [
                        torch.ones(10, 10, device=self.device) * (i + 1),
                        torch.zeros(1, device=self.device),
                        torch.zeros(size, device=self.device),
                    ]
                )
            return tensors

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            # Graph 0: allreduce(ones * 1) across size ranks → ones * size
            g0_ar = float(size)
            g0_sum = 100.0 * size
            # Graph 1 input is graph 0's allreduce output (ones * size),
            # then allreduced again → ones * size * size
            g1_ar = float(size * size)
            g1_sum = 100.0 * size * size
            return [
                torch.ones(10, 10, device=self.device) * g0_ar,
                torch.tensor([g0_sum], device=self.device),
                torch.full((size,), g0_sum, device=self.device),
                torch.ones(10, 10, device=self.device) * g1_ar,
                torch.tensor([g1_sum], device=self.device),
                torch.full((size,), g1_sum, device=self.device),
            ]

        def pipeline(b: GraphTestBuilder) -> list[PipelineStep]:
            def copy_g0_output_to_g1_input() -> None:
                b.inputs[TENSORS_PER_CAPTURE].copy_(b.inputs[0])

            return [
                (b.graphs[0], b.streams[0]),
                copy_g0_output_to_g1_input,
                (b.graphs[1], b.streams[1]),
            ]

        def assert_graph(b: GraphTestBuilder) -> None:
            self.assertEqual(len(b.graph_infos), num_graphs)
            for info in b.graph_infos:
                ar_kernels = info.kernels_with_name("AllReduce")
                ag_kernels = info.kernels_with_name("AllGather")
                reduce_kernels = info.kernels_with_name("reduce_kernel")
                self.assertEqual(len(ar_kernels), 1)
                self.assertEqual(len(ag_kernels), 1)
                self.assertEqual(len(reduce_kernels), 1)
                self.assertTrue(
                    info.has_path(ar_kernels[0].id, reduce_kernels[0].id),
                    "AllReduce must precede reduce (sum)",
                )
                self.assertTrue(
                    info.has_path(reduce_kernels[0].id, ag_kernels[0].id),
                    "reduce (sum) must precede AllGather",
                )
                self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)

        (
            GraphTestBuilder(self)
            .with_comms(2 * num_graphs)
            .with_streams(num_graphs)
            .add_capture(
                create_capture(0, 1, 2, comm0_idx=0, comm1_idx=1),
                stream=0,
            )
            .add_capture(
                create_capture(3, 4, 5, comm0_idx=2, comm1_idx=3),
                stream=1,
            )
            .run_custom_schedule(
                pipeline,
                inputs=make_inputs,
                expected=make_expected,
                graph_assertions=assert_graph,
            )
        )

    @skip_unless_ncclx
    def test_multiple_graphs_event_sync(self) -> None:
        """Three complex CUDA graphs synchronized via CUDA events.

        Graphs 0 and 1 run concurrently (fork). Their completion is
        synchronized via CUDA events (NOT full device sync). A copy
        transfers graph 0's allreduce output into graph 2's input, then
        graph 2 runs — all chained through events.

        Pipeline DAG:
          graph0 on stream0  ─┐
                               ├─ copy (event-synced, stream2) ─→ graph2 (stream2)
          graph1 on stream1  ─┘
        """

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                # Graph 0
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
                # Graph 1
                torch.ones(10, 10, device=self.device) * 2,
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
                # Graph 2 (input overwritten by event-synced copy)
                torch.zeros(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            g0_ar = float(size)
            g0_sum = 100.0 * size
            g1_ar = 2.0 * size
            g1_sum = 200.0 * size
            # Graph 2 input is graph 0's allreduce output (ones * size),
            # then allreduced again → ones * size * size
            g2_ar = float(size * size)
            g2_sum = 100.0 * size * size
            return [
                torch.ones(10, 10, device=self.device) * g0_ar,
                torch.tensor([g0_sum], device=self.device),
                torch.full((size,), g0_sum, device=self.device),
                torch.ones(10, 10, device=self.device) * g1_ar,
                torch.tensor([g1_sum], device=self.device),
                torch.full((size,), g1_sum, device=self.device),
                torch.ones(10, 10, device=self.device) * g2_ar,
                torch.tensor([g2_sum], device=self.device),
                torch.full((size,), g2_sum, device=self.device),
            ]

        def pipeline(b: GraphTestBuilder) -> list[PipelineStep]:
            def copy_g0_to_g2() -> None:
                b.inputs[2 * TENSORS_PER_CAPTURE].copy_(b.inputs[0])

            return [
                # Fork: graphs 0 and 1 run concurrently
                [(b.graphs[0], b.streams[0]), (b.graphs[1], b.streams[1])],
                # Event-synced copy: stream2 waits on events from both
                # graphs (NOT torch.cuda.synchronize())
                (copy_g0_to_g2, b.streams[2]),
                # Graph 2 chained after copy via events
                (b.graphs[2], b.streams[2]),
            ]

        (
            GraphTestBuilder(self)
            .with_comms(2 * self.NUM_GRAPHS)
            .with_streams(self.NUM_GRAPHS)
            .add_capture(
                create_capture(0, 1, 2, comm0_idx=0, comm1_idx=1),
                stream=0,
            )
            .add_capture(
                create_capture(3, 4, 5, comm0_idx=2, comm1_idx=3),
                stream=1,
            )
            .add_capture(
                create_capture(6, 7, 8, comm0_idx=4, comm1_idx=5),
                stream=2,
            )
            .run_custom_schedule(
                pipeline,
                inputs=make_inputs,
                expected=make_expected,
                graph_assertions=self._assert_complex_graphs,
            )
        )

    @skip_unless_ncclx
    def test_multiple_graphs_external_event_sync(self) -> None:
        """Two graphs replayed concurrently, synchronized mid-execution via
        an external CUDA event captured into both graphs.

        Graph 0: allreduce(sync) on inputs[0] → RECORD event → sum → allgather
        Graph 1: allreduce(sync) on inputs[3] → WAIT event → sum(inputs[0]) → allgather

        Graph 1's sum reads graph 0's allreduce output (inputs[0]), so the
        event is required for correctness. Both graphs replay concurrently;
        the synchronization happens entirely on-device with no host-side sync.
        """
        sync_event: torch.cuda.Event = torch.cuda.Event(external=True)

        def capture_g0(b: GraphTestBuilder) -> None:
            stream0 = torch.cuda.Stream()
            stream1 = torch.cuda.Stream()
            initial = torch.cuda.current_stream()

            stream0.wait_stream(initial)
            with torch.cuda.stream(stream0):
                # Sync collective
                b.comms[0].all_reduce(
                    b.inputs[0], torchcomms.ReduceOp.SUM, async_op=False
                )
                # Record event after allreduce so graph 1 can read inputs[0]
                sync_event.record(stream0)

            stream1.wait_stream(stream0)
            with torch.cuda.stream(stream1):
                torch.sum(
                    b.inputs[0].flatten(),
                    dim=0,
                    keepdim=True,
                    out=b.inputs[1],
                )
                # Async collective
                _wait(
                    b.comms[1].all_gather_single(
                        b.inputs[2], b.inputs[1], async_op=True
                    )
                )

            initial.wait_stream(stream1)

        def capture_g1(b: GraphTestBuilder) -> None:
            stream0 = torch.cuda.Stream()
            stream1 = torch.cuda.Stream()
            initial = torch.cuda.current_stream()

            stream0.wait_stream(initial)
            with torch.cuda.stream(stream0):
                # Independent sync collective
                b.comms[2].all_reduce(
                    b.inputs[3], torchcomms.ReduceOp.SUM, async_op=False
                )

            # Wait for both: local allreduce AND graph 0's allreduce via event
            stream1.wait_stream(stream0)
            stream1.wait_event(sync_event)
            with torch.cuda.stream(stream1):
                # Read graph 0's allreduce output (inputs[0])
                torch.sum(
                    b.inputs[0].flatten(),
                    dim=0,
                    keepdim=True,
                    out=b.inputs[4],
                )
                # Async collective
                _wait(
                    b.comms[3].all_gather_single(
                        b.inputs[5], b.inputs[4], async_op=True
                    )
                )

            initial.wait_stream(stream1)

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                # Graph 0: allreduce input, sum output, allgather output
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
                # Graph 1: allreduce input, sum output, allgather output
                torch.ones(10, 10, device=self.device) * 2,
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            # Graph 0: allreduce(ones) → ones * size
            g0_ar = float(size)
            g0_sum = 100.0 * size
            # Graph 1: allreduce(ones * 2) → ones * 2 * size (independent)
            g1_ar = 2.0 * size
            # Graph 1's sum reads graph 0's allreduce output (inputs[0]),
            # which is ones * size, so sum = 100 * size
            g1_sum = 100.0 * size
            return [
                torch.ones(10, 10, device=self.device) * g0_ar,
                torch.tensor([g0_sum], device=self.device),
                torch.full((size,), g0_sum, device=self.device),
                torch.ones(10, 10, device=self.device) * g1_ar,
                torch.tensor([g1_sum], device=self.device),
                torch.full((size,), g1_sum, device=self.device),
            ]

        num_graphs: int = 2

        def assert_graph(b: GraphTestBuilder) -> None:
            self.assertEqual(len(b.graph_infos), num_graphs)
            for info in b.graph_infos:
                ar_kernels = info.kernels_with_name("AllReduce")
                self.assertEqual(len(ar_kernels), 1)
                self.assertGreater(len(info.nodes_of_type("EVENT_WAIT")), 0)
                self.assertGreater(len(info.nodes_of_type("EVENT_RECORD")), 0)
                self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)
            # Graph 0 must have AllGather; graph 1 must have AllGather
            for info in b.graph_infos:
                ag_kernels = info.kernels_with_name("AllGather")
                self.assertEqual(len(ag_kernels), 1)

        (
            GraphTestBuilder(self)
            .with_comms(4)
            .with_streams(num_graphs)
            .add_capture(capture_g0, stream=0)
            .add_capture(capture_g1, stream=1)
            .run_concurrent(
                inputs=make_inputs,
                expected=make_expected,
                graph_assertions=assert_graph,
            )
        )


if __name__ == "__main__":
    unittest.main()
