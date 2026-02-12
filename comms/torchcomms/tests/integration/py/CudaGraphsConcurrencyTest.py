# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import unittest

import torch
import torchcomms
from torchcomms.tests.helpers.py.cuda_graph_test_helpers import (
    create_capture,
    CudaGraphTestBase,
    GraphTestBuilder,
    PipelineStep,
    skip_unless_ncclx,
)
from torchcomms.tests.integration.py.TorchCommTestHelpers import get_rank_and_size


class TestGraphConcurrency(CudaGraphTestBase):
    """Tests graph replay running concurrently with non-graphable GPU work.

    Each graph uses create_capture: allreduce(sync) → sum → allgather(async)
    with intra-graph stream dependencies, rather than a trivial single
    collective (which is already covered by TestSingleGraph).
    """

    @skip_unless_ncclx
    def test_graph_parallel_with_nongraphable(self) -> None:
        """Complex CUDA graph replay running concurrently with non-graphable compute."""
        nongraph_input: torch.Tensor = torch.ones(10, 10, device=self.device) * 2
        nongraph_output: torch.Tensor = torch.zeros(10, 10, device=self.device)

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device) * size,
                torch.tensor([100.0 * size], device=self.device),
                torch.full((size,), 100.0 * size, device=self.device),
            ]

        def nongraph_matmul() -> None:
            torch.matmul(nongraph_input, nongraph_input, out=nongraph_output)

        def pipeline(b: GraphTestBuilder) -> list[PipelineStep]:
            return [
                [
                    (b.graphs[0], b.streams[0]),
                    (nongraph_matmul, b.streams[1]),
                ]
            ]

        def assert_graph(b: GraphTestBuilder) -> None:
            info = b.graph_infos[0]
            self.assertEqual(len(info.kernels_with_name("AllReduce")), 1)
            self.assertEqual(len(info.kernels_with_name("AllGather")), 1)
            self.assertEqual(len(info.kernels_with_name("reduce_kernel")), 1)
            self.assertGreater(len(info.nodes_of_type("EVENT_WAIT")), 0)
            self.assertGreater(len(info.nodes_of_type("EVENT_RECORD")), 0)
            self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)

        (
            GraphTestBuilder(self)
            .with_comms(2)
            .with_streams(2)
            .add_capture(create_capture(0, 1, 2, comm0_idx=0, comm1_idx=1))
            .run_custom_schedule(
                pipeline,
                inputs=make_inputs,
                expected=make_expected,
                graph_assertions=assert_graph,
            )
        )

        # matmul of 10x10 matrix of 2s: each element = 2*2*10 = 40
        torch.testing.assert_close(
            nongraph_output,
            torch.full((10, 10), 40.0, device=nongraph_output.device),
        )

    @skip_unless_ncclx
    def test_graph_parallel_with_nongraphable_collective(self) -> None:
        """Complex graph replay concurrently with a non-graphable collective
        on a separate comm, testing comm resource isolation under concurrency."""
        nongraph_input: torch.Tensor = torch.ones(10, 10, device=self.device) * 3

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device) * size,
                torch.tensor([100.0 * size], device=self.device),
                torch.full((size,), 100.0 * size, device=self.device),
            ]

        def pipeline(b: GraphTestBuilder) -> list[PipelineStep]:
            def nongraph_allreduce() -> None:
                nongraph_input.fill_(3.0)
                b.comms[2].all_reduce(
                    nongraph_input, torchcomms.ReduceOp.SUM, async_op=False
                )

            return [
                [
                    (b.graphs[0], b.streams[0]),
                    (nongraph_allreduce, b.streams[1]),
                ]
            ]

        (
            GraphTestBuilder(self)
            .with_comms(3)  # 0,1 for graph; 2 for non-graphable collective
            .with_streams(2)
            .add_capture(create_capture(0, 1, 2, comm0_idx=0, comm1_idx=1))
            .run_custom_schedule(
                pipeline,
                inputs=make_inputs,
                expected=make_expected,
            )
        )

        _, size = get_rank_and_size()
        torch.testing.assert_close(
            nongraph_input,
            torch.full((10, 10), 3.0 * size, device=self.device),
        )

    @skip_unless_ncclx
    def test_multiple_graphs_parallel_with_nongraphable(self) -> None:
        """Multiple complex graph replays running concurrently with
        non-graphable compute, testing high stream concurrency with
        intra-graph streams nested inside."""
        nongraph_input: torch.Tensor = torch.ones(10, 10, device=self.device) * 2
        nongraph_output: torch.Tensor = torch.zeros(10, 10, device=self.device)
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
            tensors: list[torch.Tensor] = []
            for i in range(num_graphs):
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

        def nongraph_matmul() -> None:
            torch.matmul(nongraph_input, nongraph_input, out=nongraph_output)

        def pipeline(b: GraphTestBuilder) -> list[PipelineStep]:
            return [
                [
                    (b.graphs[0], b.streams[0]),
                    (b.graphs[1], b.streams[1]),
                    (nongraph_matmul, b.streams[2]),
                ]
            ]

        def assert_graph(b: GraphTestBuilder) -> None:
            self.assertEqual(len(b.graph_infos), num_graphs)
            for info in b.graph_infos:
                self.assertEqual(len(info.kernels_with_name("AllReduce")), 1)
                self.assertEqual(len(info.kernels_with_name("AllGather")), 1)
                self.assertEqual(len(info.kernels_with_name("reduce_kernel")), 1)
                self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)

        (
            GraphTestBuilder(self)
            .with_comms(2 * num_graphs)
            .with_streams(3)  # 2 for graphs, 1 for non-graphable
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

        torch.testing.assert_close(
            nongraph_output,
            torch.full((10, 10), 40.0, device=nongraph_output.device),
        )

    @skip_unless_ncclx
    def test_graph_then_nongraphable_event_sync(self) -> None:
        """Complex graph replay followed by non-graphable work that reads
        the graph's output, synchronized via CUDA events (not full device
        sync)."""
        nongraph_output: torch.Tensor = torch.zeros(10, 10, device=self.device)

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device),
                torch.zeros(1, device=self.device),
                torch.zeros(size, device=self.device),
            ]

        def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
            size = b.comms[0].get_size()
            return [
                torch.ones(10, 10, device=self.device) * size,
                torch.tensor([100.0 * size], device=self.device),
                torch.full((size,), 100.0 * size, device=self.device),
            ]

        def pipeline(b: GraphTestBuilder) -> list[PipelineStep]:
            def scale_graph_output() -> None:
                # Reads graph's allreduce output (inputs[0]) — requires
                # event sync to see the correct value
                torch.mul(b.inputs[0], 2.0, out=nongraph_output)

            return [
                # Graph replays on stream 0
                (b.graphs[0], b.streams[0]),
                # Event-synced: stream 1 waits on stream 0's event,
                # then reads graph's output (NOT full device sync)
                (scale_graph_output, b.streams[1]),
            ]

        (
            GraphTestBuilder(self)
            .with_comms(2)
            .with_streams(2)
            .add_capture(create_capture(0, 1, 2, comm0_idx=0, comm1_idx=1))
            .run_custom_schedule(
                pipeline,
                inputs=make_inputs,
                expected=make_expected,
            )
        )

        _, size = get_rank_and_size()
        # nongraph_output = inputs[0] * 2 = ones * size * 2
        torch.testing.assert_close(
            nongraph_output,
            torch.full((10, 10), 2.0 * size, device=self.device),
        )


if __name__ == "__main__":
    unittest.main()
