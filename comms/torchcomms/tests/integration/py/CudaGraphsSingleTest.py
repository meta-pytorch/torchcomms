#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
import torchcomms
from torchcomms.tests.helpers.py.cuda_graph_test_helpers import (
    _wait,
    CudaGraphTestBase,
    GraphTestBuilder,
    skip_unless_ncclx,
)


class TestSingleGraph(CudaGraphTestBase):
    """Tests capturing collectives into a single CUDA graph with varying async
    patterns, streams, and communicators."""

    @skip_unless_ncclx
    def test_single_allreduce(self) -> None:
        """Single all_reduce with sync and async variants."""
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):

                def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                    _wait(
                        b.comms[0].all_reduce(
                            b.inputs[0], torchcomms.ReduceOp.SUM, async_op=_async
                        )
                    )

                def assert_graph(b: GraphTestBuilder) -> None:
                    info = b.graph_infos[0]
                    ar_kernels = info.kernels_with_name("AllReduce")
                    self.assertEqual(len(ar_kernels), 1)
                    self.assertGreater(len(info.nodes_of_type("EVENT_WAIT")), 0)
                    self.assertGreater(len(info.nodes_of_type("EVENT_RECORD")), 0)
                    self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)

                (
                    GraphTestBuilder(self)
                    .add_capture(capture)
                    .run_serial(
                        inputs=lambda b: [torch.ones(10, 10, device=self.device)],
                        expected=lambda b: [b.inputs[0] * b.comms[0].get_size()],
                        graph_assertions=assert_graph,
                    )
                )

    @skip_unless_ncclx
    def test_multiple_allreduce(self) -> None:
        """Multiple all_reduce ops, each on a separate tensor."""
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):

                def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                    for inp in b.inputs:
                        _wait(
                            b.comms[0].all_reduce(
                                inp, torchcomms.ReduceOp.SUM, async_op=_async
                            )
                        )

                def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
                    return [
                        torch.ones(10, 10, device=self.device) * (i + 1)
                        for i in range(self.NUM_OPS)
                    ]

                def assert_graph(b: GraphTestBuilder) -> None:
                    info = b.graph_infos[0]
                    ar_kernels = info.kernels_with_name("AllReduce")
                    self.assertEqual(len(ar_kernels), self.NUM_OPS)
                    for i in range(len(ar_kernels) - 1):
                        self.assertTrue(
                            info.has_path(ar_kernels[i].id, ar_kernels[i + 1].id),
                            f"AllReduce kernels {i} and {i + 1} should be sequential",
                        )

                GraphTestBuilder(self).add_capture(capture).run_serial(
                    inputs=make_inputs,
                    expected=lambda b: [
                        inp * b.comms[0].get_size() for inp in b.inputs
                    ],
                    graph_assertions=assert_graph,
                )

    @skip_unless_ncclx
    def test_multiple_allreduce_async_wait_at_end(self) -> None:
        """Multiple all_reduce ops with async_op=True, wait all at end."""

        def capture(b: GraphTestBuilder) -> None:
            works = [
                b.comms[0].all_reduce(inp, torchcomms.ReduceOp.SUM, async_op=True)
                for inp in b.inputs
            ]
            for work in works:
                _wait(work)

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            return [
                torch.ones(10, 10, device=self.device) * (i + 1)
                for i in range(self.NUM_OPS)
            ]

        def assert_graph(b: GraphTestBuilder) -> None:
            info = b.graph_infos[0]
            ar_kernels = info.kernels_with_name("AllReduce")
            self.assertEqual(len(ar_kernels), self.NUM_OPS)
            for i in range(len(ar_kernels) - 1):
                self.assertTrue(
                    info.has_path(ar_kernels[i].id, ar_kernels[i + 1].id),
                    f"AllReduce kernels {i} and {i + 1} should be sequential",
                )

        GraphTestBuilder(self).add_capture(capture).run_serial(
            inputs=make_inputs,
            expected=lambda b: [inp * b.comms[0].get_size() for inp in b.inputs],
            graph_assertions=assert_graph,
        )

    @skip_unless_ncclx
    def test_multiple_allreduce_mixed(self) -> None:
        """Multiple all_reduce ops with mixed async_op values."""

        def capture(b: GraphTestBuilder) -> None:
            works = []
            for i, inp in enumerate(b.inputs):
                is_async = i % 2 == 0
                work = b.comms[0].all_reduce(
                    inp, torchcomms.ReduceOp.SUM, async_op=is_async
                )
                if is_async:
                    works.append(work)
            for work in works:
                _wait(work)

        def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
            return [
                torch.ones(10, 10, device=self.device) * (i + 1)
                for i in range(self.NUM_OPS)
            ]

        def assert_graph(b: GraphTestBuilder) -> None:
            info = b.graph_infos[0]
            ar_kernels = info.kernels_with_name("AllReduce")
            self.assertEqual(len(ar_kernels), self.NUM_OPS)
            for i in range(len(ar_kernels) - 1):
                self.assertTrue(
                    info.has_path(ar_kernels[i].id, ar_kernels[i + 1].id),
                    f"AllReduce kernels {i} and {i + 1} should be sequential",
                )

        GraphTestBuilder(self).add_capture(capture).run_serial(
            inputs=make_inputs,
            expected=lambda b: [inp * b.comms[0].get_size() for inp in b.inputs],
            graph_assertions=assert_graph,
        )

    @skip_unless_ncclx
    def test_multiple_streams_single_comm(self) -> None:
        """Multiple allreduce ops on different streams, same comm."""
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):

                def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                    initial_stream = torch.cuda.current_stream()
                    for inp, stream in zip(b.inputs, b.streams):
                        stream.wait_stream(initial_stream)
                        with torch.cuda.stream(stream):
                            _wait(
                                b.comms[0].all_reduce(
                                    inp, torchcomms.ReduceOp.SUM, async_op=_async
                                )
                            )
                        initial_stream.wait_stream(stream)

                def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
                    return [
                        torch.ones(10, 10, device=self.device) * (i + 1)
                        for i in range(self.NUM_OPS)
                    ]

                def assert_graph(b: GraphTestBuilder) -> None:
                    info = b.graph_infos[0]
                    ar_kernels = info.kernels_with_name("AllReduce")
                    self.assertEqual(len(ar_kernels), self.NUM_OPS)
                    for i in range(len(ar_kernels) - 1):
                        self.assertTrue(
                            info.has_path(ar_kernels[i].id, ar_kernels[i + 1].id),
                            f"AllReduce kernels {i} and {i + 1} should be sequential"
                            " (same comm enforces ordering)",
                        )

                GraphTestBuilder(self).with_streams(self.NUM_OPS).add_capture(
                    capture
                ).run_serial(
                    inputs=make_inputs,
                    expected=lambda b: [
                        inp * b.comms[0].get_size() for inp in b.inputs
                    ],
                    graph_assertions=assert_graph,
                )

    @skip_unless_ncclx
    def test_multiple_streams_multiple_comms(self) -> None:
        """Odd/even allreduce pattern across two comms, each with own stream."""
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):

                def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                    initial_stream = torch.cuda.current_stream()
                    for i, (inp, stream) in enumerate(zip(b.inputs, b.streams)):
                        comm = b.comms[i % len(b.comms)]
                        stream.wait_stream(initial_stream)
                        with torch.cuda.stream(stream):
                            _wait(
                                comm.all_reduce(
                                    inp, torchcomms.ReduceOp.SUM, async_op=_async
                                )
                            )
                        initial_stream.wait_stream(stream)

                def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
                    return [
                        torch.ones(10, 10, device=self.device) * (i + 1)
                        for i in range(self.NUM_OPS)
                    ]

                def assert_graph(b: GraphTestBuilder) -> None:
                    info = b.graph_infos[0]
                    ar_kernels = info.kernels_with_name("AllReduce")
                    self.assertEqual(len(ar_kernels), self.NUM_OPS)
                    self.assertEqual(len(info.nodes_of_type("MEMCPY")), 0)

                GraphTestBuilder(self).with_comms(2).with_streams(
                    self.NUM_OPS
                ).add_capture(capture).run_serial(
                    inputs=make_inputs,
                    expected=lambda b: [
                        inp * b.comms[0].get_size() for inp in b.inputs
                    ],
                    graph_assertions=assert_graph,
                )

    @skip_unless_ncclx
    def test_two_streams_two_comms_with_dependency(self) -> None:
        """
        Two streams, two comms with dependency:
        stream0: allreduce(inp[0], comm0) -> sum -> inp[1]
        stream1: allgather(inp[1], comm1) -> inp[2]
        """
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):

                def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
                    rank = b.comms[0].get_rank()
                    size = b.comms[0].get_size()
                    return [
                        torch.ones(10, 10, device=self.device) * (rank + 1),
                        torch.zeros(1, device=self.device),
                        torch.zeros(size, device=self.device),
                    ]

                def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
                    size = b.comms[0].get_size()
                    sum_value = 10 * 10 * sum(range(1, size + 1))
                    return [
                        torch.ones(10, 10, device=self.device)
                        * sum(range(1, size + 1)),
                        torch.tensor(
                            [sum_value], device=self.device, dtype=torch.float32
                        ),
                        torch.full(
                            (size,), sum_value, device=self.device, dtype=torch.float32
                        ),
                    ]

                def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                    initial_stream = torch.cuda.current_stream()

                    b.streams[0].wait_stream(initial_stream)
                    with torch.cuda.stream(b.streams[0]):
                        _wait(
                            b.comms[0].all_reduce(
                                b.inputs[0], torchcomms.ReduceOp.SUM, async_op=_async
                            )
                        )
                        # no intermediate alloc/memcpy as opposed to b.inputs[1].fill_(b.inputs[0].sum())
                        torch.sum(
                            b.inputs[0].flatten(),
                            dim=0,
                            keepdim=True,
                            out=b.inputs[1],
                        )

                    b.streams[1].wait_stream(b.streams[0])
                    with torch.cuda.stream(b.streams[1]):
                        _wait(
                            b.comms[1].all_gather_single(
                                b.inputs[2], b.inputs[1], async_op=_async
                            )
                        )

                    initial_stream.wait_stream(b.streams[1])

                def assert_graph(b: GraphTestBuilder) -> None:
                    info = b.graph_infos[0]
                    ar_kernels = info.kernels_with_name("AllReduce")
                    ag_kernels = info.kernels_with_name("AllGather")
                    self.assertEqual(len(ar_kernels), 1)
                    self.assertEqual(len(ag_kernels), 1)
                    reduce_kernels = info.kernels_with_name("reduce_kernel")
                    self.assertEqual(len(reduce_kernels), 1)
                    self.assertTrue(
                        info.has_path(ar_kernels[0].id, reduce_kernels[0].id),
                        "AllReduce must precede reduce (sum)",
                    )

                GraphTestBuilder(self).with_comms(2).with_streams(2).add_capture(
                    capture
                ).run_serial(
                    inputs=make_inputs,
                    expected=make_expected,
                    graph_assertions=assert_graph,
                )


if __name__ == "__main__":
    unittest.main()
