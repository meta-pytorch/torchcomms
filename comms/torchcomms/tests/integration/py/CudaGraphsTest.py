# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import os
import unittest
from collections.abc import Callable

import torch
import torchcomms
from torch import Tensor

# Skip decorator for NCCLX-only tests
# pyre-fixme[5]: Global expression must be annotated.
# pyre-ignore[56]
requires_ncclx = unittest.skipIf(
    os.getenv("TEST_BACKEND") != "ncclx", "Skipping NCCLX-only tests"
)


class TestCudaGraphs(unittest.TestCase):
    NUM_REPLAYS = 3
    NUM_OPS = 5
    TENSOR_SHAPE = (10, 10)

    def setUp(self) -> None:
        self.backend = os.environ.get("TEST_BACKEND", "")
        self.device = torch.accelerator.current_accelerator()
        # Use unique comm name per test to avoid store prefix collision
        comm_name = f"comm_{self._testMethodName}"
        self.comm = torchcomms.new_comm(self.backend, self.device, name=comm_name)
        self.comm1: torchcomms.TorchComm | None = None

    def tearDown(self) -> None:
        torch.accelerator.synchronize()
        if self.comm1 is not None:
            self.comm1.finalize()
        self.comm.finalize()

    def _setup_single_input(self) -> tuple[list[Tensor], list[Tensor]]:
        """Create single input tensor and its expected all-reduced output."""
        inp = torch.ones(*self.TENSOR_SHAPE, device=self.device)
        expected = inp * self.comm.get_size()
        return [inp], [expected]

    def _setup_multiple_inputs(self) -> tuple[list[Tensor], list[Tensor]]:
        """Create multiple input tensors and their expected all-reduced outputs."""
        inputs = [
            torch.ones(*self.TENSOR_SHAPE, device=self.device) * (i + 1)
            for i in range(self.NUM_OPS)
        ]
        expected = [inp * self.comm.get_size() for inp in inputs]
        return inputs, expected

    def _run_graph_test(
        self,
        setup_fn: Callable[[], tuple[list[Tensor], list[Tensor]]],
        graph_body_fn: Callable[[list[Tensor]], None],
    ) -> None:
        """
        Framework for CUDA graph tests.

        Args:
            setup_fn: Returns (inputs, expected_outputs) - one expected per input tensor
            graph_body_fn: The operations to capture in the graph
        """
        inputs, expected_list = setup_fn()
        original_inputs = [inp.clone() for inp in inputs]

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                graph_body_fn(inputs)

            for _ in range(self.NUM_REPLAYS):
                # Reset inputs
                for inp, orig in zip(inputs, original_inputs):
                    inp.copy_(orig)
                graph.replay()
                # Verify each tensor
                for inp, expected in zip(inputs, expected_list):
                    torch.testing.assert_close(inp, expected)
        finally:
            # Must reset graph before communicator destruction (NCCL 2.26+)
            graph.reset()

    @requires_ncclx
    def test_single_allreduce_sync(self) -> None:
        """Single all_reduce with async_op=False."""

        def graph_body(inputs: list[Tensor]) -> None:
            self.comm.all_reduce(inputs[0], torchcomms.ReduceOp.SUM, async_op=False)

        self._run_graph_test(self._setup_single_input, graph_body)

    @requires_ncclx
    def test_single_allreduce_async(self) -> None:
        """Single all_reduce with async_op=True."""

        def graph_body(inputs: list[Tensor]) -> None:
            work = self.comm.all_reduce(
                inputs[0], torchcomms.ReduceOp.SUM, async_op=True
            )
            work.wait()

        self._run_graph_test(self._setup_single_input, graph_body)

    @requires_ncclx
    def test_multiple_allreduce_sync(self) -> None:
        """Multiple all_reduce ops with async_op=False, each on separate tensor."""

        def graph_body(inputs: list[Tensor]) -> None:
            for inp in inputs:
                self.comm.all_reduce(inp, torchcomms.ReduceOp.SUM, async_op=False)

        self._run_graph_test(self._setup_multiple_inputs, graph_body)

    @requires_ncclx
    def test_multiple_allreduce_async_wait_at_end(self) -> None:
        """Multiple all_reduce ops with async_op=True, wait all at end."""

        def graph_body(inputs: list[Tensor]) -> None:
            works = []
            for inp in inputs:
                work = self.comm.all_reduce(inp, torchcomms.ReduceOp.SUM, async_op=True)
                works.append(work)
            for work in works:
                work.wait()

        self._run_graph_test(self._setup_multiple_inputs, graph_body)

    @requires_ncclx
    def test_multiple_allreduce_mixed(self) -> None:
        """Multiple all_reduce ops with mixed async_op values."""

        def graph_body(inputs: list[Tensor]) -> None:
            works = []
            for i, inp in enumerate(inputs):
                async_op = i % 2 == 0  # alternating True/False
                work = self.comm.all_reduce(
                    inp, torchcomms.ReduceOp.SUM, async_op=async_op
                )
                if async_op:
                    works.append(work)
            for work in works:
                work.wait()

        self._run_graph_test(self._setup_multiple_inputs, graph_body)

    @requires_ncclx
    def test_multiple_streams_single_comm(self) -> None:
        """Multiple allreduce ops on different streams, same comm."""
        streams: list[torch.cuda.Stream] = [
            torch.cuda.Stream() for _ in range(self.NUM_OPS)
        ]

        def graph_body(inputs: list[Tensor]) -> None:
            initial_stream = torch.cuda.current_stream()
            for inp, stream in zip(inputs, streams):
                # Branch out from initial stream
                stream.wait_stream(initial_stream)
                with torch.cuda.stream(stream):
                    self.comm.all_reduce(inp, torchcomms.ReduceOp.SUM, async_op=False)
                # Rejoin initial stream
                initial_stream.wait_stream(stream)

        self._run_graph_test(self._setup_multiple_inputs, graph_body)

    @requires_ncclx
    def test_multiple_streams_multiple_comms(self) -> None:
        """Odd/even allreduce pattern across two comms, each with own stream."""
        self.comm1 = torchcomms.new_comm(
            self.backend, self.device, name=f"{self._testMethodName}_comm1"
        )
        streams: list[torch.cuda.Stream] = [
            torch.cuda.Stream() for _ in range(self.NUM_OPS)
        ]

        def graph_body(inputs: list[Tensor]) -> None:
            assert self.comm1 is not None
            comm1 = self.comm1
            initial_stream = torch.cuda.current_stream()
            for i, (inp, stream) in enumerate(zip(inputs, streams)):
                comm = self.comm if i % 2 == 0 else comm1
                # Branch out from initial stream
                stream.wait_stream(initial_stream)
                with torch.cuda.stream(stream):
                    comm.all_reduce(inp, torchcomms.ReduceOp.SUM, async_op=False)
                # Rejoin initial stream
                initial_stream.wait_stream(stream)

        self._run_graph_test(self._setup_multiple_inputs, graph_body)

    @requires_ncclx
    def test_two_streams_two_comms_with_dependency(self) -> None:
        """
        Two streams, two comms with dependency:
        stream0: allreduce(inp[0], comm0) -> sum -> inp[1]
        stream1: allgather(inp[1], comm1) -> inp[2]
        """
        self.comm1 = torchcomms.new_comm(
            self.backend, self.device, name=f"{self._testMethodName}_comm1"
        )
        stream0: torch.cuda.Stream = torch.cuda.Stream()
        stream1: torch.cuda.Stream = torch.cuda.Stream()

        def setup() -> tuple[list[Tensor], list[Tensor]]:
            rank = self.comm.get_rank()
            size = self.comm.get_size()

            # inp[0]: allreduce input (rank-specific)
            # inp[1]: sum_result placeholder (scalar tensor)
            # inp[2]: allgather output placeholder
            inp_allreduce = torch.ones(*self.TENSOR_SHAPE, device=self.device) * (
                rank + 1
            )
            sum_result = torch.zeros(1, device=self.device)
            allgather_output = torch.zeros(size, device=self.device)

            inputs = [inp_allreduce, sum_result, allgather_output]

            # Expected values
            allreduce_expected = torch.ones(
                *self.TENSOR_SHAPE, device=self.device
            ) * sum(range(1, size + 1))
            sum_value = (
                self.TENSOR_SHAPE[0] * self.TENSOR_SHAPE[1] * sum(range(1, size + 1))
            )
            sum_expected = torch.tensor([float(sum_value)], device=self.device)
            allgather_expected = torch.full(
                (size,), float(sum_value), device=self.device
            )

            expected = [allreduce_expected, sum_expected, allgather_expected]
            return inputs, expected

        def graph_body(inputs: list[Tensor]) -> None:
            assert self.comm1 is not None
            comm1 = self.comm1
            initial_stream = torch.cuda.current_stream()

            # Branch out from initial stream to stream0
            stream0.wait_stream(initial_stream)
            with torch.cuda.stream(stream0):
                self.comm.all_reduce(inputs[0], torchcomms.ReduceOp.SUM, async_op=False)
                inputs[1].fill_(inputs[0].sum())

            # stream1 waits for stream0 (dependency)
            stream1.wait_stream(stream0)
            with torch.cuda.stream(stream1):
                comm1.all_gather_single(inputs[2], inputs[1], async_op=False)

            # Rejoin initial stream before capture ends
            initial_stream.wait_stream(stream1)

        self._run_graph_test(setup, graph_body)


if __name__ == "__main__":
    unittest.main()
