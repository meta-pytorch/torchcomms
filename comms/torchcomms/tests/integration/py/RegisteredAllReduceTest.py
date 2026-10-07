#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for persistent registered all-reduce via the TorchComm RCCLX backend."""

import gc
import os
import unittest

import torch
import torchcomms
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class RegisteredAllReduceTest(unittest.TestCase):
    REGISTERED_BYTES = 512 * 1024
    ELEM_COUNT = REGISTERED_BYTES // 2

    def setUp(self) -> None:
        self.requests = []
        self.graphs = []
        os.environ.setdefault("TEST_BACKEND", "rcclx")
        self.wrapper = TorchCommTestWrapper()
        self.comm = self.wrapper.get_torchcomm()
        self.rank = self.comm.get_rank()
        self.size = self.comm.get_size()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            self.skipTest("registered all-reduce requires a CUDA/HIP device")
        self.device = torch.device(f"cuda:{self.rank % device_count}")
        if self.size != 4:
            self.skipTest(f"registered all-reduce v1 requires TP4, got {self.size}")

    def tearDown(self) -> None:
        for graph in reversed(self.graphs):
            graph.reset()
        for request in reversed(self.requests):
            request.close()
        del self.comm
        del self.wrapper

    def _new_request(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torchcomms.RegisteredAllReduce]:
        input_tensor = torch.full(
            (self.ELEM_COUNT,),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        output_tensor = torch.empty_like(input_tensor)
        request = self.comm.registered_all_reduce(
            input_tensor, output_tensor, self.REGISTERED_BYTES
        )
        self.requests.append(request)
        return input_tensor, output_tensor, request

    def _assert_sum(self, output_tensor: torch.Tensor) -> None:
        expected = torch.full_like(
            output_tensor, float(self.size * (self.size + 1) // 2)
        )
        torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)

    def test_one_shot_exact_output_and_close(self) -> None:
        input_tensor, output_tensor, request = self._new_request()

        result = request.all_reduce(
            input_tensor, out=output_tensor, registered_input=True
        )
        self.assertIsNone(result)
        torch.cuda.synchronize()
        self._assert_sum(output_tensor)
        self.assertEqual(request.output.data_ptr(), output_tensor.data_ptr())

        request.close()
        self.assertTrue(request.closed)
        request.close()
        with self.assertRaises(RuntimeError):
            request.all_reduce(input_tensor, out=output_tensor, registered_input=True)

    def test_strong_tensor_lifetime(self) -> None:
        input_tensor, output_tensor, request = self._new_request()
        input_ptr = input_tensor.data_ptr()
        output_ptr = output_tensor.data_ptr()
        registered_input = request.input
        registered_output = request.output

        del input_tensor
        del output_tensor
        gc.collect()

        self.assertEqual(request.input.data_ptr(), input_ptr)
        self.assertEqual(request.output.data_ptr(), output_ptr)
        request.all_reduce(
            registered_input, out=registered_output, registered_input=True
        )
        torch.cuda.synchronize()
        self._assert_sum(registered_output)
        request.close()

    def test_unsynchronized_back_to_back_calls(self) -> None:
        input_tensor, output_tensor, request = self._new_request()

        input_tensor.fill_(float(self.rank + 1))
        first = request.all_reduce(
            input_tensor, out=output_tensor, registered_input=True
        )
        input_tensor.fill_(float(2 * (self.rank + 1)))
        second = request.all_reduce(
            input_tensor, out=output_tensor, registered_input=True
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        torch.cuda.synchronize()
        expected = torch.full_like(output_tensor, float(self.size * (self.size + 1)))
        torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)
        request.close()

    def test_graph_capture_replay(self) -> None:
        inner = 8192
        activation = torch.full(
            (64, inner),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        weight = torch.full(
            (inner, 4096),
            1.0 / inner,
            dtype=torch.bfloat16,
            device=self.device,
        )
        input_tensor = torch.empty((64, 4096), dtype=torch.bfloat16, device=self.device)
        output_tensor = torch.empty_like(input_tensor)
        request = self.comm.registered_all_reduce(
            input_tensor, output_tensor, self.REGISTERED_BYTES
        )
        self.requests.append(request)

        # Warm up both model operations on their execution stream before graph
        # capture, matching the 42B decode graph setup.
        execution_stream = torch.cuda.Stream(device=self.device)
        execution_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(execution_stream):
            torch.mm(activation, weight, out=input_tensor)
            request.all_reduce(input_tensor, out=output_tensor, registered_input=True)
        execution_stream.synchronize()
        self._assert_sum(output_tensor)

        graph = torch.cuda.CUDAGraph()
        self.graphs.append(graph)
        with torch.cuda.graph(graph, stream=execution_stream):
            torch.mm(activation, weight, out=input_tensor)
            request.all_reduce(input_tensor, out=output_tensor, registered_input=True)

        with torch.cuda.stream(execution_stream):
            graph.replay()
        execution_stream.synchronize()
        expected_input = torch.full_like(input_tensor, float(self.rank + 1))
        torch.testing.assert_close(input_tensor, expected_input, rtol=0, atol=0)
        self._assert_sum(output_tensor)

        activation.fill_(float(2 * (self.rank + 1)))
        execution_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(execution_stream):
            graph.replay()

        # Teardown the graph before close(). The final replay is deliberately
        # not synchronized here; request finalization must drain its pinned
        # stream before releasing registered mappings.
        graph.reset()
        request.close()
        expected = torch.full_like(output_tensor, float(self.size * (self.size + 1)))
        torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)

    def test_rejects_unregistered_input_output_and_registered_input_false(self) -> None:
        input_tensor, output_tensor, request = self._new_request()
        other_input = torch.empty_like(input_tensor)
        other_output = torch.empty_like(output_tensor)

        with self.assertRaises(RuntimeError):
            request.all_reduce(other_input, out=output_tensor, registered_input=True)
        with self.assertRaises(RuntimeError):
            request.all_reduce(input_tensor, out=other_output, registered_input=True)
        with self.assertRaises(RuntimeError):
            request.all_reduce(input_tensor, out=output_tensor, registered_input=False)
        request.close()

    def test_asymmetric_invalid_registration_fails_all_ranks(self) -> None:
        input_tensor = torch.full(
            (self.ELEM_COUNT,),
            float(self.rank + 1),
            dtype=torch.bfloat16,
            device=self.device,
        )
        output_tensor = (
            input_tensor if self.rank == 0 else torch.empty_like(input_tensor)
        )

        with self.assertRaises(RuntimeError):
            self.comm.registered_all_reduce(
                input_tensor, output_tensor, self.REGISTERED_BYTES
            )

    def test_rejects_unsupported_dtype_op_and_capacity(self) -> None:
        float_input = torch.empty(
            (self.REGISTERED_BYTES // 4,), dtype=torch.float32, device=self.device
        )
        float_output = torch.empty_like(float_input)
        float_request = self.comm.registered_all_reduce(
            float_input, float_output, self.REGISTERED_BYTES
        )
        self.requests.append(float_request)
        with self.assertRaises(RuntimeError):
            float_request.all_reduce(
                float_input, out=float_output, registered_input=True
            )
        float_request.close()

        input_tensor, output_tensor, request = self._new_request()
        with self.assertRaises(RuntimeError):
            request.all_reduce(
                input_tensor,
                torchcomms.ReduceOp.PRODUCT,
                out=output_tensor,
                registered_input=True,
            )
        request.close()

        too_small = torch.empty((1024,), dtype=torch.bfloat16, device=self.device)
        too_small_out = torch.empty_like(too_small)
        with self.assertRaises(RuntimeError):
            self.comm.registered_all_reduce(too_small, too_small_out, too_small.nbytes)


if __name__ == "__main__":
    unittest.main()
