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
        # capture, matching the [redacted] decode graph setup.
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

        # Any positive multiple of 16 bytes registers; 2002 bytes does not.
        unaligned = torch.empty((1001,), dtype=torch.bfloat16, device=self.device)
        unaligned_out = torch.empty_like(unaligned)
        with self.assertRaises(RuntimeError):
            self.comm.registered_all_reduce(unaligned, unaligned_out, unaligned.nbytes)

    def test_gated_residual_norm_epilogue_interleaves_with_plain(self) -> None:
        self._check_gated_residual_norm(rows=64, hidden=8192, post_weight=True)

    def test_gated_residual_norm_epilogue_without_post_weight(self) -> None:
        # [redacted] decode (8192-wide) and SBD BS8 draft (4096-wide) rows, without
        # a post-norm weight as [redacted] builds the epilogue.
        if self.size != 4:
            self.skipTest("8192- and 4096-wide epilogue rows are four-rank only")
        for rows, hidden in ((10, 8192), (64, 8192), (112, 4096), (160, 4096)):
            with self.subTest(rows=rows, hidden=hidden):
                self._check_gated_residual_norm(rows, hidden, post_weight=False)

    def _check_gated_residual_norm(
        self, rows: int, hidden: int, post_weight: bool
    ) -> None:
        elements = rows * hidden
        input_tensor = torch.full(
            (elements,), float(self.rank + 1), dtype=torch.bfloat16, device=self.device
        )
        output_tensor = torch.empty_like(input_tensor)
        request = self.comm.registered_all_reduce(
            input_tensor, output_tensor, input_tensor.nbytes
        )
        self.requests.append(request)
        generator = torch.Generator(device=self.device).manual_seed(11)

        def rand(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, device=self.device)

        post_norm_weight = (
            (1.0 + 0.2 * rand(hidden)).to(torch.bfloat16) if post_weight else None
        )
        pre_weight = (1.0 + 0.2 * rand(hidden)).to(torch.bfloat16)
        gate = rand(hidden) / 0.3
        gate_beta = torch.sigmoid(gate)
        gate_alpha = torch.sqrt(
            torch.clamp(torch.sigmoid(-gate) * (1.0 + gate_beta), min=1e-3)
        )
        residual_in = rand(elements)
        residual_out = torch.empty_like(residual_in)
        router_out = torch.empty_like(residual_in)
        norm = torchcomms.RegisteredAllReduceGatedResidualNorm(
            residual_in,
            residual_out,
            post_norm_weight,
            pre_weight,
            gate_alpha,
            gate_beta,
            1e-8,
            1e-5,
            router_out=router_out,
        )

        request.all_reduce(input_tensor, out=output_tensor, registered_input=True)
        torch.cuda.synchronize()
        expected_sum = float(self.size * (self.size + 1) // 2)
        torch.testing.assert_close(
            output_tensor, torch.full_like(output_tensor, expected_sum), rtol=0, atol=0
        )

        request.all_reduce(
            input_tensor,
            out=output_tensor,
            registered_input=True,
            gated_residual_norm=norm,
        )
        torch.cuda.synchronize()

        def rms_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
            mean = x.double().pow(2).mean(dim=1, keepdim=True)
            return torch.rsqrt(mean + eps).float()

        branch = torch.full((rows, hidden), expected_sum, device=self.device)
        normed = branch * rms_scale(branch, 1e-8)
        if post_norm_weight is not None:
            normed = normed * post_norm_weight.float()
        normed = normed.to(torch.bfloat16)
        stream = (
            gate_alpha * residual_in.view(rows, hidden) + gate_beta * normed.float()
        )
        pre = stream.to(torch.bfloat16).float()
        expected = (pre * rms_scale(pre, 1e-5) * pre_weight.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            residual_out.view(rows, hidden), stream, rtol=1e-6, atol=1e-6
        )

        def ordered(values: torch.Tensor) -> torch.Tensor:
            bits = values.contiguous().view(torch.int16).int()
            return torch.where(bits < 0, -(bits & 0x7FFF), bits)

        ulp = (ordered(output_tensor.view(rows, hidden)) - ordered(expected)).abs()
        self.assertLessEqual(int(ulp.max()), 1)
        self.assertTrue(
            torch.equal(
                router_out.view(rows, hidden), output_tensor.view(rows, hidden).float()
            )
        )

        request.all_reduce(input_tensor, out=output_tensor, registered_input=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output_tensor, torch.full_like(output_tensor, expected_sum), rtol=0, atol=0
        )
        request.close()


class RegisteredAllReduceTP2RelayTest(unittest.TestCase):
    """Two-rank requests large enough for the relay route, which sends part of
    each payload through the node's other GPUs (RCCLX picks it internally)."""

    def setUp(self) -> None:
        self.requests = []
        os.environ.setdefault("TEST_BACKEND", "rcclx")
        self.wrapper = TorchCommTestWrapper()
        self.comm = self.wrapper.get_torchcomm()
        self.rank = self.comm.get_rank()
        if self.comm.get_size() != 2:
            self.skipTest("the relay route needs a 2-rank communicator")
        device_count = torch.cuda.device_count()
        if device_count < 3:
            self.skipTest("the relay route needs GPUs outside the pair")
        self.device = torch.device(f"cuda:{self.rank % device_count}")
        torch.cuda.set_device(self.device)

    def tearDown(self) -> None:
        for request in reversed(self.requests):
            if not request.closed:
                request.close()
        del self.comm
        del self.wrapper

    def _request(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torchcomms.RegisteredAllReduce]:
        input_tensor = torch.empty(count, dtype=torch.bfloat16, device=self.device)
        output_tensor = torch.empty_like(input_tensor)
        request = self.comm.registered_all_reduce(
            input_tensor, output_tensor, input_tensor.nbytes
        )
        self.requests.append(request)
        return input_tensor, output_tensor, request

    def _fill(self, count: int, rank: int, salt: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(1000 * salt + rank)
        return torch.randn(count, generator=gen, device=self.device).to(torch.bfloat16)

    def _want(self, count: int, salt: int) -> torch.Tensor:
        total = self._fill(count, 0, salt).float() + self._fill(count, 1, salt).float()
        return total.to(torch.bfloat16)

    def test_prefill_sizes_match_rank_order_sum(self) -> None:
        for mib, salt in ((72, 1), (144, 2)):
            count = mib * 1024 * 1024 // 2
            input_tensor, output_tensor, request = self._request(count)
            input_tensor.copy_(self._fill(count, self.rank, salt))
            request.all_reduce(input_tensor, out=output_tensor, registered_input=True)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(output_tensor, self._want(count, salt)), f"{mib} MiB"
            )
            request.close()

    def test_decode_and_prefill_requests_interleave(self) -> None:
        decode = self._request(294912 // 2)
        prefill = self._request(36 * 1024 * 1024 // 2)
        for salt, (input_tensor, output_tensor, request) in enumerate(
            (decode, prefill, decode, prefill), start=3
        ):
            count = input_tensor.numel()
            input_tensor.copy_(self._fill(count, self.rank, salt))
            request.all_reduce(input_tensor, out=output_tensor, registered_input=True)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output_tensor, self._want(count, salt)))

    def test_relay_sized_graph_replay(self) -> None:
        count = 72 * 1024 * 1024 // 2
        input_tensor, output_tensor, request = self._request(count)
        staged = torch.empty_like(input_tensor)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            input_tensor.copy_(staged)
            request.all_reduce(input_tensor, out=output_tensor, registered_input=True)
        for salt in (7, 8):
            staged.copy_(self._fill(count, self.rank, salt))
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            self.assertTrue(torch.equal(output_tensor, self._want(count, salt)))
        graph.reset()
        request.close()


if __name__ == "__main__":
    unittest.main()
