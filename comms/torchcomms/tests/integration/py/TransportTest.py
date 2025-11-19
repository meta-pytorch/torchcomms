#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import pickle
import unittest

import torch
from torchcomms._comms_ncclx import RdmaMemory, RdmaTransport


class TransportTest(unittest.TestCase):
    def setUp(self):
        if not RdmaTransport.supported():
            self.skipTest("RdmaTransport is not supported on this system")

    def test_construct(self) -> None:
        _ = RdmaTransport(torch.device("cuda:0"))

    def test_rdma_memory_from_tensor(self) -> None:
        tensor = torch.arange(1024, dtype=torch.uint8, device="cuda:0")
        compare_tensor = torch.zeros_like(tensor, device="cuda:1")

        tensor_mem = RdmaMemory(tensor)
        compare_mem = RdmaMemory(compare_tensor)

        tensor_view = tensor_mem.to_view()
        compare_view = compare_mem.to_view()

        self.assertEqual(tensor_view.size(), tensor.nbytes)
        self.assertAlmostEqual(tensor_view.size(), compare_view.size())

    def bind_and_connect(self, server: RdmaTransport, client: RdmaTransport) -> None:
        server_url = server.bind()
        client_url = client.bind()

        self.assertIsNotNone(server_url)
        self.assertIsNotNone(client_url)
        self.assertNotEqual(server_url, "")
        self.assertNotEqual(client_url, "")

        server_result = server.connect(client_url)
        client_result = client.connect(server_url)

        self.assertEqual(
            server_result, 0, "Server connect should return commSuccess (0)"
        )
        self.assertEqual(
            client_result, 0, "Client connect should return commSuccess (0)"
        )

        self.assertTrue(server.connected())
        self.assertTrue(client.connected())

    def test_bind_and_connect(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest(
                f"Test requires at least 2 CUDA devices, found {torch.cuda.device_count()}"
            )

        server_device = torch.device("cuda:0")
        client_device = torch.device("cuda:1")

        server_transport = RdmaTransport(server_device)
        client_transport = RdmaTransport(client_device)

        self.bind_and_connect(server_transport, client_transport)

    def run_send_recv(
        self,
        device1: str,
        device2: str,
    ) -> None:
        transport_device_1 = "cuda:0" if device1 == "cpu" else device1
        transport_device_2 = "cuda:0" if device2 == "cpu" else device2
        transport1 = RdmaTransport(torch.device(transport_device_1))
        transport2 = RdmaTransport(torch.device(transport_device_2))

        self.bind_and_connect(transport1, transport2)

        tensor1 = torch.arange(1024, dtype=torch.uint8, device=device1)
        tensor2 = torch.zeros_like(tensor1, device=device2)

        self.assertEqual(tensor1.nbytes, tensor2.nbytes)

        tensor1_mem = RdmaMemory(tensor1)
        tensor2_mem = RdmaMemory(tensor2)

        res = transport1.write(tensor1_mem.to_view(), tensor2_mem.to_remote_buffer())

        self.assertEqual(res, 0)
        self.assertTrue(torch.allclose(tensor1.cpu(), tensor2.cpu()))

        del transport1
        del transport2
        del tensor1_mem
        del tensor2_mem

    def check_multi_gpu(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest(
                f"Test requires at least 2 CUDA devices, found {torch.cuda.device_count()}"
            )

    def test_write_gpu_to_gpu(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cuda:0", "cuda:1")

    def test_write_gpu_to_gpu_2(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cuda:0", "cuda:0")

    def test_write_cpu_to_gpu(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cpu", "cuda:1")

    def test_write_cpu_to_gpu_2(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cpu", "cuda:0")

    def test_write_gpu_to_cpu(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cuda:1", "cpu")

    def test_write_gpu_to_cpu_2(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cuda:0", "cpu")

    def test_write_cpu_to_cpu(self) -> None:
        self.check_multi_gpu()
        self.run_send_recv("cpu", "cpu")

    def test_rdma_remote_buffer_pickle(self) -> None:
        self.check_multi_gpu()

        transport1 = RdmaTransport(torch.device("cuda:0"))
        transport2 = RdmaTransport(torch.device("cuda:1"))
        self.bind_and_connect(transport1, transport2)

        tensor1 = torch.arange(1024, dtype=torch.uint8, device="cuda:0")
        tensor2 = torch.zeros_like(tensor1, device="cuda:1")

        tensor1_mem = RdmaMemory(tensor1)
        tensor2_mem = RdmaMemory(tensor2)
        remote_buffer = tensor2_mem.to_remote_buffer()

        pickled = pickle.dumps(remote_buffer)
        unpickled_remote_buffer = pickle.loads(pickled)

        res = transport1.write(tensor1_mem.to_view(), unpickled_remote_buffer)

        self.assertEqual(res, 0, "Write transfer should succeed")
        self.assertTrue(
            torch.allclose(tensor1.cpu(), tensor2.cpu()),
            "Data should be correctly transferred after unpickling",
        )

        del transport1
        del transport2
        del tensor1_mem
        del tensor2_mem


if __name__ == "__main__" and os.environ["TEST_BACKEND"] == "ncclx":
    unittest.main()
