#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
from torchcomms._comms_ncclx import RdmaTransport


class TransportTest(unittest.TestCase):
    def setUp(self):
        if not RdmaTransport.supported():
            self.skipTest("RdmaTransport is not supported on this system")

    def test_construct(self) -> None:
        _ = RdmaTransport(torch.device("cuda:0"))

    def test_bind_and_connect(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest(
                f"Test requires at least 2 CUDA devices, found {torch.cuda.device_count()}"
            )

        server_device = torch.device("cuda:0")
        client_device = torch.device("cuda:1")

        server_transport = RdmaTransport(server_device)
        client_transport = RdmaTransport(client_device)

        server_url = server_transport.bind()
        client_url = client_transport.bind()

        self.assertIsNotNone(server_url)
        self.assertIsNotNone(client_url)
        self.assertNotEqual(server_url, "")
        self.assertNotEqual(client_url, "")

        server_result = server_transport.connect(client_url)
        client_result = client_transport.connect(server_url)

        self.assertEqual(
            server_result, 0, "Server connect should return commSuccess (0)"
        )
        self.assertEqual(
            client_result, 0, "Client connect should return commSuccess (0)"
        )

        self.assertTrue(server_transport.connected())
        self.assertTrue(client_transport.connected())


if __name__ == "__main__" and os.environ["TEST_BACKEND"] == "ncclx":
    unittest.main()
