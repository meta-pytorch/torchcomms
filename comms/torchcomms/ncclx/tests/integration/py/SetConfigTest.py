#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import unittest

import torch
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class SetConfigTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.ncclx_backend = self.torchcomm.get_backend_impl()
        self.rank = self.torchcomm.get_rank()
        self.size = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        del self.torchcomm
        del self.wrapper

    def _do_sendrecv(self):
        send_rank = (self.rank + 1) % self.size
        recv_rank = (self.rank - 1 + self.size) % self.size
        send_tensor = torch.ones(1024, device=self.device) * self.rank
        recv_tensor = torch.zeros(1024, device=self.device)
        batch = self.torchcomm.batch_op_create()
        batch.send(send_tensor, send_rank)
        batch.recv(recv_tensor, recv_rank)
        work = batch.issue(True)
        work.wait()
        return recv_tensor

    def test_set_config_basic(self):
        self.ncclx_backend.set_config({"sendrecvAlgo": "orig"})

    def test_set_multiple_algos(self):
        self.ncclx_backend.set_config(
            {
                "allgatherAlgo": "ctdirect",
                "allreduceAlgo": "ctdirect",
                "sendrecvAlgo": "ctp2p",
            }
        )

    def test_reject_immutable_hint(self):
        with self.assertRaises(RuntimeError):
            self.ncclx_backend.set_config({"useCtran": "1"})

    def test_reject_invalid_key(self):
        with self.assertRaises(RuntimeError):
            self.ncclx_backend.set_config({"nonExistentKey": "value"})

    def test_sendrecv_after_config_override(self):
        self.ncclx_backend.set_config({"sendrecvAlgo": "orig"})
        recv = self._do_sendrecv()
        expected_rank = (self.rank - 1 + self.size) % self.size
        self.assertTrue(
            torch.allclose(recv, torch.ones(1024, device=self.device) * expected_rank)
        )

    def test_sendrecv_with_sequential_overrides(self):
        self.ncclx_backend.set_config({"sendrecvAlgo": "orig"})
        recv1 = self._do_sendrecv()

        self.ncclx_backend.set_config({"sendrecvAlgo": "ctp2p"})
        recv2 = self._do_sendrecv()

        expected_rank = (self.rank - 1 + self.size) % self.size
        expected = torch.ones(1024, device=self.device) * expected_rank
        self.assertTrue(torch.allclose(recv1, expected))
        self.assertTrue(torch.allclose(recv2, expected))
