#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
import torchcomms


class TestFactory(unittest.TestCase):
    def test_factory(self):
        print(torchcomms)
        print(dir(torchcomms))

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["TORCHCOMM_GLOO_HOSTNAME"] = "localhost"

        comm = torchcomms.new_comm("gloo", torch.device("cpu"), "my_comm")
        comm.finalize()
        backend = comm.get_backend_impl()
        print(backend)

        from torchcomms._comms_gloo import TorchCommGloo

        # if backend was lazily loaded backend will not have the right type
        self.assertIsInstance(backend, TorchCommGloo)

    def test_factory_missing(self):
        with self.assertRaisesRegex(ModuleNotFoundError, "failed to find backend"):
            torchcomms.new_comm("invalid", torch.device("cuda"), "my_comm")
