#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import tempfile
import unittest

import torch
import torchcomms
from torchcomms._comms import _get_store


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

    def test_duplciate_store(self):
        _, path = tempfile.mkstemp()
        os.environ["TORCHCOMM_STORE_PATH"] = path

        _get_store("custom_backend", "name")
        with self.assertRaisesRegex(
            RuntimeError, r"Store prefix has been reused.*custom_backend.*name"
        ):
            _get_store("custom_backend", "name")
