# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import os
import unittest

import torch
import torchcomms


class TestFactory(unittest.TestCase):
    def test_factory(self) -> None:
        print(torchcomms)
        print(dir(torchcomms))

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        comm = torchcomms.new_comm("rcclx", torch.device("hip"), name="my_comm")
        comm.finalize()
        backend = comm.get_backend_impl()
        print(backend)

        from torchcomms._comms_rcclx import TorchCommRCCLX

        # if backend was lazily loaded backend will not have the right type
        self.assertIsInstance(backend, TorchCommRCCLX)

    def test_factory_missing(self) -> None:
        with self.assertRaisesRegex(ModuleNotFoundError, "failed to find backend"):
            torchcomms.new_comm("invalid", torch.device("hip"), name="my_comm")
