# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
import importlib
import os
import sys
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torchcomms


# Unit test for TorchCommMCCL backend initialization and finalzation.
# finalization test should also be added.
class TestMcclBackend(unittest.TestCase):
    def setUp(self) -> None:
        """Set up common environment variables for MCCL initialization."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

    def test_factory(self):
        print(torchcomms)
        print(dir(torchcomms))

        device_id = 0
        comm = torchcomms.new_comm("mccl", torch.device(f"cuda:{device_id}"), "my_comm")
        backend = comm.get_backend_impl()
        print(backend)

        from torchcomms._comms_mccl import TorchCommMCCL

        # if backend was lazily loaded backend will not have the right type
        self.assertIsInstance(backend, TorchCommMCCL)
        comm.finalize()
        backend = None
        comm = None

    def test_c10d_registration_is_lazy(self) -> None:
        self.assertNotIn("torchcomms._comms_mccl", sys.modules)
        mccl_plugin = importlib.import_module("torchcomms.mccl")
        self.assertNotIn("torchcomms._comms_mccl", sys.modules)

        process_group = MagicMock()
        process_group.bound_device_id = torch.device("cuda:3")
        opts = MagicMock()
        opts.process_group = process_group
        opts.group_rank = 1
        opts.group_size = 4
        opts.group_id = "test_mccl_registration"
        opts.store = dist.HashStore()
        opts.timeout = timedelta(seconds=45)
        opts.enable_reconfigure = True
        backend_options = object()
        wrapped_backend = object()

        with (
            patch.object(dist.Backend, "register_backend") as register_backend,
            patch(
                "torch.distributed.distributed_c10d._create_torchcomms_backend",
                return_value=wrapped_backend,
            ) as create_backend,
        ):
            mccl_plugin._register_c10d_backend()
            creator = register_backend.call_args.args[1]
            self.assertIs(creator(opts, backend_options), wrapped_backend)
            create_backend.assert_called_once_with(
                "mccl",
                "cuda",
                group_rank=1,
                group_size=4,
                group_name="test_mccl_registration",
                store=opts.store,
                device_id=torch.device("cuda:3"),
                backend_options=backend_options,
                timeout=opts.timeout,
                enable_reconfigure=opts.enable_reconfigure,
            )

            create_backend.reset_mock()
            opts.process_group = None
            self.assertIs(creator(opts, backend_options), wrapped_backend)

        self.assertIn("torchcomms._comms_mccl", sys.modules)
        register_backend.assert_called_once_with(
            "mccl",
            creator,
            extended_api=True,
            devices=["cuda"],
        )
        create_backend.assert_called_once_with(
            "mccl",
            "cuda",
            group_rank=1,
            group_size=4,
            group_name="test_mccl_registration",
            store=opts.store,
            device_id=None,
            backend_options=backend_options,
            timeout=opts.timeout,
            enable_reconfigure=opts.enable_reconfigure,
        )

    def test_factory_missing(self):
        with self.assertRaisesRegex(ModuleNotFoundError, "failed to find backend"):
            torchcomms.new_comm("invalid", torch.device("cuda"), "my_comm")

    def test_init_with_tcp_store(self):
        """Test that initialization succeeds when passing a TCPStore explicitly."""
        device_id = 0
        store = dist.TCPStore(
            host_name="localhost",
            port=0,
            world_size=1,
            is_master=True,
            wait_for_workers=False,
        )
        comm = torchcomms.new_comm(
            "mccl",
            torch.device(f"cuda:{device_id}"),
            "my_comm_with_store",
            store=store,
        )
        backend = comm.get_backend_impl()

        from torchcomms._comms_mccl import TorchCommMCCL

        self.assertIsInstance(backend, TorchCommMCCL)
        comm.finalize()
        backend = None
        comm = None

    def test_init_with_non_cuda_device(self):
        """Test that initialization fails gracefully with non-CUDA device."""
        with self.assertRaisesRegex(
            RuntimeError, "TorchCommMCCL only supports CUDA devices"
        ):
            torchcomms.new_comm("mccl", torch.device("cpu"), "my_comm")
