#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import os
import unittest

import torch
import torch.distributed as dist
import torchcomms
from torchcomms.device_mesh import _flatten_with_comm, init_device_mesh

try:
    from torch.distributed._mesh_layout import _MeshLayout

    HAS_MESH_LAYOUT = True
except ImportError:
    HAS_MESH_LAYOUT = False


class DeviceMeshTest(unittest.TestCase):
    """Test class for DeviceMesh compatibility in torchcomms."""

    def test_init(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

        comm = torchcomms.new_comm(backend, device, name="comms_test_init")

        try:
            device_mesh = init_device_mesh(
                mesh_dim_comms=(comm,),
                mesh_dim_names=("main",),
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                comm.finalize()
                return
            raise

        group = device_mesh.get_group("main")
        self.assertEqual(group.group_name, "main")

        t = torch.ones(10, device=device, dtype=torch.int32)
        dist.all_reduce(t, group=group)

        # Device-aware synchronization
        if device.type == "cuda":
            torch.cuda.synchronize()
        # No synchronization needed for CPU

        self.assertEqual(t[0].item(), comm.get_size())

        comm.finalize()

    @unittest.skipIf(
        torch.cuda.device_count() < 4, "Skipping non GPU situations for now"
    )
    def test_2_d_parallel(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        world_size = torch.cuda.device_count()
        dp_degree = 2
        tp_degree = world_size // dp_degree
        mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(
            dp_degree, tp_degree
        )

        comm = torchcomms.new_comm(
            backend,
            device,
            name="comms_test_2_d_parallel",
            timeout=datetime.timedelta(seconds=60),
        )

        # Get current rank to determine which groups this rank belongs to
        cur_rank = comm.get_rank()

        # For TP communication: find which row contains current rank
        tp_ranks = None
        for row in mesh.tolist():
            if cur_rank in row:
                tp_ranks = row
                break

        # For DP communication: find which column contains current rank
        dp_ranks = None
        mesh_transposed = mesh.transpose(0, 1)
        for col in mesh_transposed.tolist():
            if cur_rank in col:
                dp_ranks = col
                break

        # Create communicators using the new single-list API
        tp_comm = comm.split(tp_ranks, "tp")
        dp_comm = comm.split(dp_ranks, "dp")

        sub_comms = {"dp": dp_comm, "tp": tp_comm}

        try:
            device_mesh_2d = init_device_mesh(
                mesh_dim_comms=(dp_comm, tp_comm),
                mesh_dim_names=("dp", "tp"),
                _global_comm=comm,
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                for sub_comm in sub_comms.values():
                    sub_comm.finalize()
                comm.finalize()
                return
            raise

        cur_rank = comm.get_rank()
        for dim, sub_comm in sub_comms.items():
            sub_mesh = device_mesh_2d[dim]
            self.assertEqual(sub_mesh.get_rank(), cur_rank)
            self.assertEqual(sub_mesh.size(), sub_comm.get_size())
            sub_group = sub_mesh.get_group()
            self.assertEqual(sub_group.group_name, dim)

            t = torch.ones(10, device=device, dtype=torch.int32)
            dist.all_reduce(t, group=sub_group)

            # Device-aware synchronization
            if device.type == "cuda":
                torch.cuda.synchronize()
            # No synchronization needed for CPU

            self.assertEqual(t[0].item(), sub_comm.get_size())
            sub_comm.finalize()
        comm.finalize()

    @unittest.skipIf(
        torch.cuda.device_count() < 8 or not HAS_MESH_LAYOUT,
        "Skipping non GPU situations for now",
    )
    def test_n_d_parallel(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        world_size = torch.cuda.device_count()
        pp_degree = 2
        ep_degree = 2
        cp_degree = world_size // (pp_degree * ep_degree)
        mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(
            pp_degree, cp_degree, ep_degree
        )

        comm = torchcomms.new_comm(
            backend,
            device,
            name="comms_test_n_d_parallel",
            timeout=datetime.timedelta(seconds=60),
        )

        # Get current rank to determine which groups this rank belongs to
        cur_rank = comm.get_rank()

        mesh_dim_names = ["pp", "cp", "ep"]
        ranks_per_dim = {}
        comm_per_dim = {}
        for idx, dim_name in enumerate(mesh_dim_names):
            # Calculate global ranks mapping for this mesh dimension
            global_ranks = mesh.transpose(idx, -1).reshape(-1, mesh.size(idx)).tolist()
            for row in global_ranks:
                if cur_rank in row:
                    ranks_per_dim[dim_name] = row
                    break

        # Create communicators using the new single-list API
        for dim_name, ranks in ranks_per_dim.items():
            comm_per_dim[dim_name] = comm.split(ranks, dim_name)

        try:
            device_mesh_3d = init_device_mesh(
                mesh_dim_comms=(
                    comm_per_dim["pp"],
                    comm_per_dim["cp"],
                    comm_per_dim["ep"],
                ),
                mesh_dim_names=("pp", "cp", "ep"),
                _global_comm=comm,
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                for sub_comm in comm_per_dim.values():
                    sub_comm.finalize()
                comm.finalize()
                return
            raise

        flatten_mesh = [
            mesh.view(pp_degree * cp_degree, ep_degree),
            mesh.view(pp_degree, cp_degree * ep_degree),
        ]

        flattened_mesh_dim_names = ["pp_cp", "cp_ep"]
        flatten_mesh_dim_names = {"pp_cp": ["pp", "cp"], "cp_ep": ["cp", "ep"]}
        flatten_ranks_per_dim = {}
        # For DP communication: find which column contains current rank
        for idx, dim_name in enumerate(flattened_mesh_dim_names):
            # Calculate global ranks mapping for this mesh dimension
            global_ranks = flatten_mesh[idx].transpose(idx, -1).tolist()

            for row in global_ranks:
                if cur_rank in row:
                    flatten_ranks_per_dim[dim_name] = row
                    break

        for flatten_dim_name, ranks in flatten_ranks_per_dim.items():
            comm_per_dim[flatten_dim_name] = comm.split(ranks, flatten_dim_name)
            sizes = []
            strides = []
            # This is important because we need to make sure the layout is correct
            for dim_name in flatten_mesh_dim_names[flatten_dim_name]:
                layout = device_mesh_3d[dim_name]._layout
                sizes.append(layout.sizes)
                strides.append(layout.strides)
            flatten_layout = _MeshLayout(tuple(sizes), tuple(strides))
            _flatten_with_comm(
                device_mesh_3d,
                flatten_dim_name,
                comm_per_dim[flatten_dim_name],
                ranks,
                flatten_layout,
            )

        dims_to_test = ["cp", "pp_cp", "cp_ep"]
        cur_rank = comm.get_rank()
        for dim_mesh_name in dims_to_test:
            sub_comm = comm_per_dim[dim_mesh_name]
            sub_mesh = device_mesh_3d[dim_mesh_name]
            self.assertEqual(sub_mesh.get_rank(), cur_rank)
            self.assertEqual(sub_mesh.size(), sub_comm.get_size())
            sub_group = sub_mesh.get_group()
            self.assertEqual(sub_group.group_name, dim_mesh_name)

            t = torch.ones(10, device=device, dtype=torch.int32)
            dist.all_reduce(t, group=sub_group)

            # Device-aware synchronization
            if device.type == "cuda":
                torch.cuda.synchronize()
            # No synchronization needed for CPU

            self.assertEqual(t[0].item(), sub_comm.get_size())

        for sub_comm in comm_per_dim.values():
            sub_comm.finalize()
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
