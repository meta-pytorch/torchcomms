#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for object collectives."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class ObjectCollectivesTest(unittest.TestCase):
    """Test class for object collective operations using distwrap."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize distwrap once for all tests."""
        rank, _ = get_rank_and_size()
        device = get_device(rank)
        backend = get_backend()

        dist.init_process_group(
            backend=backend,
            use_torchcomms=use_torchcomms(),
        )

        if device.type == "cuda":
            torch.cuda.set_device(device)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up distwrap after all tests."""
        dist.destroy_process_group()

    def test_all_gather_object(self) -> None:
        """Test all_gather_object gathers objects from all ranks."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()

        # Each rank contributes its rank as an object
        obj = {"rank": rank, "data": f"from_rank_{rank}"}
        object_list = [None] * num_ranks

        dist.all_gather_object(object_list, obj)

        # Verify all objects were gathered
        for i in range(num_ranks):
            obj_i = object_list[i]
            if obj_i is None:
                raise AssertionError(f"object_list[{i}] is None")
            self.assertEqual(obj_i["rank"], i)
            self.assertEqual(obj_i["data"], f"from_rank_{i}")

    def test_all_gather_object_with_list(self) -> None:
        """Test all_gather_object with list objects."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()

        # Each rank contributes a list
        obj = [rank, rank * 2, rank * 3]
        object_list = [None] * num_ranks

        dist.all_gather_object(object_list, obj)

        # Verify all lists were gathered
        for i in range(num_ranks):
            self.assertEqual(object_list[i], [i, i * 2, i * 3])

    def test_gather_object(self) -> None:
        """Test gather_object gathers objects to root rank."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        root = 0

        # Each rank contributes its data
        obj = f"data_from_rank_{rank}"

        if rank == root:
            object_gather_list = [None] * num_ranks
        else:
            object_gather_list = None

        dist.gather_object(obj, object_gather_list=object_gather_list, dst=root)

        # Only root rank should have the gathered objects
        if rank == root:
            if object_gather_list is None:
                raise AssertionError("object_gather_list is None")
            for i in range(num_ranks):
                self.assertEqual(object_gather_list[i], f"data_from_rank_{i}")

    def test_gather_object_non_zero_root(self) -> None:
        """Test gather_object with non-zero root rank."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for this test")

        root = num_ranks - 1  # Last rank is root

        obj = {"value": rank * 10}

        if rank == root:
            object_gather_list = [None] * num_ranks
        else:
            object_gather_list = None

        dist.gather_object(obj, object_gather_list=object_gather_list, dst=root)

        if rank == root:
            if object_gather_list is None:
                raise AssertionError("object_gather_list is None")
            for i in range(num_ranks):
                self.assertEqual(object_gather_list[i], {"value": i * 10})

    def test_scatter_object_list(self) -> None:
        """Test scatter_object_list scatters objects from root."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        root = 0

        # Root prepares objects to scatter
        if rank == root:
            scatter_input = [f"object_for_rank_{i}" for i in range(num_ranks)]
        else:
            scatter_input = None

        scatter_output = [None]

        dist.scatter_object_list(
            scatter_output, scatter_object_input_list=scatter_input, src=root
        )

        # Each rank should receive its designated object
        self.assertEqual(scatter_output[0], f"object_for_rank_{rank}")

    def test_scatter_object_list_with_dicts(self) -> None:
        """Test scatter_object_list with dictionary objects."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        root = 0

        if rank == root:
            scatter_input = [{"id": i, "value": i * 100} for i in range(num_ranks)]
        else:
            scatter_input = None

        scatter_output = [None]

        dist.scatter_object_list(
            scatter_output, scatter_object_input_list=scatter_input, src=root
        )

        self.assertEqual(scatter_output[0], {"id": rank, "value": rank * 100})

    def test_broadcast_object_list(self) -> None:
        """Test broadcast_object_list broadcasts objects from root."""
        rank = dist.get_rank()
        root = 0

        if rank == root:
            object_list = ["broadcast_string", 123, {"key": "value"}, [1, 2, 3]]
        else:
            object_list = [None, None, None, None]

        dist.broadcast_object_list(object_list, src=root)

        # All ranks should have the same objects
        self.assertEqual(object_list[0], "broadcast_string")
        self.assertEqual(object_list[1], 123)
        self.assertEqual(object_list[2], {"key": "value"})
        self.assertEqual(object_list[3], [1, 2, 3])

    def test_broadcast_object_list_non_zero_root(self) -> None:
        """Test broadcast_object_list with non-zero root."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for this test")

        root = num_ranks - 1

        if rank == root:
            object_list = [{"source": "last_rank"}, 999]
        else:
            object_list = [None, None]

        dist.broadcast_object_list(object_list, src=root)

        self.assertEqual(object_list[0], {"source": "last_rank"})
        self.assertEqual(object_list[1], 999)

    def test_broadcast_object_list_complex_objects(self) -> None:
        """Test broadcast_object_list with complex nested objects."""
        rank = dist.get_rank()
        root = 0

        complex_obj = {
            "nested": {"level1": {"level2": [1, 2, 3]}},
            "tuple_as_list": [1, "two", 3.0],
            "none_value": None,
        }

        if rank == root:
            object_list = [complex_obj]
        else:
            object_list = [None]

        dist.broadcast_object_list(object_list, src=root)

        obj_0 = object_list[0]
        if obj_0 is None:
            raise AssertionError("obj_0 is None")
        if not isinstance(obj_0, dict):
            raise AssertionError(f"obj_0 is not a dict: {type(obj_0)}")
        nested = obj_0["nested"]
        if not isinstance(nested, dict):
            raise AssertionError(f"nested is not a dict: {type(nested)}")
        level1 = nested["level1"]
        if not isinstance(level1, dict):
            raise AssertionError(f"level1 is not a dict: {type(level1)}")
        self.assertEqual(level1["level2"], [1, 2, 3])
        self.assertEqual(obj_0["tuple_as_list"], [1, "two", 3.0])
        self.assertIsNone(obj_0["none_value"])


if __name__ == "__main__":
    unittest.main()
