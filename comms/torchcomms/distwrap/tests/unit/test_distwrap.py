# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import unittest

import torch.distributed
import torchcomms.distwrap


class ForwardedAttributesTest(unittest.TestCase):
    """Tests for attributes forwarded from torch.distributed."""

    def test_get_rank_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.get_rank, torch.distributed.get_rank)

    def test_get_world_size_is_forwarded(self) -> None:
        self.assertIs(
            torchcomms.distwrap.get_world_size, torch.distributed.get_world_size
        )

    def test_is_initialized_is_forwarded(self) -> None:
        self.assertIs(
            torchcomms.distwrap.is_initialized, torch.distributed.is_initialized
        )

    def test_is_available_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.is_available, torch.distributed.is_available)

    def test_get_process_group_ranks_is_forwarded(self) -> None:
        self.assertIs(
            torchcomms.distwrap.get_process_group_ranks,
            torch.distributed.get_process_group_ranks,
        )

    def test_ProcessGroup_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.ProcessGroup, torch.distributed.ProcessGroup)

    def test_GroupMember_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.GroupMember, torch.distributed.GroupMember)

    def test_ReduceOp_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.ReduceOp, torch.distributed.ReduceOp)

    def test_P2POp_is_exported(self) -> None:
        # P2POp is a custom wrapper class, not a direct forwarding from torch.distributed
        self.assertTrue(callable(torchcomms.distwrap.P2POp))

    def test_group_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.group, torch.distributed.group)

    def test_Store_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.Store, torch.distributed.Store)

    def test_HashStore_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.HashStore, torch.distributed.HashStore)

    def test_Work_is_forwarded(self) -> None:
        self.assertIs(torchcomms.distwrap.Work, torch.distributed.Work)

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with self.assertRaises(AttributeError) as context:
            # pyre-ignore[16]: Intentionally accessing nonexistent attribute
            _ = torchcomms.distwrap.nonexistent_attribute

        self.assertIn("has no attribute", str(context.exception))
        self.assertIn("nonexistent_attribute", str(context.exception))


class ExportedFunctionsTest(unittest.TestCase):
    """Tests that all expected functions are exported and callable."""

    def test_destroy_process_group_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.destroy_process_group))

    def test_init_process_group_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.init_process_group))

    def test_new_group_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.new_group))

    def test_split_group_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.split_group))

    def test_all_reduce_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.all_reduce))

    def test_broadcast_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.broadcast))

    def test_reduce_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.reduce))

    def test_all_gather_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.all_gather))

    def test_all_gather_into_tensor_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.all_gather_into_tensor))

    def test_reduce_scatter_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.reduce_scatter))

    def test_reduce_scatter_tensor_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.reduce_scatter_tensor))

    def test_all_to_all_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.all_to_all))

    def test_all_to_all_single_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.all_to_all_single))

    def test_scatter_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.scatter))

    def test_gather_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.gather))

    def test_barrier_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.barrier))

    def test_send_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.send))

    def test_recv_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.recv))

    def test_isend_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.isend))

    def test_irecv_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.irecv))

    def test_batch_isend_irecv_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.batch_isend_irecv))

    def test_all_gather_object_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.all_gather_object))

    def test_gather_object_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.gather_object))

    def test_scatter_object_list_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.scatter_object_list))

    def test_broadcast_object_list_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.broadcast_object_list))

    def test_new_window_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.new_window))

    def test_alltoallv_dedup_init_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.alltoallv_dedup_init))

    def test_alltoallv_dedup_exec_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.alltoallv_dedup_exec))

    def test_alltoallv_dynamic_dispatch_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.alltoallv_dynamic_dispatch))

    def test_alltoallv_dynamic_combine_is_exported(self) -> None:
        self.assertTrue(callable(torchcomms.distwrap.alltoallv_dynamic_combine))


class AllExportsTest(unittest.TestCase):
    """Tests that __all__ matches actual exports."""

    def test_all_exports_are_accessible(self) -> None:
        for name in torchcomms.distwrap.__all__:
            self.assertTrue(
                hasattr(torchcomms.distwrap, name),
                f"'{name}' in __all__ but not accessible",
            )


if __name__ == "__main__":
    unittest.main()
