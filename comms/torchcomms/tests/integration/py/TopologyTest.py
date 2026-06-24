#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)

# This test is hardcoded for a single-node.
# With every rank on one node, the world communicator and every
# subcommunicator created via split() are single-node; only local_node_ranks
# varies with the group size.
EXPECTED_NUM_RANKS = 4


class TopologyTest(unittest.TestCase):
    """Tests for CommTopology, including subcommunicators created via split().

    Hardcoded for a 1-node x 4-rank layout so the expected topology is known
    exactly without an independent oracle.
    """

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

        # This test hardcodes expectations for a single-node, 4-rank run.
        if self.num_ranks != EXPECTED_NUM_RANKS:
            # tearDown won't run on a setUp skip, so clean up explicitly to
            # finalize the comm created above.
            self.tearDown()
            self.skipTest(
                f"TopologyTest requires exactly {EXPECTED_NUM_RANKS} ranks on a "
                f"single node, got {self.num_ranks}"
            )

        # Skip on backends that do not implement get_topology(). The base
        # backend raises, surfaced as a RuntimeError on the Python side.
        try:
            self.torchcomm.get_topology()
        except RuntimeError as e:
            # tearDown won't run on a setUp skip, so clean up explicitly to
            # finalize the comm created above.
            self.tearDown()
            self.skipTest(f"backend does not implement get_topology(): {e}")

    def tearDown(self):
        """Clean up after each test."""
        self.torchcomm = None
        self.wrapper = None

    def _assert_topology(self, comm, num_nodes, local_node_ranks, description=""):
        """Assert comm.get_topology() matches the expected single-node layout.

        For a single-node launch every communicator is single-node and uniform,
        so only num_nodes and local_node_ranks need to be supplied."""
        topo = comm.get_topology()
        self.assertEqual(
            topo.num_nodes, num_nodes, f"num_nodes mismatch for {description}"
        )
        self.assertEqual(
            topo.local_node_ranks,
            local_node_ranks,
            f"local_node_ranks mismatch for {description}",
        )
        # All ranks share one node, so the topology is always uniform and single.
        self.assertTrue(topo.uniform, f"expected uniform topology for {description}")
        self.assertTrue(
            topo.is_single_node(), f"expected single-node for {description}"
        )
        return topo

    def test_world_topology(self):
        """World communicator: 1 node, all 4 ranks."""
        self._assert_topology(
            self.torchcomm,
            num_nodes=1,
            local_node_ranks=4,
            description="world communicator",
        )

    def test_contiguous_split_topology(self):
        """Contiguous split into the first half {0, 1}; ranks 2, 3 get None."""
        split_size = self.num_ranks // 2  # 2
        in_group = self.rank < split_size
        ranks = list(range(split_size)) if in_group else []

        child = self.torchcomm.split(ranks, name="contiguous_topology_comm")

        if in_group:
            self.assertIsNotNone(child, f"rank {self.rank} expected a child comm")
            # 2 ranks, same node.
            self._assert_topology(
                child,
                num_nodes=1,
                local_node_ranks=2,
                description="contiguous split child",
            )
            child.finalize()
        else:
            self.assertIsNone(child, f"rank {self.rank} should not have a child comm")

    def test_non_contiguous_split_topology(self):
        """Non-contiguous split of even ranks {0, 2}; odd ranks get None."""
        in_group = self.rank % 2 == 0
        ranks = list(range(0, self.num_ranks, 2)) if in_group else []

        child = self.torchcomm.split(ranks, name="noncontig_topology_comm")

        if in_group:
            self.assertIsNotNone(child, f"rank {self.rank} expected a child comm")
            # 2 even ranks, same node.
            self._assert_topology(
                child,
                num_nodes=1,
                local_node_ranks=2,
                description="non-contiguous split child",
            )
            child.finalize()
        else:
            self.assertIsNone(child, f"rank {self.rank} should not have a child comm")

    def test_single_rank_split_topology(self):
        """A communicator split down to a single rank: 1 node, 1 rank."""
        child = self.torchcomm.split([self.rank], name="single_rank_topology_comm")

        self.assertIsNotNone(child, f"rank {self.rank} expected a single-rank comm")
        self._assert_topology(
            child,
            num_nodes=1,
            local_node_ranks=1,
            description="single-rank split child",
        )
        child.finalize()

    def test_multi_level_split_topology(self):
        """Two levels of splitting: {0,1,2,3} -> {0,1} -> {0}."""
        first_size = self.num_ranks // 2  # 2
        in_first = self.rank < first_size
        first_ranks = list(range(first_size)) if in_first else []

        first_comm = self.torchcomm.split(first_ranks, name="first_level_topology_comm")

        if not in_first:
            self.assertIsNone(first_comm, f"rank {self.rank} should not have a child")
            return

        self.assertIsNotNone(
            first_comm, f"rank {self.rank} expected a first-level comm"
        )
        # First level: ranks {0, 1}, same node.
        self._assert_topology(
            first_comm,
            num_nodes=1,
            local_node_ranks=2,
            description="first-level split child",
        )

        first_rank = first_comm.get_rank()
        second_size = first_comm.get_size() // 2  # 1
        in_second = first_rank < second_size
        second_ranks = list(range(second_size)) if in_second else []

        second_comm = first_comm.split(second_ranks, name="second_level_topology_comm")

        if in_second:
            self.assertIsNotNone(
                second_comm, f"first-level rank {first_rank} expected a child"
            )
            # Second level: single rank {0}, 1 node, 1 rank.
            self._assert_topology(
                second_comm,
                num_nodes=1,
                local_node_ranks=1,
                description="second-level split child",
            )
            second_comm.finalize()
        else:
            self.assertIsNone(
                second_comm,
                f"first-level rank {first_rank} should not have a child",
            )

        first_comm.finalize()

    def test_compute_topology_disabled(self):
        """With the compute_topology hint off, get_topology() must raise.

        Disabling the hint skips the init-time host-hash all-gather, so the
        communicator behaves like a backend that never implemented
        get_topology() — the call raises rather than returning stale/default
        data. std::logic_error surfaces as RuntimeError on the Python side.

        The wrapper finalizes its own comm in __del__, so we drop the wrapper
        rather than calling finalize() here (double finalize() raises)."""
        wrapper = TorchCommTestWrapper(hints={"compute_topology": "false"})
        try:
            with self.assertRaises(RuntimeError):
                wrapper.get_torchcomm().get_topology()
        finally:
            wrapper = None

    def test_compute_topology_enabled_explicitly(self):
        """Setting the hint to true matches the default-on behavior."""
        wrapper = TorchCommTestWrapper(hints={"compute_topology": "true"})
        try:
            self._assert_topology(
                wrapper.get_torchcomm(),
                num_nodes=1,
                local_node_ranks=EXPECTED_NUM_RANKS,
                description="world communicator with compute_topology=true",
            )
        finally:
            wrapper = None


if __name__ == "__main__":
    unittest.main(failfast=True)
