#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
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

        # The topology reports two launcher-assigned id lists for the on-node
        # ranks: global (world) rank ids from RANK, and node-local rank ids from
        # LOCAL_RANK. On this single-node run every member's world rank, global
        # rank, and node-local rank coincide, so the expected id list is the
        # same for both fields.
        global_rank = os.environ.get("RANK")
        self.global_rank_ids_predictable = (
            global_rank is not None and int(global_rank) == self.rank
        )
        local_rank = os.environ.get("LOCAL_RANK")
        self.local_rank_ids_predictable = (
            local_rank is not None and int(local_rank) == self.rank
        )

        # This test hardcodes expected results for a single-node, 4-rank run.
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

    def _assert_topology(
        self,
        comm,
        num_nodes,
        local_node_ranks,
        expected_rank_ids=None,
        description="",
    ):
        """Assert comm.get_topology() matches the expected single-node layout.

        For a single-node launch every communicator is single-node and uniform,
        so only num_nodes and local_node_ranks need to be supplied.

        expected_rank_ids, when given, is the expected sorted list of member
        ranks for both local_node_global_rank_ids and local_node_local_rank_ids
        (equal here since every rank equals its world rank on a single node).
        Values are asserted only when the matching env var is predictable (see
        self.global_rank_ids_predictable / self.local_rank_ids_predictable);
        otherwise only the length is checked."""
        topo = comm.get_topology()
        self.assertEqual(
            topo.num_nodes, num_nodes, f"num_nodes mismatch for {description}"
        )
        self.assertEqual(
            topo.local_node_ranks,
            local_node_ranks,
            f"local_node_ranks mismatch for {description}",
        )
        self.assertEqual(
            len(topo.local_node_global_rank_ids),
            local_node_ranks,
            f"local_node_global_rank_ids length mismatch for {description}",
        )
        self.assertEqual(
            len(topo.local_node_local_rank_ids),
            local_node_ranks,
            f"local_node_local_rank_ids length mismatch for {description}",
        )
        if expected_rank_ids is not None:
            if self.global_rank_ids_predictable:
                self.assertEqual(
                    sorted(topo.local_node_global_rank_ids),
                    expected_rank_ids,
                    f"local_node_global_rank_ids mismatch for {description}",
                )
            if self.local_rank_ids_predictable:
                self.assertEqual(
                    sorted(topo.local_node_local_rank_ids),
                    expected_rank_ids,
                    f"local_node_local_rank_ids mismatch for {description}",
                )
        # All ranks share one node, so the topology is always uniform and single.
        self.assertTrue(topo.uniform, f"expected uniform topology for {description}")
        self.assertTrue(
            topo.is_single_node(), f"expected single-node for {description}"
        )

    def test_world_topology(self):
        """World communicator: 1 node, all 4 ranks."""
        self._assert_topology(
            self.torchcomm,
            num_nodes=1,
            local_node_ranks=4,
            expected_rank_ids=list(range(EXPECTED_NUM_RANKS)),
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
            # 2 ranks, same node. World ranks {0, 1} participate.
            self._assert_topology(
                child,
                num_nodes=1,
                local_node_ranks=2,
                expected_rank_ids=ranks,
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
            # 2 even ranks, same node: ranks {0, 2} participate
            self._assert_topology(
                child,
                num_nodes=1,
                local_node_ranks=2,
                expected_rank_ids=ranks,
                description="non-contiguous split child",
            )
            child.finalize()
        else:
            self.assertIsNone(child, f"rank {self.rank} should not have a child comm")

    def test_single_rank_split_topology(self):
        """A communicator split down to a single rank: 1 node, 1 rank."""
        child = self.torchcomm.split([self.rank], name="single_rank_topology_comm")

        self.assertIsNotNone(child, f"rank {self.rank} expected a single-rank comm")
        # Only this world rank participates.
        self._assert_topology(
            child,
            num_nodes=1,
            local_node_ranks=1,
            expected_rank_ids=[self.rank],
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
        # First level: ranks {0, 1}
        self._assert_topology(
            first_comm,
            num_nodes=1,
            local_node_ranks=2,
            expected_rank_ids=first_ranks,
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
            # Second level: 1 rank
            self._assert_topology(
                second_comm,
                num_nodes=1,
                local_node_ranks=1,
                expected_rank_ids=[self.rank],
                description="second-level split child",
            )
            second_comm.finalize()
        else:
            self.assertIsNone(
                second_comm,
                f"first-level rank {first_rank} should not have a child",
            )

        first_comm.finalize()

    def test_chained_intra_node_split(self):
        """Build an intra-node comm from the topology, then split it again
           using the remap-by-position rule.

        Level 1 splits the world comm on local_node_global_rank_ids — on the
        world comm those world ranks are exactly the comm-local ranks split()
        expects.

        Level 2 splits that child by parity into tensor-parallel groups;
        the child is renumbered, so we map each world rank to its comm-local
        rank by its position in the (sorted) list that formed the child,
        then group by the parity of that comm-local rank.

        Single-node, 4-rank run: level 1 yields a 4-rank intra-node comm
        [0,1,2,3]; level 2 yields two 2-rank groups, even comm-local {0,2} and
        odd {1,3}, which on this single-node launch are world ranks {0,2}/{1,3}.
        """
        # split() takes world ranks, so local_node_global_rank_ids is only a
        # valid membership list when RANK is exposed and equals the world rank.
        if not self.global_rank_ids_predictable:
            self.skipTest(
                "RANK not exposed, so local_node_global_rank_ids is not a valid "
                "split() membership list"
            )

        # Level 1: intra-node comm. local_node_global_rank_ids comes ordered by
        # comm rank and includes this rank, so it is the exact membership split()
        # expects; no sorting needed.
        topo = self.torchcomm.get_topology()
        node_world_ranks = topo.local_node_global_rank_ids
        intra_node = self.torchcomm.split(node_world_ranks, name="intra_node_chain")

        self.assertIsNotNone(
            intra_node, f"rank {self.rank} expected an intra-node comm"
        )
        self._assert_topology(
            intra_node,
            num_nodes=1,
            local_node_ranks=EXPECTED_NUM_RANKS,
            expected_rank_ids=list(range(EXPECTED_NUM_RANKS)),
            description="intra-node comm (chain level 1)",
        )

        # Level 2: remap this rank's world id to its comm-local rank within
        # intra_node (its position in the list that formed the child), then
        # build the parity group in the child's comm-local numbering.
        my_local = node_world_ranks.index(self.rank)
        tp_ranks = [i for i in range(len(node_world_ranks)) if i % 2 == my_local % 2]
        tp_group = intra_node.split(tp_ranks, name="tp_group_chain")

        # Every rank lands in a group, so no rank gets None here.
        self.assertIsNotNone(tp_group, f"rank {self.rank} expected a tp group")
        # The group's members are the world ranks at the parity positions of
        # node_world_ranks, i.e. its on-node world ranks of the same parity.
        expected_group_world_ranks = sorted(node_world_ranks[i] for i in tp_ranks)
        self._assert_topology(
            tp_group,
            num_nodes=1,
            local_node_ranks=EXPECTED_NUM_RANKS // 2,
            expected_rank_ids=expected_group_world_ranks,
            description="tensor-parallel group (chain level 2)",
        )

        tp_group.finalize()
        intra_node.finalize()

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
                expected_rank_ids=list(range(EXPECTED_NUM_RANKS)),
                description="world communicator with compute_topology=true",
            )
        finally:
            wrapper = None


if __name__ == "__main__":
    unittest.main(failfast=True)
