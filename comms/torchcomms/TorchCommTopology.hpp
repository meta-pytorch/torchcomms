// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace torch::comms {

/**
 * CommTopology - Node layout of a communicator's ranks.
 *
 * Backends that can determine this (e.g. via a hostname hash exchange at
 * init) populate it so higher layers can make topology-aware decisions:
 * keeping tensor-parallel groups intra-node, sizing DeviceMesh dimensions,
 * choosing hybrid-shard layouts, or selecting intra-node fast-path
 * collectives.
 *
 * The default value describes an unknown topology and is deliberately
 * conservative: isSingleNode() returns false, so callers will not take an
 * intra-node-only fast path unless a backend has affirmatively reported a
 * single-node communicator.
 *
 * Note on node identity: "node" means a distinct host as observed by the
 * backend. A backend that folds the Linux boot_id into its host hash will
 * treat two containers on the same physical host as different nodes.
 */
struct CommTopology {
  // Number of distinct hosts spanned by this communicator.
  int num_nodes{0};

  // Number of ranks co-located on the local rank's node.
  int local_node_ranks{0};

  // True if every node hosts the same number of ranks. When false, the
  // num_nodes/local_node_ranks factorization is not meaningful for the whole
  // communicator and consumers should fall back to a topology-agnostic path.
  bool uniform{false};

  // The question most callers actually ask: are all ranks on one node?
  // Returns false for the default (unknown) topology.
  bool isSingleNode() const {
    return num_nodes == 1;
  }
};

} // namespace torch::comms
