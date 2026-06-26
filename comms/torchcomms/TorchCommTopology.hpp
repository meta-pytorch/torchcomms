// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

namespace torch::comms {

/**
 * CommTopology - Node layout of a communicator's ranks.
 *
 * Backends that can determine this (e.g. via a hostname hash exchange at
 * init) populate it so higher layers can make topology-aware decisions.
 *
 * The default value describes an unknown topology and is deliberately
 * conservative: isSingleNode() returns false.
 *
 * Note on node identity: "node" means a distinct host as observed by the
 * backend. A backend that folds the Linux boot_id into its host hash will
 * treat two containers on the same physical host as different nodes.
 */
struct CommTopology {
  // Number of distinct hosts spanned by this communicator.
  int num_nodes{0};

  // Node index of every rank in the communicator, indexed by comm-local rank:
  // node_ids[r] is the node hosting rank r, in [0, num_nodes). Nodes are
  // numbered by first appearance in comm-rank order (rank 0's node is 0), so
  // the numbering is deterministic and identical on every rank. Size equals the
  // communicator size. Unlike the local_node_* fields, this maps the whole
  // communicator, so callers can build inter-node groups (e.g. the HSDP
  // replication group or per-node leaders), not just the caller's intra-node
  // group.
  std::vector<int> node_ids;

  // Number of ranks co-located on the local rank's node.
  int local_node_ranks{0};

  // Global (world) rank ids co-located on the local rank's node.
  std::vector<int> local_node_global_rank_ids;

  // Local rank ids co-located on the local rank's node.
  std::vector<int> local_node_local_rank_ids;

  // True if every node hosts the same number of ranks.
  bool uniform{false};

  // Are all ranks on the same node?
  bool isSingleNode() const {
    return num_nodes == 1;
  }
};

} // namespace torch::comms
