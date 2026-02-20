// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

/**
 * Result of topology discovery — identifies NVLink peers and provides
 * the global-to-NVL-local rank mapping.
 *
 * Redundant fields are intentionally omitted; consumers derive them:
 *   - nvlNRanks        = nvlPeerRanks.size() + 1
 *   - nvlLocalRank     = globalToNvlLocal.at(myRank)
 *   - typePerRank[r]   = SELF if r==myRank, P2P_NVL if in globalToNvlLocal,
 *                         P2P_IBGDA otherwise
 *   - ibgdaPeerRanks   = all ranks except self (universal fallback)
 */
struct TopologyResult {
  /// Global ranks of NVLink-connected peers (excluding self), sorted.
  std::vector<int> nvlPeerRanks;

  /// Maps global rank → NVL-local index for all ranks in the NVL domain
  /// (including self).
  std::unordered_map<int, int> globalToNvlLocal;

  /// MNNVL fabric cluster UUID (all zeros if fabric info unavailable).
  char clusterUuid[NvmlFabricInfo::kUuidLen]{};

  /// MNNVL fabric clique ID (0 if fabric info unavailable).
  unsigned int cliqueId{0};

  /// Whether MNNVL fabric info was available for this rank.
  bool fabricAvailable{false};
};

/**
 * Discovers multi-GPU topology via bootstrap allGather.
 *
 * Two-tier NVLink detection (following NCCL's MNNVL pattern):
 *
 *   Tier 1 — MNNVL fabric (GB200):
 *     Both ranks have NVML fabric info and share the same clusterUuid
 *     + cliqueId → same NVLink domain → NVL peer.
 *
 *   Tier 2 — Same-host + cudaDeviceCanAccessPeer (H100 and earlier):
 *     Both ranks on the same hostname → query CUDA peer access.
 *
 *   Fallback → IBGDA.
 *
 * Usage:
 *   auto topo = TopologyDiscovery::discover(myRank, nRanks, deviceId,
 * bootstrap);
 *   // topo.nvlPeerRanks, topo.globalToNvlLocal, etc.
 */
class TopologyDiscovery {
 public:
  static TopologyResult discover(
      int myRank,
      int nRanks,
      int deviceId,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap);
};

} // namespace comms::pipes
