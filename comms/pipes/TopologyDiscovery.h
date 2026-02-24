// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

/**
 * Callable that checks whether deviceA can access deviceB via P2P.
 * Used for Tier 2 (same-host) NVLink detection.
 * Return true if P2P access is possible.
 */
using PeerAccessFn = std::function<bool(int deviceA, int deviceB)>;

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
 * Per-rank topology info used by classify().
 *
 * This struct captures the per-rank inputs needed for topology classification
 * without requiring CUDA or NVML. It enables unit testing of the
 * classification logic with synthetic data.
 */
struct RankTopologyInfo {
  char hostname[64]{};
  int cudaDevice{0};
  NvmlFabricInfo fabricInfo;
};

/**
 * Callable that gathers local topology info for a given CUDA device.
 * Returns a RankTopologyInfo populated with hostname, cudaDevice, and
 * NvmlFabricInfo. Injectable for testing without real CUDA/NVML/gethostname.
 */
using LocalInfoFn = std::function<RankTopologyInfo(int deviceId)>;

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
 *   TopologyDiscovery topo;  // default: real CUDA + NVML + gethostname
 *   auto result = topo.discover(myRank, nRanks, deviceId, bootstrap);
 *
 * For testing:
 *   TopologyDiscovery topo(myPeerAccessFn, myLocalInfoFn);
 *   auto result = topo.discover(myRank, nRanks, deviceId, bootstrap);
 */
class TopologyDiscovery {
 public:
  /**
   * Default constructor: uses real CUDA + NVML + gethostname for local
   * info gathering and cudaDeviceCanAccessPeer for Tier 2 detection.
   */
  TopologyDiscovery();

  /**
   * Constructor with custom peer access function.
   * Uses real CUDA + NVML + gethostname for local info gathering.
   *
   * @param peerAccessFn  Custom peer access function for Tier 2 detection.
   *                      Pass an empty std::function to skip Tier 2.
   */
  explicit TopologyDiscovery(PeerAccessFn peerAccessFn);

  /**
   * Constructor with custom peer access and local info functions.
   * Fully injectable for testing without real hardware.
   *
   * @param peerAccessFn  Custom peer access function for Tier 2 detection.
   *                      Pass an empty std::function to skip Tier 2.
   * @param localInfoFn   Custom function to gather per-rank topology info.
   */
  TopologyDiscovery(PeerAccessFn peerAccessFn, LocalInfoFn localInfoFn);

  /**
   * Discover topology using local info gathering and bootstrap allGather.
   *
   * @param myRank      This rank's global index.
   * @param nRanks      Total number of ranks.
   * @param deviceId    CUDA device index.
   * @param bootstrap   Bootstrap interface for allGather.
   */
  TopologyResult discover(
      int myRank,
      int nRanks,
      int deviceId,
      ctran::bootstrap::IBootstrap& bootstrap);

  /**
   * Classify pre-populated rank topology info into NVL peers.
   *
   * This is the core classification logic extracted from discover() for
   * testability. It classifies peers using Tier 1 (MNNVL fabric match)
   * and Tier 2 (same-host + peer access).
   *
   * Tier 2 requires a non-empty peerAccessFn (set via constructor).
   * If not set, Tier 2 is skipped.
   *
   * @param myRank       This rank's global index.
   * @param nRanks       Total number of ranks.
   * @param allInfo      Pre-populated per-rank topology info (size == nRanks).
   */
  TopologyResult
  classify(int myRank, int nRanks, std::vector<RankTopologyInfo>& allInfo);

 private:
  PeerAccessFn peerAccessFn_;
  LocalInfoFn localInfoFn_;
};

} // namespace comms::pipes
