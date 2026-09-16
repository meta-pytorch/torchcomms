// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/topology/NvmlFabricInfo.h"
#include "comms/prims/transport/Transport.cuh"

namespace comms::prims {

/**
 * Callable that checks whether deviceA can access deviceB via P2P.
 * Used for Tier 2 (same-host) NVLink detection.
 * Return true if P2P access is possible.
 */
using PeerAccessFn = std::function<bool(int deviceA, int deviceB)>;

/** Returns whether FABRIC memory handles are usable for a CUDA device. */
using FabricHandleAccessFn = std::function<bool(int deviceId)>;

/**
 * Controls whether Multi-Node NVLink (MNNVL) is used for cross-host
 * NVLink communication.
 *
 * Follows NCCL's NCCL_MNNVL_ENABLE semantics.
 */
enum class MnnvlMode {
  // Disable MNNVL support. Cross-host NVLink (Tier 1) is skipped even on
  // MNNVL-capable hardware. Only same-host cudaDeviceCanAccessPeer (Tier 2)
  // is used for NVLink peer detection.
  kDisabled = 0,

  // Enable MNNVL support. Initialization will fail if MNNVL is not supported
  // (i.e., fabric info is unavailable).
  kEnabled = 1,

  // Automatic detection (default). Use MNNVL if available, fall back to Tier 2
  // if not.
  kAuto = 2,
};

/**
 * Configuration for topology discovery.
 *
 * Controls MNNVL overrides following NCCL's NCCL_MNNVL_ENABLE,
 * NCCL_MNNVL_UUID, and NCCL_MNNVL_CLIQUE_ID semantics. UUID and clique ID
 * overrides only take effect on MNNVL-capable hardware (GB200). On H100 and
 * earlier, NVLink connectivity is determined by same-host +
 * cudaDeviceCanAccessPeer regardless of these settings.
 */
struct TopologyConfig {
  // Controls whether MNNVL (cross-host NVLink) is used.
  // Follows NCCL's NCCL_MNNVL_ENABLE semantics:
  //   - kDisabled: never use MNNVL, even if hardware supports it
  //   - kEnabled: require MNNVL; fail if not supported
  //   - kAuto (default): use MNNVL if available, fall back otherwise
  MnnvlMode mnnvlMode{MnnvlMode::kAuto};

  // Override MNNVL cluster UUID.
  // Follows NCCL's NCCL_MNNVL_UUID semantics:
  //   - std::nullopt (default): use hardware-reported cluster UUID from NVML
  //   - 64-bit integer: the value is written into both the upper and lower
  //     64-bit halves of the 128-bit cluster UUID.
  std::optional<int64_t> mnnvlUuid;

  // Override MNNVL clique ID.
  // Follows NCCL's NCCL_MNNVL_CLIQUE_ID semantics:
  //   - std::nullopt (default): use hardware-reported clique ID from NVML
  //   - 32-bit integer: override clique ID. Ranks with the same
  //     <clusterUuid, cliqueId> pair form an NVLink clique.
  std::optional<int> mnnvlCliqueId;

  // Optional logical NVL rank group supplied by an upper layer.
  //
  // When set, both MNNVL fabric matching and same-host P2P discovery are
  // constrained to this rank set. This lets CTRAN pass CommStateX's effective
  // local/NVL group, which may be a virtual clique smaller than the raw MNNVL
  // fabric domain.
  std::optional<std::vector<int>> logicalNvlRanks;

  // Disable all P2P NVLink transport (both Tier 1 MNNVL and Tier 2 same-host).
  // Follows NCCL's NCCL_P2P_DISABLE semantics (PATH_LOC — self only):
  //   - false (default): use NVLink when available (MNNVL or peer access)
  //   - true: skip both Tier 1 and Tier 2; all non-self peers fall back to
  //     IBGDA
  bool p2pDisable{false};
};

/** Selects the effective domains exposed to MCCL collectives. */
enum class TopologyDomainMode : std::uint8_t {
  kSystem = 0,
  kNoLocal = 1,
  kVirtual = 2,
};

/**
 * Configuration for canonical all-rank topology discovery.
 *
 * enableNvlFabricDomains is an upper-layer rollout gate. When false it takes
 * precedence over mnnvlMode and keeps discovery on same-host P2P domains.
 * localDeviceRack is a rank-local fact and is therefore excluded from the
 * communicator-wide policy comparison.
 */
struct CanonicalTopologyConfig {
  MnnvlMode mnnvlMode{MnnvlMode::kAuto};
  std::optional<int64_t> mnnvlUuid;
  std::optional<int> mnnvlCliqueId;
  bool p2pDisable{false};
  bool enableNvlFabricDomains{true};
  bool mnnvlTrunkDisable{false};
  TopologyDomainMode domainMode{TopologyDomainMode::kSystem};
  int virtualDomainSize{0};
  std::string localDeviceRack;
};

inline constexpr std::uint32_t kCanonicalTopologyWireMagic = 0x50544f50;
inline constexpr std::uint16_t kCanonicalTopologyWireVersion = 1;
inline constexpr std::uint16_t kCanonicalTopologyPreambleSize = 16;
inline constexpr std::uint16_t kCanonicalTopologyWireSize = 192;
inline constexpr std::size_t kCanonicalTopologyNameLength = 64;

/**
 * Fixed preamble exchanged before any version-dependent record.
 *
 * Version negotiation is intentionally unsupported. Its layout is frozen so
 * incompatible peers can fail before exchanging version-dependent records.
 */
struct alignas(4) CanonicalTopologyPreamble {
  std::uint32_t magic{kCanonicalTopologyWireMagic};
  std::uint16_t version{kCanonicalTopologyWireVersion};
  std::uint16_t headerSize{kCanonicalTopologyPreambleSize};
  std::uint16_t recordSize{kCanonicalTopologyWireSize};
  std::uint8_t status{0};
  std::uint8_t reserved{0};
  std::int32_t rank{-1};
};

/** Fixed-width policy embedded in every gathered rank record. */
struct alignas(8) CanonicalTopologyPolicyWire {
  std::int64_t mnnvlUuid{0};
  std::int32_t mnnvlCliqueId{0};
  std::int32_t virtualDomainSize{0};
  std::uint8_t mnnvlMode{0};
  std::uint8_t hasMnnvlUuid{0};
  std::uint8_t hasMnnvlCliqueId{0};
  std::uint8_t p2pDisable{0};
  std::uint8_t enableNvlFabricDomains{0};
  std::uint8_t mnnvlTrunkDisable{0};
  std::uint8_t domainMode{0};
  std::uint8_t reserved{0};
};

/** Fixed-width record exchanged by canonical topology discovery. */
struct alignas(8) CanonicalRankTopologyInfo {
  std::uint32_t magic{kCanonicalTopologyWireMagic};
  std::uint16_t version{kCanonicalTopologyWireVersion};
  std::uint16_t recordSize{kCanonicalTopologyWireSize};
  std::int32_t rank{-1};
  std::int32_t cudaDevice{-1};
  std::array<char, NvmlFabricInfo::kUuidLen> clusterUuid{};
  std::uint32_t cliqueId{0};
  std::uint8_t fabricInfoAvailable{0};
  std::uint8_t fabricHandleAvailable{0};
  std::array<std::uint8_t, 2> reserved{};
  CanonicalTopologyPolicyWire policy{};
  std::array<char, kCanonicalTopologyNameLength> hostname{};
  std::array<char, kCanonicalTopologyNameLength> deviceRack{};
};

static_assert(sizeof(CanonicalTopologyPolicyWire) == 24);
static_assert(std::is_standard_layout_v<CanonicalTopologyPolicyWire>);
static_assert(std::is_trivially_copyable_v<CanonicalTopologyPolicyWire>);
static_assert(offsetof(CanonicalTopologyPolicyWire, mnnvlUuid) == 0);
static_assert(offsetof(CanonicalTopologyPolicyWire, mnnvlCliqueId) == 8);
static_assert(offsetof(CanonicalTopologyPolicyWire, virtualDomainSize) == 12);
static_assert(offsetof(CanonicalTopologyPolicyWire, mnnvlMode) == 16);
static_assert(offsetof(CanonicalTopologyPolicyWire, reserved) == 23);
static_assert(
    sizeof(CanonicalTopologyPreamble) == kCanonicalTopologyPreambleSize);
static_assert(std::is_standard_layout_v<CanonicalTopologyPreamble>);
static_assert(std::is_trivially_copyable_v<CanonicalTopologyPreamble>);
static_assert(offsetof(CanonicalTopologyPreamble, magic) == 0);
static_assert(offsetof(CanonicalTopologyPreamble, version) == 4);
static_assert(offsetof(CanonicalTopologyPreamble, headerSize) == 6);
static_assert(offsetof(CanonicalTopologyPreamble, recordSize) == 8);
static_assert(offsetof(CanonicalTopologyPreamble, status) == 10);
static_assert(offsetof(CanonicalTopologyPreamble, rank) == 12);
static_assert(sizeof(CanonicalRankTopologyInfo) == kCanonicalTopologyWireSize);
static_assert(std::is_standard_layout_v<CanonicalRankTopologyInfo>);
static_assert(std::is_trivially_copyable_v<CanonicalRankTopologyInfo>);
static_assert(offsetof(CanonicalRankTopologyInfo, magic) == 0);
static_assert(offsetof(CanonicalRankTopologyInfo, version) == 4);
static_assert(offsetof(CanonicalRankTopologyInfo, recordSize) == 6);
static_assert(offsetof(CanonicalRankTopologyInfo, rank) == 8);
static_assert(offsetof(CanonicalRankTopologyInfo, cudaDevice) == 12);
static_assert(offsetof(CanonicalRankTopologyInfo, clusterUuid) == 16);
static_assert(offsetof(CanonicalRankTopologyInfo, cliqueId) == 32);
static_assert(offsetof(CanonicalRankTopologyInfo, policy) == 40);
static_assert(offsetof(CanonicalRankTopologyInfo, hostname) == 64);
static_assert(offsetof(CanonicalRankTopologyInfo, deviceRack) == 128);

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

  /// Whether this result uses MNNVL fabric. Canonical discovery sets this only
  /// for communicator-wide activation; legacy discovery retains its local
  /// capability semantics.
  bool fabricAvailable{false};
};

/**
 * Canonical topology shared by MCCL and MultiPeerTransport.
 *
 * rankInfo and ranksByDomain are identical on every rank. mptTopology is the
 * exact rank-local projection to pass to MultiPeerTransport.
 */
struct CanonicalTopologyResult {
  std::vector<CanonicalRankTopologyInfo> rankInfo;
  std::vector<std::vector<int>> ranksByDomain;
  TopologyResult mptTopology;
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
 *     Skipped when TopologyConfig::p2pDisable is true (set by
 * NCCL_P2P_DISABLE).
 *
 *   Both tiers are skipped when p2pDisable is true, matching NCCL's PATH_LOC
 *   semantics where all inter-GPU P2P is disabled.
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
 *
 * TopologyConfig:
 *
 * MNNVL overrides (following NCCL's NCCL_MNNVL_ENABLE / NCCL_MNNVL_UUID /
 * NCCL_MNNVL_CLIQUE_ID):
 *   These optional parameters override the hardware-reported fabric info
 *   from NVML. They only take effect when fabric info is available (i.e.,
 *   on MNNVL-capable hardware like GB200).
 *
 *   mnnvlUuid:
 *   - std::nullopt (default): use hardware-reported cluster UUID
 *   - 64-bit integer: the value is written into both the upper and lower
 *     64-bit halves of the 128-bit cluster UUID, matching NCCL's
 *     NCCL_MNNVL_UUID semantics.
 *
 *   mnnvlCliqueId:
 *   - std::nullopt (default): use hardware-reported clique ID
 *   - 32-bit integer: override clique ID. Ranks with the same
 *     <clusterUuid, cliqueId> pair form an NVLink clique.
 *
 *   logicalNvlRanks:
 *   - std::nullopt (default): discover every NVLink-reachable peer
 *   - rank set: restrict both MNNVL and same-host P2P discovery to the given
 *     logical group, preserving the upper layer's local/NVL rank mapping.
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

  /** Constructor with all hardware probes injectable for testing. */
  TopologyDiscovery(
      PeerAccessFn peerAccessFn,
      LocalInfoFn localInfoFn,
      FabricHandleAccessFn fabricHandleAccessFn);

  /**
   * Discover topology using local info gathering and bootstrap allGather.
   *
   * @param myRank      This rank's global index.
   * @param nRanks      Total number of ranks.
   * @param deviceId    CUDA device index.
   * @param bootstrap   Bootstrap interface for allGather.
   * @param topoConfig  Optional MNNVL overrides and logical rank constraints.
   */
  TopologyResult discover(
      int myRank,
      int nRanks,
      int deviceId,
      meta::comms::IBootstrap& bootstrap,
      const TopologyConfig& topoConfig = {});

  /**
   * Classify pre-populated rank topology info into NVL peers.
   *
   * This is the core classification logic extracted from discover() for
   * testability. It applies TopologyConfig overrides (MnnvlMode, UUID,
   * clique ID) to allInfo[myRank], then classifies peers using Tier 1
   * (MNNVL fabric match) and Tier 2 (same-host + peer access).
   *
   * Tier 2 requires a non-empty peerAccessFn (set via constructor).
   * If not set, Tier 2 is skipped.
   *
   * @param myRank       This rank's global index.
   * @param nRanks       Total number of ranks.
   * @param allInfo      Pre-populated per-rank topology info (size == nRanks).
   *                     allInfo[myRank] may be modified by TopologyConfig
   *                     overrides.
   * @param topoConfig   Optional MNNVL overrides and logical rank constraints.
   */
  TopologyResult classify(
      int myRank,
      int nRanks,
      std::vector<RankTopologyInfo>& allInfo,
      const TopologyConfig& topoConfig = {});

  /**
   * Discover and validate one communicator-wide topology snapshot.
   *
   * Every rank must collectively agree to call this API before any rank enters
   * it. Selecting between discover() and discoverCanonical() per rank would
   * execute different allGather sequences and can hang initialization.
   *
   * The final allGather exchanges one same-host reachability row per rank.
   * The returned mptTopology is derived from ranksByDomain and must be passed
   * to MultiPeerTransport rather than rediscovering topology there.
   */
  CanonicalTopologyResult discoverCanonical(
      int myRank,
      int nRanks,
      int deviceId,
      meta::comms::IBootstrap& bootstrap,
      const CanonicalTopologyConfig& topoConfig = {});

  /**
   * Pure canonical classification entry point for synthetic tests.
   * peerReachability stores nRanks row-major bit-packed rows, each containing
   * ceil(nRanks / 8) bytes.
   */
  CanonicalTopologyResult classifyCanonical(
      int myRank,
      int nRanks,
      std::vector<CanonicalRankTopologyInfo> rankInfo,
      const std::vector<std::uint8_t>& peerReachability,
      const CanonicalTopologyConfig& topoConfig = {});

 private:
  PeerAccessFn peerAccessFn_;
  LocalInfoFn localInfoFn_;
  FabricHandleAccessFn fabricHandleAccessFn_;
};

} // namespace comms::prims
