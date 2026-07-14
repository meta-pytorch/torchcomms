// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "comms/uniflow/Result.h"

namespace uniflow {

/// Bandwidth data for a single PCIe link (used during sysfs discovery).
struct PcieLinkInfo {
  uint32_t speedMbpsPerLane{0}; // Mbps per lane, encoding overhead included
  uint16_t width{0}; // e.g. 16 for x16

  /// Total effective bandwidth in MB/s (speed * width / 8).
  uint32_t bandwidthMBps() const noexcept;
};

/// Node types in the topology graph.
enum class NodeType : uint8_t { GPU, CPU, NIC };

/// Path quality between two nodes, ordered best to worst.
/// Path type = worst segment along the route (same as NCCL).
enum class PathType : uint8_t {
  NVL, // NVLink (through NVSwitch)
  C2C, // Chip-to-chip (e.g. Grace Hopper)
  PIX, // Same PCIe switch
  PXB, // Multiple PCIe switches, same root complex
  PXN, // PCIe + NVLink proxy through peer GPU
  PHB, // PCIe host bridge, same NUMA
  SYS, // Cross NUMA
  DIS, // Disconnected
};

const char* pathTypeToString(PathType p) noexcept;

/// NIC device name filter using NCCL_IB_HCA-style syntax.
///
/// Filter string format: [^][=]<name>[:<port>][,<name>[:<port>],...]
///
/// Prefix modifiers:
///   (none) - include devices matching any entry prefix (default)
///   =      - include devices whose name exactly matches an entry
///   ^      - exclude devices matching any entry prefix
///   ^=     - exclude devices whose name exactly matches an entry
///
/// Examples:
///   "mlx5_0,mlx5_1"   - include devices starting with mlx5_0 or mlx5_1
///   "=mlx5_0"          - include only exactly mlx5_0
///   "^bnxt_re"          - exclude devices starting with bnxt_re
///   "mlx5_0:1"          - include mlx5_0 port 1 only
///
/// Default-constructed filter matches everything.
class NicFilter {
 public:
  NicFilter() = default;
  explicit NicFilter(std::string_view filterStr);

  /// Check if a device name (and optional port) passes the filter.
  bool matches(const std::string& devName, int port = -1) const;

  bool empty() const {
    return entries_.empty();
  }

 private:
  enum class MatchMode : uint8_t {
    PrefixInclude,
    ExactInclude,
    PrefixExclude,
    ExactExclude,
  };
  struct Entry {
    std::string name;
    int port{-1};
  };
  bool isExactMode() const noexcept;
  MatchMode mode_{MatchMode::PrefixInclude};
  std::vector<Entry> entries_;
};

/// A weighted edge in the topology graph.
struct TopoLink {
  PathType type; // Path type this link contributes to BFS
  uint32_t bw; // Bandwidth in MB/s
  int peerNodeId; // Index into Topology nodes
};

/// A node in the topology graph with type-specific payload.
struct TopoNode {
  NodeType type;
  int id{-1}; // Index in Topology nodes
  std::string name; // e.g. "cuda:0", "cpu:0", "mlx5_0"

  std::vector<TopoLink> links; // Adjacency list

  struct GpuData {
    int cudaDeviceId{-1};
    std::string bdf;
    int numaNode{-1};
    int sm{0}; // computeCapabilityMajor * 10 + minor
  };

  struct CpuData {
    int numaId{-1};
  };

  struct NicData {
    std::string bdf;
    int numaNode{-1};
    int port{-1}; // Active port numbers
    uint32_t portSpeedMbps{0}; // RDMA port speed from ibv_query_port (Mbps)
    std::string netdevName; // Backing netdev (e.g. "beth0"); empty = unknown
  };

  std::variant<GpuData, CpuData, NicData> data;
};

/// Pre-computed path between two topology nodes.
struct TopoPath {
  PathType type{PathType::DIS};
  uint32_t bw{0}; // Bottleneck bandwidth in MB/s
  std::optional<int> proxyNode; // For PXN: proxy GPU node id
};

/// Controls which path types are considered when querying paths.
/// discover() always detects all link types; this config filters at query time.
struct PathFilter {
  bool allowC2C{false}; // Allow C2C paths (e.g. Grace Hopper)
  bool allowPxn{false}; // Allow PXN proxy routes (GPU→NVLink→GPU→PCIe→NIC)
};

/// Graph-based topology covering GPUs, CPUs, NICs, and NVSwitches.
///
/// Pure data structure: holds the graph, the all-pairs path matrix, and
/// query methods. Construction is done from the outside via the mutation
/// API (`addNode`, `addLink`, `setP2pMatrix`, ..., then `recomputePaths`).
/// Discovery backends live in `comms/uniflow/drivers/` — see
/// `drivers/TopologyDiscovery.h` for the public entry point.
///
/// After populating, all path queries are O(1) lookups into the
/// pre-computed all-pairs path matrix.
class Topology {
 public:
  Topology() = default;

  // --- Status ---
  Status available() const {
    return status;
  }

  // --- Node count queries ---

  size_t gpuCount() const {
    return gpuNodeIds_.size();
  }

  size_t nicCount() const {
    return nicNodeIds_.size();
  }

  size_t numaNodeCount() const {
    return cpuNodeIds_.size();
  }

  // --- Pre-computed path lookups (O(1) after recomputePaths) ---

  /// Returns the pre-computed path, filtered by PathFilter.
  /// By default (C2C and PXN disabled), returns the BFS-only path.
  const TopoPath&
  getPath(int srcNodeId, int dstNodeId, const PathFilter& filter = {}) const;

  // --- P2P connectivity ---

  /// Check if hardware can do intra-node P2P.
  bool canGpuAccess(int device, int peerDevice) const;

  // --- Node access ---

  const TopoNode& getNode(int nodeId) const;
  const TopoNode& getGpuNode(int cudaDeviceId) const;
  const TopoNode& getNicNode(int nicIndex) const;
  const TopoNode& getCpuNode(int numaId) const;

  /// Returns the NUMA node of the calling thread's CPU.
  int getCurrentCpuNumaNode() const;

  /// Returns the NUMA node id of the NIC with the given device name (e.g.
  /// "mlx5_0"), or -1 if unknown. O(1) lookup into a map built at discovery.
  int nicNumaNode(const std::string& nicName) const;

  /// Check if a NIC passes the given filter.
  bool filterNic(int nicIndex, const NicFilter& filter) const;

  /// Returns true if the NIC's backing netdev name starts with @p netdevPrefix.
  /// An empty prefix matches everything (predicate disabled).
  bool matchesNetdevPrefix(int nicIndex, std::string_view netdevPrefix) const;

  /// Returns true if any NIC has a known backing netdev name. Backends that do
  /// not populate netdev names return false, letting callers skip netdev-prefix
  /// filtering instead of warning on every NIC selection.
  bool hasNetdevNames() const;

  // --- NIC selection ---
  /// When @p netdevPrefix is non-empty, only NICs whose backing netdev name
  /// starts with the prefix (e.g. "beth") are considered; if none match, it
  /// falls back to filter-only selection and logs a warning.
  std::vector<std::string> selectCpuNics(
      const NicFilter& filter = {},
      std::string_view netdevPrefix = "") const;

  /// Select NICs with the best path from the given NUMA node, filtered by
  /// NicFilter. Returns an empty vector when the NUMA node is unknown.
  std::vector<std::string> selectCpuNicsForNuma(
      int numaId,
      const NicFilter& filter = {},
      size_t maxNics = 0) const;

  /// Select a bounded union of best CPU NICs for every known NUMA node.
  /// This preserves transfer-time buffer-NUMA scheduling without keeping every
  /// equivalent CPU NIC in the transport.
  std::vector<std::string> selectCpuNicsForNumaNodes(
      const NicFilter& filter = {},
      size_t maxNicsPerNuma = 0) const;

  /// Select NICs closest to the given GPU, filtered by NicFilter.
  /// Returns multiple NICs when they share the same best path (e.g. GB200:
  /// 2 NICs per GPU). See selectCpuNics for @p netdevPrefix semantics.
  std::vector<std::string> selectGpuNics(
      int cudaDeviceId,
      const NicFilter& filter = {},
      std::string_view netdevPrefix = "") const;

  // --- Mutation API (used by discovery backends) ---

  /// Reset to an empty topology (status, nodes, links, paths, indices).
  void clear();

  /// Set the discovery status (typically Ok() at the end of discovery).
  void setStatus(Status s) {
    status = std::move(s);
  }

  /// Append a node. Returns the assigned nodeId.
  int addNode(TopoNode node);

  /// Append a bidirectional link between two existing nodes.
  void addLink(int srcId, int dstId, PathType type, uint32_t bw);

  /// Register a GPU node id under its cudaDeviceId (appended).
  void registerGpuNode(int cudaDeviceId, int nodeId);

  /// Register a CPU node id under its NUMA id (appended).
  void registerCpuNode(int numaId, int nodeId);

  /// Register a NIC node id under its NIC index (appended).
  void registerNicNode(int nicIndex, int nodeId);

  /// Install the GPU→GPU peer-access matrix. Must be square of size
  /// `gpuCount()`.
  void setP2pMatrix(std::vector<std::vector<bool>> matrix);

  /// Recompute all-pairs paths (BFS), C2C overrides, and PXN overrides.
  /// Call after all nodes and links have been added.
  void recomputePaths();

 private:
  void computePaths();
  void computeC2cPaths();
  void computePxnPaths();

  Status status{ErrCode::TopologyDisconnect};

  std::vector<TopoNode> nodes_;
  // BFS paths (no C2C, no PXN). This is the baseline.
  std::vector<std::vector<TopoPath>> paths_;
  // Sparse overrides: only populated for node pairs where C2C/PXN
  // provides a better path than the baseline BFS.
  std::map<std::pair<int, int>, TopoPath> c2cPaths_;
  std::map<std::pair<int, int>, TopoPath> pxnPaths_;

  // Index maps for quick lookup
  std::vector<int> gpuNodeIds_; // gpuNodeIds_[cudaDeviceId] = nodeId
  std::vector<int> nicNodeIds_; // nicNodeIds_[nicIndex] = nodeId
  std::unordered_map<std::string, int> nicNameToNuma_; // nic name -> NUMA id
  std::vector<int> cpuNodeIds_; // cpuNodeIds_[numaId] = nodeId
  std::vector<std::vector<bool>> p2pMatrix_;
};

} // namespace uniflow
