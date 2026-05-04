// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/drivers/nvml/NvmlApi.h"
#include "comms/uniflow/drivers/sysfs/SysfsApi.h"

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
/// After discover(), all path queries are O(1) lookups into a pre-computed
/// all-pairs path matrix. NIC discovery uses IbvApi (ibverbs wrapper).
class Topology {
 public:
  // --- Singleton ---
  /// Returns the singleton Topology instance. Driver overrides are only
  /// honored on the first call for Unit Test — subsequent calls return the
  /// existing instance regardless of arguments (static local semantics).
  static Topology& get(
      std::shared_ptr<CudaApi> cudaApi = nullptr,
      std::shared_ptr<NvmlApi> nvmlApi = nullptr,
      std::shared_ptr<IbvApi> ibvApi = nullptr,
      std::shared_ptr<SysfsApi> sysfsApi = nullptr);

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

  // --- Pre-computed path lookups (O(1) after discover) ---

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

  /// Check if a NIC passes the given filter.
  bool filterNic(int nicIndex, const NicFilter& filter) const;

  // --- NIC selection ---

  /// Select NICs with the best path from their own NUMA node, filtered by
  /// NicFilter. Returns multiple NICs when they share the same best path
  /// type and bandwidth.
  std::vector<std::string> selectCpuNics(const NicFilter& filter = {}) const;

  /// Select NICs closest to the given GPU, filtered by NicFilter.
  /// Returns multiple NICs when they share the same best path (e.g. GB200:
  /// 2 NICs per GPU).
  std::vector<std::string> selectGpuNics(
      int cudaDeviceId,
      const NicFilter& filter = {}) const;

  friend class TopologyTest;

 private:
  Topology(
      std::shared_ptr<CudaApi> cudaApi,
      std::shared_ptr<NvmlApi> nvmlApi,
      std::shared_ptr<IbvApi> ibvApi,
      std::shared_ptr<SysfsApi> sysfsApi);

  /// Probe GPUs, NICs, NUMA nodes, build graph, and compute all-pairs paths.
  /// Always detects C2C links and computes PXN routes.
  Status discover();

  struct DiscoveryData;
  Status discoverHardware(DiscoveryData& data);
  void buildNodes(const DiscoveryData& data);
  void buildP2pMatrix();
  void buildEdges(const DiscoveryData& data);
  void computePaths();
  void computeC2cPaths();
  void computePxnPaths();
  int addNode(TopoNode node);
  void addLink(int srcId, int dstId, PathType type, uint32_t bw);

  Status status{ErrCode::TopologyDisconnect};
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<NvmlApi> nvmlApi_;
  std::shared_ptr<IbvApi> ibvApi_;
  std::shared_ptr<SysfsApi> sysfsApi_;

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
  std::vector<int> cpuNodeIds_; // cpuNodeIds_[numaId] = nodeId
  std::vector<std::vector<bool>> p2pMatrix_;
};

} // namespace uniflow
