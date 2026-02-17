// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace comms::pipes {

/**
 * Path type hierarchy for topology-aware NIC selection.
 * Lower values = better (shorter PCIe path between GPU and NIC).
 */
enum class PathType {
  PIX = 4, // Same PCIe switch (direct connection)
  PXB = 5, // Multiple PCIe bridges (same NUMA)
  PHB = 8, // Through PCIe Host Bridge (same socket, same PCI domain)
  NODE = 9, // Same NUMA node, different PCI domains (no common PCI ancestor)
  SYS = 10, // Cross-NUMA (different sockets)
  DIS = 11, // Disconnected / unknown
};

/**
 * Convert PathType enum to human-readable string.
 */
inline const char* pathTypeToString(PathType pt) {
  switch (pt) {
    case PathType::PIX:
      return "PIX";
    case PathType::PXB:
      return "PXB";
    case PathType::PHB:
      return "PHB";
    case PathType::NODE:
      return "NODE";
    case PathType::SYS:
      return "SYS";
    case PathType::DIS:
      return "DIS";
    default:
      return "UNKNOWN";
  }
}

/**
 * NIC candidate information for topology-aware selection.
 */
struct NicCandidate {
  std::string name; // Device name (e.g., "mlx5_0")
  std::string pcie; // PCIe bus ID
  PathType pathType{PathType::DIS}; // Topology path type to GPU
  int bandwidthGbps{0}; // Link bandwidth in Gb/s
  int numaNode{-1}; // NUMA node (-1 if unknown)
  int nhops{-1}; // PCIe hops between GPU and NIC (-1 if unknown)
};

/**
 * NicDiscovery - Topology-aware RDMA NIC selection for GPUs.
 *
 * Discovers and selects the best RDMA NIC for a given GPU based on
 * PCIe topology analysis (prefers closest NIC to GPU).
 *
 * Usage:
 *   NicDiscovery discovery(0);  // Discovery happens in constructor
 *   const auto& candidates = discovery.getCandidates();
 *   std::string nicName = candidates[0].name;  // Best NIC first
 *
 * This class only discovers and ranks NICs - it does not manage
 * any ibv_context*. The caller should open the selected device.
 */
class NicDiscovery {
 public:
  /**
   * Constructor - performs NIC discovery for the given CUDA device.
   *
   * Discovery runs immediately, selecting the best NIC based on
   * PCIe topology. After construction, call getCandidates()
   * to retrieve the ranked NIC list.
   *
   * @param cudaDevice CUDA device index for GPU topology analysis
   * @throws std::runtime_error if no suitable NIC found
   */
  explicit NicDiscovery(
      int cudaDevice,
      const std::vector<std::string>& ibHca = {});

  /**
   * Get all discovered NIC candidates, sorted best-to-worst.
   */
  const std::vector<NicCandidate>& getCandidates() const {
    return candidates_;
  }

  /**
   * Get the GPU's PCIe bus ID string.
   */
  const std::string& getGpuPciBusId() const {
    return gpuPciBusId_;
  }

  /**
   * Get the GPU's NUMA node.
   */
  int getGpuNumaNode() const {
    return gpuNumaNode_;
  }

  // Static utility functions

  /**
   * Normalize PCIe address to lowercase for sysfs compatibility.
   * CUDA returns uppercase (e.g., "0000:1B:00.0") but sysfs uses lowercase.
   */
  static std::string normalizePcieAddress(const std::string& pciBusId);

  /**
   * Get PCIe bus ID string from CUDA device.
   */
  static std::string getCudaPciBusId(int cudaDevice);

  /**
   * Get NUMA node for a PCIe device.
   *
   * @param pciBusId PCIe bus ID (will be normalized internally)
   * @return NUMA node number, or -1 if unknown
   */
  static int getNumaNodeForPcie(const std::string& pciBusId);

  /**
   * Get NUMA node for an IB device.
   *
   * @param devName IB device name (e.g., "mlx5_0")
   * @return NUMA node number, or -1 if unknown
   */
  static int getNumaNodeForIbDev(const char* devName);

  /**
   * Get PCIe bus ID for an IB device.
   *
   * @param devName IB device name (e.g., "mlx5_0")
   * @return PCIe bus ID string (e.g., "0000:18:00.0")
   */
  static std::string getPcieForIbDev(const char* devName);

 private:
  // CUDA device index
  int cudaDevice_;

  // GPU topology info (lazily initialized by initGpuTopology())
  std::string gpuPciBusId_;
  std::string gpuPcieNormalized_;
  std::vector<std::string> gpuAncestorChain_;
  std::unordered_set<std::string> gpuAncestors_;
  int gpuNumaNode_{-1};

  // IB HCA allowlist filter (empty = no filtering)
  std::unordered_set<std::string> ibHcaFilter_;

  // Discovered candidates (populated during discovery)
  std::vector<NicCandidate> candidates_;

  // Private helpers
  void discover();
  void initGpuTopology();
  std::pair<PathType, int> computePathType(
      const std::string& nicPcie,
      int nicNuma) const;
};

} // namespace comms::pipes
