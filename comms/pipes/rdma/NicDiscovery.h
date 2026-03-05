// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "comms/pipes/rdma/IbHcaParser.h"

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
 * Get the NUMA node of the calling thread via getcpu(2) syscall.
 *
 * @return NUMA node number, or -1 on failure
 */
int getCurrentNumaNode();

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
 * NicDiscovery - Base class for topology-aware RDMA NIC selection.
 *
 * Discovers and selects the best RDMA NIC based on PCIe/NUMA topology
 * analysis. Subclasses define the topology ranking strategy:
 *
 *   // GPU-anchored: fine-grained PCIe topology ranking
 *   GpuNicDiscovery discovery(0);
 *
 *   // CPU-anchored: NUMA affinity ranking
 *   CpuNicDiscovery discovery(getCurrentNumaNode());
 *
 * This class only discovers and ranks NICs - it does not manage
 * any ibv_context*. The caller should open the selected device.
 */
class NicDiscovery {
 public:
  virtual ~NicDiscovery() = default;

  /**
   * Get all discovered NIC candidates, sorted best-to-worst.
   */
  const std::vector<NicCandidate>& getCandidates() const {
    return candidates_;
  }

  /**
   * Get the anchor's NUMA node.
   * For GPU-anchored: the GPU's NUMA node.
   * For CPU-anchored: the NUMA node passed to the constructor.
   */
  int getAnchorNumaNode() const {
    return anchorNumaNode_;
  }

  // Static utility functions

  /**
   * Normalize PCIe address to lowercase for sysfs compatibility.
   * CUDA returns uppercase (e.g., "0000:1B:00.0") but sysfs uses lowercase.
   */
  static std::string normalizePcieAddress(const std::string& pciBusId);

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

 protected:
  // Protected constructor — only subclasses construct.
  explicit NicDiscovery(const std::string& ibHcaEnv);

  // Anchor NUMA node (GPU's NUMA for GPU-anchored, pass-in for CPU)
  int anchorNumaNode_{-1};

  // IB HCA filter (empty = no filtering)
  IbHcaParser ibHcaParser_;

  // Discovered candidates (populated during discovery)
  std::vector<NicCandidate> candidates_;

  /**
   * Discover and rank NICs. Called by subclass constructors after
   * subclass-specific initialization is complete.
   */
  void discover();

  /**
   * Compute the path type between the anchor device and a NIC.
   * Subclasses override to implement their ranking strategy.
   *
   * @param nicPcie NIC's PCIe bus ID
   * @param nicNuma NIC's NUMA node
   * @return (PathType, hop count) pair
   */
  virtual std::pair<PathType, int> computePathType(
      const std::string& nicPcie,
      int nicNuma) const = 0;

  /**
   * Return a description of the anchor for log messages.
   * E.g., "GPU 0000:1B:00.0" or "CPU NUMA 0".
   */
  virtual std::string anchorDescription() const = 0;
};

/**
 * GpuNicDiscovery - GPU-anchored NIC selection using PCIe topology.
 *
 * Ranks NICs by PCIe ancestor walk from the CUDA GPU device,
 * producing fine-grained path types (PIX/PXB/PHB/NODE/SYS).
 */
class GpuNicDiscovery : public NicDiscovery {
 public:
  /**
   * Create a GPU-anchored NIC discovery.
   *
   * @param cudaDevice CUDA device index
   * @param ibHcaEnv NCCL_IB_HCA-style filter string (empty = no filtering)
   * @throws std::runtime_error if no suitable NIC found
   */
  explicit GpuNicDiscovery(int cudaDevice, const std::string& ibHcaEnv = {});

  /**
   * Get the anchor GPU's PCIe bus ID string.
   */
  const std::string& getAnchorPciBusId() const {
    return anchorPciBusId_;
  }

  /**
   * Get PCIe bus ID string from CUDA device.
   */
  static std::string getCudaPciBusId(int cudaDevice);

 private:
  void initGpuTopology();

  std::pair<PathType, int> computePathType(
      const std::string& nicPcie,
      int nicNuma) const override;

  std::string anchorDescription() const override;

  int cudaDevice_;
  std::string anchorPciBusId_;
  std::vector<std::string> anchorAncestorChain_;
  std::unordered_set<std::string> anchorAncestors_;
};

/**
 * CpuNicDiscovery - CPU-anchored NIC selection using NUMA affinity.
 *
 * Ranks NICs by NUMA affinity to the given node (NODE for same-NUMA,
 * SYS for cross-NUMA). No CUDA dependency.
 */
class CpuNicDiscovery : public NicDiscovery {
 public:
  /**
   * Create a CPU-anchored NIC discovery.
   *
   * The NUMA node is validated against sysfs; throws
   * std::invalid_argument if the node does not exist.
   *
   * @param numaNode NUMA node to anchor on
   * @param ibHcaEnv NCCL_IB_HCA-style filter string (empty = no filtering)
   * @throws std::runtime_error if no suitable NIC found
   * @throws std::invalid_argument if numaNode is invalid
   */
  explicit CpuNicDiscovery(int numaNode, const std::string& ibHcaEnv = {});

 private:
  std::pair<PathType, int> computePathType(
      const std::string& nicPcie,
      int nicNuma) const override;

  std::string anchorDescription() const override;
};

} // namespace comms::pipes
