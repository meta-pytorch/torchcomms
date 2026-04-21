// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// MultiPeerTransportAmd — Unified NVL + IBGDA transport for AMD GPUs
// =============================================================================
//
// AMD equivalent of comms::pipes::MultiPeerTransport.
// Auto-detects topology and creates:
//   - P2P_NVL (IPC/XGMI) transport for same-node peers
//   - P2P_IBGDA_AMD (RDMA) transport for cross-node peers
//   - SELF transport for self-rank
//
// Single-node: all peers use NVL (no IBGDA needed)
// Multi-node:  local peers use NVL, remote peers use IBGDA

#pragma once

#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/amd/MultiPeerNvlTransportAmd.h"

// Forward declaration to avoid including IBGDA headers
namespace pipes_gda {
class MultipeerIbgdaTransportAmd;
struct MultipeerIbgdaTransportAmdConfig;
} // namespace pipes_gda

namespace comms::pipes {

struct MultiPeerTransportAmdConfig {
  // NVL transport config
  MultiPeerNvlTransportAmdConfig nvlConfig;

  // IBGDA transport config (only used for multi-node)
  int hipDevice{0};
  uint32_t ibgdaQpDepth{128};
};

class MultiPeerTransportAmd {
 public:
  MultiPeerTransportAmd(
      int myRank,
      int nRanks,
      int localRank,
      int localSize,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultiPeerTransportAmdConfig& config);

  ~MultiPeerTransportAmd();

  MultiPeerTransportAmd(const MultiPeerTransportAmd&) = delete;
  MultiPeerTransportAmd& operator=(const MultiPeerTransportAmd&) = delete;

  // Collective: all ranks must call
  void exchange();

  // Get unified device-accessible Transport array (one per rank)
  DeviceSpan<Transport> getDeviceTransports();

  int getMyRank() const {
    return myRank_;
  }
  int getNRanks() const {
    return nRanks_;
  }
  bool isHybrid() const {
    return !remotePeerRanks_.empty();
  }
  const std::vector<int>& getLocalPeerRanks() const {
    return localPeerRanks_;
  }
  const std::vector<int>& getRemotePeerRanks() const {
    return remotePeerRanks_;
  }
  pipes_gda::MultipeerIbgdaTransportAmd* getIbgdaTransport() const {
    return ibgdaTransport_.get();
  }

 private:
  void detectTopology();
  void initializeTransportsArray();

  const int myRank_;
  const int nRanks_;
  const int localRank_;
  const int localSize_;
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  MultiPeerTransportAmdConfig config_;

  // Topology
  std::vector<int> localPeerRanks_; // same-node peers (NVL)
  std::vector<int> remotePeerRanks_; // cross-node peers (IBGDA)

  // Sub-transports
  std::unique_ptr<MultiPeerNvlTransportAmd> nvlTransport_;
  std::unique_ptr<pipes_gda::MultipeerIbgdaTransportAmd> ibgdaTransport_;

  // Unified device Transport array
  void* transportsDevice_{nullptr};
  bool initialized_{false};
};

} // namespace comms::pipes
