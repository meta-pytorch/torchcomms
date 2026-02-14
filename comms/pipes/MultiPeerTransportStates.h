// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

// Forward declaration — include MultiPeerDeviceHandle.cuh to use
// get_device_handle()
struct MultiPeerDeviceHandle;

struct MultiPeerTransportStatesConfig {
  MultiPeerNvlTransportConfig nvlConfig;
  MultipeerIbgdaTransportConfig ibgdaConfig;
};

/**
 * MultiPeerTransportStates - Host-side wrapper unifying NVLink, IBGDA, and
 * Self transports.
 *
 * IBGDA is the universal transport created for ALL non-self peers.
 * NVL is additionally created for NVLink-connected peers and is preferred
 * when available. get_transport_type() returns the preferred transport.
 *
 * Construction:
 *   1. Discovers topology (NVLink peers) via bootstrap allGather
 *      + cudaDeviceCanAccessPeer
 *   2. Creates MultiPeerNvlTransport for NVLink-reachable peers
 *      (using IntraNodeBootstrapAdapter for local rank mapping)
 *   3. Always creates MultipeerIbgdaTransport for ALL peers
 *      (using full global rank space)
 *
 * Usage:
 *   auto states = MultiPeerTransportStates(myRank, nRanks, bootstrap, config);
 *   states.exchange();                            // COLLECTIVE
 *   auto handle = states.get_device_handle();     // For kernels
 */
class MultiPeerTransportStates {
 public:
  MultiPeerTransportStates(
      int myRank,
      int nRanks,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      const MultiPeerTransportStatesConfig& config);

  ~MultiPeerTransportStates();

  // Non-copyable, non-movable
  MultiPeerTransportStates(const MultiPeerTransportStates&) = delete;
  MultiPeerTransportStates& operator=(const MultiPeerTransportStates&) = delete;
  MultiPeerTransportStates(MultiPeerTransportStates&&) = delete;
  MultiPeerTransportStates& operator=(MultiPeerTransportStates&&) = delete;

  /**
   * COLLECTIVE: exchanges NVLink memory handles and IBGDA RDMA info.
   * All nRanks must call this.
   */
  void exchange();

  // --- Topology queries ---

  /** @return Preferred transport type for the given peer rank. */
  TransportType get_transport_type(int peerRank) const;

  /** @return True if peerRank is reachable via NVLink. */
  bool is_nvl_peer(int peerRank) const;

  /** @return True if IBGDA is the preferred transport for peerRank. */
  bool is_ibgda_peer(int peerRank) const;

  /** @return True if IBGDA transport is available for peerRank (all non-self).
   */
  bool has_ibgda(int peerRank) const {
    return peerRank != myRank_;
  }

  /** @return True if IBGDA is the preferred transport (no NVL available). */
  bool prefers_ibgda(int peerRank) const {
    return typePerRank_[peerRank] == TransportType::P2P_IBGDA;
  }

  /** @return This rank's global rank index. */
  int my_rank() const {
    return myRank_;
  }

  /** @return Total number of ranks in the communicator. */
  int n_ranks() const {
    return nRanks_;
  }

  /** @return Global ranks of NVL peers (excluding self). */
  const std::vector<int>& nvl_peer_ranks() const {
    return nvlPeerRanks_;
  }

  /** @return Global ranks of all non-self peers (IBGDA covers everyone). */
  const std::vector<int>& ibgda_peer_ranks() const {
    return ibgdaPeerRanks_;
  }

  // --- Host-side transport accessors ---

  /**
   * @param globalPeerRank Global rank of the NVL peer.
   * @return P2pNvlTransportDevice handle (by value) for the given peer.
   */
  P2pNvlTransportDevice get_p2p_nvl_transport_device(int globalPeerRank) const;

  /**
   * @param globalPeerRank Global rank of the IBGDA peer.
   * @return Non-owning pointer to GPU-allocated P2pIbgdaTransportDevice.
   */
  P2pIbgdaTransportDevice* get_p2p_ibgda_transport_device(
      int globalPeerRank) const;

  /** @return A stateless P2pSelfTransportDevice handle. */
  P2pSelfTransportDevice get_p2p_self_transport_device() const;

  // --- Device handle (for passing to kernels) ---

  /**
   * @return MultiPeerDeviceHandle suitable for passing to CUDA kernels.
   * @throws std::runtime_error if exchange() has not been called.
   */
  MultiPeerDeviceHandle get_device_handle() const;

 private:
  const int myRank_;
  const int nRanks_;
  std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap_;

  // --- Topology (populated in constructor) ---
  std::vector<int> nvlPeerRanks_;
  std::vector<int> ibgdaPeerRanks_;
  std::vector<TransportType> typePerRank_;

  // --- NVLink rank mapping ---
  std::unordered_map<int, int> globalToNvlLocal_;
  int nvlLocalRank_{-1};
  int nvlNRanks_{0};

  // --- Sub-transports ---
  std::shared_ptr<ctran::bootstrap::IBootstrap> nvlBootstrapAdapter_;
  std::unique_ptr<MultiPeerNvlTransport> nvlTransport_;
  std::unique_ptr<MultipeerIbgdaTransport> ibgdaTransport_;

  // --- GPU-allocated arrays for device handle ---
  P2pNvlTransportDevice* nvlTransportsGpu_{nullptr};
  TransportType* typePerRankGpu_{nullptr};
  int* globalToNvlIndexGpu_{nullptr};
  int* globalToIbgdaIndexGpu_{nullptr};
  bool deviceHandleBuilt_{false};

  // --- Private helpers ---
  void discover_topology();
  void build_device_handle();
  void free_device_handle();
};

} // namespace comms::pipes
