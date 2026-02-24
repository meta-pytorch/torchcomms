// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

// Forward declaration — include MultiPeerDeviceHandle.cuh to use
// get_device_handle()
struct MultiPeerDeviceHandle;

struct MultiPeerTransportConfig {
  MultiPeerNvlTransportConfig nvlConfig;
  MultipeerIbgdaTransportConfig ibgdaConfig;

  // MNNVL topology overrides for UUID and clique ID.
  // See TopologyConfig for field-level documentation.
  TopologyConfig topoConfig;
};

/**
 * MultiPeerTransport - Host-side wrapper unifying NVLink, IBGDA, and
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
 *      (using NvlBootstrapAdapter for local rank mapping)
 *   3. Always creates MultipeerIbgdaTransport for ALL peers
 *      (using full global rank space)
 *
 * Usage:
 *   auto transport = MultiPeerTransport(myRank, nRanks, deviceId, bootstrap,
 * config); transport.exchange();                            // COLLECTIVE auto
 * handle = transport.get_device_handle();     // For kernels
 */
class MultiPeerTransport {
 public:
  MultiPeerTransport(
      int myRank,
      int nRanks,
      int deviceId,
      std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
      const MultiPeerTransportConfig& config);

  ~MultiPeerTransport();

  // Non-copyable, non-movable
  MultiPeerTransport(const MultiPeerTransport&) = delete;
  MultiPeerTransport& operator=(const MultiPeerTransport&) = delete;
  MultiPeerTransport(MultiPeerTransport&&) = delete;
  MultiPeerTransport& operator=(MultiPeerTransport&&) = delete;

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

  /** @return This rank's local index within the NVL peer group. */
  int nvl_local_rank() const {
    return nvlLocalRank_;
  }

  /** @return Number of ranks in the NVL peer group (including self). */
  int nvl_n_ranks() const {
    return nvlNRanks_;
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

  // --- IBGDA buffer registration (delegates to ibgdaTransport_) ---

  /**
   * Register a user-provided buffer for IBGDA RDMA access.
   *
   * @param ptr Pointer to GPU memory
   * @param size Size of the buffer in bytes
   * @return IbgdaLocalBuffer with valid lkey for local RDMA operations
   * @throws std::runtime_error if no IBGDA transport or registration fails
   */
  IbgdaLocalBuffer registerIbgdaBuffer(void* ptr, size_t size);

  /**
   * Deregister a previously registered IBGDA buffer.
   *
   * @param ptr Pointer to the buffer to deregister
   */
  void deregisterIbgdaBuffer(void* ptr);

  /**
   * Collectively exchange IBGDA buffer info with all peers.
   *
   * COLLECTIVE OPERATION: All ranks MUST call this with their local buffer.
   * Returns remote buffer info for all IBGDA peers.
   *
   * @param localBuf Local buffer registered with registerIbgdaBuffer()
   * @return Vector of remote buffers, one per IBGDA peer (size = nRanks - 1)
   */
  std::vector<IbgdaRemoteBuffer> exchangeIbgdaBuffer(
      const IbgdaLocalBuffer& localBuf);

  // --- NVL recv buffer IPC exchange (for zero-copy variant) ---

  /**
   * Collectively exchange NVL buffer pointers within the NVL peer group.
   *
   * COLLECTIVE OPERATION: All NVL ranks MUST call this with their local buffer.
   * Uses GpuMemHandler's IPC exchange pattern (cudaIpcGetMemHandle /
   * cudaIpcOpenMemHandle or fabric handles on GB200).
   *
   * The returned vector is indexed by NVL peer index (not global rank).
   * Use nvl_peer_ranks() to map NVL indices to global ranks.
   *
   * @param localBuf Local GPU buffer pointer (allocated with cudaMalloc or
   * similar)
   * @param size Size of the buffer in bytes
   * @return Vector of IPC-mapped peer buffer pointers (size = nvlNRanks_)
   *         Entry at nvlLocalRank_ is localBuf (not IPC-mapped).
   */
  std::vector<void*> exchangeNvlBuffer(void* localBuf, size_t size);

  /**
   * Unmap previously exchanged NVL buffers.
   *
   * Call this before freeing the local buffer to clean up IPC mappings.
   *
   * @param mappedPtrs Vector returned from exchangeNvlBuffer()
   */
  void unmapNvlBuffers(const std::vector<void*>& mappedPtrs);

 private:
  const int myRank_;
  const int nRanks_;
  const int deviceId_;
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

  // --- GPU-allocated transport array for device handle ---
  Transport* transportsGpu_{nullptr};
  bool deviceHandleBuilt_{false};

  // --- Private helpers ---
  void build_device_handle();
  void free_device_handle();
};

} // namespace comms::pipes
