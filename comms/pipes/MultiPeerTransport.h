// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/GpuMemHandler.h"
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

  // When true, IBGDA transport is never constructed and all non-self peers
  // are routed over NVLink. Requires all ranks in the same NVL domain.
  bool disableIb{false};
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
  /// When topo is provided, bypasses TopologyDiscovery and uses the
  /// pre-computed topology directly (primarily for unit testing).
  MultiPeerTransport(
      int myRank,
      int nRanks,
      int deviceId,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultiPeerTransportConfig& config,
      std::optional<TopologyResult> topo = std::nullopt);

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
    return ibgdaTransport_ != nullptr && peerRank != myRank_;
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

  /** @return NVL bootstrap adapter for NVL-scoped collective ops.
   *  Used by HostWindow for GpuMemHandler NVL exchange. */
  std::shared_ptr<meta::comms::IBootstrap> nvl_bootstrap() const {
    return nvlBootstrapAdapter_;
  }

  /** @return This rank's local index within the NVL peer group. */
  int nvl_local_rank() const {
    return nvlLocalRank_;
  }

  /** @return Number of ranks in the NVL peer group (including self). */
  int nvl_n_ranks() const {
    return nvlNRanks_;
  }

  /** @return NVL local rank for the given global rank.
   *  @throws std::out_of_range if globalRank is not in the NVL group. */
  int global_to_nvl_local(int globalRank) const {
    return globalToNvlLocal_.at(globalRank);
  }

  // --- External buffer configuration ---

  /**
   * Set external NVL data buffers for reuse instead of internal allocation.
   *
   * Call BEFORE exchange(). Delegates to
   * MultiPeerNvlTransport::setExternalDataBuffers().
   */
  void setExternalNvlDataBuffers(ExternalStagingBuffers externalStagingBuffers);

  // --- Host-side transport accessors ---

  /**
   * @param globalPeerRank Global rank of the NVL peer.
   * @return Pointer to P2pNvlTransportDevice on device memory for the given
   * peer.
   */
  P2pNvlTransportDevice get_p2p_nvl_transport_device(int globalPeerRank) const;

  /**
   * @return Pointer to the device-side Transport array from NVL transport,
   *   indexed by NVL local rank. Returns nullptr if no NVL transport.
   */
  Transport* /*nullable*/ get_nvl_transports_array() const;

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
  IbgdaLocalBuffer localRegisterIbgdaBuffer(void* ptr, size_t size);

  /**
   * Deregister a previously registered IBGDA buffer.
   *
   * @param ptr Pointer to the buffer to deregister
   */
  void localDeregisterIbgdaBuffer(void* ptr);

  /**
   * Collectively exchange IBGDA buffer info with all peers.
   *
   * COLLECTIVE OPERATION: All ranks MUST call this with their local buffer.
   * Returns remote buffer info for all IBGDA peers.
   *
   * @param localBuf Local buffer registered with localRegisterIbgdaBuffer()
   * @return Vector of remote buffers, one per IBGDA peer (size = nRanks - 1)
   */
  std::vector<IbgdaRemoteBuffer> exchangeIbgdaBuffer(
      const IbgdaLocalBuffer& localBuf);

  /**
   * Collectively exchange a user-provided GPU buffer with NVL peers via IPC.
   *
   * COLLECTIVE OPERATION: All NVL ranks MUST call this with their buffer.
   * Supports both cudaMalloc'd and cuMem-allocated buffers (e.g. from
   * ncclMemAlloc). Three exchange paths are auto-detected:
   * - cudaMalloc buffers: cudaIpcMemHandle path
   * - cuMem with CU_MEM_HANDLE_TYPE_FABRIC: fabric handle path
   * - cuMem with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR (no fabric):
   *   POSIX FD path via pidfd_getfd (Linux 5.6+, intra-host only)
   *
   * @param localPtr GPU pointer (cudaMalloc or ncclMemAlloc)
   * @param size Size of the buffer in bytes
   * @return Vector of mapped peer pointers (size = nvlNRanks_), indexed by
   *         NVL local rank. Self entry is the original localPtr. Other entries
   *         are IPC-mapped pointers to peer buffers.
   */
  std::vector<void*> exchangeNvlBuffer(void* localPtr, std::size_t size);

  /**
   * Unmap NVL IPC-mapped peer buffers obtained from exchangeNvlBuffer().
   *
   * @param mappedPtrs Vector returned by exchangeNvlBuffer()
   */
  void unmapNvlBuffers(const std::vector<void*>& mappedPtrs);

 private:
  const int myRank_;
  const int nRanks_;
  const int deviceId_;
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;

  // --- Topology (populated in constructor) ---
  std::vector<int> nvlPeerRanks_;
  std::vector<int> ibgdaPeerRanks_;
  std::vector<TransportType> typePerRank_;

  // --- NVLink rank mapping ---
  std::unordered_map<int, int> globalToNvlLocal_;
  int nvlLocalRank_{-1};
  int nvlNRanks_{0};

  // --- Sub-transports ---
  std::shared_ptr<meta::comms::IBootstrap> nvlBootstrapAdapter_;
  std::unique_ptr<MultiPeerNvlTransport> nvlTransport_;
  std::unique_ptr<MultipeerIbgdaTransport> ibgdaTransport_;

  // --- GPU-allocated transport array for device handle ---
  Transport* transportsGpu_{nullptr};
  bool deviceHandleBuilt_{false};

  // --- Private helpers ---
  void initFromTopology(
      TopologyResult topo,
      const MultiPeerTransportConfig& config);
  void build_device_handle();
  void free_device_handle();

  // Memory type detection for exchangeNvlBuffer tri-path support.
  enum class NvlMemMode { kCudaIpc, kFabric, kPosixFd };
  NvlMemMode detectNvlMemMode(void* ptr) const;

  // Handle exchange helpers
  std::vector<void*> exchangeNvlBufferCudaIpc(void* localPtr);
  std::vector<void*> exchangeNvlBufferFabric(void* localPtr, std::size_t size);
  std::vector<void*> exchangeNvlBufferPosixFd(void* localPtr, std::size_t size);

  // Track NVL exchange state for proper cleanup in unmapNvlBuffers
  struct NvlExchangeRecord {
    NvlMemMode mode;
    // cuMem state for cleanup (used by kFabric and kPosixFd paths):
    std::vector<CUdeviceptr> cuMemPeerPtrs;
    std::vector<CUmemGenericAllocationHandle> cuMemPeerAllocHandles;
    std::vector<size_t> cuMemPeerSizes;
    // POSIX FD exported by this rank — kept open until unmap so that peers
    // can complete pidfd_getfd imports before the fd is closed.
    int localExportedFd{-1};
  };
  // Keyed by the mappedPtrs vector's data pointer (first element address)
  std::unordered_map<void*, NvlExchangeRecord> nvlExchangeRecords_;
};

} // namespace comms::pipes
