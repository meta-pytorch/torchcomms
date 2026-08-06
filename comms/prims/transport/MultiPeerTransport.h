// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. On AMD,
// `comms/prims/memory/GpuMemHandler.h` (included below) brings in the HIP
// runtime headers under `#ifdef __HIP_PLATFORM_AMD__` and provides the
// stand-in types needed for the CUDA driver types referenced via
// `NvlMemExchange.h` (`NvlPeerMem`).
#ifndef __HIP_PLATFORM_AMD__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/memory/NvlMemExchange.h"
#include "comms/prims/topology/TopologyDiscovery.h"
#include "comms/prims/transport/IbTransportConfig.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"
#include "comms/prims/transport/self/P2pSelfTransportDevice.cuh"

namespace comms::prims {

namespace detail {

enum class PrimsChannelMode : uint32_t {
  kEager = 0,
  kLazyPrefix = 1,
};

enum class PrimsTransportRoute : uint8_t {
  kSelf = 0,
  kNvl = 1,
  kIbgda = 2,
  kIbrc = 3,
};

struct ChannelProtocolRecord {
  PrimsChannelMode mode{PrimsChannelMode::kEager};
  uint32_t channelCapacity{0};

  bool operator==(const ChannelProtocolRecord&) const = default;
};

static_assert(sizeof(ChannelProtocolRecord) == 8);
static_assert(std::is_trivially_copyable_v<ChannelProtocolRecord>);

void validateChannelProtocolRecords(
    std::span<const ChannelProtocolRecord> records);

void exchangeAndValidateChannelProtocol(
    meta::comms::IBootstrap& bootstrap,
    int rank,
    int nRanks,
    const ChannelProtocolRecord& localRecord);

void validatePrimsTransportRoutes(
    std::span<const PrimsTransportRoute> routeMatrix,
    int nRanks);

void exchangeAndValidatePrimsTransportRoutes(
    meta::comms::IBootstrap& bootstrap,
    int rank,
    int nRanks,
    std::span<const PrimsTransportRoute> localRoutes);

} // namespace detail

struct PeerChannelDemand {
  int peerRank;
  uint32_t ibChannels;
};

struct MultiPeerTransportConfig {
  MultiPeerNvlTransportConfig nvlConfig;
  MultipeerIbTransportConfig ibConfig;

  // Selects the IB backend for non-NVL peers. Default remains IBGDA.
  IbBackendMode ibMode{IbBackendMode::kIbgda};

  // MNNVL topology overrides for UUID and clique ID.
  // See TopologyConfig for field-level documentation.
  TopologyConfig topoConfig;

  // When true, no IB transport is constructed and all non-self peers
  // are routed over NVLink. Requires all ranks in the same NVL domain.
  bool disableIb{false};
};

/**
 * MultiPeerTransport - Host-side wrapper unifying NVLink, IBGDA, IBRC, and
 * Self transports.
 *
 * NVL is created for NVLink-connected peers. The selected IB backend is
 * created when at least one peer prefers IB; IBGDA can then also serve any
 * non-self peer as an explicit fallback. get_transport_type() returns the
 * preferred transport.
 *
 * Construction:
 *   1. Discovers topology (NVLink peers) via bootstrap allGather
 *      + cudaDeviceCanAccessPeer
 *   2. Creates MultiPeerNvlTransport for NVLink-reachable peers
 *      (using NvlBootstrapAdapter for local rank mapping)
 *   3. Creates the selected IB backend when the topology has IB peers
 *      (using the full global rank space)
 *
 * Usage:
 *   auto transport = MultiPeerTransport(myRank, nRanks, deviceId, bootstrap,
 * config); transport.exchange(); // COLLECTIVE
 * handle = transport.get_device_handle(demands); // For kernels
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
   * COLLECTIVE: exchanges NVLink memory handles and IB RDMA info.
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

  /** @return Global ranks whose preferred transport is IB. */
  const std::vector<int>& ib_peer_ranks() const {
    return ibPeerRanks_;
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
   * @return True after collective multimem initialization succeeds.
   *
   * This is a cached local query and never starts a collective operation.
   */
  bool has_multimem_nvl_transport() const;

  /**
   * Collectively initialize the multimem NVL transport when all ranks are
   * eligible. Returns false for disabled/ineligible communicators so the
   * dispatcher can select a fallback algorithm.
   *
   * PRECONDITION: all NVL ranks call this in lockstep.
   * @throws std::runtime_error on bootstrap or multicast setup failure.
   */
  bool initialize_multimem_nvl_transport() const;

  /**
   * Return the device handle for the copy-based (staging) multimem NVL
   * transport. Delegates to
   * MultiPeerNvlTransport::getMultimemNvlTransportDevice(). Used by the cnvlmm
   * staging path.
   *
   * Call initialize_multimem_nvl_transport() collectively before this cached
   * getter. It throws when initialization has not succeeded.
   *
   * This getter is local and never touches bootstrap.
   *
   * @throws std::runtime_error if no NVL transport, multimem NVL is not
   * initialized.
   */
  MultimemNvlTransportDevice get_multimem_nvl_transport_device() const;

  /**
   * @param globalPeerRank Global rank of the IBGDA peer.
   * @return Non-owning pointer to a prepared GPU transport slot.
   * @throws std::runtime_error if get_device_handle() has not prepared the
   * peer.
   */
  P2pIbgdaTransportDevice* get_p2p_ibgda_transport_device(
      int globalPeerRank) const;

  /** @return A stateless P2pSelfTransportDevice handle. */
  P2pSelfTransportDevice get_p2p_self_transport_device() const;

  // --- Device handle (for passing to kernels) ---

  /**
   * Compatibility overload for callers without channel geometry. Each valid
   * IB peer is materialized at full configured capacity.
   */
  MultiPeerDeviceHandle get_device_handle(const std::vector<int>& peers);

  /**
   * Ensure the requested peer/channel prefixes are ready and return the
   * stable device handle.
   *
   * A positive demand performs the peer's first connection when needed and
   * grows an existing connection otherwise. Duplicate peers use their maximum
   * demand and are processed by rank.
   * Both endpoints must demand each IB edge in the same connect round.
   * Channel-eager mode promotes every positive demand to full capacity.
   * Capture-time growth completes on graph-external device work before this
   * call returns; graph replay performs no allocation or connection.
   */
  MultiPeerDeviceHandle get_device_handle(
      std::span<const PeerChannelDemand> demands);

  /** @return Configured IB channel capacity for each peer. */
  uint32_t ib_channel_capacity() const {
    return ibChannelCapacity_;
  }

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

  IbgdaLocalBuffer allocateIbCounterBuffer(std::size_t size, void** hostPtr);
  IbgdaLocalBuffer registerIbCounterBuffer(
      const IbgdaLocalBuffer& buffer,
      std::size_t size);
  void freeIbCounterBuffer(IbgdaLocalBuffer& buffer, void*& hostPtr) noexcept;

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
   * @param localPtr GPU pointer (cudaMalloc or ncclMemAlloc). The actual
   *                 buffer length is derived from the allocation itself
   *                 (via `cuMemGetAddressRange` on the VMM path) -- callers
   *                 do not need to pass a size.
   * @return Vector of mapped peer pointers (size = nvlNRanks_), indexed by
   *         NVL local rank. Self entry is the original localPtr. Other entries
   *         are IPC-mapped pointers to peer buffers.
   */
  std::vector<void*> exchangeNvlBuffer(void* localPtr);

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
  std::vector<int> ibPeerRanks_;
  std::vector<TransportType> typePerRank_;

  // --- NVLink rank mapping ---
  std::unordered_map<int, int> globalToNvlLocal_;
  int nvlLocalRank_{-1};
  int nvlNRanks_{0};

  // --- Sub-transports ---
  std::shared_ptr<meta::comms::IBootstrap> nvlBootstrapAdapter_;
  std::unique_ptr<MultiPeerNvlTransport> nvlTransport_;
  // Exactly one IB backend is constructed, selected by MultiPeerTransportConfig
  // ::ibMode (kIbgda by default, kIbrc selects the CPU-proxy skeleton backend).
  // IBRC functional entry points fail fast until the backend is implemented.
  std::unique_ptr<MultipeerIbgdaTransport> ibgdaTransport_;
  std::unique_ptr<MultipeerIbrcTransport> ibrcTransport_;
  detail::PrimsChannelMode channelMode_{detail::PrimsChannelMode::kEager};
  uint32_t ibChannelCapacity_{0};

  // --- GPU-allocated transport array for device handle ---
  Transport* transportsGpu_{nullptr};
  bool deviceHandleBuilt_{false};

  // --- Private helpers ---
  void initFromTopology(
      TopologyResult topo,
      const MultiPeerTransportConfig& config);
  MultiPeerDeviceHandle make_device_handle() const;
  void materializePeerChannels(std::span<const PeerChannelDemand> demands);
  void build_device_handle();
  void free_device_handle();

  // Memory type detection for exchangeNvlBuffer tri-path support.
  enum class NvlMemMode { kCudaIpc, kFabric, kPosixFd };
  NvlMemMode detectNvlMemMode(void* ptr) const;

  // Track NVL exchange state for proper cleanup in unmapNvlBuffers. The
  // NvlPeerMem owns the peer memory state produced by the NvlMemExchange
  // helpers: for VMM modes its `vmmMappings` (RAII peer VAs) and
  // `vmmPeerHandles` (imported handles released via cuMemRelease), and for
  // cudaIpc mode the peer pointers in `peerPtrs` (closed via
  // cudaIpcCloseMemHandle).
  struct NvlExchangeRecord {
    NvlMemMode mode{NvlMemMode::kCudaIpc};
    NvlPeerMem mem;
  };
  // Keyed by the self (local) pointer, i.e. mappedPtrs[nvlLocalRank_].
  std::unordered_map<void*, NvlExchangeRecord> nvlExchangeRecords_;
};

} // namespace comms::prims
