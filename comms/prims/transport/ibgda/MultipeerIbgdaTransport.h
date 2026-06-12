// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/ibverbx/Ibvcore.h"

// Host-side DOCA APIs are NVIDIA-only. On AMD, `DocaCompat.h`
// translates both device-side and host-side `doca_*` symbols to the
// `pipes_gda_*` APIs implemented in `amd/pipes_gda/PipesGdaHost.{h,cc}`,
// backed by HSA + raw mlx5dv + libibverbs.
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/DocaCompat.h"
#else
#include <doca_gpunetio_host.h>
#endif

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/transport/MultiPeerIbTransport.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#ifndef __HIP_PLATFORM_AMD__
#include "comms/utils/CudaRAII.h"
#endif

// Forward declarations for device types (defined in .cuh files)
namespace comms::prims {
class P2pIbgdaTransportDevice;
struct MultipeerIbgdaDeviceTransport;
struct P2pIbgdaTransportBuildParams;
struct PeerQpPayload;
struct PeerBufferPayload;
struct PeerBufferSizes;
} // namespace comms::prims

namespace comms::prims {

// AddressFamily and the shared transport config now live in
// MultiPeerIbTransport.h. Keep the historical name as an alias so callers and
// the ctor signature below are unchanged.
using MultipeerIbgdaTransportConfig = MultipeerIbTransportConfig;

// The exchange wire structs and allGather caps (kMaxRanksForAllGather,
// kMaxQpsPerPeerPerNic) now live in MultiPeerIbTransport.h so they are shared
// across IB backends. Keep the historical IBGDA names as aliases so callers and
// the .cc are unchanged.
using IbgdaTransportExchInfo = IbTransportExchInfo;
using IbgdaTransportExchInfoAll = IbTransportExchInfoAll;

/**
 * MultipeerIbgdaTransport - Host-side multi-peer RDMA transport manager
 *
 * Manages GPU-initiated RDMA (IBGDA) communication across multiple ranks using
 * DOCA GPUNetIO high-level APIs. This transport enables CUDA kernels to
 * directly issue RDMA operations without CPU involvement.
 *
 * ARCHITECTURE:
 * =============
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  Host Control Path                                                  │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  MultipeerIbgdaTransport (this class)                               │
 *   │  ├── IbvDevice (RDMA device management)                             │
 *   │  ├── IbvPd (Protection Domain)                                      │
 *   │  ├── IbvMr[] (Memory regions - data per peer)                       │
 *   │  ├── doca_gpu (GPU context for DOCA)                                │
 *   │  ├── doca_gpu_verbs_qp[] (High-level QPs per peer)                  │
 *   │  └── IBootstrap (Collective exchange)                               │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  GPU Data Path                                                      │
 *   ├─────────────────────────────────────────────────────────────────────┤
 *   │  MultipeerIbgdaDeviceTransport (returned by getDeviceTransport())   │
 *   │  └── P2pIbgdaTransportDevice[] (per-peer handles)                   │
 *   │      ├── doca_gpu_dev_verbs_qp* (GPU QP handle)                     │
 *   │      └── put() / wait_local() device methods                        │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * USAGE:
 * ======
 *
 *   // Host setup
 *   MultipeerIbgdaTransportConfig config{
 *       .cudaDevice = 0,
 *       .dataBufferSize = 1 << 20,  // 1 MB per peer
 *   };
 *   MultipeerIbgdaTransport transport(myRank, nRanks, bootstrap, config);
 *   transport.exchange();  // Collective - all ranks must call
 *
 *   // Get device handle for kernel (requires including .cuh header)
 *   auto* deviceTransportPtr = transport.getDeviceTransportPtr();
 *
 * NIC SELECTION:
 * ==============
 *
 * The transport selects the RDMA NIC in order of priority:
 * 1. config.gpuNicMap - explicit GPU-to-NIC mapping (map from GPU index to NIC
 * names)
 * 2. Auto-discovery - selects the NIC with closest NUMA affinity to the GPU
 *    (optionally filtered by config.ibHca allowlist)
 *
 * COMMUNICATOR SEMANTICS:
 * =======================
 *
 * - Constructor: Local operation (allocates resources)
 * - exchange(): COLLECTIVE operation (all ranks must call)
 * - getDeviceTransportPtr(): Local operation (after exchange completes)
 */
class MultipeerIbgdaTransport
    : public MultiPeerIbTransport<MultipeerIbgdaTransport> {
 public:
  /**
   * Constructor - Initialize multi-peer IBGDA transport
   */
  MultipeerIbgdaTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbgdaTransportConfig& config);

  /**
   * Destructor - Release all resources
   */
  ~MultipeerIbgdaTransport();

  // Non-copyable, non-movable
  MultipeerIbgdaTransport(const MultipeerIbgdaTransport&) = delete;
  MultipeerIbgdaTransport& operator=(const MultipeerIbgdaTransport&) = delete;
  MultipeerIbgdaTransport(MultipeerIbgdaTransport&&) = delete;
  MultipeerIbgdaTransport& operator=(MultipeerIbgdaTransport&&) = delete;

  /**
   * exchange - Exchange connection info and connect QPs
   *
   * COLLECTIVE OPERATION: All ranks MUST call this before using
   * getDeviceTransportPtr().
   */
  void exchange();

  /**
   * getDeviceTransport - Get multi-peer device transport wrapper
   *
   * Returns a MultipeerIbgdaDeviceTransport wrapper that provides convenient
   * access to per-peer transport handles with rank-to-index mapping.
   * Use .get(peerRank) to get the transport for a specific peer.
   *
   * NOTE: Requires including MultipeerIbgdaDeviceTransport.cuh in CUDA files.
   * For non-CUDA code, use getP2pTransportDevice(peerRank) instead.
   *
   * @return MultipeerIbgdaDeviceTransport wrapper (include .cuh header to use)
   */
  MultipeerIbgdaDeviceTransport getDeviceTransport() const;

  /**
   * getP2pTransportDevice - Get P2P transport for a specific peer rank
   *
   * Returns a pointer to the P2pIbgdaTransportDevice for the given peer rank.
   * This method handles the rank-to-index mapping internally and provides
   * explicit peer selection without requiring CUDA headers.
   *
   * In lazy mode, this materializes the requested peer before returning, so
   * the returned pointer is ready for kernel use.
   *
   * @param peerRank Global rank of the peer (must be != myRank and < nRanks)
   * @return Pointer to P2pIbgdaTransportDevice for the specified peer
   */
  P2pIbgdaTransportDevice* getP2pTransportDevice(int peerRank);

  /**
   * Lazily materialize one peer and return after its GPU device transport slot
   * is populated. No-op in eager mode or if the peer is already materialized.
   *
   * For ring-style setup where all ranks need to expose multiple peers before
   * connecting, use queuePeerForMaterialization() followed by connectPeers().
   *
   * @param peerRank Global rank of the peer to materialize
   */
  void materializePeer(int peerRank);

  /**
   * Queue a peer for lazy materialization. No network I/O happens here.
   * Call connectPeers() to complete all queued peers.
   *
   * No-op in eager mode or if the peer is already materialized.
   *
   * @param peerRank Global rank of the peer to queue
   */
  void queuePeerForMaterialization(int peerRank);

  /**
   * Connect all queued peers. In lazy mode, this MUST be called after
   * queuePeerForMaterialization() and BEFORE fetching transport pointers for
   * kernel launch.
   *
   * Processes peers in sorted rank order to avoid deadlock for >2 ranks.
   * No-op in eager mode or if no peers are queued.
   *
   * Example:
   *   transport->queuePeerForMaterialization(prev_rank);
   *   transport->queuePeerForMaterialization(next_rank);
   *   transport->connectPeers();
   *   prev = transport->getP2pTransportDevice(prev_rank);
   *   next = transport->getP2pTransportDevice(next_rank);
   *   launchKernel<<<...>>>(prev, next);
   */
  void connectPeers();

  /**
   * Check whether a peer's staging buffers have been allocated and its
   * GPU device transport slot populated. In eager mode, returns true for
   * all valid peers after exchange().
   *
   * @param peerRank Global rank of the peer
   * @return true if the peer is ready for kernel use
   */
  bool isPeerMaterialized(int peerRank) const;

  /**
   * getDeviceTransportPtr - Get pointer to device transport array
   *
   * Returns a pointer to the GPU memory containing the per-peer transport
   * handles. Each element corresponds to a peer (indexed by peer rank mapping).
   * Prefer getP2pTransportDevice(peerRank) for explicit peer selection.
   *
   * @return Pointer to P2pIbgdaTransportDevice array in GPU memory
   */
  P2pIbgdaTransportDevice* getDeviceTransportPtr() const;

  /**
   * Return the GPU pointer for a peer's device transport slot without
   * triggering lazy materialization. The slot may be zeroed if the peer
   * has not been materialized yet.
   *
   * @param peerRank Global rank of the peer
   * @return Pointer to P2pIbgdaTransportDevice slot (may be zeroed)
   */
  P2pIbgdaTransportDevice* getP2pTransportDeviceSlot(int peerRank) const;

  // numPeers()/myRank()/nRanks() are inherited from MultiPeerIbTransport.

  // registerBuffer()/deregisterBuffer()/exchangeBuffer() are inherited from
  // MultiPeerIbTransport (the refcounted MR registry lives in the base; the
  // backend supplies the register_mr_on_nics()/lookup_alloc_base()/
  // deregister_mr() hooks below).

  /**
   * Get the number of QP sets per (peer, NIC).
   * Total QPs to a peer = numQpsPerPeerPerNic() * numNics().
   */
  int numQpsPerPeerPerNic() const;

  // numNics() is inherited from MultiPeerIbTransport.

  /**
   * Get the GID index being used
   */
  int getGidIndex() const;

 private:
  // Helper methods
  void initDocaGpu();
  void openIbDevice();
  void allocateResources();
  void registerMemory();
  void createQpGroups();
  void allocate_send_recv_buffers();
  void exchange_send_recv_buffers();
  void cleanup_send_recv_buffers();
  void cleanup();
  // Connect a QP to a peer (or self for loopback). The nic argument selects
  // which local NIC's AH attrs / port to use; the peerInfo carries the
  // remote-side GID / LID / qpn. At numNics_=1 nic is always 0.
  void connectQp(
      doca_gpu_verbs_qp_hl* qpHl,
      const IbgdaTransportExchInfo& peerInfo,
      int nic);
  // rankToPeerIndex()/peerIndexToRank() are inherited from
  // MultiPeerIbTransport.

  // Per-peer helpers shared by eager exchange() and lazy materializePeer()
  void createPeerQps(int peerIndex);
  void connectPeerLoopback(int peerIndex);
  P2pIbgdaTransportBuildParams buildPeerTransportParams(int peerIndex) const;

  PeerBufferSizes computePeerBufferSizes() const;

  void doMaterializePeer(int peerRank);

  PeerQpPayload buildLocalQpPayload(int peerIndex) const;
  void allocatePeerBuffers(int peerIndex, PeerBufferPayload& payload);
  template <typename T>
  T exchangeWithPeer(int peerRank, const T& localPayload, int tag);
  void connectPeerMainQps(int peerIndex, const PeerQpPayload& remotePayload);
  void applyRemoteViews(int peerIndex, const PeerBufferPayload& remotePayload);
  void cleanupPeerOnFailure(int peerIndex);

  // The MR registry (register/deregister/exchangeBuffer + the cache) lives
  // entirely in MultiPeerIbTransport; it registers on the base-owned
  // nics_[*].ibvPd, which openIbDevice() fills.

  // myRank_/nRanks_/bootstrap_ are inherited (protected) from
  // MultiPeerIbTransport.

  // Configuration
  MultipeerIbgdaTransportConfig config_;

  // DOCA GPU context (shared across NICs).
  doca_gpu* docaGpu_{nullptr};

  // numNics_ is inherited (protected) from MultiPeerIbTransport;
  // nicDevices_.size() == numNics_ after openIbDevice().

  // Per-NIC host-side IB verbs resources. qpGroups and loopbackCompanionQps
  // are indexed [peer * numQpsPerPeerPerNic + q].
  // Backend-specific (DOCA) per-NIC state. The generic per-NIC resources
  // (device name, context, PD, GID) live in MultiPeerIbTransport::nics_,
  // index-aligned with this vector; openIbDevice() fills both.
  struct NicDocaResources {
    doca_verbs_ah_attr* ahAttr{nullptr};
    ibverbx::ibv_mr* sinkMr{nullptr};
    std::vector<doca_gpu_verbs_qp_group_hl*> qpGroups;
    std::vector<doca_gpu_verbs_qp_hl*> loopbackCompanionQps;
  };
  std::vector<NicDocaResources> nicDoca_;

  // Sink buffer for RDMA atomic return values (discarded).
  // DOCA's OPCODE_ATOMIC_FA requires a local address for the fetch-add
  // return value. We don't need it, so we use a small "sink" buffer.
  // Allocated via cuMemCreate with gpuDirectRDMACapable=1 so it can be
  // registered as an IB MR on all platforms (including aarch64/SMMU).
  void* sinkBuffer_{nullptr};
  std::size_t sinkBufferSize_{0};
  std::size_t sinkBufferAllocSize_{0};
  std::uint64_t sinkBufferHandle_{0};

  // The refcounted MR cache (CachedMr + registeredBuffers_) lives in
  // MultiPeerIbTransport.

  // GPU PCIe bus ID.
  std::string gpuPciBusId_;
  // GID index + active MTU are common across NICs (same config knob, same
  // fabric/HCA generation in multi-NIC platforms like GB200/GB300).
  int gidIndex_{3};
  ibverbx::ibv_mtu localMtu_{ibverbx::IBV_MTU_4096};

  // Per-peer device transports (GPU accessible)
  P2pIbgdaTransportDevice* peerTransportsGpu_{nullptr};
  std::size_t peerTransportSize_{0};

  // All GPU allocations from buildDeviceTransportsOnGpu (freed in cleanup)
  std::vector<void*> gpuAllocations_;

  // Exchange info received from peers
  std::vector<IbgdaTransportExchInfo> peerExchInfo_;

  // Slot-index signal/counter buffers.
  // Eager mode: bulk-allocated in exchange(). Null in lazy mode.
  void* signalInboxGpu_{nullptr};
  void* counterGpu_{nullptr};
  // Per-peer views into signal/counter (populated by both modes).
  std::vector<IbgdaRemoteBuffer> signalRemoteViews_;
  std::vector<IbgdaLocalBuffer> signalLocalViews_;
  std::vector<IbgdaLocalBuffer> counterViews_;

  // Per-peer send/recv buffer views (populated by both modes).
  struct SendRecvPeerBuffers {
    IbgdaLocalBuffer sendStaging;
    IbgdaLocalBuffer recvStaging;
    IbgdaLocalBuffer signal;
    IbgdaLocalBuffer counter;
    std::optional<DeviceSpan<IbSendRecvState::ProgressSlot>> state;
    IbgdaRemoteBuffer remoteRecvStaging;
    IbgdaRemoteBuffer remoteSignal;
  };
  std::vector<SendRecvPeerBuffers> sendRecvPeerBuffers_;

  // Eager mode: bulk allocations for all peers. Null in lazy mode.
  std::unique_ptr<meta::comms::DeviceBuffer> sendStagingBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> recvStagingBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> signalBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> counterBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> stateBulk_;
  IbgdaLocalBuffer recvStagingBulkReg_;
  IbgdaLocalBuffer signalBulkReg_;
  IbgdaLocalBuffer counterBulkReg_;

  // Lazy mode: per-peer contiguous allocation (staging + signal + counter +
  // state + slot-index). Null for unmaterialized peers.
  // Empty in eager mode.
  std::vector<std::unique_ptr<meta::comms::DeviceBuffer>> lazyPeerBufs_;

  // Lazy mode: set to true after writeDeviceTransportSlot completes.
  std::vector<bool> peerMaterialized_;

  // Lazy mode: set after a failed connectPeers() attempt. Retrying peer
  // materialization is unsafe because peer-side state may be asymmetric.
  bool materializationFailed_{false};

  // Queued peers awaiting connectPeers().
  std::vector<int> pendingPeers_;
};

} // namespace comms::prims
