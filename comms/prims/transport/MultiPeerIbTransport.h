// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/rdma/DataDirectMode.h"

namespace meta::comms {
class DeviceBuffer;
} // namespace meta::comms

namespace comms::prims {

/**
 * IP address family for RoCE GID selection (similar to NCCL_IB_ADDR_FAMILY).
 */
enum class AddressFamily {
  IPV4, // IPv4
  IPV6, // IPv6
};

/**
 * Shared configuration for the multi-peer IB transports (IBGDA, IBRC). Every
 * field is backend-agnostic IB transport config. IMPORTANT: all ranks must use
 * identical configuration values.
 */
struct MultipeerIbTransportConfig {
  // CUDA device index for GPU operations
  int cudaDevice{0};

  // Override GID index for RoCE. If not set, auto-discovers a valid RoCEv2 GID.
  std::optional<int> gidIndex;

  // IP address family for the InfiniBand GID (similar to NCCL_IB_ADDR_FAMILY).
  // Used to determine the address type for RoCE connections when gidIndex is
  // not explicitly set. Has no effect on InfiniBand (non-RoCE) links.
  AddressFamily addressFamily{AddressFamily::IPV6};

  // GPU-to-NIC mapping for RDMA device selection. Maps CUDA device index to a
  // list of NIC names (first element is preferred). If empty, uses
  // topology-aware auto-discovery. (Data buffers are NOT managed by the
  // transport; users allocate and register them.)
  std::map<int, std::vector<std::string>> gpuNicMap;

  // IB HCA filter string (NCCL_IB_HCA format) for NIC filtering during
  // auto-discovery. If empty, all discovered NICs are considered. Only used
  // during auto-discovery (not when gpuNicMap has a mapping for the GPU).
  std::string ibHca;

  // Per-peer data buffer size in bytes. Raw put()/signal() users interpret
  // this as the exported per-peer RDMA buffer size; send()/recv() users
  // interpret it as the size of one logical staging slot.
  std::size_t dataBufferSize{0};

  // Number of signal slots managed by the transport (per peer), for the
  // slot-index API. Independent of send/recv's private signal buffers.
  int numSignalSlots{0};

  // Number of counter slots managed by the transport (per peer), for the
  // slot-index API. Independent of send/recv's private counter buffers.
  int numCounterSlots{0};

  // Maximum number of physical block groups that may own IB QP resources.
  // Device-side IB QP selection uses ThreadGroup::block_id and requires
  // block_id < maxGroups.
  int maxGroups{64};

  // Send/recv configuration. When set, the transport allocates a private
  // pipelined staging ring plus private signal/counter state for send()/recv().
  // When nullopt (default), only the raw put/signal APIs are available.
  struct SendRecvConfig {
    // Maximum number of block-groups that may participate in one send()/recv()
    // call. Sizes the private signal/counter/step arrays and caps
    // active_blocks. A value of 0 inherits the top-level maxGroups.
    int maxGroups{0};

    // Number of logical slots in the send/recv staging ring. Total staging
    // bytes per peer per direction: pipelineDepth * dataBufferSize.
    int pipelineDepth{2};
  };
  std::optional<SendRecvConfig> sendRecv;

  // Queue pair depth (outstanding WQEs per peer). BNXT bumps the default
  // because qpDepth also sizes msn_tbl_sz on bnxt_re.
#ifdef NIC_BNXT
  uint32_t qpDepth{2048};
#else
  uint32_t qpDepth{1024};
#endif

  // Number of main QPs owned by one physical block on each NIC. IBGDA executes
  // on the selected QP directly; IBRC enqueues to the selected QP's CPU-proxy
  // command queue.
  int qpsPerBlockPerNic{1};

  int numQpsPerPeerPerNic() const {
    return maxGroups * qpsPerBlockPerNic;
  }

  // mlx5 Data-Direct: register MRs through the NIC's data-direct (BAR1) PCIe
  // path for ~2x NIC<->HBM RDMA-write BW on GB300 (NCCL's NCCL_IB_DATA_DIRECT).
  // The single shared comms::prims::DataDirectMode (see DataDirectMode.h) — the
  // same enum NIC discovery uses — so this field both selects the discovery
  // mode and gates the registration path. Disabled disables discovery's DD
  // probing too; Only/Both take effect only on a DD-capable NIC (a no-op
  // otherwise). The caller should tunnel NCCL_IB_DATA_DIRECT (0/1/2) into this
  // field.
  DataDirectMode enableDataDirect{DataDirectMode::Only};

  // PCIe Relaxed Ordering on eligible (bulk data) MRs so NIC<->HBM DMA TLPs
  // pipeline instead of strict-ordering to ~half rate (NCCL's
  // NCCL_IB_PCI_RELAXED_ORDERING). Only applied to MRs the caller marks
  // relaxed-ordering-eligible (data, not signal/counter). The caller should
  // tunnel NCCL_IB_PCI_RELAXED_ORDERING into this field.
  enum class PciRelaxedOrderingMode {
    Disabled, // strict ordering on every MR
    Enabled, // relaxed ordering on eligible MRs
    Auto, // relaxed ordering on eligible MRs when supported (NCCL default)
  };
  PciRelaxedOrderingMode enablePciRelaxedOrdering{PciRelaxedOrderingMode::Auto};

  // InfiniBand Verbs Timeout for QP ACK timeout (4.096us * 2^timeout). Valid
  // 1-31; 0 or >=32 is infinite. Default 20 (similar to NCCL_IB_TIMEOUT).
  uint8_t timeout{20};

  // InfiniBand retry count for QP transport errors (NCCL_IB_RETRY_CNT).
  uint8_t retryCount{7};

  // InfiniBand traffic class field (similar to NCCL_IB_TC).
  uint8_t trafficClass{224};

  // InfiniBand Service Level (similar to NCCL_IB_SL).
  uint8_t serviceLevel{0};

  // Minimum RNR NAK Timer field (ibv_qp_attr.min_rnr_timer); NCCL
  // IbvQpUtils=12.
  uint8_t minRnrTimer{12};

  // RNR retry count (ibv_qp_attr.rnr_retry); 7 means infinite.
  uint8_t rnrRetry{7};

  // When true, defer per-peer state (QPs, staging, signal buffers) to first
  // use via materializePeer(). When false (default), allocate eagerly at
  // exchange() time.
  bool ibLazyConnect{false};

  // Timeout (ms) for the bilateral exchange in materializePeer().
  uint32_t materializePeerTimeoutMs{30000};
};

// Whether Data-Direct MR registration applies for a NIC: Data-Direct is
// requested via config (not Disabled) and the NIC is DD-capable.
// registerBuffer() selects the Data-Direct registration path exactly when this
// holds (and the mlx5dv symbol is available). Exposed as a free function so the
// config -> registration tunnel can be unit-tested without a NIC.
inline bool dataDirectActiveForNic(
    const MultipeerIbTransportConfig& config,
    bool nicIsDataDirect) {
  return config.enableDataDirect != DataDirectMode::Disabled && nicIsDataDirect;
}

// Whether PCIe Relaxed Ordering applies for a NIC: requested via config (not
// Disabled) and the NIC accepts the IBV_ACCESS_RELAXED_ORDERING access flag
// (probed during openNics). registerBuffer() sets the flag exactly when this
// holds, so on a NIC whose driver rejects it both Auto and Enabled fall back to
// strict ordering instead of failing registration. Free function so the
// config -> registration gating is unit-testable without a NIC.
inline bool relaxedOrderingActiveForNic(
    const MultipeerIbTransportConfig& config,
    bool nicRelaxedOrderingCapable) {
  return config.enablePciRelaxedOrdering !=
      MultipeerIbTransportConfig::PciRelaxedOrderingMode::Disabled &&
      nicRelaxedOrderingCapable;
}

/**
 * Transport connection information for RDMA QP setup.
 *
 * This struct is exchanged ONCE during the bootstrap phase to establish
 * RDMA connectivity between peers. Contains immutable connection parameters
 * that define how to reach a peer's QP.
 */
struct IbTransportExchInfo {
  // Queue Pair Number for RDMA connection
  uint32_t qpn{0};

  // Global Identifier for RoCE routing (16 bytes)
  uint8_t gid[16]{};

  // GID index used
  int gidIndex{0};

  // Local Identifier (for IB, not used in RoCE)
  uint16_t lid{0};

  // Port active MTU. Used to negotiate path MTU: min(local, remote).
  ibverbx::ibv_mtu mtu{ibverbx::IBV_MTU_4096};
};

/**
 * Maximum number of ranks supported for allGather-based exchange.
 * This limit exists because we use fixed-size arrays for QPN exchange.
 */
constexpr int kMaxRanksForAllGather = 128;

// Eager allGather QPN exchange uses a compact fixed-size wire format. Larger
// block-owned QP shapes must use lazy peer materialization.
constexpr int kMaxEagerExchangeQpsPerPeerPerNic = 128;

constexpr int kMaxIbGroups = 64;
constexpr int kMaxIbQpsPerBlockPerNic = 128;
constexpr int kMaxIbQpsPerPeerPerNic = kMaxIbGroups * kMaxIbQpsPerBlockPerNic;

/**
 * Transport exchange info for allGather-based exchange.
 *
 * Each rank contributes this structure containing per-NIC GID/LID and the
 * per-(target_rank, q) QPN this rank uses on that NIC.
 */
struct IbTransportExchInfoAll {
  // Per-NIC public info shared with peers (wire format). nicInfo[n] holds
  // this rank's NIC n's GID, LID, and the QPNs it uses to connect to each
  // (target_rank, q). Indices [numNics .. kMaxNicsPerGpu) are zero-init and
  // never read by peers (both ranks must agree on numNics — validated at
  // exchange time).
  struct NicWireInfo {
    uint8_t gid[16]{};
    uint16_t lid{0};
    // QPN this rank uses on this NIC to connect to (target_rank, q).
    // qpnForRank[myRank][*] is unused (set to 0).
    uint32_t qpnForRank[kMaxRanksForAllGather]
                       [kMaxEagerExchangeQpsPerPeerPerNic]{};
  };
  NicWireInfo nicInfo[kMaxNicsPerGpu]{};

  // Common (shared across NICs on this rank).
  int gidIndex{0};
  ibverbx::ibv_mtu mtu{ibverbx::IBV_MTU_4096};

  // Number of NICs (rails) used by this rank.
  // Must match across all ranks (validated at exchange time).
  int numNics{1};

  // Number of QPs per (peer, NIC) used by this rank.
  int numQpsPerPeerPerNic{1};

  // Block-owned QP shape.
  int maxGroups{64};
  int qpsPerBlockPerNic{1};
};

// Bootstrap tags for the two-phase bilateral exchange in lazy materialization.
constexpr int kIbPeerQpExchangeTag = 0;
constexpr int kIbPeerBufferExchangeTag = 1;

// Wire formats for bilateral peer materialization. Split into two phases: QP
// info first (to connect), then buffer info (acts as QP-ready barrier).
struct PeerQpPayload {
  struct NicQpInfo {
    uint8_t gid[16]{};
    uint16_t lid{0};
    uint32_t qpns[kMaxIbQpsPerPeerPerNic]{};
  };
  NicQpInfo nicInfo[kMaxNicsPerGpu]{};
  int gidIndex{0};
  int mtu{0};
  int numNics{0};
  int numQpsPerPeerPerNic{0};
  int maxGroups{0};
  int qpsPerBlockPerNic{0};
};

struct PeerBufferPayload {
  IbgdaBufferExchInfo recvStaging;
  IbgdaBufferExchInfo srSignal;
  IbgdaBufferExchInfo slotSignal;
  IbgdaBufferExchInfo slotDiscard;
};

// Which memory a NIC completion counter lives in. Shared by the slot counter
// (#16) and the send/recv NIC_DONE counter:
//   Device     - GPU device memory, allocated and registered by the transport.
//                The NIC bumps it via a loopback RDMA atomic (IBGDA).
//   HostPinned - host-mapped (cudaHostAllocMapped) memory, allocated by the
//                transport; the CPU progress thread writes it and the device
//                reads via the mapped pointer (IBRC). Never MR-registered.
enum class IbCounterStorage {
  Device,
  HostPinned,
};

// Per-peer send/recv staging-ring views. Eager mode owns the bulk allocations
// and slices these; the device side reads them via sendRecvStateForPeer().
struct IbSendRecvPeerBuffers {
  IbgdaLocalBuffer sendStaging;
  IbgdaLocalBuffer recvStaging;
  IbgdaLocalBuffer signal;
  IbgdaLocalBuffer counter;
  IbgdaLocalBuffer counterCompletion;
  // DeviceSpan has a const data_ member (no copy-assign), so wrap in optional
  // and emplace() the per-peer slice.
  std::optional<DeviceSpan<IbSendRecvState::ProgressSlot>> state;
  IbgdaRemoteBuffer remoteRecvStaging;
  IbgdaRemoteBuffer remoteSignal;
};

/**
 * MultiPeerIbTransportBase - backend-agnostic host control plane shared by the
 * multi-peer IB transports (IBGDA today, IBRC next).
 *
 * This is a NON-template base so its (heavy) method bodies live in
 * MultiPeerIbTransport.cc and are compiled exactly once, reused by every
 * backend with no per-backend wiring. It owns rank state + rank<->peerIndex
 * mapping, the full refcounted MR registry
 * (registerBuffer/deregisterBuffer/exchangeBuffer), the generic per-NIC IB
 * resources (NicResources), the bilateral bootstrap exchange, and the lazy
 * materialization queue/state. It NEVER calls into a backend; the small piece
 * of control flow that does (the connect loop) lives in the CRTP layer below.
 *
 * MR registration is generic — it resolves the allocation, exports a DMA-BUF
 * via the platform helper, and registers one MR per NIC on the base-owned PDs
 * (nics_[n].ibvPd), which the backend fills during NIC bring-up. No backend
 * hook is involved.
 */
class MultiPeerIbTransportBase {
 public:
  /** @return Number of peers (nRanks - 1). */
  int numPeers() const {
    return nRanks_ - 1;
  }

  /** @return This rank's id. */
  int myRank() const {
    return myRank_;
  }

  /** @return Total number of ranks. */
  int nRanks() const {
    return nRanks_;
  }

  /**
   * @return Number of NICs (rails) in use. Resolved by the shared base during
   * construction.
   */
  int numNics() const {
    return numNics_;
  }

  /**
   * registerBuffer - Register a user GPU buffer for RDMA, refcounted per
   * allocation. Containment fast-path returns cached per-NIC lkeys without any
   * driver call; on a miss it resolves the allocation, exports a DMA-BUF, and
   * registers one MR per NIC on the base-owned PDs.
   *
   * @return IbgdaLocalBuffer carrying one lkey per NIC.
   */
  // @param relaxedOrdering eligible for PCIe Relaxed Ordering (gated by
  //   config.enablePciRelaxedOrdering). Only bulk data (staging) MRs pass true;
  //   signal/counter MRs stay strict. Data-Direct is applied automatically on
  //   DD-capable NICs regardless of this flag.
  IbgdaLocalBuffer
  registerBuffer(void* ptr, std::size_t size, bool relaxedOrdering = false);

  /** deregisterBuffer - Decrement refcount; deregister all per-NIC MRs at 0. */
  void deregisterBuffer(void* ptr);

  /**
   * exchangeBuffer - COLLECTIVE. allGather a registered buffer's addr + per-NIC
   * rkeys; return one IbgdaRemoteBuffer per peer (indexed by peerIndexToRank).
   */
  std::vector<IbgdaRemoteBuffer> exchangeBuffer(
      const IbgdaLocalBuffer& localBuf);

  /** Queue a peer for lazy materialization (no network I/O). */
  void queuePeerForMaterialization(int peerRank);

  /** @return true if the peer is ready for kernel use (always true eager). */
  bool isPeerMaterialized(int peerRank) const;

 protected:
  MultiPeerIbTransportBase(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      MultipeerIbTransportConfig config);

  // Non-virtual protected dtor: the base is never owned/deleted polymorphically
  // (the dispatcher holds the concrete backend type). Defined out-of-line in
  // the .cc so the unique_ptr<DeviceBuffer> members destruct against a complete
  // type.
  ~MultiPeerIbTransportBase();

  MultiPeerIbTransportBase(const MultiPeerIbTransportBase&) = delete;
  MultiPeerIbTransportBase& operator=(const MultiPeerIbTransportBase&) = delete;

  int rankToPeerIndex(int rank) const {
    return (rank < myRank_) ? rank : (rank - 1);
  }
  int peerIndexToRank(int peerIndex) const {
    return (peerIndex < myRank_) ? peerIndex : (peerIndex + 1);
  }

  // Generic NIC bring-up: resolve NIC names (config.gpuNicMap / topology
  // auto-discovery), open device + PD, and query GID + port (active MTU, link
  // layer, port state). Fills nics_
  // (deviceName/ibvCtx/ibvPd/localGid/linkLayer) and localMtu_ (from NIC 0),
  // using gidIndex_. No backend hook — each backend builds its address handles
  // afterwards from nics_[n].linkLayer + config_.addressFamily.
  void openNics();

  // ---- shared eager-exchange scaffolding ----
  // Collective allGather of this rank's exchange info. The backend fills the
  // backend-specific QPN/GID/LID fields of localInfo; the base guards the rank
  // count, places localInfo at myRank_, allGathers, and returns all ranks'
  // info (indexed by global rank).
  std::vector<IbTransportExchInfoAll> allGatherExchInfo(
      const IbTransportExchInfoAll& localInfo);

  // Validate every peer agrees on numNics (same-rail pairing precondition) and
  // numQpsPerPeerPerNic. Throws std::runtime_error on mismatch.
  void validatePeerTopology(
      const std::vector<IbTransportExchInfoAll>& allInfo) const;

  // Bilateral bootstrap exchange of a fixed-size payload with one peer. The
  // typed wrapper is header-only (so it can instantiate with backend-private
  // payload types); the heavy logic (lower-rank-recvs-first to avoid deadlock,
  // honoring materializePeerTimeoutMs) lives in exchangeRawWithPeer in the .cc.
  template <typename T>
  T exchangeWithPeer(int peerRank, const T& localPayload, int tag) {
    T remotePayload{};
    exchangeRawWithPeer(
        peerRank, &localPayload, &remotePayload, sizeof(T), tag);
    return remotePayload;
  }
  void exchangeRawWithPeer(
      int peerRank,
      const void* localPayload,
      void* remotePayload,
      std::size_t bytes,
      int tag);

  // ---- shared send/recv staging-ring lifecycle (eager mode) ----
  // Backend-agnostic host send/recv buffer management, shared by IBGDA (Device
  // counter, NIC loopback atomic) and IBRC (Host counter, CPU proxy). Staging
  // = pipelineDepth * dataBufferSize per direction; signal/state are sized off
  // maxGroups. Per-peer staging + signal are device-registered; recvStaging +
  // signal are collectively exchanged so peers can RDMA into our ring.
  bool sendRecvBuffersEnabled() const {
    return config_.sendRecv.has_value();
  }
  IbSendRecvState sendRecvStateForPeer(int peerIndex) const;
  // Allocate + register the per-peer staging/signal/state bulks and slice them.
  // counterStorage selects the NIC_DONE counter: Device (transport-allocated,
  // registered) or HostPinned (transport-allocated host-mapped, never
  // registered).
  void allocateSendRecvBuffersEager(IbCounterStorage counterStorage);
  // COLLECTIVE. allGather recvStaging + signal so each peer holds our remote
  // views. Must be called after allocateSendRecvBuffersEager().
  void exchangeSendRecvBuffersEager();
  void cleanupSendRecvBuffers() noexcept;

  // ---- per-peer (lazy) send/recv: shared by IBGDA + IBRC ----
  // Allocate + register ONE peer's send/recv rings on demand (lazy connect) and
  // fill the outbound payload's recvStaging/srSignal exch info. counterStorage
  // selects the NIC_DONE counter: Device (a registered slice of the contiguous
  // per-peer buffer; NIC loopback atomic — IBGDA) or HostPinned (a separate
  // host-mapped allocation written by the CPU proxy — IBRC). The per-peer
  // buffer is dedicated to this peer pair, so no numPeers slicing is needed.
  void allocateSendRecvBufferForPeer(
      int peerIndex,
      PeerBufferPayload& payload,
      IbCounterStorage counterStorage);
  // Apply a peer's payload: remote recvStaging/signal views are used whole.
  void applyRemoteSendRecvBuffer(
      int peerIndex,
      const PeerBufferPayload& remotePayload);
  // Per-peer teardown: deregister + free this peer's lazy allocation and reset
  // its views. Safe on an unmaterialized peer.
  void cleanupSendRecvBufferForPeer(int peerIndex) noexcept;

  const MultipeerIbTransportConfig::SendRecvConfig& sendRecvConfig() const;
  void validateSendRecvConfig() const;
  std::size_t sendRecvStagingBytesPerPeer() const;
  std::size_t sendRecvSignalBytesPerPeer() const;
  std::size_t sendRecvCounterBytesPerPeer() const;
  std::size_t sendRecvStateBytesPerPeer() const;

  void allocateSignalCounterResources(
      IbCounterStorage counterStorage,
      bool allocateDiscardSignal);
  void cleanupSignalCounterResources() noexcept;
  void cleanupPeerSignalCounterResources(int peerIndex) noexcept;
  void allocatePeerSignalCounterResources(
      int peerIndex,
      PeerBufferPayload& payload,
      IbCounterStorage counterStorage,
      bool allocateDiscardSignal);
  void applyRemoteSignalCounterResources(
      int peerIndex,
      const PeerBufferPayload& remotePayload,
      bool hasDiscardSignal);

  IbgdaRemoteBuffer slotRemoteSignalView(int peerIndex) const;
  IbgdaLocalBuffer slotLocalSignalView(int peerIndex) const;
  IbgdaLocalBuffer slotCounterDeviceView(int peerIndex) const;
  IbgdaLocalBuffer slotCounterHostView(int peerIndex) const;
  IbgdaRemoteBuffer slotDiscardSignalRemoteView(int peerIndex) const;

  // Cached MR entry: one MR per (CUDA allocation, NIC), refcounted. Multiple
  // user buffers within the same allocation share one MR set.
  struct CachedMr {
    std::array<ibverbx::ibv_mr*, kMaxNicsPerGpu> mrs{};
    std::size_t allocSize{0};
    int refs{0};
    // Effective PCIe Relaxed Ordering the MRs were registered with (the
    // caller's request resolved against config). Part of the cache key: a
    // containment hit must resolve to the same value, else the access-flag
    // (ordering) semantics would silently differ from what the caller asked
    // for.
    bool relaxedOrdering{false};
  };

  const int myRank_{-1};
  const int nRanks_{0};
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  MultipeerIbTransportConfig config_;

  // Number of NICs (rails) in use; resolved by the base constructor.
  int numNics_{1};

  // Generic IB facts captured during NIC bring-up (openNics()): the RoCE GID
  // index (resolved from config in the ctor) and the negotiated active MTU
  // (NIC 0's). Read by backends when building address handles / connecting QPs.
  int gidIndex_{3};
  ibverbx::ibv_mtu localMtu_{ibverbx::IBV_MTU_4096};

  // Per-NIC generic IB resources (device name, context, PD, GID, link layer).
  // openNics() fills these; the base registers MRs on the PDs. The backend
  // keeps only its backend-specific per-NIC state (e.g. DOCA AH attrs and QP
  // groups), index-aligned with this vector.
  struct NicResources {
    std::string deviceName;
    ibverbx::ibv_context* ibvCtx{nullptr};
    ibverbx::ibv_pd* ibvPd{nullptr};
    ibverbx::ibv_gid localGid{};
    int linkLayer{0}; // ibverbx::IBV_LINK_LAYER_* (IB vs Ethernet/RoCE)
    // This NIC exposes a Data-Direct (`_dma`) variant, so data MRs can register
    // through the PCIe (BAR1) path. Copied from the discovery NicCandidate.
    bool isDataDirect{false};
    // This NIC's driver accepts IBV_ACCESS_RELAXED_ORDERING (probed once during
    // openNics). registerBuffer() applies Relaxed Ordering only when every NIC
    // is capable, so an unsupporting NIC falls back to strict ordering instead
    // of failing every data-MR registration.
    bool relaxedOrderingCapable{false};
  };
  std::vector<NicResources> nics_;

  // True iff every opened NIC accepts IBV_ACCESS_RELAXED_ORDERING (AND of
  // nics_[n].relaxedOrderingCapable, computed once in openNics). The MR cache
  // keys on a single effective-ordering bool per allocation, so Relaxed
  // Ordering must be uniform across NICs; gating on this aggregate keeps it so.
  bool relaxedOrderingCapable_{false};

  // Maps allocation base address -> cached MR covering the full allocation.
  // Ordered map enables O(log n) containment lookup via upper_bound.
  std::map<uintptr_t, CachedMr> registeredBuffers_;

  // Shared send/recv staging-ring state (eager mode). Owns the bulk
  // allocations; sendRecvPeerBuffers_ slices them per peer.
  std::vector<IbSendRecvPeerBuffers> sendRecvPeerBuffers_;
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvSendStagingBulk_;
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvRecvStagingBulk_;
  // Signal + device-counter control regions packed into one granularity-aligned
  // allocation (both Data-Direct-registered; share one aligned MR).
  // Host-counter configs put only the signal region here. See
  // allocateSendRecvBuffersEager.
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvControlBulk_;
  // Device-local progress state (never RDMA-registered / shared): separate,
  // natural size, no alignment needed.
  std::unique_ptr<meta::comms::DeviceBuffer> sendRecvStateBulk_;
  IbgdaLocalBuffer sendRecvRecvStagingBulkReg_;
  IbgdaLocalBuffer sendRecvSignalBulkReg_;
  IbgdaLocalBuffer sendRecvCounterBulkReg_;
  IbCounterStorage sendRecvCounterStorage_{IbCounterStorage::Device};

  // Lazy materialization state machine.
  std::vector<int> pendingPeers_;
  std::vector<bool> peerMaterialized_;
  bool materializationFailed_{false};

 private:
  struct DeviceSlotAllocation {
    void* ptr{nullptr};
    std::size_t bytes{0};
    bool registered{false};
    // On AMD the signal-inbox/discard buffers are host-pinned (device-memory
    // MR registration via peer-mem is unreliable); free accordingly.
    bool isHostPinned{false};

    DeviceSlotAllocation() = default;
    DeviceSlotAllocation(const DeviceSlotAllocation&) = delete;
    DeviceSlotAllocation& operator=(const DeviceSlotAllocation&) = delete;
    DeviceSlotAllocation(DeviceSlotAllocation&& other) noexcept
        : ptr(std::exchange(other.ptr, nullptr)),
          bytes(std::exchange(other.bytes, 0)),
          registered(std::exchange(other.registered, false)),
          isHostPinned(std::exchange(other.isHostPinned, false)) {}
    DeviceSlotAllocation& operator=(DeviceSlotAllocation&& other) noexcept {
      ptr = std::exchange(other.ptr, nullptr);
      bytes = std::exchange(other.bytes, 0);
      registered = std::exchange(other.registered, false);
      isHostPinned = std::exchange(other.isHostPinned, false);
      return *this;
    }
  };

  struct CounterSlotAllocation {
    void* hostPtr{nullptr};
    void* devicePtr{nullptr};
    std::size_t bytes{0};
    bool registered{false};

    CounterSlotAllocation() = default;
    CounterSlotAllocation(const CounterSlotAllocation&) = delete;
    CounterSlotAllocation& operator=(const CounterSlotAllocation&) = delete;
    CounterSlotAllocation(CounterSlotAllocation&& other) noexcept
        : hostPtr(std::exchange(other.hostPtr, nullptr)),
          devicePtr(std::exchange(other.devicePtr, nullptr)),
          bytes(std::exchange(other.bytes, 0)),
          registered(std::exchange(other.registered, false)) {}
    CounterSlotAllocation& operator=(CounterSlotAllocation&& other) noexcept {
      hostPtr = std::exchange(other.hostPtr, nullptr);
      devicePtr = std::exchange(other.devicePtr, nullptr);
      bytes = std::exchange(other.bytes, 0);
      registered = std::exchange(other.registered, false);
      return *this;
    }
  };

  void freeDeviceSlotAllocation(DeviceSlotAllocation& allocation) noexcept;
  DeviceSlotAllocation allocateDeviceSlotAllocation(
      std::size_t bytes,
      const char* label);
  void freeCounterSlotAllocation(CounterSlotAllocation& allocation) noexcept;
  CounterSlotAllocation allocateCounterSlotAllocation(
      IbCounterStorage storage,
      std::size_t bytes,
      const char* label);
  IbgdaLocalBuffer registerSlotMemory(
      void* registrationPtr,
      void* devicePtr,
      std::size_t bytes,
      bool& registered);
  IbgdaBufferExchInfo registeredSlotMemoryExchInfo(void* registrationPtr) const;

  std::vector<IbgdaRemoteBuffer> slotRemoteSignalViews_;
  std::vector<IbgdaLocalBuffer> slotLocalSignalViews_;
  std::vector<IbgdaLocalBuffer> slotCounterDeviceViews_;
  std::vector<IbgdaLocalBuffer> slotCounterHostViews_;
  std::vector<IbgdaRemoteBuffer> slotDiscardSignalRemoteViews_;

  DeviceSlotAllocation slotSignalAllocation_;
  CounterSlotAllocation slotCounterAllocation_;
  DeviceSlotAllocation slotDiscardSignalAllocation_;
  // Host-mapped send/recv NIC_DONE counter (counterStorage == Host). Owns the
  // host-pinned allocation; sliced per peer into IbSendRecvPeerBuffers.counter.
  CounterSlotAllocation sendRecvHostCounterAllocation_;
  std::vector<DeviceSlotAllocation> lazySlotSignalAllocations_;
  std::vector<CounterSlotAllocation> lazySlotCounterAllocations_;
  std::vector<DeviceSlotAllocation> lazySlotDiscardSignalAllocations_;
  // Lazy per-peer send/recv allocations: one contiguous device buffer per
  // materialized peer (sendStaging|recvStaging|signal|state, plus the counter
  // when device-resident). Empty in eager mode. Shared by IBGDA (Device
  // counter) and IBRC, which additionally allocates a per-peer host-mapped
  // NIC_DONE counter below.
  std::vector<std::unique_ptr<meta::comms::DeviceBuffer>> lazyPeerBufs_;
  std::vector<CounterSlotAllocation> lazySendRecvHostCounters_;
};

/**
 * MultiPeerIbTransport<Backend> - CRTP layer over MultiPeerIbTransportBase.
 *
 * Holds ONLY the small piece of control plane that must call into the concrete
 * backend: the lazy connect loop (connectPeers) drives the backend's per-peer
 * doMaterializePeer()/cleanupPeerOnFailure() hooks via a static `backend()`
 * downcast (no vtable). Each backend derives as
 *   `class MultipeerIbgdaTransport : public
 * MultiPeerIbTransport<MultipeerIbgdaTransport>`.
 * All backend-agnostic state and methods are inherited from the non-template
 * base, so they are compiled once (in MultiPeerIbTransport.cc) and shared.
 */
template <typename Backend>
class MultiPeerIbTransport : public MultiPeerIbTransportBase {
 public:
  /** Materialize one peer (queue + connect). No-op in eager mode. */
  void materializePeer(int peerRank) {
    queuePeerForMaterialization(peerRank);
    connectPeers();
  }

  /** Connect all queued peers in sorted order (deadlock-safe for >2 ranks). */
  void connectPeers();

 protected:
  MultiPeerIbTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      MultipeerIbTransportConfig config)
      : MultiPeerIbTransportBase(
            myRank,
            nRanks,
            std::move(bootstrap),
            std::move(config)) {}

  ~MultiPeerIbTransport() = default;

  // CRTP downcast for static dispatch into backend hooks.
  Backend& backend() {
    return static_cast<Backend&>(*this);
  }
  const Backend& backend() const {
    return static_cast<const Backend&>(*this);
  }
};

template <typename Backend>
void MultiPeerIbTransport<Backend>::connectPeers() {
  if (materializationFailed_) {
    pendingPeers_.clear();
    throw std::runtime_error(
        "MultiPeerIbTransport: lazy peer materialization previously failed; "
        "retry is not supported");
  }
  if (pendingPeers_.empty()) {
    return;
  }
  // Sorted order avoids deadlock for >2 ranks (both sides connect in the same
  // global order).
  std::sort(pendingPeers_.begin(), pendingPeers_.end());

  std::vector<int> peers;
  peers.swap(pendingPeers_);
  std::vector<int> touchedPeerIndexes;
  touchedPeerIndexes.reserve(peers.size());

  try {
    for (int peerRank : peers) {
      if (isPeerMaterialized(peerRank)) {
        continue;
      }
      touchedPeerIndexes.push_back(rankToPeerIndex(peerRank));
      backend().doMaterializePeer(peerRank);
    }
  } catch (...) {
    materializationFailed_ = true;
    for (int peerIndex : touchedPeerIndexes) {
      backend().cleanupPeerOnFailure(peerIndex);
    }
    throw;
  }
}

} // namespace comms::prims
