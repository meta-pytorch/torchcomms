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
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

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

  // Send/recv configuration. When set, the transport allocates a private
  // pipelined staging ring plus private signal/counter state for send()/recv().
  // When nullopt (default), only the raw put/signal APIs are available.
  struct SendRecvConfig {
    // Maximum number of block-groups that may participate in one send()/recv()
    // call. Sizes the private signal/counter/step arrays and caps
    // active_blocks.
    int maxGroups{128};

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

  // Number of QP sets per (peer, NIC). Total QPs to a peer =
  // numQpsPerPeerPerNic * numNics. Multiple QPs let different GPU blocks use
  // independent QPs.
  int numQpsPerPeerPerNic{1};

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

/**
 * Maximum number of QP sets per (peer, NIC) for multi-QP support.
 */
constexpr int kMaxQpsPerPeerPerNic = 128;

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
    uint32_t qpnForRank[kMaxRanksForAllGather][kMaxQpsPerPeerPerNic]{};
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
    uint32_t qpns[kMaxQpsPerPeerPerNic]{};
  };
  NicQpInfo nicInfo[kMaxNicsPerGpu]{};
  int gidIndex{0};
  int mtu{0};
  int numNics{0};
  int numQpsPerPeerPerNic{0};
};

struct PeerBufferPayload {
  IbgdaBufferExchInfo recvStaging;
  IbgdaBufferExchInfo srSignal;
  IbgdaBufferExchInfo slotSignal;
  IbgdaBufferExchInfo slotDiscard;
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
  IbgdaLocalBuffer registerBuffer(void* ptr, std::size_t size);

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
  // (the dispatcher holds the concrete backend type).
  ~MultiPeerIbTransportBase() = default;

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

  // Cached MR entry: one MR per (CUDA allocation, NIC), refcounted. Multiple
  // user buffers within the same allocation share one MR set.
  struct CachedMr {
    std::array<ibverbx::ibv_mr*, kMaxNicsPerGpu> mrs{};
    std::size_t allocSize{0};
    int refs{0};
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
  };
  std::vector<NicResources> nics_;

  // Maps allocation base address -> cached MR covering the full allocation.
  // Ordered map enables O(log n) containment lookup via upper_bound.
  std::map<uintptr_t, CachedMr> registeredBuffers_;

  // Lazy materialization state machine.
  std::vector<int> pendingPeers_;
  std::vector<bool> peerMaterialized_;
  bool materializationFailed_{false};
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
