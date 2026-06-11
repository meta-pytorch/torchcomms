// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

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

/**
 * MultiPeerIbTransportBase - backend-agnostic host control plane shared by the
 * multi-peer IB transports (IBGDA today, IBRC next).
 *
 * This is a NON-template base so its (heavy) method bodies live in
 * MultiPeerIbTransport.cc and are compiled exactly once, reused by every
 * backend with no per-backend wiring. It owns rank state + rank<->peerIndex
 * mapping and the full refcounted MR registry
 * (registerBuffer/deregisterBuffer/exchangeBuffer), plus the generic per-NIC IB
 * resources (NicResources). It NEVER calls into a backend.
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
   * @return Number of NICs (rails) in use. Resolved by the backend at
   * construction time (see backend ctor for the resolution rules).
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

 protected:
  MultiPeerIbTransportBase(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap);

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

  // Number of NICs (rails) in use; set by the backend at construction.
  int numNics_{1};

  // Per-NIC generic IB resources (device name, context, PD, GID). The backend
  // fills these during NIC bring-up; the base registers MRs on the PDs. The
  // backend keeps only its backend-specific per-NIC state (e.g. DOCA AH attrs
  // and QP groups), index-aligned with this vector.
  struct NicResources {
    std::string deviceName;
    ibverbx::ibv_context* ibvCtx{nullptr};
    ibverbx::ibv_pd* ibvPd{nullptr};
    ibverbx::ibv_gid localGid{};
  };
  std::vector<NicResources> nics_;

  // Maps allocation base address -> cached MR covering the full allocation.
  // Ordered map enables O(log n) containment lookup via upper_bound.
  std::map<uintptr_t, CachedMr> registeredBuffers_;
};

/**
 * MultiPeerIbTransport<Backend> - CRTP layer over MultiPeerIbTransportBase.
 *
 * Each backend derives as
 *   `class MultipeerIbgdaTransport : public
 * MultiPeerIbTransport<MultipeerIbgdaTransport>`.
 * The base can statically call backend-specific hooks via `backend()` with no
 * vtable. All backend-agnostic state and methods are inherited from the
 * non-template base, so they are compiled once (in MultiPeerIbTransport.cc) and
 * shared. Backend-coupled control flow (the exchange/lazy skeleton) moves up in
 * later slices and will use `backend()` hooks.
 */
template <typename Backend>
class MultiPeerIbTransport : public MultiPeerIbTransportBase {
 protected:
  MultiPeerIbTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap)
      : MultiPeerIbTransportBase(myRank, nRanks, std::move(bootstrap)) {}

  ~MultiPeerIbTransport() = default;

  // CRTP downcast for static dispatch into backend hooks (used by later
  // slices).
  Backend& backend() {
    return static_cast<Backend&>(*this);
  }
  const Backend& backend() const {
    return static_cast<const Backend&>(*this);
  }
};

} // namespace comms::prims
