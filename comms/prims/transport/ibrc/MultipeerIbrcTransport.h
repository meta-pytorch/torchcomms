// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/transport/MultiPeerIbTransport.h"

namespace comms::prims {

/**
 * MultipeerIbrcTransport - CPU-proxy IBRC backend.
 *
 * The IBRC backend posts RDMA work from a CPU progress thread that drains
 * GPU-written command-queue rings, updating host-mapped completion counters.
 * It derives from the shared CRTP base MultiPeerIbTransport<Backend> so the
 * host control plane can be wired in as the backend-specific pieces land.
 *
 * This is still incomplete: per-lane RC QP exchange/connect is implemented, but
 * GPU<->CPU command-queue rings, the CPU progress thread, host-mapped
 * completion counters, and the device transport are not yet ported. The common
 * (inherited) API works; the backend is selectable
 * (NCCL_CTRAN_PIPES_IB_MODE=ibrc), but exchange() still throws after QPs are
 * connected until the remaining data path lands.
 *
 * IBRC supports both eager exchange() and lazy per-peer materialization from
 * day one: the base's lazy connect loop drives the doMaterializePeer() hook
 * below, so there is no design-level eager-only restriction.
 */
class MultipeerIbrcTransport
    : public MultiPeerIbTransport<MultipeerIbrcTransport> {
 public:
  MultipeerIbrcTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbTransportConfig& config);

  ~MultipeerIbrcTransport();

  // Non-copyable, non-movable
  MultipeerIbrcTransport(const MultipeerIbrcTransport&) = delete;
  MultipeerIbrcTransport& operator=(const MultipeerIbrcTransport&) = delete;
  MultipeerIbrcTransport(MultipeerIbrcTransport&&) = delete;
  MultipeerIbrcTransport& operator=(MultipeerIbrcTransport&&) = delete;

  /**
   * exchange - COLLECTIVE. Connect QPs eagerly, then build command queues,
   * device transports, and the CPU progress thread once those slices land.
   */
  void exchange();

  // numPeers() / myRank() / nRanks() / numNics() are inherited from
  // MultiPeerIbTransport(Base). Buffer registration/exchange and lazy
  // materialization are intentionally blocked by MultiPeerTransport until the
  // IBRC backend initializes the required resources.

 private:
  // Lazy per-peer materialization hook. The shared base owns queueing,
  // ordering, and failure rollback; IBRC fills in per-peer QPs here and will
  // add ring/device setup in the next implementation slice.
  void doMaterializePeer(int peerRank);
  void cleanupPeerOnFailure(int peerIndex);

  struct PeerQpResource {
    ibverbx::ibv_cq* cq{nullptr};
    ibverbx::ibv_qp* qp{nullptr};
    int nic{0};
    int qpSlot{0};
  };

  struct PeerResources {
    std::vector<PeerQpResource> qpResources;
    bool qpsConnected{false};
  };

  void cleanup();
  void cleanupPeerQps(int peerIndex) noexcept;
  void destroyPeerQps(std::vector<PeerQpResource>& qpResources) noexcept;
  void closeNics() noexcept;

  void createPeerQps(int peerIndex);
  PeerQpPayload buildLocalQpPayload(int peerIndex) const;
  void connectPeerQps(int peerIndex, const PeerQpPayload& remotePayload);
  void connectPeerQp(
      PeerQpResource& qpResource,
      uint32_t remoteQpn,
      const uint8_t* remoteGid,
      uint16_t remoteLid,
      int remoteMtu);
  void exchangeAndConnectQps();
  PeerQpResource& qpResourceAt(int peerIndex, int nic, int qpSlot);
  const PeerQpResource& qpResourceAt(int peerIndex, int nic, int qpSlot) const;

  // MultiPeerIbTransport drives the shared control plane and calls the private
  // hooks above.
  friend class MultiPeerIbTransport<MultipeerIbrcTransport>;

  std::vector<PeerResources> peerResources_;
};

} // namespace comms::prims
