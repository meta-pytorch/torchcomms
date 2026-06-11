// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/transport/MultiPeerIbTransport.h"

namespace comms::prims {

/**
 * MultipeerIbrcTransport - CPU-proxy IBRC backend (skeleton).
 *
 * The IBRC backend posts RDMA work from a CPU progress thread that drains
 * GPU-written command-queue rings, updating host-mapped completion counters.
 * It derives from the shared CRTP base MultiPeerIbTransport<Backend> so the
 * host control plane can be wired in as the backend-specific pieces land.
 *
 * This is currently a SKELETON: the IBRC-specific machinery (per-lane RC QPs +
 * atomic sink, GPU<->CPU command-queue rings, the CPU progress thread,
 * host-mapped completion counters, and the device transport) is not yet ported.
 * The backend is selectable (NCCL_CTRAN_PIPES_IB_MODE=ibrc) and constructs on
 * the shared base, but exchange(), buffer registration/exchange, lazy
 * materialization, and device-handle use are not yet functional.
 */
class MultipeerIbrcTransport
    : public MultiPeerIbTransport<MultipeerIbrcTransport> {
 public:
  MultipeerIbrcTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbTransportConfig& config);

  ~MultipeerIbrcTransport() = default;

  // Non-copyable, non-movable
  MultipeerIbrcTransport(const MultipeerIbrcTransport&) = delete;
  MultipeerIbrcTransport& operator=(const MultipeerIbrcTransport&) = delete;
  MultipeerIbrcTransport(MultipeerIbrcTransport&&) = delete;
  MultipeerIbrcTransport& operator=(MultipeerIbrcTransport&&) = delete;

  /**
   * exchange - COLLECTIVE. Allocate command queues, connect QPs, build device
   * transports, and start the CPU progress thread. NOT YET IMPLEMENTED.
   */
  void exchange();

  // numPeers() / myRank() / nRanks() / numNics() are inherited from
  // MultiPeerIbTransport(Base). Buffer registration/exchange and lazy
  // materialization are intentionally blocked by MultiPeerTransport until the
  // IBRC backend initializes the required resources.

 private:
  // Lazy per-peer materialization hook. Not yet implemented
  // (doMaterializePeer throws).
  void doMaterializePeer(int peerRank);
  void cleanupPeerOnFailure(int peerIndex);

  // MultiPeerIbTransport drives the shared control plane and calls the private
  // hooks above.
  friend class MultiPeerIbTransport<MultipeerIbrcTransport>;
};

} // namespace comms::prims
