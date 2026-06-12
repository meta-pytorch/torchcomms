// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/transport/MultiPeerIbTransport.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"

namespace comms::prims {

/**
 * MultipeerIbrcTransport - CPU-proxy IBRC backend.
 *
 * The IBRC backend posts RDMA work from a CPU progress thread that drains
 * GPU-written command-queue rings, updating host-mapped completion counters.
 * It derives from the shared CRTP base MultiPeerIbTransport<Backend> so the
 * host control plane can be wired in as the backend-specific pieces land.
 *
 * This is still incomplete: per-peer RC QP exchange/connect and GPU-visible
 * command-queue resources are implemented, but the CPU progress thread,
 * device transport, and HostWindow counter plumbing are not yet ported. The
 * common (inherited) API works; the backend is selectable
 * (NCCL_CTRAN_PIPES_IB_MODE=ibrc), but exchange() still throws after QPs and
 * command queues are ready until the remaining data path lands.
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
  // ordering, and failure rollback; IBRC fills in per-peer QPs and command
  // queues here, then later slices will attach the device transport.
  void doMaterializePeer(int peerRank);
  void cleanupPeerOnFailure(int peerIndex);

  struct PeerQpResource {
    ibverbx::ibv_cq* cq{nullptr};
    ibverbx::ibv_qp* qp{nullptr};
    int nic{0};
    int qpSlot{0};
  };

  struct MappedAllocation {
    void* host{nullptr};
    void* device{nullptr};
    std::size_t bytes{0};

    MappedAllocation() = default;
    ~MappedAllocation();

    MappedAllocation(const MappedAllocation&) = delete;
    MappedAllocation& operator=(const MappedAllocation&) = delete;

    MappedAllocation(MappedAllocation&& other) noexcept;
    MappedAllocation& operator=(MappedAllocation&& other) noexcept;

    void reset() noexcept;
  };

  struct IbrcCmdState {
    uint64_t seq{kIbrcInvalidReadySeq};
    uint64_t counterValue{0};
    uint32_t counterId{0};
    uint16_t flags{0};
  };

  struct IbrcCmdQueueHost {
    MappedAllocation control;
    std::vector<IbrcCmdState> cmdStates;
    IbrcDesc* descsHost{nullptr};
    uint64_t* piHost{nullptr};
    uint64_t* ciHost{nullptr};
    IbrcCmdQueueDevice device{};
    uint32_t nic{0};
    uint32_t qpSlot{0};
    uint64_t nextToPoll{0};
    uint64_t nextToComplete{0};
  };

  struct PeerResources {
    std::vector<PeerQpResource> qpResources;
    std::vector<IbrcCmdQueueHost> cmdQueues;
    bool qpsConnected{false};
    bool cmdQueuesAllocated{false};
  };

  void cleanup();
  void initializeControlResources();
  void cleanupPeerCmdQueues(int peerIndex) noexcept;
  void cleanupPeerQps(int peerIndex) noexcept;
  void destroyPeerQps(std::vector<PeerQpResource>& qpResources) noexcept;
  void closeNics() noexcept;

  void allocateCmdQueuesForAllPeers();
  void allocatePeerCmdQueues(int peerIndex);
  std::size_t allocatedCmdQueueCount() const;
  MappedAllocation allocateMapped(std::size_t bytes, const char* label);

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
  MappedAllocation statusControl_;
  std::vector<IbrcNicStatus*> statusHostByNic_;
  std::vector<IbrcNicStatus*> statusDeviceByNic_;
  uint32_t cmdQueueDepth_{kIbrcDefaultCmdQueueDepth};
  std::size_t cmdQueuePiOffset_{0};
  std::size_t cmdQueueCiOffset_{0};
  std::size_t cmdQueueControlBytes_{0};
};

} // namespace comms::prims
