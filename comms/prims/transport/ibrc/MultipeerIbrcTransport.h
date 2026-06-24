// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

// `meta::comms::DeviceBuffer`: HIP shim on AMD, CUDA RAII on NVIDIA (mirrors
// MultipeerIbgdaTransport.h).
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include "comms/utils/CudaRAII.h"
#endif

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/MultiPeerIbTransport.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"

namespace comms::prims {

class P2pIbrcTransportDevice;

/**
 * MultipeerIbrcTransport - CPU-proxy IBRC backend.
 *
 * The IBRC backend posts RDMA work from a CPU progress thread that drains
 * GPU-written command-queue rings, updating host-mapped completion counters.
 * It derives from the shared CRTP base MultiPeerIbTransport<Backend> so the
 * host control plane can be wired in as the backend-specific pieces land.
 *
 * Per-peer RC QP exchange/connect, GPU-visible command-queue resources, the
 * host progress loop, device enqueue transport, host-side device transport
 * construction, and proxy-completion local counters are implemented. The
 * counter path follows NCCL GIN proxy style by polling the normal CQ and then
 * updating host-mapped counter memory directly, instead of adding IBGDA-style
 * companion counter QPs.
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

  P2pIbrcTransportDevice* getP2pTransportDeviceSlot(int peerRank) const;

  // Per-peer device handle accessor used by Ring/SendRecv algorithms (the
  // counterpart of IBGDA's getP2pTransportDevice). IBRC builds all slots
  // eagerly, so this returns the slot pointer directly.
  P2pIbrcTransportDevice* getP2pTransportDevice(int peerRank) const;

 private:
  // Lazy per-peer materialization hook. The shared base owns queueing,
  // ordering, and failure rollback; IBRC fills in per-peer QPs and command
  // queues here, then later slices will attach the device transport.
  void doMaterializePeer(int peerRank);
  void cleanupPeerOnFailure(int peerIndex);

  struct PeerQpResource {
    ibverbx::ibv_cq* cq{nullptr};
    ibverbx::ibv_qp* qp{nullptr};
    ibverbx::ibv_mr* signalAtomicSinkMr{nullptr};
    std::unique_ptr<uint64_t> signalAtomicSink;
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
    uint64_t counterAddr{0};
    uint64_t counterValue{0};
    uint16_t flags{0};
    // Set when the peer-facing WR for this descriptor has completed (CQE
    // reaped), or for descriptors that post no WR. Retirement (nextToComplete/
    // ci advance) happens strictly in seq order in drainCompletedCommands().
    bool peerCompleted{false};
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
    MappedAllocation cmdQueueDevices;
    MappedAllocation blockQpState;
    bool qpsConnected{false};
    bool cmdQueuesAllocated{false};
  };

  void cleanup();
  void initializeControlResources();
  void cleanupPeerCmdQueues(int peerIndex) noexcept;
  void cleanupPeerQps(int peerIndex) noexcept;
  void destroyPeerQps(std::vector<PeerQpResource>& qpResources) noexcept;
  void closeNics() noexcept;

  void startProgressThread();
  void stopProgressThread() noexcept;
  void progressLoop() noexcept;
  bool progressOnce();
  bool pollOneCmdQueueDescriptor(int peerIndex, IbrcCmdQueueHost& cmdQueue);
  bool pollCmdQueueCompletions(int peerIndex, IbrcCmdQueueHost& cmdQueue);
  bool drainCompletedCommands(int peerIndex, IbrcCmdQueueHost& cmdQueue);
  void postDescriptor(
      int peerIndex,
      IbrcCmdQueueHost& cmdQueue,
      const IbrcDesc& desc,
      uint64_t seq);
  void publishQueueError(
      int peerIndex,
      const IbrcCmdQueueHost& cmdQueue,
      uint32_t errorCode,
      const char* reason) noexcept;
  // Publish a transport-level error to every NIC status block, independent of
  // any specific command queue, so all device wait paths observe the failure.
  void publishTransportError(uint32_t errorCode, const char* reason) noexcept;

  void allocateCmdQueuesForAllPeers();
  void allocatePeerCmdQueues(int peerIndex);
  void initializeDeviceTransportSlots();
  void updatePeerDeviceTransport(int peerIndex) noexcept;
  std::size_t allocatedCmdQueueCount() const;
  MappedAllocation allocateMapped(std::size_t bytes, const char* label);

  // ---- Pipelined send/recv staging (eager mode only) ----
  //
  // Host send/recv buffer management is shared with IBGDA in
  // MultiPeerIbTransportBase. IBRC delegates to
  // allocateSendRecvBuffersEager(IbCounterStorage::HostPinned) — the NIC_DONE
  // counter is host-mapped and updated by the CPU proxy (NCCL GIN style)
  // instead of an IBGDA companion-QP loopback counter — plus
  // exchangeSendRecvBuffersEager(), sendRecvStateForPeer(), and
  // cleanupSendRecvBuffers().

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
  // Per-peer publish flag (release in allocatePeerCmdQueues, acquire in
  // progressOnce) so the progress thread never reads a half-moved cmdQueues.
  // Separate array: std::atomic can't live in the movable PeerResources vector.
  std::unique_ptr<std::atomic<bool>[]> peerQueuesPublished_;
  MappedAllocation statusControl_;
  MappedAllocation p2pTransportDevices_;
  std::vector<IbrcNicStatus*> statusHostByNic_;
  std::vector<IbrcNicStatus*> statusDeviceByNic_;
  uint32_t cmdQueueDepth_{kIbrcDefaultCmdQueueDepth};
  std::size_t cmdQueuePiOffset_{0};
  std::size_t cmdQueueCiOffset_{0};
  std::size_t cmdQueueControlBytes_{0};
  std::atomic<bool> stopProgress_{false};
  std::thread progressThread_;

  // Send/recv staging state (eager mode) lives in MultiPeerIbTransportBase
  // (sendRecvPeerBuffers_ + bulks); IBRC delegates allocation/exchange/cleanup.
};

} // namespace comms::prims
