// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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
#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/MultiPeerIbTransport.h"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"
#include "comms/prims/transport/ibrc/IbrcTypes.h"
#include "comms/prims/transport/ibrc/P2pIbrcHostLanes.h"
#include "comms/prims/transport/ibrc/P2pIbrcHostWriter.h"

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
  // `abort` is the owning communicator's device handle. It is baked into every
  // per-peer device slot so the device-side waits on the CPU proxy terminate on
  // abort instead of trapping. A default-constructed handle keeps the legacy
  // cycle-deadline trap; see kIbrcDefaultDeviceTimeoutCycles.
  //
  // `hostAborted` is the same abort seen from the CPU, and it is separate only
  // because AbortDevice::isAborted() is __device__-only. Every writer handed
  // out by getHostWriter() is armed with it, so a host-driven collective cannot
  // forget to and fall back to the writer's ten-minute deadline.
  MultipeerIbrcTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbTransportConfig& config,
      comms::fault_tolerance::AbortDevice abort = {},
      std::function<bool()> hostAborted = nullptr);

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

  // Per-peer device handle accessor used by Ring/SendRecv algorithms. The
  // requested peer is materialized before its device slot is returned.
  P2pIbrcTransportDevice* getP2pTransportDevice(int peerRank);

  // Host-side writer into a peer's IBRC command-queue ring, so a CPU thread can
  // post RDMA put/signal without a kernel; the CPU proxy drains host-produced
  // descriptors unchanged. `queueIndex` selects one (qpSlot, nic) ring. The
  // peer must already be materialized and remain materialized while the
  // returned writer is used; P2pIbrcHostWriter is a non-owning view of the
  // queue's mapped host memory. Do not concurrently drive the same ring from
  // device code.
  // Throws if a live P2pIbrcHostLanes holds this peer: the two would be
  // separate producers on rings only one may drive, which the backpressure
  // check cannot survive -- it corrupts descriptors rather than failing, so it
  // is refused here. Use lanes or bare writers for a peer, never both.
  P2pIbrcHostWriter getHostWriter(int peerRank, uint32_t queueIndex = 0) const;

  /**
   * How many command-queue rings this peer has, i.e. the largest lane count
   * getHostLanes() will accept. Equals numNics() * the per-NIC QP count, so
   * callers clamp against it rather than probing until getHostWriter throws.
   */
  std::size_t hostLaneCapacity(int peerRank) const;

  /**
   * Lanes 0..numLanes-1 onto one peer, for splitting a single transfer across
   * NICs. Because queues are [qpSlot * numNics + nic], lane l lands on NIC
   * l % numNics, so consecutive lanes alternate NICs before reusing one.
   *
   * @throws std::runtime_error if numLanes is not in [1,
   * hostLaneCapacity].
   */
  P2pIbrcHostLanes getHostLanes(int peerRank, int numLanes) const;

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
    MappedAllocation channelState;
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
  std::vector<int> selectProgressCpus() const;
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
  /*
   * Non-expired while a P2pIbrcHostLanes owns that peer's rings.
   *
   * getHostLanes() always hands out command queues [0, numLanes), so a second
   * issue for the same peer overlaps completely, and two producers on one ring
   * break P2pIbrcHostWriter's backpressure check -- it reads the free space
   * before a fetch-add it cannot roll back, which is only sound with a single
   * producer. A weak_ptr rather than a flag the lanes object clears: it cannot
   * dangle, and it needs no cooperation from an object that outlives the call
   * that built it.
   *
   * Guarded by its own mutex because getHostLanes() is const and the check and
   * the claim have to be one step. Taken once per communicator, not per call.
   */
  /*
   * getHostWriter() without the ring-claim check. getHostLanes() builds its own
   * writers after taking the claim, so the public entry point would refuse the
   * very object it is constructing.
   */
  P2pIbrcHostWriter makeHostWriter(int peerRank, uint32_t queueIndex) const;

  mutable std::mutex hostLanesMutex_;
  mutable std::vector<std::weak_ptr<void>> hostLanesIssued_;
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
  std::vector<int> progressCpus_;
  comms::fault_tolerance::AbortDevice abortDevice_;
  std::function<bool()> hostAborted_;

  // Send/recv staging state (eager mode) lives in MultiPeerIbTransportBase
  // (sendRecvPeerBuffers_ + bulks); IBRC delegates allocation/exchange/cleanup.
};

} // namespace comms::prims
