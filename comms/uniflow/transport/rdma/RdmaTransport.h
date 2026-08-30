// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>

#include "comms/uniflow/transport/rdma/CopyEngine.h"
#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/Transport.h"
#include "comms/uniflow/transport/rdma/RdmaResources.h"

namespace uniflow {

constexpr uint8_t kRdmaVersion{3};

// Forward declarations.
class RdmaRegistrationHandle;
class RdmaRemoteRegistrationHandle;
class RdmaSlab;
class RdmaSlabPool;

/*
 * RDMA transport configuration.
 *
 * Encapsulates all tunable parameters for RDMA queue pair setup.
 * Passed from factory to transport at creation time. Defaults match
 * production values from ctran/ibverbx.
 *
 * TODO: Make configurable via runtime config (env vars or config file),
 * similar to NCCL_IB_TIMEOUT / NCCL_IB_RETRY_CNT / NCCL_IB_PKEY cvars.
 */
struct RdmaTransportConfig {
  uint32_t numQps{1}; /* Total QPs distributed round-robin across NICs. */
  /* RoCE GID table index. -1 = auto-select a RoCEv2 entry by scanning the GID
   * table; >= 0 forces that index (3 is the Mellanox RoCEv2 convention). */
  int16_t gidIndex{-1};
  uint8_t timeout{14}; /* IB timeout exponent for QP retransmission. */
  uint8_t retryCnt{7}; /* Number of retries before reporting an error. */
  uint8_t trafficClass{0}; /* Traffic class for GRH (QoS / DSCP). */
  uint16_t pkeyIndex{0}; /* Partition key index for QP INIT. */
  /* Max outstanding send WRs per QP. The default leaves enough SQ depth for
   * high-throughput multi-NIC CPU RDMA without requiring a benchmark override.
   */
  uint32_t maxWr{2048};
  /* Max outstanding RDMA READ/atomic operations per QP. This directly controls
   * requester-side READ pipelining and responder-side inbound READ capacity.
   */
  uint32_t maxRdAtomic{8};
  uint32_t maxSge{1}; /* Max scatter/gather entries per WR. */
  uint32_t maxInlineData{16}; /* Max inline data bytes per WR. */
  size_t chunkSize{512 * 1024}; /* Transfer chunk size in bytes (512KB). */
  uint16_t pipelineDepth{2}; /* Send/recv pipeline depth (D staging slabs). */
  RdmaSlabPoolConfig slabPoolConfig{.slabNum = 0}; /* Disabled by default. */
  /* Register GPU memory over the mlx5 Data Direct path (NIC<->GPU via PCIe
   * BAR1) instead of the standard dmabuf path (which traverses the Grace C2C
   * link on GB300). Only applies to NICs that expose a data-direct sysfs path;
   * other NICs fall back to the standard path. Default off = existing behavior.
   */
  bool dataDirect{false};
};

/*
 * RDMA transport connection metadata — serialization and deserialization.
 *
 * Wire format (packed, memcpy-based):
 *   Header (14 bytes: version, numQps, numNics, domainId)
 *   + NicInfo[numNics]
 *   + QpInfo[numQps]
 *
 * QP-to-NIC mapping: QP i belongs to NIC (i % numNics).
 */
struct RdmaTransportInfo {
  /* Header in the wire format. */
  struct __attribute__((packed)) Header {
    uint8_t version{kRdmaVersion};
    uint8_t numNics{0};
    uint32_t numQps{0};
    uint64_t domainId{0};
    uint32_t slabSize{0};
  };
  /* Per-NIC addressing info in the wire format. */
  struct __attribute__((packed)) NicInfo {
    uint16_t lid{0}; /* Remote LID (used for IB link layer). */
    uint8_t linkLayer{0}; /* IBV_LINK_LAYER_ETHERNET or _INFINIBAND. */
    uint8_t mtu{0}; /* ibv_mtu enum value for MTU negotiation. */
    ibv_gid gid{}; /* Remote GID (used for RoCE / GRH). */
  };

  /* Per-QP connection parameters in the wire format. */
  struct __attribute__((packed)) QpInfo {
    uint32_t qpNum{0}; /* Queue pair number assigned by the device. */
    uint32_t psn{0}; /* Starting packet sequence number (random). */
  };

  struct RegisteredBuffer {
    uint64_t addr{0};
    uint32_t length{0};
    std::vector<uint32_t> rkeys;

    size_t size() const {
      return sizeof(addr) + sizeof(length) + rkeys.size() * sizeof(uint32_t);
    }

    static constexpr size_t expectedSize(uint8_t numNics) {
      return sizeof(addr) + sizeof(length) + numNics * sizeof(uint32_t);
    }

    void reset();
    size_t serialize(uint8_t* data) const;
    size_t deserialize(const uint8_t* data, uint8_t numNics);
  };

  /* Header fields. */
  Header header;

  /* Deserialized per-NIC and per-QP data. */
  std::vector<NicInfo> nicInfos;
  std::vector<QpInfo> qpInfos;

  RegisteredBuffer ctrl;
  RegisteredBuffer slab;

  /*
   * Serialize this info into a TransportInfo byte vector.
   * Only the header, nicInfos, and qpInfos are serialized.
   */
  TransportInfo serialize() const;

  /*
   * Deserialize a TransportInfo byte vector into an RdmaTransportInfo.
   * Returns an error on malformed or unsupported data.
   */
  static Result<RdmaTransportInfo> deserialize(std::span<const uint8_t> data);

  void reset();
};

// ---------------------------------------------------------------------------
// RdmaTransport
// ---------------------------------------------------------------------------

/*
 * Point-to-point RDMA transport over RC (Reliable Connection) queue pairs.
 *
 * Supports multiple NICs with round-robin QP distribution: QP i is created
 * on NIC (i % numNics), each NIC gets its own completion queue. This enables
 * rail-optimized traffic across multi-NIC hosts.
 *
 * Lifecycle:
 *   1. Construct with IbvApi, NIC resources, and config.
 *   2. Call bind() to allocate CQs/QPs and get local TransportInfo.
 *   3. Exchange TransportInfo with the remote peer (out-of-band).
 *   4. Call connect(remoteInfo) to transition QPs to RTS.
 *   5. Use put/get/send/recv for data transfer.
 *   6. Call shutdown() to tear down (also called by destructor).
 *
 * Thread safety: not thread-safe. Caller must synchronize access.
 */
class RdmaTransport : public Transport {
 public:
  /*
   * Construct an RDMA transport.
   *
   * @param ibvApi  IbvApi instance (injectable for testing via MockIbvApi).
   * @param nics    Per-NIC resources from the factory. Must be non-empty.
   *                Borrowed — must outlive this transport.
   * @param config  Transport configuration (numQps, QP attributes, etc.).
   *                numQps must be in [1, 255].
   */
  RdmaTransport(
      std::shared_ptr<IbvApi> ibvApi,
      std::shared_ptr<CudaApi> cudaApi,
      std::shared_ptr<CudaDriverApi> cudaDriverApi,
      EventBase* evb,
      std::shared_ptr<std::vector<NicResources>> nics,
      uint64_t domainId,
      RdmaTransportConfig config = {},
      std::shared_ptr<RdmaSlabPool> slabPool = nullptr);

  ~RdmaTransport() override;

  RdmaTransport(const RdmaTransport&) = delete;
  RdmaTransport& operator=(const RdmaTransport&) = delete;
  RdmaTransport(RdmaTransport&&) = delete;
  RdmaTransport& operator=(RdmaTransport&&) = delete;

  /* Returns "rdma_<nic1>_<nic2>_...". */
  const std::string& name() const noexcept override {
    return name_;
  }

  TransportType transportType() const noexcept override {
    return TransportType::RDMA;
  }

  /* Returns the current transport state. */
  TransportState state() const noexcept override {
    return state_;
  }

  /*
   * Allocate RDMA resources and return serialized connection parameters.
   *
   * Creates one CQ per NIC and numQps RC queue pairs (round-robin across
   * NICs), transitions each QP to INIT state, and serializes addressing
   * info into a TransportInfo byte vector.
   *
   * Must be called exactly once before connect(). Returns an empty vector
   * and sets state to Error on failure.
   */
  TransportInfo bind() override;

  /*
   * Establish connection with a remote peer.
   *
   * Deserializes the remote peer's TransportInfo, validates QP count match,
   * negotiates path MTU as min(local, remote), then transitions each QP
   * through RTR -> RTS. Sets GRH conditionally based on remote link layer.
   *
   * Precondition: bind() must have been called successfully.
   */
  Status connect(std::span<const uint8_t> remoteInfo) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> put(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> get(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> send(
      RegisteredSegment::Span src,
      const RequestOptions& options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> send(
      Segment::Span src,
      const RequestOptions& options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> recv(
      Segment::Span dst,
      const RequestOptions& options = {}) override;

  /*
   * Tear down RDMA resources. Transitions QPs to ERROR state for graceful
   * drain, then destroys all QPs and CQs. Safe to call multiple times.
   * Also called by the destructor.
   */
  void shutdown() override;

 private:
  // --- Private methods ---

  enum class IOType {
    Put,
    Get,
    Send,
    Recv,
  };

  /// Tracks completion of a batch of RDMA work requests from a single
  /// call. All fields are accessed only on the EventBase thread, so no
  /// synchronization is needed.
  class Task {
   public:
    explicit Task(IOType type) : type_(type) {}
    std::future<Status> get_future() {
      return promise_.get_future();
    }

    void postFinished() {
      postFinished_ = true;
    }

    void posted(uint32_t numWrs) {
      remainingWrs_ += numWrs;
    }

    void recordCompletion(Result<uint32_t> numWrs) {
      if (numWrs.hasError()) {
        if (!error_) {
          error_ = std::move(numWrs).error();
        }
        postFinished_ = true;
        return;
      }
      remainingWrs_ -= numWrs.value();
    }

    IOType type() const {
      return type_;
    }

    bool isDone() const {
      return isFullyDrained() || hasError();
    }

    bool isFullyDrained() const {
      return postFinished_ && remainingWrs_ == 0;
    }

    bool hasError() const {
      return error_.has_value();
    }

    void fulfill() {
      if (!fulfilled_) {
        fulfilled_ = true;
        if (error_) {
          promise_.set_value(std::move(*error_));
        } else {
          promise_.set_value(Ok());
        }
      }
    }

   private:
    IOType type_{};
    bool fulfilled_{false};
    bool postFinished_{false};
    uint32_t remainingWrs_{0};
    std::optional<Err> error_;
    std::promise<Status> promise_;
  };

  /// Work request descriptor for a single RDMA chunk.
  /// Holds the ibv_send_wr, ibv_sge, and handle pointers needed for posting.
  /// The caller must keep the underlying Segment alive until the async task
  /// completes, so storing handle pointers here is safe.
  struct SendWr {
    ibv_send_wr wr{};
    ibv_sge sge{};
    const RdmaRegistrationHandle* localHandle{nullptr};
    const RdmaRemoteRegistrationHandle* remoteHandle{nullptr};
  };

  struct PutGetTransfer {
    uint32_t taskId{};
    std::unique_ptr<std::vector<SendWr>> sendWrs;
    size_t idx{0};
    std::shared_ptr<Task> task;
  };

  struct SendRecvTransfer {
    explicit SendRecvTransfer(IOType opType, Segment::Span data);
    ~SendRecvTransfer();

    IOType opType;
    Segment::Span data;
    uint32_t taskId{};
    std::optional<cudaStream_t> stream;
    uint32_t initStep{0};
    uint64_t totalSteps{0};

    uint64_t copied{0};
    uint64_t notified{0};
    uint64_t done{0};

    std::vector<RdmaSlab> localSlabs;
    std::shared_ptr<Task> task;
    std::optional<CopyEngine> copyEngine;

    void* ptr(size_t offset) const {
      return static_cast<char*>(data.mutable_data()) + offset;
    }
  };

  struct PendingTransfer {
    std::unique_ptr<PutGetTransfer> putGetTransfer{nullptr};
    std::unique_ptr<SendRecvTransfer> sendRecvTransfer{nullptr};
  };

  struct PendingCompletion {
    uint32_t taskId;
    std::shared_ptr<Task> task;
  };

  // --- Caller thread methods ---

  /// Validates requests and builds a flat list of 512K-chunk SendWrs.
  /// Called on the caller thread before dispatch to EventBase.
  /// Returns error if any request has invalid handles or mismatched sizes.
  Result<std::unique_ptr<std::vector<SendWr>>> buildSendWrs(
      std::span<const TransferRequest> requests,
      ibv_wr_opcode opcode);

  /// Validates a single request and extracts matching registration handles.
  Status preprocessRequest(
      const TransferRequest& request,
      RdmaRegistrationHandle const** localHandle,
      RdmaRemoteRegistrationHandle const** remoteHandle) const;

  // --- EventBase thread methods ---

  /// Entry point: dispatches buildSendWrs result to EventBase for posting.
  std::future<Status> rdmaPutGetTransfer(
      std::span<const TransferRequest> requests,
      ibv_wr_opcode opcode,
      const RequestOptions& options);

  std::future<Status> rdmaSendRecvTransfer(
      IOType ioType,
      Segment::Span data,
      const RequestOptions& options);

  /// Unified IO processing loop. Runs exclusively on EventBase thread.
  /// Posts WRs from pendingTransfers_, polls CQs, fulfills promises in order.
  void ioLooper() noexcept;

  bool putGetIoProcess(PutGetTransfer& entry) noexcept;
  bool sendRecvIoProcess(SendRecvTransfer& entry) noexcept;

  /// Computes per-QP available SQ capacity. When bufNuma >= 0 (host memory with
  /// a known NUMA node), QPs whose NIC is on a different NUMA node are capped
  /// to zero so put/get only targets NUMA-local NICs; falls back to all NICs if
  /// none are NUMA-local.
  uint32_t getQpAvail(std::vector<uint32_t>& qpAvail, int bufNuma = -1);

  uint32_t assignToQps(
      const uint32_t remaining,
      const uint32_t totalAvail,
      std::vector<uint32_t>& qpAvail,
      std::vector<uint32_t>& qpAssigned);

  /// Distributes SendWrs across QPs proportional to available capacity
  /// and posts them.
  ///
  /// Returns:
  ///   - Ok(n):  n WRs committed. If n == 0, all QPs are full (poll & retry).
  ///   - Error:  postSend failed on a QP. The task is already errored and
  ///             postFinished. Caller must NOT post more WRs — just let
  ///             the poll chain drain remaining CQEs (from earlier QPs
  ///             and the flush WR).
  ///
  /// Correctness invariant: numWrsPerQp_[q] is incremented by exactly
  /// the number of WRs that will generate CQEs (directly or via flush),
  /// ensuring pollCompletions will decrement it back to zero.
  Result<uint32_t>
  spray(std::vector<SendWr>& wrs, size_t& idx, uint32_t taskId, Task& task);

  /// Posts a chain of WRs to a single QP. On partial failure, posts a
  /// flush WR so the HCA generates a CQE for the consumed unsignaled WRs.
  ///
  /// Returns: number of WRs committed to numWrsPerQp_ (including flush),
  ///          or 0 on complete failure (nothing was consumed).
  ///
  /// Error behavior:
  ///   - Partial failure (consumed > 0): flush WR posted, returns consumed+1.
  ///     task->recordCompletion(error) is called. task->posted(consumed+1)
  ///     is called.
  ///   - Flush failure: transport set to Error state.
  ///   - Complete failure (consumed == 0): returns 0.
  ///     task->recordCompletion(error) is called. No counter changes.
  uint32_t postSend(
      uint32_t qpIdx,
      ibv_send_wr* head,
      uint32_t count,
      uint32_t taskId,
      Task& task,
      std::optional<uint16_t> slot = std::nullopt);

  /// Polls all CQs and routes completions to their tasks via inflightTasks_.
  Status pollCompletions();

  // --- Copy-based send/recv pipeline (cursor model) ---

  /// Ctrl buffer helpers.
  std::atomic_ref<uint32_t> ctrlCts(uint16_t slotIdx);
  std::atomic_ref<uint32_t> ctrlNotify(uint16_t slotIdx, uint32_t qpIdx);
  uint64_t remoteCtrlCtsAddr(uint16_t slotIdx) const;
  uint64_t remoteCtrlNotifyAddr(uint16_t slotIdx, uint32_t qpIdx) const;
  size_t ctrlBufferSize() const;

  // --- Send pipeline ---

  Status sendProgress(SendRecvTransfer& transfer);
  void
  sendCopyProgress(SendRecvTransfer& transfer, uint32_t depth, size_t slabSize);
  Status sendTransmitProgress(
      SendRecvTransfer& transfer,
      uint32_t depth,
      size_t slabSize);
  Result<size_t> postSlabTransfer(
      SendRecvTransfer& transfer,
      RdmaSlab& localSlab,
      uint16_t remoteSlab,
      size_t len,
      uint32_t slot,
      uint32_t taskId);

  // --- Recv pipeline ---

  Status recvProgress(SendRecvTransfer& transfer);
  void recvNotifyProgress(
      SendRecvTransfer& transfer,
      uint32_t depth,
      size_t slabSize);
  void
  recvCopyProgress(SendRecvTransfer& transfer, uint32_t depth, size_t slabSize);
  Status recvDoneProgress(SendRecvTransfer& transfer, uint32_t depth);
  Result<bool>
  postCts(uint32_t slot, uint16_t slabIdx, uint32_t taskId, Task& task);

  friend class GetQpAvailNumaTest;

  const std::shared_ptr<IbvApi> ibvApi_;
  const std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<CudaDriverApi> cudaDriverApi_;
  std::shared_ptr<DeviceAdapter> deviceAdapter_;
  EventBase* evb_{nullptr};

  std::string name_;
  std::shared_ptr<std::vector<NicResources>> nicsHandle_;
  std::span<NicResources> nics_;
  const RdmaTransportConfig config_;

  std::vector<ibv_cq*> cqs_;
  std::vector<ibv_qp*> qps_;
  std::vector<uint32_t> psns_;

  uint64_t domainId_{0};
  uint64_t remoteDomainId_{0};

  uint8_t remoteNumNics_{0};

  /// Monotonically increasing Task ID. Accessed only on EventBase thread.
  uint32_t nextTaskId_{0};

  /// Number of pending CQEs for each nic. Accessed only on EventBase thread.
  std::vector<int> numPendingCqe_;

  /// Number of pending CQEs for each slot. Accessed only on EventBase thread.
  std::vector<int> slotPendingCqe_;

  /// Transfers awaiting WR posting. Accessed only on EventBase.
  std::deque<PendingTransfer> pendingTransfers_;

  /// Transfers awaiting CQ completion. Accessed only on EventBase.
  std::deque<PendingCompletion> pendingCompletions_;

  /// Whether ioLooper is scheduled on the EventBase. Accessed only on
  /// EventBase.
  bool ioLooperScheduled_{false};

  /// Maps taskId → task for in-flight requests. Accessed only on EventBase.
  std::unordered_map<uint32_t, std::shared_ptr<Task>> inflightTasks_;

  /// Per-QP inflight WR count. Accessed only on EventBase thread.
  std::vector<uint32_t> numWrsPerQp_;

  /// Maps (NIC index, ibv QP number) → QP index for completion routing.
  /// QP numbers are only unique within an RDMA device.
  std::unordered_map<uint64_t, uint32_t> qpNumToIdx_;

  /// State of the transport.
  TransportState state_{TransportState::Disconnected};

  /// transport info. set up and return by bind()
  RdmaTransportInfo info_;

  /// guard for shutdown()
  std::atomic<bool> shutdown_{false};

  // --- Send/recv state and pool ---

  std::shared_ptr<RdmaSlabPool> slabPool_;

  // Ctrl buffer for send/recv pipeline flow control.
  void* ctrlBuffer_{nullptr};
  std::vector<ibv_mr*> ctrlMrs_;

  // Remote peer's channel info (populated by connect)
  RdmaTransportInfo::RegisteredBuffer remoteCtrlBuffer_;
  RdmaTransportInfo::RegisteredBuffer remoteSlabBuffer_;
};

// ---------------------------------------------------------------------------
// RdmaTransportFactory
// ---------------------------------------------------------------------------

/*
 * Factory for creating RdmaTransport instances.
 *
 * Owns device-level RDMA resources: opens one or more IB devices by name,
 * creates a protection domain per device, and queries port attributes
 * (LID, GID, MTU, link layer). Supports multi-NIC setups.
 *
 * Port discovery: if portNum is not specified (nullopt), the factory
 * auto-discovers the first active port on each device.
 *
 * Throws std::runtime_error from the constructor if any device cannot be
 * opened or queried. Previously opened devices are cleaned up on failure.
 *
 * TODO: Replace manual cleanup with EXIT_SCOPE macro (core/Utils.h).
 */
class RdmaTransportFactory : public TransportFactory {
 public:
  /*
   * Query whether RDMA transport is available on this platform.
   *
   * Probes for RDMA hardware by loading libibverbs and verifying that
   * every IB device has at least one active port. Returns Ok() on
   * success, or an error describing why RDMA is not available.
   *
   * @param ibvApi  Optional IbvApi instance for dependency injection
   *                (testing). If nullptr, a default instance is created.
   */
  static Status supported(std::shared_ptr<IbvApi> ibvApi = nullptr);

  /*
   * Open RDMA devices and initialize per-NIC resources.
   *
   * @param deviceNames  List of IB device names (e.g., {"mlx5_0", "mlx5_1"}).
   *                     Must be non-empty.
   * @param config       Transport config (numQps, gidIndex, QP attributes).
   * @param ibvApi       IbvApi instance for dependency injection. If nullptr,
   *                     a default IbvApi is created and init()'d.
   * @param portNum      Physical port number. If nullopt, auto-discovers the
   *                     first active port on each device.
   */
  explicit RdmaTransportFactory(
      const std::vector<std::string>& deviceNames,
      EventBase* evb,
      RdmaTransportConfig config = {},
      std::shared_ptr<IbvApi> ibvApi = nullptr,
      std::shared_ptr<CudaDriverApi> cudaDriverApi = nullptr,
      std::shared_ptr<CudaApi> cudaApi = nullptr,
      std::optional<uint8_t> portNum = std::nullopt,
      std::shared_ptr<DeviceAdapter> deviceAdapter = nullptr);

  ~RdmaTransportFactory() override = default;

  RdmaTransportFactory(const RdmaTransportFactory&) = delete;
  RdmaTransportFactory& operator=(const RdmaTransportFactory&) = delete;
  RdmaTransportFactory(RdmaTransportFactory&&) = delete;
  RdmaTransportFactory& operator=(RdmaTransportFactory&&) = delete;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& segment) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t segmentLength,
      std::span<const uint8_t> payload) override;

  /*
   * Create a new RdmaTransport with all NICs owned by this factory.
   * Uses numQps from the factory's RdmaTransportConfig.
   */
  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> peerTopology) override;

  /* Return the topology information */
  std::vector<uint8_t> getTopology() override;

  uint64_t dmaBufFallbackCount() const {
    return dmaBufFallbackCount_.load(std::memory_order_relaxed);
  }

 private:
  Status canConnect(std::span<const uint8_t> peerTopology) override;

  std::shared_ptr<IbvApi> ibvApi_;
  std::shared_ptr<CudaDriverApi> cudaDriverApi_;
  std::shared_ptr<CudaApi> cudaApi_;

  EventBase* evb_{nullptr};
  uint64_t domainId_{0};
  std::atomic<uint64_t> dmaBufFallbackCount_{0};
  std::shared_ptr<std::vector<NicResources>> nicsHandle_;
  std::shared_ptr<DeviceAdapter> deviceAdapter_;
  const RdmaTransportConfig config_;
  std::shared_ptr<RdmaSlabPool> slabPool_;

  /// Set once the "Data Direct requested but unavailable" fallback has been
  /// logged, so this per-factory-invariant warning fires at most once rather
  /// than on every registerSegment call. Atomic so the check-then-set latch is
  /// race-free if registerSegment is ever called concurrently.
  std::atomic<bool> dataDirectFallbackWarned_{false};
};

} // namespace uniflow
