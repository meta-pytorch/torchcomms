// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstring>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>

#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/Transport.h"

namespace uniflow {

constexpr uint8_t kRdmaVersion{1};

// Forward declarations.
class RdmaRegistrationHandle;
class RdmaRemoteRegistrationHandle;

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
  uint8_t gidIndex{3}; /* GID table index for RoCE addressing (3 = RoCEv2). */
  uint8_t timeout{14}; /* IB timeout exponent for QP retransmission. */
  uint8_t retryCnt{7}; /* Number of retries before reporting an error. */
  uint8_t trafficClass{0}; /* Traffic class for GRH (QoS / DSCP). */
  uint16_t pkeyIndex{0}; /* Partition key index for QP INIT. */
  uint32_t maxWr{128}; /* Max outstanding send WRs per QP. */
  uint32_t maxSge{1}; /* Max scatter/gather entries per WR. */
  uint32_t maxInlineData{16}; /* Max inline data bytes per WR. */
  size_t chunkSize{512 * 1024}; /* Transfer chunk size in bytes (512KB). */
};

/*
 * Per-NIC RDMA resources.
 *
 * Populated by RdmaTransportFactory during device initialization.
 * Ownership: the factory owns the underlying ibverbs objects (context, PD).
 * Transports borrow these resources — they must not outlive the factory.
 */
struct NicResources {
  ibv_context* ctx{nullptr}; /* Opened device context. */
  ibv_pd* pd{nullptr}; /* Protection domain on this device. */
  uint16_t lid{0}; /* Local identifier (IB fabrics). */
  ibv_gid gid{}; /* GID for RoCE / IB with GRH. */
  ibv_mtu mtu{IBV_MTU_4096}; /* Active MTU from port query. */
  int linkLayer{IBV_LINK_LAYER_ETHERNET}; /* IB or Ethernet (RoCE). */
  uint8_t portNum{1}; /* Physical port number on the HCA. */
  bool dmaBufSupported{false}; /* Kernel supports DMA-BUF MR registration. */
};

/*
 * RDMA transport connection metadata — serialization and deserialization.
 *
 * Wire format (packed, memcpy-based):
 *   Header (11 bytes: version, numQps, numNics, domainId)
 *   + NicInfo[numNics]
 *   + QpInfo[numQps]
 *
 * QP-to-NIC mapping: QP i belongs to NIC (i % numNics).
 */
struct RdmaTransportInfo {
  /* Header in the wire format. */
  struct __attribute__((packed)) Header {
    uint8_t version{kRdmaVersion};
    uint8_t numQps{0};
    uint8_t numNics{0};
    uint64_t domainId{0};
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

  /* Header fields. */
  Header header;

  /* Deserialized per-NIC and per-QP data. */
  std::vector<NicInfo> nicInfos;
  std::vector<QpInfo> qpInfos;

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
   *                numQps must be <= 255.
   */
  RdmaTransport(
      std::shared_ptr<IbvApi> ibvApi,
      EventBase* evb,
      std::vector<NicResources> nics,
      uint64_t domainId,
      RdmaTransportConfig config = {});

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
      std::span<TransferRequest> requests,
      RequestOptions options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> get(
      std::span<TransferRequest> requests,
      RequestOptions options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> send(
      RegisteredSegment::Span src,
      RequestOptions options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Status> send(Segment::Span src, RequestOptions options = {})
      override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Result<size_t>> recv(
      RegisteredSegment::Span dst,
      RequestOptions options = {}) override;

  /* Not yet implemented. Returns ErrCode::NotImplemented. */
  std::future<Result<size_t>> recv(
      Segment::Span dst,
      RequestOptions options = {}) override;

  /*
   * Tear down RDMA resources. Transitions QPs to ERROR state for graceful
   * drain, then destroys all QPs and CQs. Safe to call multiple times.
   * Also called by the destructor.
   */
  void shutdown() override;

 private:
  /// Tracks completion of a batch of RDMA work requests from a single
  /// rdmaTransfer() call. All fields are accessed only on the EventBase
  /// thread, so no synchronization is needed.
  class Task {
   public:
    std::future<Status> get_future() {
      return promise_.get_future();
    }

    bool isComplete() {
      return isPromiseSet_ && postFinished_ && remainingWrs_ == 0;
    }

    void postFinished() {
      postFinished_ = true;
    }

    void posted(uint32_t numWrs) {
      remainingWrs_ += numWrs;
    }

    void complete(Result<uint32_t> numWrs) {
      if (numWrs.hasError()) {
        postFinished();
        setPromise(std::move(numWrs).error());
        return;
      }
      remainingWrs_ -= numWrs.value();
      if (remainingWrs_ == 0 && postFinished_) {
        setPromise(Ok());
      }
    }

   private:
    bool isPromiseSet_{false};
    bool postFinished_{false};
    uint32_t remainingWrs_{0};
    std::promise<Status> promise_;

    void setPromise(Status status) {
      if (!isPromiseSet_) {
        isPromiseSet_ = true;
        promise_.set_value(std::move(status));
      }
    }
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

  struct PendingRequests {
    uint32_t taskId;
    std::unique_ptr<std::vector<SendWr>> sendWrs;
  };

  // --- Caller thread methods ---

  /// Validates requests and builds a flat list of 512K-chunk SendWrs.
  /// Called on the caller thread before dispatch to EventBase.
  /// Returns error if any request has invalid handles or mismatched sizes.
  Result<std::unique_ptr<std::vector<SendWr>>> buildSendWrs(
      std::span<TransferRequest> requests,
      ibv_wr_opcode opcode);

  /// Validates a single request and extracts matching registration handles.
  Status preprocessRequest(
      const TransferRequest& request,
      RdmaRegistrationHandle const** localHandle,
      RdmaRemoteRegistrationHandle const** remoteHandle) const;

  // --- EventBase thread methods ---

  /// Entry point: dispatches buildSendWrs result to EventBase for posting.
  std::future<Status> rdmaTransfer(
      std::span<TransferRequest> requests,
      ibv_wr_opcode opcode,
      RequestOptions options);

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
  Result<uint32_t> spray(
      std::vector<SendWr>& wrs,
      size_t& idx,
      uint32_t taskId,
      std::shared_ptr<Task>& task);

  /// Posts a chain of WRs to a single QP. On partial failure, posts a
  /// flush WR so the HCA generates a CQE for the consumed unsignaled WRs.
  ///
  /// Returns: number of WRs committed to numWrsPerQp_ (including flush),
  ///          or 0 on complete failure (nothing was consumed).
  ///
  /// Error behavior:
  ///   - Partial failure (consumed > 0): flush WR posted, returns consumed+1.
  ///     task->complete(error) is called. task->posted(consumed+1) is called.
  ///   - Flush failure: transport set to Error state.
  ///   - Complete failure (consumed == 0): returns 0.
  ///     task->complete(error) is called. No counter changes.
  uint32_t postSend(
      uint32_t qpIdx,
      ibv_send_wr* head,
      uint32_t count,
      uint32_t taskId,
      std::shared_ptr<Task>& task);

  /// Polls all CQs and dispatches completions to their tasks.
  /// Re-dispatches itself on EventBase if the given task is still in-flight.
  void pollCompletions(uint32_t taskId, bool retry);

  const std::shared_ptr<IbvApi> ibvApi_;
  EventBase* evb_{nullptr};

  std::string name_;
  const std::vector<NicResources> nics_;
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
  std::vector<int> numPendingCqe_{0};

  /// pending requests queue for post-order guarantee. Accessed only on
  /// EventBase.
  std::queue<PendingRequests> pendingRequests_;

  /// Maps taskId → task for in-flight requests. Accessed only on EventBase.
  std::unordered_map<uint32_t, std::shared_ptr<Task>> inflightTasks_;

  /// Per-QP inflight WR count. Accessed only on EventBase thread.
  std::vector<uint32_t> numWrsPerQp_;

  /// Maps ibv QP number → QP index for completion routing.
  std::unordered_map<uint32_t, uint32_t> qpNumToIdx_;

  TransportState state_{TransportState::Uninitialized};
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
      std::optional<uint8_t> portNum = std::nullopt);

  ~RdmaTransportFactory() override;

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

 private:
  Status canConnect(std::span<const uint8_t> peerTopology) override;

  /* Find the first active port on the device. Returns 0 on failure. */
  uint8_t findActivePort(ibv_context* ctx);

  std::shared_ptr<IbvApi> ibvApi_;
  std::shared_ptr<CudaDriverApi> cudaDriverApi_;
  EventBase* evb_{nullptr};
  uint64_t domainId_{0};
  size_t pageSize_{0};
  std::vector<NicResources> nics_;
  const RdmaTransportConfig config_;
};

} // namespace uniflow
