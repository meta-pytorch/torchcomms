// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "comms/uniflow/controller/TcpController.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/Transport.h"

namespace uniflow {

// Defined in TcpTransport.cpp.
struct TcpOpState;

class TcpRemoteRegistrationHandle;

struct TcpTransportInfo {
  struct __attribute__((packed)) Header {
    uint16_t port{0};
    uint16_t hostLen{0};
  };

  std::string host{"127.0.0.1"};
  uint16_t port{0};

  TransportInfo serialize() const;
  static Result<TcpTransportInfo> deserialize(std::span<const uint8_t> data);
};

class CudaApi;

/// Factory-shared registry mapping a locally-assigned segment id to the
/// registered host buffer. registerSegment() (on the factory) populates it;
/// every TcpTransport created by that factory shares it so an inbound WRITE or
/// READ_REQUEST naming a segId can be resolved to the local buffer.
/// Thread-safe.
class TcpSegmentRegistry {
 public:
  struct Entry {
    void* ptr{nullptr};
    size_t len{0};
    MemoryType memType{MemoryType::DRAM};
    int deviceId{-1};
  };

  void
  add(uint64_t segId, void* ptr, size_t len, MemoryType memType, int deviceId) {
    std::lock_guard<std::mutex> lk(mu_);
    segs_[segId] = Entry{ptr, len, memType, deviceId};
  }

  std::optional<Entry> find(uint64_t segId) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = segs_.find(segId);
    if (it == segs_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void erase(uint64_t segId) {
    std::lock_guard<std::mutex> lk(mu_);
    segs_.erase(segId);
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, Entry> segs_;
};

/// Tracks a single in-flight request awaiting its reply. Shared completion
/// state is aggregated per put()/get() call in TcpOpState.
struct TcpInflight {
  std::shared_ptr<TcpOpState> state;
  void* dst{nullptr}; // READ destination; nullptr for WRITE.
  size_t len{0};
  bool isRead{false};
  MemoryType memType{MemoryType::DRAM};
  int deviceId{-1};
  // Caller's CUDA stream (as void*) for VRAM staging on the get() consumer
  // path, so the H2D is ordered against the framework's stream (not the null
  // stream). nullptr => null stream.
  void* stream{nullptr};
};

/// One queued outbound frame. `onSent`, when set, is completed by the sender
/// thread once the frame is flushed to the socket (two-sided send()).
struct TcpOutItem {
  std::vector<uint8_t> bytes;
  std::shared_ptr<TcpOpState> onSent;
};

/// A posted recv() awaiting a matching inbound SEND frame.
struct TcpPendingRecv {
  void* dst{nullptr};
  size_t cap{0};
  std::shared_ptr<TcpOpState> state;
  MemoryType memType{MemoryType::DRAM};
  int deviceId{-1};
};

/// Native, self-contained TCP data transport.
///
/// Establishes one full-duplex data connection per peer (deterministic
/// listener/dialer role by host:port ordering). Three threads run per
/// connection:
///  - reader: blocking recv + demultiplex; never blocks on a send, so it always
///    drains inbound (avoids the mutual-READ deadlock).
///  - sender: drains an outbound frame queue and performs all socket sends
///    (requests, ACKs, READ replies), serialized by construction.
///
/// put() copies a local buffer into a peer segment (WRITE + ACK). get() is
/// emulated as a pull: READ_REQUEST -> peer replies READ_REPLY with the data.
///
/// Independent of NIXL/UCX/folly. send()/recv() implement a two-sided
/// rendezvous (FIFO-matched via pendingRecvs_/unmatchedSends_).
class TcpTransport : public Transport {
 public:
  TcpTransport(
      int deviceId,
      EventBase* evb,
      std::shared_ptr<TcpSegmentRegistry> registry,
      controller::TcpSocketConfig config = {},
      std::string host = "127.0.0.1",
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  ~TcpTransport() override;

  TcpTransport(const TcpTransport&) = delete;
  TcpTransport& operator=(const TcpTransport&) = delete;
  TcpTransport(TcpTransport&&) = delete;
  TcpTransport& operator=(TcpTransport&&) = delete;

  const std::string& name() const noexcept override {
    return name_;
  }

  TransportType transportType() const noexcept override {
    return TransportType::TCP;
  }

  TransportState state() const noexcept override {
    return state_;
  }

  TransportInfo bind() override;
  Status connect(std::span<const uint8_t> remoteInfo) override;

  std::future<Status> put(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  std::future<Status> get(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  std::future<Status> send(
      RegisteredSegment::Span src,
      const RequestOptions& options = {}) override;

  std::future<Status> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {}) override;

  std::future<Status> send(
      Segment::Span src,
      const RequestOptions& options = {}) override;

  std::future<Status> recv(
      Segment::Span dst,
      const RequestOptions& options = {}) override;

  void shutdown() override;

 private:
  Result<const TcpRemoteRegistrationHandle*> findRemoteHandle(
      const RemoteRegisteredSegment::Span& span) const;

  // Reader thread: blocking recv + demultiplex until the connection closes.
  void readerLoop() noexcept;
  // Sender thread: drains outQueue_ and performs all socket sends.
  void senderLoop() noexcept;
  // Handles one fully-received inbound frame (reader thread).
  void handleFrame(std::span<const uint8_t> frame) noexcept;
  // Queues one fire-and-forget framed message for the sender thread.
  void enqueueFrame(std::vector<uint8_t> frame);
  // Queues a framed message whose completion is reported via `onSent` once the
  // frame is flushed (two-sided send()).
  void enqueueSendFrame(
      std::vector<uint8_t> frame,
      std::shared_ptr<TcpOpState> onSent);
  // Shared bodies for the zero-copy and copy-based send/recv overloads.
  std::future<Status>
  sendImpl(const void* data, size_t len, MemoryType memType, int deviceId);
  std::future<Status>
  recvImpl(void* dst, size_t cap, MemoryType memType, int deviceId);
  // Host-staging copies used for VRAM segments (plain memcpy for DRAM):
  // hostFromDevice = D2H, deviceFromHost = H2D. Synchronous. `stream` (a
  // cudaStream_t as void*, nullptr => null stream) orders the copy against the
  // caller's framework stream, matching the RDMA CopyEngine.
  Status hostFromDevice(
      void* hostDst,
      const void* devSrc,
      size_t len,
      int deviceId,
      void* stream = nullptr);
  Status deviceFromHost(
      void* devDst,
      const void* hostSrc,
      size_t len,
      int deviceId,
      void* stream = nullptr);
  // Fails every pending put/get/recv/queued-send; marks the conn broken.
  void failAllPending(const char* message);

  [[maybe_unused]] int deviceId_{-1};
  EventBase* evb_{nullptr};
  std::shared_ptr<TcpSegmentRegistry> registry_;
  std::shared_ptr<CudaApi> cudaApi_;
  controller::TcpSocketConfig config_;
  std::string name_{"tcp"};
  TransportState state_{TransportState::Disconnected};

  std::string host_{"127.0.0.1"};
  uint16_t port_{0};

  std::unique_ptr<controller::AsyncTcpServer> server_;
  std::unique_ptr<controller::Conn> dataConn_;

  std::thread reader_;
  std::thread sender_;
  std::atomic<bool> running_{false};
  std::atomic<bool> connBroken_{false};

  // Outbound frame queue drained by the sender thread. Decoupling all sends
  // from the reader thread is what prevents a mutual-READ deadlock (two peers'
  // readers both blocked mid-send while neither drains its recv).
  std::mutex outMu_;
  std::condition_variable outCv_;
  std::deque<TcpOutItem> outQueue_;
  bool outClosed_{false};

  std::mutex inflightMu_;
  std::unordered_map<uint64_t, TcpInflight> inflight_;
  std::atomic<uint64_t> nextReqId_{1};

  // Two-sided send/recv rendezvous (FIFO-matched on the single stream):
  // inbound SEND frames waiting for a posted recv, and posted recvs waiting
  // for an inbound SEND frame.
  std::mutex recvMu_;
  std::deque<TcpPendingRecv> pendingRecvs_;
  std::deque<std::vector<uint8_t>> unmatchedSends_;
};

class TcpTransportFactory : public TransportFactory {
 public:
  static Status supported();

  // @param host  Local IP the transport binds to and advertises to peers.
  //              Defaults to loopback; pass a routable IP (e.g. an eth2 IPv6)
  //              for cross-host transfers.
  explicit TcpTransportFactory(
      int deviceId,
      EventBase* evb,
      controller::TcpSocketConfig config = {},
      std::string host = "127.0.0.1",
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& segment) override;

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t segmentLength,
      std::span<const uint8_t> payload) override;

  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> peerTopology) override;

  std::vector<uint8_t> getTopology() override;

  Status canConnect(std::span<const uint8_t> peerTopology) override;

 private:
  int deviceId_{-1};
  EventBase* evb_{nullptr};
  controller::TcpSocketConfig config_;
  std::string host_{"127.0.0.1"};
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<TcpSegmentRegistry> registry_{
      std::make_shared<TcpSegmentRegistry>()};
  std::atomic<uint64_t> nextSegId_{1};
};

} // namespace uniflow
