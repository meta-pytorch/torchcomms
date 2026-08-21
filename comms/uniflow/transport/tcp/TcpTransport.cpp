// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpTransport.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"
#include "comms/uniflow/transport/tcp/TcpWireProtocol.h"

namespace uniflow {

namespace {
// Fallback when TcpSocketConfig::connTimeout is unset.
constexpr int kDefaultHandshakeTimeoutSeconds = 30;
} // namespace

namespace {

// The controller frames each TcpConn message with a 4-byte length and caps it
// at 64 MiB; keep each chunk (header + payload) safely under that so large
// put/get transfers are split across multiple frames.
constexpr size_t kMaxChunkSize = 4u << 20; // 4 MiB
std::vector<uint8_t> makeHeaderFrame(TcpOp op, uint64_t reqId) {
  TcpMsgHeader header;
  header.op = static_cast<uint8_t>(op);
  header.reqId = reqId;
  return serializeTcpHeader(header);
}

// Representation-independent ordering key for a bind host. Parses IPv6 then
// IPv4 to raw address bytes so different textual forms of the same address
// (e.g. 2001:db8::1 vs its fully-expanded form) order and compare equal on both
// peers; falls back to the raw string for non-numeric hosts. Used only for
// deterministic listener/dialer role assignment.
std::string hostOrderKey(const std::string& host) {
  in6_addr a6{};
  if (::inet_pton(AF_INET6, host.c_str(), &a6) == 1) {
    return "6:" + std::string(reinterpret_cast<const char*>(&a6), sizeof(a6));
  }
  in_addr a4{};
  if (::inet_pton(AF_INET, host.c_str(), &a4) == 1) {
    return "4:" + std::string(reinterpret_cast<const char*>(&a4), sizeof(a4));
  }
  return "s:" + host;
}

} // namespace

// ---------------------------------------------------------------------------
// TcpTransportInfo
// ---------------------------------------------------------------------------

TransportInfo TcpTransportInfo::serialize() const {
  Header header{
      .port = port,
      .hostLen = static_cast<uint16_t>(host.size()),
  };
  TransportInfo data(sizeof(Header) + host.size());
  std::memcpy(data.data(), &header, sizeof(header));
  if (!host.empty()) {
    std::memcpy(data.data() + sizeof(header), host.data(), host.size());
  }
  return data;
}

Result<TcpTransportInfo> TcpTransportInfo::deserialize(
    std::span<const uint8_t> data) {
  if (data.size() < sizeof(Header)) {
    return Err(ErrCode::InvalidArgument, "tcp transport info is truncated");
  }

  Header header;
  std::memcpy(&header, data.data(), sizeof(header));
  if (data.size() != sizeof(header) + header.hostLen) {
    return Err(ErrCode::InvalidArgument, "tcp transport info size mismatch");
  }

  TcpTransportInfo info;
  info.port = header.port;
  info.host.assign(
      reinterpret_cast<const char*>(data.data() + sizeof(header)),
      header.hostLen);
  return info;
}

// ---------------------------------------------------------------------------
// TcpTransport
// ---------------------------------------------------------------------------

TcpTransport::TcpTransport(
    int deviceId,
    EventBase* evb,
    std::shared_ptr<TcpSegmentRegistry> registry,
    controller::TcpSocketConfig config,
    std::string host,
    std::shared_ptr<CudaApi> cudaApi)
    : deviceId_(deviceId),
      evb_(evb),
      registry_(std::move(registry)),
      cudaApi_(std::move(cudaApi)),
      config_(std::move(config)) {
  if (!registry_) {
    registry_ = std::make_shared<TcpSegmentRegistry>();
  }
  if (!cudaApi_) {
    cudaApi_ = std::make_shared<CudaApi>();
  }
  if (!host.empty()) {
    host_ = std::move(host);
  }
}

Status TcpTransport::hostFromDevice(
    void* hostDst,
    const void* devSrc,
    size_t len,
    int deviceId,
    void* stream) {
  if (len == 0) {
    return Ok();
  }
  auto s = static_cast<cudaStream_t>(stream);
  CudaDeviceGuard guard(*cudaApi_, deviceId);
  auto st =
      cudaApi_->memcpyAsync(hostDst, devSrc, len, cudaMemcpyDeviceToHost, s);
  if (!st) {
    return st;
  }
  return cudaApi_->streamSynchronize(s);
}

Status TcpTransport::deviceFromHost(
    void* devDst,
    const void* hostSrc,
    size_t len,
    int deviceId,
    void* stream) {
  if (len == 0) {
    return Ok();
  }
  auto s = static_cast<cudaStream_t>(stream);
  CudaDeviceGuard guard(*cudaApi_, deviceId);
  auto st =
      cudaApi_->memcpyAsync(devDst, hostSrc, len, cudaMemcpyHostToDevice, s);
  if (!st) {
    return st;
  }
  return cudaApi_->streamSynchronize(s);
}

TcpTransport::~TcpTransport() {
  shutdown();
}

TransportInfo TcpTransport::bind() {
  std::lock_guard<std::mutex> lk(lifecycleMu_);
  if (shutdown_.load(std::memory_order_acquire)) {
    UNIFLOW_LOG_ERROR("TcpTransport::bind: transport is already shut down");
    return TransportInfo{};
  }
  server_ = std::make_unique<controller::AsyncTcpServer>(
      host_ + ":0", config_, *evb_);
  auto status = server_->init();
  if (!status) {
    UNIFLOW_LOG_ERROR(
        "TcpTransport::bind: server init failed: {}", status.error().message());
    state_ = TransportState::Error;
    server_.reset();
    return TransportInfo{};
  }
  port_ = static_cast<uint16_t>(server_->getPort());
  state_ = TransportState::Initialized;

  TcpTransportInfo info;
  info.host = host_;
  info.port = port_;
  return info.serialize();
}

Status TcpTransport::connect(std::span<const uint8_t> remoteInfo) {
  // Held across the handshake wait below, so a concurrent shutdown() cannot
  // interleave with the installation of dataConn_/reader_/sender_ at the end of
  // this function. The flag check is what stops a connect() queued behind a
  // completed shutdown() from bringing the transport back to life.
  std::lock_guard<std::mutex> lk(lifecycleMu_);
  if (shutdown_.load(std::memory_order_acquire)) {
    return Err(
        ErrCode::NotConnected, "tcp connect: transport is already shut down");
  }
  if (state_ != TransportState::Initialized) {
    return Err(
        ErrCode::InvalidArgument, "tcp connect: transport must be bound first");
  }

  auto peerResult = TcpTransportInfo::deserialize(remoteInfo);
  if (!peerResult) {
    state_ = TransportState::Error;
    return std::move(peerResult).error();
  }
  const auto peer = std::move(peerResult).value();

  // Deterministic role: the smaller (host, port) listens, the other dials.
  // Compare by the parsed binary address (hostOrderKey), not the raw string, so
  // different textual forms of the same IPv6 address can't make the two peers
  // disagree on ordering. Reject the degenerate identical-endpoint case, which
  // would otherwise leave both peers dialing and nobody accepting (silent
  // hang).
  const auto localKey = std::make_tuple(hostOrderKey(host_), port_);
  const auto peerKey = std::make_tuple(hostOrderKey(peer.host), peer.port);
  if (localKey == peerKey) {
    state_ = TransportState::Error;
    return Err(
        ErrCode::ConnectionFailed,
        "tcp connect: local and peer bind address are identical; cannot assign "
        "distinct listener/dialer roles");
  }
  const bool listener = localKey < peerKey;

  std::unique_ptr<controller::Conn> conn;
  // Both handshake waits are bounded. AsyncAccept::accept() queues a promise
  // that is resolved only by an inbound connection or by teardown, so an
  // unbounded get() here wedges this thread forever if the dialing peer dies
  // between the bind-info exchange and its own connect(). Nothing would rescue
  // it: server_ is owned by this transport, whose connect() is the thing
  // parked. AsyncAccept also ignores acceptRetryCnt and AsyncConnect ignores
  // connectRetries, so no configured bound applies on its own.
  //
  // This matters beyond TCP: MultiTransport::connect() connects every
  // registered transport, so on AMD a wedged TCP handshake would stall
  // connection setup even for jobs whose data path is RDMA.
  const auto handshakeTimeout = config_.connTimeout.value_or(
      std::chrono::seconds{kDefaultHandshakeTimeoutSeconds});

  if (listener) {
    if (!server_) {
      state_ = TransportState::Error;
      return Err(ErrCode::ConnectionFailed, "tcp connect: no server bound");
    }
    auto future = server_->accept();
    if (future.wait_for(handshakeTimeout) != std::future_status::ready) {
      // shutdown() resolves the queued promise with nullptr (via teardown), so
      // the get() below returns immediately instead of blocking. Safe from this
      // thread: AsyncAccept::shutdown marshals teardown onto the EventBase
      // thread and waits when called from outside the loop.
      UNIFLOW_LOG_ERROR(
          "TcpTransport::connect: no peer dialed in within {}s; tearing down "
          "listener {}:{}",
          handshakeTimeout.count(),
          host_,
          port_);
      server_->shutdown();
    }
    conn = future.get();
  } else {
    controller::AsyncTcpClient client(config_, *evb_);
    auto future = client.connect(peer.host + ":" + std::to_string(peer.port));
    if (future.wait_for(handshakeTimeout) != std::future_status::ready) {
      UNIFLOW_LOG_ERROR(
          "TcpTransport::connect: dial to {}:{} did not complete within {}s",
          peer.host,
          peer.port,
          handshakeTimeout.count());
      // No teardown hook on the client side; abandon the attempt. The future
      // owns its own state, so letting it go out of scope is safe.
      state_ = TransportState::Error;
      return Err(ErrCode::ConnectionFailed, "tcp connect: dial timed out");
    }
    conn = future.get();
  }

  if (!conn) {
    state_ = TransportState::Error;
    return Err(
        ErrCode::ConnectionFailed, "tcp connect: data connection failed");
  }

  dataConn_ = std::move(conn);
  running_.store(true, std::memory_order_release);
  reader_ = std::thread([this]() { readerLoop(); });
  sender_ = std::thread([this]() { senderLoop(); });
  state_ = TransportState::Connected;
  UNIFLOW_LOG_INFO(
      "TcpTransport: connected (listener={}) {}:{} <-> {}:{}",
      listener,
      host_,
      port_,
      peer.host,
      peer.port);
  return Ok();
}

Result<const TcpRemoteRegistrationHandle*> TcpTransport::findRemoteHandle(
    const RemoteRegisteredSegment::Span& span) const {
  for (const auto& handle : span.handles_) {
    if (auto* tcp =
            dynamic_cast<const TcpRemoteRegistrationHandle*>(handle.get())) {
      return tcp;
    }
  }
  return Err(
      ErrCode::InvalidArgument, "tcp: no TCP remote registration handle found");
}

std::future<Status> TcpTransport::put(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp put: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  auto state = std::make_shared<TcpOpState>();
  auto future = state->promise.get_future();

  // Validate and count chunks up front so the completion counter is exact.
  size_t totalChunks = 0;
  for (const auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      state->fail(
          Err(ErrCode::InvalidArgument,
              "tcp put: local and remote buffer sizes must match"));
      return future;
    }
    const size_t len = req.local.size();
    totalChunks += (len == 0) ? 1 : (len + kMaxChunkSize - 1) / kMaxChunkSize;
  }
  state->remaining = totalChunks;

  for (const auto& req : requests) {
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      state->fail(std::move(remoteHandle).error());
      return future;
    }
    const uint64_t segId = remoteHandle.value()->segId();
    const uint64_t baseOffset = static_cast<uint64_t>(req.remote.remoteOffset_);
    const size_t len = req.local.size();
    const bool vram = req.local.memType() == MemoryType::VRAM;
    const int deviceId = req.local.deviceId();
    const auto* src = static_cast<const uint8_t*>(req.local.data());

    size_t off = 0;
    do {
      const size_t chunk = std::min(kMaxChunkSize, len - off);
      const uint64_t reqId = nextReqId_.fetch_add(1, std::memory_order_relaxed);
      if (auto admitted =
              admitInflight(reqId, TcpInflight{state, nullptr, chunk, false});
          admitted.hasError()) {
        state->fail(std::move(admitted));
        return future;
      }
      std::vector<uint8_t> frame(sizeof(TcpMsgHeader) + chunk);
      TcpMsgHeader header;
      header.op = static_cast<uint8_t>(TcpOp::Write);
      header.reqId = reqId;
      header.segId = segId;
      header.offset = baseOffset + off;
      header.len = static_cast<uint64_t>(chunk);
      std::memcpy(frame.data(), &header, sizeof(header));
      if (chunk > 0) {
        Status st = Ok();
        if (vram) {
          st = hostFromDevice(
              frame.data() + sizeof(header),
              src + off,
              chunk,
              deviceId,
              options.stream.has_value()
                  ? static_cast<void*>(options.stream.value())
                  : nullptr);
        } else {
          std::memcpy(frame.data() + sizeof(header), src + off, chunk);
        }
        if (!st) {
          {
            std::lock_guard<std::mutex> lk(inflightMu_);
            inflight_.erase(reqId);
          }
          state->fail(std::move(st));
          return future;
        }
      }
      if (!enqueueFrame(std::move(frame), /*mayBlock=*/true)) {
        state->fail(
            Err(ErrCode::NotConnected,
                "tcp put: transport closed before the write was queued"));
        return future;
      }
      off += chunk;
    } while (off < len);
  }

  return future;
}

std::future<Status> TcpTransport::get(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp get: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  auto state = std::make_shared<TcpOpState>();
  auto future = state->promise.get_future();

  size_t totalChunks = 0;
  for (const auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      state->fail(
          Err(ErrCode::InvalidArgument,
              "tcp get: local and remote buffer sizes must match"));
      return future;
    }
    const size_t len = req.local.size();
    totalChunks += (len == 0) ? 1 : (len + kMaxChunkSize - 1) / kMaxChunkSize;
  }
  state->remaining = totalChunks;

  for (const auto& req : requests) {
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      state->fail(std::move(remoteHandle).error());
      return future;
    }
    const uint64_t segId = remoteHandle.value()->segId();
    const uint64_t baseOffset = static_cast<uint64_t>(req.remote.remoteOffset_);
    const size_t len = req.local.size();
    const MemoryType memType = req.local.memType();
    const int deviceId = req.local.deviceId();
    auto* dst = static_cast<uint8_t*>(req.local.mutable_data());

    size_t off = 0;
    do {
      const size_t chunk = std::min(kMaxChunkSize, len - off);
      const uint64_t reqId = nextReqId_.fetch_add(1, std::memory_order_relaxed);
      if (auto admitted = admitInflight(
              reqId,
              TcpInflight{
                  state,
                  dst + off,
                  chunk,
                  true,
                  memType,
                  deviceId,
                  options.stream.has_value()
                      ? static_cast<void*>(options.stream.value())
                      : nullptr});
          admitted.hasError()) {
        state->fail(std::move(admitted));
        return future;
      }
      TcpMsgHeader header;
      header.op = static_cast<uint8_t>(TcpOp::ReadRequest);
      header.reqId = reqId;
      header.segId = segId;
      header.offset = baseOffset + off;
      header.len = static_cast<uint64_t>(chunk);
      if (!enqueueFrame(serializeTcpHeader(header), /*mayBlock=*/true)) {
        state->fail(
            Err(ErrCode::NotConnected,
                "tcp get: transport closed before the read was queued"));
        return future;
      }
      off += chunk;
    } while (off < len);
  }

  return future;
}

Status TcpTransport::admitInflight(uint64_t reqId, TcpInflight entry) {
  std::lock_guard<std::mutex> lk(inflightMu_);
  if (connBroken_.load(std::memory_order_acquire)) {
    return Err(
        ErrCode::NotConnected, "tcp: transport closed while admitting request");
  }
  if (inflight_.size() >= kMaxInflightRequests) {
    return Err(
        ErrCode::ResourceExhausted,
        "tcp: too many outstanding requests (" +
            std::to_string(kMaxInflightRequests) + ")");
  }
  inflight_[reqId] = std::move(entry);
  return Ok();
}

bool TcpTransport::enqueueFrame(std::vector<uint8_t> frame, bool mayBlock) {
  const size_t bytes = frame.size();
  {
    std::unique_lock<std::mutex> lk(outMu_);
    if (mayBlock) {
      // Caller threads absorb backpressure by waiting, which is real
      // backpressure: an application issuing put/get faster than the link
      // drains slows down instead of growing the queue. An empty queue always
      // admits, however large the frame, or a payload bigger than the cap could
      // never drain and would wedge here forever.
      outCv_.wait(lk, [this, bytes]() {
        return outClosed_ || connBroken_.load(std::memory_order_acquire) ||
            outQueue_.empty() || outBytes_ + bytes <= kMaxOutQueueBytes;
      });
    }
    // The reader thread deliberately gets no cap. It is producing replies the
    // peer is already blocked on, and a get of N bytes legitimately needs up to
    // N bytes of replies queued -- N is bounded only by the segment size, so no
    // fixed cap can tell a large honest get apart from abuse; they are the same
    // traffic. Refusing here would fail every unrelated in-flight transfer on
    // the connection. It cannot wait either: that stops it draining the socket
    // and reintroduces the mutual-READ deadlock the reader/sender split exists
    // to avoid. What bounds this queue is the drain rate, not a byte cap.
    if (outClosed_) {
      return false;
    }
    outQueue_.push_back(TcpOutItem{std::move(frame), nullptr});
    outBytes_ += bytes;
  }
  outCv_.notify_all();
  return true;
}

void TcpTransport::enqueueSendFrame(
    std::vector<uint8_t> frame,
    std::shared_ptr<TcpOpState> onSent) {
  bool closed = false;
  std::shared_ptr<TcpOpState> toFail;
  const size_t bytes = frame.size();
  {
    std::unique_lock<std::mutex> lk(outMu_);
    // send() runs on a caller thread, so it can wait for room.
    outCv_.wait(lk, [this, bytes]() {
      return outClosed_ || connBroken_.load(std::memory_order_acquire) ||
          outQueue_.empty() || outBytes_ + bytes <= kMaxOutQueueBytes;
    });
    if (outClosed_) {
      closed = true;
      // Taken over here so each path has exactly one owner: the queue takes it
      // when the frame is enqueued, this does when it cannot be.
      toFail = std::move(onSent);
    } else {
      outQueue_.push_back(TcpOutItem{std::move(frame), std::move(onSent)});
      outBytes_ += bytes;
    }
  }
  // Settled outside outMu_ so TcpOpState::mu stays a leaf lock: a caller woken
  // by this promise may re-enter the transport from its error path, and it must
  // not find a container mutex still held by the thread that woke it.
  if (closed) {
    if (toFail) {
      toFail->fail(Err(ErrCode::NotConnected, "tcp send: transport closing"));
    }
    return;
  }
  outCv_.notify_all();
}

void TcpTransport::senderLoop() noexcept {
  for (;;) {
    TcpOutItem item;
    {
      std::unique_lock<std::mutex> lk(outMu_);
      outCv_.wait(lk, [this]() { return outClosed_ || !outQueue_.empty(); });
      if (outClosed_) {
        return;
      }
      item = std::move(outQueue_.front());
      outQueue_.pop_front();
      outBytes_ -= std::min(outBytes_, item.bytes.size());
    }
    outCv_.notify_all(); // room freed; wake any producer waiting for space

    auto result = dataConn_->send(item.bytes).get();
    if (!result) {
      UNIFLOW_LOG_ERROR(
          "tcp sender: send failed: {}", result.error().message());
      // Close the out queue before unwinding. Nothing drains outQueue_ once
      // this thread returns, and enqueueFrame() gates only on outClosed_, so
      // the still-running reader thread would keep appending
      // Ack/Error/ReadReply frames -- the last up to kMaxChunkSize each -- to a
      // queue with no consumer. failAllPending() clears it exactly once, so
      // without this the growth is unbounded and driven by whatever the peer
      // keeps requesting.
      {
        std::lock_guard<std::mutex> lk(outMu_);
        outClosed_ = true;
      }
      outCv_.notify_all();
      if (item.onSent) {
        item.onSent->fail(Err(ErrCode::ConnectionFailed, "tcp: send failed"));
      }
      failAllPending("tcp: send failed");
      return;
    }
    if (item.onSent) {
      item.onSent->completeOne();
    }
  }
}

void TcpTransport::readerLoop() noexcept {
  while (running_.load(std::memory_order_acquire)) {
    std::vector<uint8_t> msg;
    auto result = dataConn_->recv(msg).get();
    if (!result) {
      // Connection closed, errored, or idle-timed-out; stop reading.
      break;
    }
    handleFrame(msg);
  }
  // The reader is the only place that resolves in-flight put/get/recv replies.
  // If it exits on a recv error (peer disconnect) while requests are still
  // outstanding and the sender is idle, nothing else would fulfill their
  // promises and callers would block forever on future.get(). Fail them here.
  // Idempotent with shutdown()'s own failAllPending().
  failAllPending("tcp: reader stopped (connection closed or read error)");
}

void TcpTransport::handleFrame(std::span<const uint8_t> frame) noexcept {
  auto headerResult = deserializeTcpHeader(frame);
  if (!headerResult) {
    UNIFLOW_LOG_ERROR(
        "tcp: dropping malformed frame: {}", headerResult.error().message());
    return;
  }
  const TcpMsgHeader header = headerResult.value();
  const auto op = static_cast<TcpOp>(header.op);
  const std::span<const uint8_t> payload = frame.subspan(sizeof(TcpMsgHeader));

  switch (op) {
    case TcpOp::Write: {
      bool ok = false;
      // Lease, not a plain lookup: it must outlive the copy below, because it
      // is what stops the owner deregistering and freeing the buffer underneath
      // us.
      auto entry = registry_->find(header.segId);
      if (entry && header.len <= entry->len &&
          header.offset <= entry->len - header.len &&
          payload.size() == header.len) {
        Status st = Ok();
        if (header.len > 0) {
          void* dst = static_cast<uint8_t*>(entry->ptr) + header.offset;
          if (entry->memType == MemoryType::VRAM) {
            st = deviceFromHost(
                dst, payload.data(), header.len, entry->deviceId);
          } else {
            std::memcpy(dst, payload.data(), header.len);
          }
        }
        ok = !st.hasError();
      }
      (void)enqueueFrame(
          makeHeaderFrame(ok ? TcpOp::Ack : TcpOp::Error, header.reqId),
          /*mayBlock=*/false);
      break;
    }

    case TcpOp::ReadRequest: {
      // Held across the read below for the same reason as the Write path.
      auto entry = registry_->find(header.segId);
      Status readStatus = Ok();
      // Bound the reply by the wire-frame cap, not just by the segment length.
      // A peer that registered a segment larger than the cap can request a read
      // whose reply exceeds kMaxMessageSize; the controller then refuses the
      // send, and senderLoop treats that as fatal -- killing the sender and
      // failing every unrelated in-flight transfer. Our own get() chunks at
      // kMaxChunkSize so a same-version peer never asks for this, which is
      // exactly the version-skew case the header version byte guards: a peer
      // built with a larger chunk size would otherwise turn a recoverable
      // per-request error into a permanently dead connection.
      if (header.len > kMaxFrameSize - sizeof(TcpMsgHeader)) {
        readStatus =
            Err(ErrCode::InvalidArgument,
                "tcp read: requested length would exceed the wire-frame cap");
      } else if (
          entry && header.len <= entry->len &&
          header.offset <= entry->len - header.len) {
        std::vector<uint8_t> reply(sizeof(TcpMsgHeader) + header.len);
        TcpMsgHeader replyHeader;
        replyHeader.op = static_cast<uint8_t>(TcpOp::ReadReply);
        replyHeader.reqId = header.reqId;
        replyHeader.segId = header.segId;
        replyHeader.offset = header.offset;
        replyHeader.len = header.len;
        std::memcpy(reply.data(), &replyHeader, sizeof(replyHeader));
        if (header.len > 0) {
          const void* src =
              static_cast<const uint8_t*>(entry->ptr) + header.offset;
          if (entry->memType == MemoryType::VRAM) {
            readStatus = hostFromDevice(
                reply.data() + sizeof(replyHeader),
                src,
                header.len,
                entry->deviceId);
          } else {
            std::memcpy(reply.data() + sizeof(replyHeader), src, header.len);
          }
        }
        if (!readStatus.hasError()) {
          (void)enqueueFrame(std::move(reply), /*mayBlock=*/false);
        }
      } else {
        readStatus = Err(ErrCode::InvalidArgument, "tcp read: bad segment");
      }
      if (readStatus.hasError()) {
        (void)enqueueFrame(
            makeHeaderFrame(TcpOp::Error, header.reqId), /*mayBlock=*/false);
      }
      break;
    }

    case TcpOp::Send: {
      if (payload.size() != header.len) {
        UNIFLOW_LOG_ERROR("tcp send: inbound payload size mismatch");
        break;
      }
      std::shared_ptr<TcpOpState> state;
      void* dst = nullptr;
      size_t cap = 0;
      MemoryType memType = MemoryType::DRAM;
      int deviceId = -1;
      void* stream = nullptr;
      bool matched = false;
      bool overflow = false;
      {
        std::lock_guard<std::mutex> lk(recvMu_);
        if (!pendingRecvs_.empty()) {
          auto pr = std::move(pendingRecvs_.front());
          pendingRecvs_.pop_front();
          state = std::move(pr.state);
          dst = pr.dst;
          cap = pr.cap;
          memType = pr.memType;
          deviceId = pr.deviceId;
          stream = pr.stream;
          matched = true;
        } else if (unmatchedBytes_ + payload.size() > kMaxUnmatchedSendBytes) {
          overflow = true;
        } else {
          unmatchedSends_.emplace_back(payload.begin(), payload.end());
          unmatchedBytes_ += payload.size();
        }
      }
      if (overflow) {
        // No backpressure is available on this path, so absorbing more would be
        // unbounded host-memory growth driven by the peer. Refuse the
        // connection instead: closing it makes the reader's next recv() fail,
        // and the reader's own failAllPending() then resolves outstanding ops.
        UNIFLOW_LOG_ERROR(
            "tcp recv: unmatched inbound sends exceed {} bytes; closing the "
            "connection",
            kMaxUnmatchedSendBytes);
        if (dataConn_) {
          dataConn_->close();
        }
        break;
      }
      if (matched && state) {
        if (payload.size() > cap) {
          state->fail(
              Err(ErrCode::InvalidArgument,
                  "tcp recv: buffer too small for incoming send"));
        } else {
          Status st = Ok();
          if (!payload.empty() && dst != nullptr) {
            if (memType == MemoryType::VRAM) {
              st = deviceFromHost(
                  dst, payload.data(), payload.size(), deviceId, stream);
            } else {
              std::memcpy(dst, payload.data(), payload.size());
            }
          }
          if (st.hasError()) {
            state->fail(std::move(st));
          } else {
            state->completeOne();
          }
        }
      }
      break;
    }

    case TcpOp::Ack:
    case TcpOp::ReadReply:
    case TcpOp::Error: {
      TcpInflight entry;
      bool found = false;
      {
        std::lock_guard<std::mutex> lk(inflightMu_);
        auto it = inflight_.find(header.reqId);
        if (it != inflight_.end()) {
          entry = std::move(it->second);
          inflight_.erase(it);
          found = true;
        }
      }
      if (!found || !entry.state) {
        break;
      }
      if (op == TcpOp::Error) {
        entry.state->fail(
            Err(ErrCode::TransportError, "tcp: peer reported an error"));
      } else if (op == TcpOp::ReadReply) {
        if (payload.size() != header.len || header.len != entry.len) {
          entry.state->fail(Err(
              ErrCode::TransportError, "tcp get: read reply size mismatch"));
          break;
        }
        // The copy into the caller's get() destination and this chunk's
        // completion must be one step: a concurrent failAllPending() resolving
        // the op (via a sibling chunk of the same multi-chunk get()) releases
        // the caller to free entry.dst, so writing it either side of that
        // resolution is a write-after-free. writeAndComplete() drops the copy
        // if the op is already resolved.
        entry.state->writeAndComplete([&]() -> Status {
          if (header.len == 0 || entry.dst == nullptr) {
            return Ok();
          }
          if (entry.memType == MemoryType::VRAM) {
            return deviceFromHost(
                entry.dst,
                payload.data(),
                header.len,
                entry.deviceId,
                entry.stream);
          }
          std::memcpy(entry.dst, payload.data(), header.len);
          return Ok();
        });
      } else { // Ack
        entry.state->completeOne();
      }
      break;
    }

    case TcpOp::Notification:
      UNIFLOW_LOG_WARN("tcp: unexpected inbound Notification frame");
      break;

    default:
      UNIFLOW_LOG_WARN(
          "tcp: unexpected opcode {}", static_cast<int>(header.op));
      break;
  }
}

void TcpTransport::failAllPending(const char* message) {
  connBroken_.store(true, std::memory_order_release);
  // Collect first, settle after every mutex is released. Fulfilling a promise
  // under a container mutex hands control to the waiting caller while this
  // thread still holds it; a caller that tears the transport down from its
  // error path would then destroy the mutex and the containers underneath this
  // function. Settling afterwards also keeps TcpOpState::mu a leaf lock.
  std::vector<std::shared_ptr<TcpOpState>> toFail;
  {
    std::lock_guard<std::mutex> lk(inflightMu_);
    for (auto& [reqId, entry] : inflight_) {
      if (entry.state) {
        toFail.push_back(std::move(entry.state));
      }
    }
    inflight_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(recvMu_);
    for (auto& pr : pendingRecvs_) {
      if (pr.state) {
        toFail.push_back(std::move(pr.state));
      }
    }
    pendingRecvs_.clear();
    unmatchedSends_.clear();
    unmatchedBytes_ = 0;
  }
  {
    std::lock_guard<std::mutex> lk(outMu_);
    for (auto& item : outQueue_) {
      if (item.onSent) {
        toFail.push_back(std::move(item.onSent));
      }
    }
    outQueue_.clear();
    outBytes_ = 0;
  }
  outCv_.notify_all(); // wake producers blocked for space so they see the break
  for (auto& state : toFail) {
    state->fail(Err(ErrCode::ConnectionFailed, message));
  }
}

std::future<Status> TcpTransport::sendImpl(
    const void* data,
    size_t len,
    MemoryType memType,
    int deviceId,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp send: not connected"));
  }
  // Staging VRAM on the null stream is the one outcome that can silently
  // transmit stale device data, so require the caller to say which stream the
  // D2H must be ordered against. Matches RdmaTransport::rdmaSendRecvTransfer.
  if (memType == MemoryType::VRAM && !options.stream.has_value()) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "VRAM transfer requires an explicit CUDA stream"));
  }
  // send() is single-frame (no chunking); reject payloads that would exceed the
  // wire-frame cap so callers get a diagnostic instead of a silent drop/hang.
  if (len > kMaxFrameSize - sizeof(TcpMsgHeader)) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "tcp send: payload exceeds the 64 MiB wire-frame cap; use put/get "
            "(which chunk) for large transfers"));
  }
  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;
  auto future = state->promise.get_future();

  std::vector<uint8_t> frame(sizeof(TcpMsgHeader) + len);
  TcpMsgHeader header;
  header.op = static_cast<uint8_t>(TcpOp::Send);
  header.len = static_cast<uint64_t>(len);
  std::memcpy(frame.data(), &header, sizeof(header));
  if (len > 0 && data != nullptr) {
    Status st = Ok();
    if (memType == MemoryType::VRAM) {
      st = hostFromDevice(
          frame.data() + sizeof(header),
          data,
          len,
          deviceId,
          static_cast<void*>(options.stream.value()));
    } else {
      std::memcpy(frame.data() + sizeof(header), data, len);
    }
    if (!st) {
      state->fail(std::move(st));
      return future;
    }
  }
  enqueueSendFrame(std::move(frame), std::move(state));
  return future;
}

std::future<Status> TcpTransport::recvImpl(
    void* dst,
    size_t cap,
    MemoryType memType,
    int deviceId,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected ||
      connBroken_.load(std::memory_order_acquire)) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "tcp recv: not connected"));
  }
  // Without an explicit stream the H2D would land on the null stream, letting
  // the caller launch kernels against a buffer the payload has not reached yet.
  // Matches RdmaTransport::rdmaSendRecvTransfer.
  if (memType == MemoryType::VRAM && !options.stream.has_value()) {
    return make_ready_future<Status>(
        Err(ErrCode::InvalidArgument,
            "VRAM transfer requires an explicit CUDA stream"));
  }
  void* const stream = options.stream.has_value()
      ? static_cast<void*>(options.stream.value())
      : nullptr;
  auto state = std::make_shared<TcpOpState>();
  state->remaining = 1;
  auto future = state->promise.get_future();

  std::vector<uint8_t> payload;
  bool matched = false;
  bool closed = false;
  {
    std::lock_guard<std::mutex> lk(recvMu_);
    // Same admission race as the inflight_ path: a failAllPending() landing
    // between the entry check above and this push_back would leave a posted
    // recv no sweep will ever see, blocking the caller forever. Checked under
    // recvMu_ so the two orderings are exclusive.
    if (connBroken_.load(std::memory_order_acquire)) {
      closed = true;
    } else if (!unmatchedSends_.empty()) {
      payload = std::move(unmatchedSends_.front());
      unmatchedSends_.pop_front();
      unmatchedBytes_ -= std::min(unmatchedBytes_, payload.size());
      matched = true;
    } else {
      pendingRecvs_.push_back(
          TcpPendingRecv{dst, cap, state, memType, deviceId, stream});
    }
  }
  if (closed) {
    state->fail(
        Err(ErrCode::NotConnected,
            "tcp recv: transport closed while posting receive"));
    return future;
  }
  if (matched) {
    if (payload.size() > cap) {
      state->fail(
          Err(ErrCode::InvalidArgument,
              "tcp recv: buffer too small for buffered send"));
    } else {
      Status st = Ok();
      if (!payload.empty() && dst != nullptr) {
        if (memType == MemoryType::VRAM) {
          st = deviceFromHost(
              dst, payload.data(), payload.size(), deviceId, stream);
        } else {
          std::memcpy(dst, payload.data(), payload.size());
        }
      }
      if (st.hasError()) {
        state->fail(std::move(st));
      } else {
        state->completeOne();
      }
    }
  }
  return future;
}

std::future<Status> TcpTransport::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  return sendImpl(
      src.data(), src.size(), src.memType(), src.deviceId(), options);
}

std::future<Status> TcpTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return recvImpl(
      dst.mutable_data(), dst.size(), dst.memType(), dst.deviceId(), options);
}

std::future<Status> TcpTransport::send(
    Segment::Span src,
    const RequestOptions& options) {
  return sendImpl(
      src.data(), src.size(), src.memType(), src.deviceId(), options);
}

std::future<Status> TcpTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return recvImpl(
      dst.mutable_data(), dst.size(), dst.memType(), dst.deviceId(), options);
}

void TcpTransport::shutdown() {
  std::lock_guard<std::mutex> lk(lifecycleMu_);
  // One-shot, but checked under the mutex rather than before taking it:
  // shutdown() is called twice in the normal flow (MultiTransport::shutdown()
  // then ~TcpTransport), and the second caller must not return while the first
  // is still tearing down -- the destructor would otherwise race it.
  if (shutdown_.exchange(true)) {
    return;
  }
  running_.store(false, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lk(outMu_);
    outClosed_ = true;
  }
  outCv_.notify_all();

  // Closing the data connection unblocks the reader's blocking recv and any
  // in-progress send on the sender thread.
  if (dataConn_) {
    dataConn_->close();
  }
  if (reader_.joinable()) {
    reader_.join();
  }
  if (sender_.joinable()) {
    sender_.join();
  }

  failAllPending("tcp transport shut down");

  // Compare-exchange, not load-then-store: a concurrent bind() failure setting
  // Error in the gap between a load and a store would be clobbered back to
  // Disconnected, so a transport that failed to bind would report as cleanly
  // closed.
  auto expected = state_.load(std::memory_order_acquire);
  while (
      expected != TransportState::Error &&
      !state_.compare_exchange_weak(expected, TransportState::Disconnected)) {
  }
}

// ---------------------------------------------------------------------------
// TcpTransportFactory
// ---------------------------------------------------------------------------

Status TcpTransportFactory::supported() {
  return Ok();
}

TcpTransportFactory::TcpTransportFactory(
    int deviceId,
    EventBase* evb,
    controller::TcpSocketConfig config,
    std::string host,
    std::shared_ptr<CudaApi> cudaApi)
    : TransportFactory(TransportType::TCP),
      deviceId_(deviceId),
      evb_(evb),
      config_(std::move(config)),
      host_(host.empty() ? std::string("127.0.0.1") : std::move(host)),
      cudaApi_(cudaApi ? std::move(cudaApi) : std::make_shared<CudaApi>()) {}

Result<std::unique_ptr<RegistrationHandle>>
TcpTransportFactory::registerSegment(Segment& segment) {
  if (segment.memType() != MemoryType::DRAM &&
      segment.memType() != MemoryType::VRAM) {
    return Err(
        ErrCode::MemoryRegistrationError,
        "tcp transport supports only DRAM and VRAM segments");
  }

  const auto segId = nextSegId_.fetch_add(1, std::memory_order_relaxed);
  registry_->add(
      segId,
      segment.mutable_data(),
      segment.len(),
      segment.memType(),
      segment.deviceId());
  return std::make_unique<TcpRegistrationHandle>(
      segId,
      static_cast<uint64_t>(segment.len()),
      [weakRegistry = std::weak_ptr<TcpSegmentRegistry>(registry_), segId]() {
        if (auto registry = weakRegistry.lock()) {
          // Blocks until no reader is mid-copy on this segment, so that when
          // the handle finishes destructing the owner can free the buffer.
          registry->erase(segId);
        }
      });
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
TcpTransportFactory::importSegment(
    size_t segmentLength,
    std::span<const uint8_t> payload) {
  auto handle =
      TcpRemoteRegistrationHandle::deserialize(segmentLength, payload);
  if (!handle) {
    return std::move(handle).error();
  }
  return std::move(handle).value();
}

Result<std::unique_ptr<Transport>> TcpTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  // Validate the peer's capability blob before building anything, the way
  // RdmaTransportFactory::createTransport does. The peer's *address* is not
  // here; it arrives later through bind()/connect().
  auto status = canConnect(peerTopology);
  if (!status) {
    return std::move(status).error();
  }
  return std::make_unique<TcpTransport>(
      deviceId_, evb_, registry_, config_, host_, cudaApi_);
}

std::vector<uint8_t> TcpTransportFactory::getTopology() {
  return TcpTopologyInfo{}.serialize();
}

Status TcpTransportFactory::canConnect(std::span<const uint8_t> peerTopology) {
  auto info = TcpTopologyInfo::deserialize(peerTopology);
  if (!info) {
    return std::move(info).error();
  }
  if (info->version != kTcpWireVersion) {
    return Err(
        ErrCode::TopologyDisconnect,
        "tcp: unsupported peer wire version " + std::to_string(info->version) +
            ", local is " + std::to_string(kTcpWireVersion));
  }
  return Ok();
}

} // namespace uniflow
