// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pTransport.h"

#include <unistd.h>

#include <algorithm>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {
namespace {

// Fulfill a completion promise without escaping a noexcept context. set_value
// can throw std::future_error if the promise was already satisfied / has no
// shared state (a logic-bug invariant violation); swallow it rather than
// terminating the noexcept caller (an EventBase task).
void setValueNoThrow(std::promise<Status>& promise, Status status) noexcept {
  try {
    promise.set_value(std::move(status));
  } catch (const std::exception& e) {
    UNIFLOW_LOG_ERROR("P2P transfer: set_value failed: {}", e.what());
  }
}

// Error-path cleanup for transfer(): drain any copies already enqueued (so the
// caller can safely release the buffers -- the GPU is no longer touching them),
// destroy the completion event if one was created, then report the failure on
// the promise. @p event may be nullptr when no event exists yet.
void drainAndFail(
    CudaApi& cudaApi,
    std::promise<Status>& promise,
    void* stream,
    void* event,
    Status status) noexcept {
  (void)cudaApi.synchronizeStream(stream);
  if (event != nullptr) {
    (void)cudaApi.destroyEvent(event);
  }
  setValueNoThrow(promise, std::move(status));
}

// Drive a recorded event to completion without blocking the EventBase worker:
// poll queryEvent, and while the event is still in-flight re-dispatch onto @p
// evb so the worker thread is freed between polls. On completion (or a query
// error) destroy the event and fulfill @p promise. Takes ownership of @p event
// and @p promise. noexcept: runs as an EventBase task.
void pollEventToCompletion(
    EventBase* evb,
    std::shared_ptr<CudaApi> cudaApi,
    void* event,
    std::promise<Status> promise) noexcept {
  auto poll = [evb,
               cudaApi = std::move(cudaApi),
               event,
               promise = std::move(promise)](auto& self) mutable noexcept {
    auto res = cudaApi->queryEvent(event);
    if (res.hasError()) {
      (void)cudaApi->destroyEvent(event);
      setValueNoThrow(promise, std::move(res).error());
      return;
    }
    if (res.value()) {
      (void)cudaApi->destroyEvent(event);
      setValueNoThrow(promise, Ok());
      return;
    }
    evb->dispatch([self = std::move(self)]() mutable noexcept { self(self); });
  };
  poll(poll);
}

} // namespace

// ---------------------------------------------------------------------------
// P2pTransport
// ---------------------------------------------------------------------------

P2pTransport::P2pTransport(
    int deviceId,
    EventBase* evb,
    std::shared_ptr<CudaApi> cudaApi)
    : deviceId_(deviceId),
      deviceName_("cuda:" + std::to_string(deviceId)),
      evb_(evb),
      cudaApi_(std::move(cudaApi)) {
  CHECK_THROW_EXCEPTION(evb_ != nullptr, std::invalid_argument);
  if (!cudaApi_) {
    cudaApi_ = std::make_shared<CudaApi>();
  }
}

TransportInfo P2pTransport::bind() {
  TransportInfo info(sizeof(int32_t));
  const int32_t devId = deviceId_;
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&devId), sizeof(devId), info.data());
  // Advance the initial Disconnected state to Initialized. A transport that is
  // already Connected is left untouched so bind() never regresses it back to
  // Initialized (which would leave peerDeviceId_ stale).
  if (state_ == TransportState::Disconnected) {
    state_ = TransportState::Initialized;
  }
  return info;
}

Status P2pTransport::connect(std::span<const uint8_t> remoteInfo) {
  if (state_ != TransportState::Initialized) {
    return Err(
        ErrCode::InvalidArgument,
        "P2P connect: transport must be bound and not already connected");
  }
  if (remoteInfo.size() != sizeof(int32_t)) {
    return Err(
        ErrCode::InvalidArgument,
        "P2P connect: expected " + std::to_string(sizeof(int32_t)) +
            " bytes, got " + std::to_string(remoteInfo.size()));
  }
  int32_t peer = -1;
  std::copy_n(
      remoteInfo.data(), sizeof(peer), reinterpret_cast<uint8_t*>(&peer));
  if (peer < 0) {
    return Err(
        ErrCode::InvalidArgument,
        "P2P connect: invalid peer device id " + std::to_string(peer));
  }

  // Enable this device to access the peer device (no-op for same device or if
  // already enabled). Covers same-process cross-device P2P; cross-process
  // imports also lazily enable peer access via ipcOpenMemHandle. Commit
  // peerDeviceId_ / Connected only after this succeeds, so a failure leaves the
  // transport in its prior (Initialized) state.
  if (peer != deviceId_) {
    CudaDeviceGuard guard(*cudaApi_, deviceId_);
    auto st = cudaApi_->deviceEnablePeerAccess(peer);
    if (st.hasError()) {
      return st;
    }
  }

  peerDeviceId_ = peer;
  state_ = TransportState::Connected;
  UNIFLOW_LOG_INFO(
      "connect: device {} connected to peer device {}",
      deviceId_,
      peerDeviceId_);
  return Ok();
}

std::future<Status> P2pTransport::transfer(
    std::vector<CopyOp> ops,
    void* stream) {
  std::promise<Status> promise;
  auto future = promise.get_future();

  evb_->dispatch([evb = evb_,
                  cudaApi = cudaApi_,
                  deviceId = deviceId_,
                  promise = std::move(promise),
                  ops = std::move(ops),
                  stream]() mutable noexcept {
    CudaDeviceGuard deviceGuard(*cudaApi, deviceId);

    for (auto& op : ops) {
      auto st =
          cudaApi->memcpyDeviceToDeviceAsync(op.dst, op.src, op.size, stream);
      if (st.hasError()) {
        drainAndFail(
            *cudaApi, promise, stream, /*event=*/nullptr, std::move(st));
        return;
      }
    }

    // Completion via a recorded event, polled and re-dispatched on the
    // EventBase (same model as NVLinkTransport). The neutral event seam
    // (create/record/query/destroy over opaque handles) keeps this library
    // vendor-free.
    auto event = cudaApi->createEvent();
    if (event.hasError()) {
      drainAndFail(
          *cudaApi,
          promise,
          stream,
          /*event=*/nullptr,
          std::move(event).error());
      return;
    }
    void* ev = event.value();

    auto recordSt = cudaApi->recordEvent(ev, stream);
    if (recordSt.hasError()) {
      drainAndFail(*cudaApi, promise, stream, ev, std::move(recordSt));
      return;
    }

    // Event recorded after the last copy; hand off to the poller, which frees
    // this worker between polls and fulfills the promise on completion.
    pollEventToCompletion(evb, cudaApi, ev, std::move(promise));
  });

  return future;
}

Result<const P2pRemoteRegistrationHandle*> P2pTransport::findRemoteHandle(
    const RemoteRegisteredSegment::Span& span) const {
  // dynamic_cast (not static_cast): the NVLink interconnect tier is shared, so
  // a handle reporting TransportType::NVLink is not guaranteed to be a P2P
  // handle.
  for (const auto& h : span.handles_) {
    if (auto* p = dynamic_cast<const P2pRemoteRegistrationHandle*>(h.get())) {
      return p;
    }
  }
  return Err(
      ErrCode::InvalidArgument, "P2P: no P2P remote registration handle found");
}

std::future<Status> P2pTransport::put(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "P2P put: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  std::vector<CopyOp> ops;
  ops.reserve(requests.size());
  for (auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      return make_ready_future<Status>(
          Err(ErrCode::InvalidArgument,
              "P2P put: local and remote buffer sizes must match"));
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      return make_ready_future<Status>(std::move(remoteHandle).error());
    }
    auto* remoteDst = static_cast<uint8_t*>(remoteHandle.value()->mappedPtr()) +
        req.remote.nvlinkOffset_;
    ops.emplace_back(CopyOp{remoteDst, req.local.data(), req.local.size()});
  }

  return transfer(std::move(ops), options.stream.value_or(nullptr));
}

std::future<Status> P2pTransport::get(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected) {
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "P2P get: not connected"));
  }
  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  std::vector<CopyOp> ops;
  ops.reserve(requests.size());
  for (auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      return make_ready_future<Status>(
          Err(ErrCode::InvalidArgument,
              "P2P get: local and remote buffer sizes must match"));
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      return make_ready_future<Status>(std::move(remoteHandle).error());
    }
    auto* remoteSrc = static_cast<uint8_t*>(remoteHandle.value()->mappedPtr()) +
        req.remote.nvlinkOffset_;
    ops.emplace_back(
        CopyOp{req.local.mutable_data(), remoteSrc, req.remote.size()});
  }

  return transfer(std::move(ops), options.stream.value_or(nullptr));
}

std::future<Status> P2pTransport::send(
    RegisteredSegment::Span /*src*/,
    const RequestOptions& /*options*/) {
  return make_ready_future<Status>(Err(ErrCode::NotImplemented, "P2P send"));
}

std::future<Status> P2pTransport::send(
    Segment::Span /*src*/,
    const RequestOptions& /*options*/) {
  return make_ready_future<Status>(Err(ErrCode::NotImplemented, "P2P send"));
}

std::future<Status> P2pTransport::recv(
    RegisteredSegment::Span /*dst*/,
    const RequestOptions& /*options*/) {
  return make_ready_future<Status>(Err(ErrCode::NotImplemented, "P2P recv"));
}

std::future<Status> P2pTransport::recv(
    Segment::Span /*dst*/,
    const RequestOptions& /*options*/) {
  return make_ready_future<Status>(Err(ErrCode::NotImplemented, "P2P recv"));
}

void P2pTransport::shutdown() {
  // Marks the transport Disconnected so no new transfers are accepted.
  // Transfers already in flight are NOT cancelled -- queued GPU copies cannot
  // be aborted mid-flight; their futures still complete via the host callback.
  // Callers needing a hard stop must wait on outstanding futures before calling
  // this.
  //
  // Single-threaded-owned: like NVLinkTransport, this only transitions state_
  // and does not reset peerDeviceId_ (peerDeviceId_ is set once in connect()
  // and is never read on the transfer path), so there is no synchronization.
  UNIFLOW_LOG_INFO("shutdown: device {}", deviceId_);
  state_ = TransportState::Disconnected;
}

// ---------------------------------------------------------------------------
// P2pTransportFactory
// ---------------------------------------------------------------------------

Status P2pTransportFactory::supported(std::shared_ptr<CudaApi> cudaApi) {
  if (!cudaApi) {
    cudaApi = std::make_shared<CudaApi>();
  }
  auto count = cudaApi->getDeviceCount();
  CHECK_RETURN(count);
  if (count.value() == 0) {
    return Err(ErrCode::ResourceExhausted, "P2P: no GPU devices found");
  }
  return Ok();
}

P2pTransportFactory::P2pTransportFactory(
    int deviceId,
    EventBase* evb,
    std::shared_ptr<CudaApi> cudaApi)
    : TransportFactory(TransportType::NVLink),
      deviceId_(deviceId),
      evb_(evb),
      cudaApi_(std::move(cudaApi)) {
  CHECK_THROW_EXCEPTION(evb_ != nullptr, std::invalid_argument);
  if (!cudaApi_) {
    cudaApi_ = std::make_shared<CudaApi>();
  }
}

Result<std::unique_ptr<RegistrationHandle>>
P2pTransportFactory::registerSegment(Segment& segment) {
  if (segment.memType() != MemoryType::VRAM) {
    return Err(
        ErrCode::InvalidArgument, "P2P registerSegment: segment must be VRAM");
  }

  CudaDeviceGuard deviceGuard(*cudaApi_, deviceId_);

  // Export the IPC handle at the *allocation base*, recording the segment's
  // offset within it. A segment may be a sub-range of a larger allocation (e.g.
  // a caching-allocator pool slice); the IPC handle maps to the allocation
  // base, so the peer opens the base and adds this offset. getMemAddressRange
  // returns the real base on AMD; on NVIDIA (and on any query failure) it
  // reports the pointer as its own base -> whole-allocation behavior (offset
  // 0), identical to the pre-existing path.
  void* segPtr = segment.mutable_data();
  void* allocBase = segPtr;
  auto range = cudaApi_->getMemAddressRange(segPtr);
  if (!range.hasError() && range.value().base != nullptr) {
    allocBase = range.value().base;
  }

  auto handle = cudaApi_->ipcGetMemHandle(allocBase);
  CHECK_RETURN(handle);

  const auto offset = reinterpret_cast<uintptr_t>(segPtr) -
      reinterpret_cast<uintptr_t>(allocBase);
  return std::make_unique<P2pRegistrationHandle>(
      handle.value(),
      static_cast<int32_t>(::getpid()),
      reinterpret_cast<uint64_t>(allocBase),
      static_cast<uint64_t>(offset),
      static_cast<uint64_t>(segment.len()));
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
P2pTransportFactory::importSegment(
    size_t segmentLength,
    std::span<const uint8_t> payload) {
  auto parsed = P2pRegistrationHandle::deserialize(payload);
  CHECK_RETURN(parsed);
  const auto& p = parsed.value();

  if (segmentLength != static_cast<size_t>(p.size)) {
    return Err(
        ErrCode::InvalidArgument,
        "P2P importSegment: segment length mismatch (expected " +
            std::to_string(segmentLength) + ", payload " +
            std::to_string(p.size) + ")");
  }

  CudaDeviceGuard deviceGuard(*cudaApi_, deviceId_);

  if (p.ownerPid == static_cast<int32_t>(::getpid())) {
    // Same-process import: reuse the exporter's pointer directly. Opening an
    // IPC handle exported by the same process is unsupported; peer access is
    // enabled by the transport's connect().
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* base = reinterpret_cast<void*>(p.base);
    return std::make_unique<P2pRemoteRegistrationHandle>(
        base,
        p.offset,
        static_cast<size_t>(p.size),
        /*ownedByIpc=*/false,
        cudaApi_);
  }

  auto mapped = cudaApi_->ipcOpenMemHandle(p.ipcHandle);
  CHECK_RETURN(mapped);
  return std::make_unique<P2pRemoteRegistrationHandle>(
      mapped.value(),
      p.offset,
      static_cast<size_t>(p.size),
      /*ownedByIpc=*/true,
      cudaApi_);
}

Result<std::unique_ptr<Transport>> P2pTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  CHECK_EXPR(canConnect(peerTopology));
  return std::make_unique<P2pTransport>(deviceId_, evb_, cudaApi_);
}

std::vector<uint8_t> P2pTransportFactory::getTopology() {
  std::vector<uint8_t> topo(sizeof(int32_t));
  const int32_t devId = deviceId_;
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&devId), sizeof(devId), topo.data());
  return topo;
}

Status P2pTransportFactory::canConnect(std::span<const uint8_t> peerTopology) {
  if (peerTopology.size() != sizeof(int32_t)) {
    return Err(
        ErrCode::InvalidArgument, "P2P canConnect: invalid topology size");
  }
  int32_t peerDev = -1;
  std::copy_n(
      peerTopology.data(),
      sizeof(peerDev),
      reinterpret_cast<uint8_t*>(&peerDev));

  // Same device is trivially reachable; no peer-access check needed.
  if (peerDev == deviceId_) {
    return Ok();
  }

  auto canAccess = cudaApi_->deviceCanAccessPeer(deviceId_, peerDev);
  CHECK_RETURN(canAccess);
  if (!canAccess.value()) {
    return Err(
        ErrCode::TopologyDisconnect,
        "P2P: P2P access not supported between devices");
  }
  return Ok();
}

} // namespace uniflow
