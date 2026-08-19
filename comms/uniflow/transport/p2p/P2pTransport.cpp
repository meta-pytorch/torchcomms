// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pTransport.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {
namespace {

// Device ids cross the wire as a raw host-order int32 (matching
// NVLinkTransport). Encode/decode are centralized here so bind()/getTopology()
// and connect()/canConnect() cannot drift on size or range validation.
std::vector<uint8_t> encodeDeviceId(int32_t deviceId) {
  std::vector<uint8_t> bytes(sizeof(deviceId));
  const auto* src = reinterpret_cast<const uint8_t*>(&deviceId);
  std::copy_n(src, sizeof(deviceId), bytes.begin());
  return bytes;
}

// Parses a device id, rejecting both a wrong-sized buffer and a negative id.
Result<int32_t> decodeDeviceId(
    std::span<const uint8_t> bytes,
    std::string_view context) {
  if (bytes.size() != sizeof(int32_t)) {
    return Err(
        ErrCode::InvalidArgument,
        std::string(context) + ": expected " + std::to_string(sizeof(int32_t)) +
            " bytes, got " + std::to_string(bytes.size()));
  }
  int32_t deviceId = -1;
  std::copy_n(
      bytes.data(), sizeof(deviceId), reinterpret_cast<uint8_t*>(&deviceId));
  if (deviceId < 0) {
    return Err(
        ErrCode::InvalidArgument,
        std::string(context) + ": invalid device id " +
            std::to_string(deviceId));
  }
  return deviceId;
}

// TODO(D110509654): the event-completion machinery below (poll -> re-dispatch
// on the EventBase) is structurally shared with NVLinkTransport::transfer, but
// the two are not interchangeable today: NVLink fulfills promises via
// CHECK_SET_PROMISE (no noexcept guard, no drain on error) and has a
// memcpyBatchAsync fast path this transport lacks. Converge them in the unified
// intra-node transport rather than extracting a helper that would have to
// preserve both error semantics.

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
    cudaStream_t stream,
    cudaEvent_t event,
    Status status) noexcept {
  (void)cudaApi.streamSynchronize(stream);
  if (event != nullptr) {
    (void)cudaApi.eventDestroy(event);
  }
  setValueNoThrow(promise, std::move(status));
}

// Drive a recorded event to completion without blocking the EventBase worker:
// poll eventQuery, and while the event is still in-flight re-dispatch onto @p
// evb so the worker thread is freed between polls. On successful completion,
// destroy the event and fulfill @p promise with Ok. On a query error the state
// of the in-flight copies is unknown, so drain @p stream (via drainAndFail)
// before reporting, so no D2D copy is still touching the caller's buffers when
// the error is observed. Takes ownership of @p event and @p promise. noexcept:
// runs as an EventBase task.
void pollEventToCompletion(
    EventBase* evb,
    std::shared_ptr<CudaApi> cudaApi,
    cudaEvent_t event,
    cudaStream_t stream,
    std::promise<Status> promise) noexcept {
  auto poll = [evb,
               cudaApi = std::move(cudaApi),
               event,
               stream,
               promise = std::move(promise)](auto& self) mutable noexcept {
    auto res = cudaApi->eventQuery(event);
    if (res.hasError()) {
      drainAndFail(*cudaApi, promise, stream, event, std::move(res).error());
      return;
    }
    if (res.value()) {
      (void)cudaApi->eventDestroy(event);
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
  auto bytes = encodeDeviceId(deviceId_);
  TransportInfo info(bytes.size());
  std::copy_n(bytes.begin(), bytes.size(), info.data());
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
  auto parsedPeer = decodeDeviceId(remoteInfo, "P2P connect");
  CHECK_RETURN(parsedPeer);
  const int32_t peer = parsedPeer.value();

  // Enable peer access in BOTH directions (no-op for same device or if already
  // enabled). Commit peerDeviceId_ / Connected only after both succeed, so a
  // failure leaves the transport Initialized.
  //
  // The reverse direction is the non-obvious one: get() reads the IPC-imported
  // peer buffer, and ROCm runs that copy on the SOURCE agent, which writes into
  // THIS device's memory. With only local->peer, put() works and get() faults
  // once the two ranks are separate processes. PeerToPeerTransferTest enables
  // both directions for the same reason.
  //
  // The helper keeps each edge paired with its own source device current.
  auto enableOneDirection = [this](int32_t from, int32_t to) {
    CudaDeviceGuard guard(*cudaApi_, from);
    return cudaApi_->deviceEnablePeerAccess(to);
  };

  if (peer != deviceId_) {
    auto forward = enableOneDirection(deviceId_, peer);
    if (forward.hasError()) {
      return forward;
    }
    auto reverse = enableOneDirection(peer, deviceId_);
    if (reverse.hasError()) {
      return reverse;
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
                  stream =
                      static_cast<cudaStream_t>(stream)]() mutable noexcept {
    CudaDeviceGuard deviceGuard(*cudaApi, deviceId);

    for (auto& op : ops) {
      auto st = cudaApi->memcpyAsync(
          op.dst, op.src, op.size, cudaMemcpyDeviceToDevice, stream);
      if (st.hasError()) {
        drainAndFail(
            *cudaApi, promise, stream, /*event=*/nullptr, std::move(st));
        return;
      }
    }

    // Completion via a recorded event, polled and re-dispatched on the
    // EventBase (same model as NVLinkTransport).
    cudaEvent_t event = nullptr;
    auto createSt = cudaApi->eventCreate(&event);
    if (createSt.hasError()) {
      drainAndFail(
          *cudaApi, promise, stream, /*event=*/nullptr, std::move(createSt));
      return;
    }

    auto recordSt = cudaApi->eventRecord(event, stream);
    if (recordSt.hasError()) {
      drainAndFail(*cudaApi, promise, stream, event, std::move(recordSt));
      return;
    }

    // Event recorded after the last copy; hand off to the poller, which frees
    // this worker between polls and fulfills the promise on completion.
    // Last use of the captured cudaApi in this lambda: move to avoid copying
    // the shared_ptr (pollEventToCompletion takes ownership by value).
    pollEventToCompletion(
        evb, std::move(cudaApi), event, stream, std::move(promise));
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

Result<std::vector<P2pTransport::CopyOp>> P2pTransport::buildCopyOps(
    std::span<const TransferRequest> requests,
    Dir dir) const {
  std::vector<CopyOp> ops;
  ops.reserve(requests.size());
  for (auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      return Err(
          ErrCode::InvalidArgument,
          "P2P: local and remote buffer sizes must match");
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    CHECK_RETURN(remoteHandle);

    // mappedPtr() is nullable (moved-from / invalid handle). Reject it here
    // rather than feeding nullptr + offset into a device-to-device memcpy.
    auto* mappedBase = static_cast<uint8_t*>(remoteHandle.value()->mappedPtr());
    if (mappedBase == nullptr) {
      return Err(ErrCode::InvalidArgument, "P2P: null mapped pointer");
    }
    auto* remotePtr = mappedBase + req.remote.nvlinkOffset_;

    if (dir == Dir::Put) {
      ops.emplace_back(CopyOp{remotePtr, req.local.data(), req.local.size()});
    } else {
      ops.emplace_back(
          CopyOp{req.local.mutable_data(), remotePtr, req.remote.size()});
    }
  }
  return ops;
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

  auto ops = buildCopyOps(requests, Dir::Put);
  if (!ops) {
    return make_ready_future<Status>(std::move(ops).error());
  }

  return transfer(std::move(ops).value(), options.stream.value_or(nullptr));
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

  auto ops = buildCopyOps(requests, Dir::Get);
  if (!ops) {
    return make_ready_future<Status>(std::move(ops).error());
  }

  return transfer(std::move(ops).value(), options.stream.value_or(nullptr));
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
  // be aborted mid-flight; their futures still complete via the event poller
  // (pollEventToCompletion).
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

#if defined(__HIP_PLATFORM_AMD__)
  // All supported AMD parts (MI300 gfx942, MI350 gfx950, MI450 gfx1250) are
  // all-to-all XGMI intra-node. Gate on this arch family so a non-XGMI part
  // never lets MultiTransport's presence-driven selectTransport prefer a slow
  // PCIe-P2P link over RDMA (hipDeviceCanAccessPeer conflates the two). See
  // comms/uniflow/amd/AMD_ROCM_IMPLEMENTATION_SUMMARY.md ("P2P / XGMI").
  //
  // Match an exact gfx token, allowing a feature-flag suffix after ':'
  // (e.g. "gfx942:sramecc+:xnack-") but not a longer digit run ("gfx12500").
  static constexpr std::array<std::string_view, 3> kAllowedArch = {
      "gfx942", "gfx950", "gfx1250"};
  auto matchesAllowed = [](std::string_view arch) {
    for (const auto& base : kAllowedArch) {
      if (arch.size() >= base.size() && arch.substr(0, base.size()) == base &&
          (arch.size() == base.size() || arch[base.size()] == ':')) {
        return true;
      }
    }
    return false;
  };

  for (int dev = 0; dev < count.value(); ++dev) {
    auto arch = cudaApi->getDeviceArch(dev);
    if (arch.hasError()) {
      UNIFLOW_LOG_WARN(
          "P2P arch gate: getDeviceArch({}) failed ({}); disabling P2P",
          dev,
          arch.error().message());
      return Err(
          ErrCode::NotImplemented,
          "P2P (XGMI) transport not supported: device arch query failed");
    }
    if (!matchesAllowed(arch.value())) {
      UNIFLOW_LOG_INFO(
          "P2P arch gate: device {} arch '{}' not in the all-XGMI family; "
          "disabling P2P for this node",
          dev,
          arch.value());
      return Err(
          ErrCode::NotImplemented,
          "P2P (XGMI) transport not supported: not an all-XGMI node");
    }
  }
#endif

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

  // p.offset, p.size, and p.base all come off the wire and drive raw pointer
  // math (mappedBase + offset, then a size-byte D2D copy). p.size is pinned to
  // segmentLength above; guard offset against wraparound here. Full bounds
  // checking is not possible: ipcOpenMemHandle does not report the mapping
  // size, and p.base is the exporter's VA. These values come from a cooperating
  // peer rank on the same node, not an untrusted source.
  if (p.offset > std::numeric_limits<uint64_t>::max() - p.size) {
    return Err(
        ErrCode::InvalidArgument, "P2P importSegment: offset + size overflow");
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
  return encodeDeviceId(deviceId_);
}

Status P2pTransportFactory::canConnect(std::span<const uint8_t> peerTopology) {
  auto parsedPeer = decodeDeviceId(peerTopology, "P2P canConnect");
  CHECK_RETURN(parsedPeer);
  const int32_t peerDev = parsedPeer.value();

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
