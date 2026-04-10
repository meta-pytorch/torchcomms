// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"
#include "comms/uniflow/transport/nvlink/NVLinkTopology.h"

#include "comms/uniflow/drivers/nvml/NvmlApi.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace uniflow {

// ---------------------------------------------------------------------------
// NVLinkTransport
// ---------------------------------------------------------------------------

NVLinkTransport::NVLinkTransport(
    int deviceId,
    EventBase* evb,
    std::shared_ptr<CudaApi> cuda_api,
    std::shared_ptr<CudaDriverApi> cuda_driver_api)
    : deviceId_(deviceId),
      deviceName_("cuda:" + std::to_string(deviceId)),
      evb_(evb),
      cuda_api_(std::move(cuda_api)),
      cuda_driver_api_(std::move(cuda_driver_api)) {
  CHECK_THROW_EXCEPTION(evb_ != nullptr, std::invalid_argument);
  if (!cuda_api_) {
    cuda_api_ = std::make_shared<CudaApi>();
  }
  if (!cuda_driver_api_) {
    cuda_driver_api_ = std::make_shared<CudaDriverApi>();
  }
}

TransportInfo NVLinkTransport::bind() {
  // Serialize local device ID as TransportInfo for the peer.
  TransportInfo info(sizeof(int32_t));
  int32_t devId = deviceId_;
  std::memcpy(info.data(), &devId, sizeof(devId));
  UNIFLOW_LOG_INFO("bind: device {}", deviceId_);
  state_ = TransportState::Initialized;
  return info;
}

Status NVLinkTransport::connect(std::span<const uint8_t> remoteInfo) {
  if (remoteInfo.size() != sizeof(int32_t)) {
    UNIFLOW_LOG_ERROR(
        "connect: invalid remote info size (expected {}, got {})",
        sizeof(int32_t),
        remoteInfo.size());
    return Err(
        ErrCode::InvalidArgument,
        "NVLink connect: expected " + std::to_string(sizeof(int32_t)) +
            " bytes, got " + std::to_string(remoteInfo.size()));
  }

  std::memcpy(&peerDeviceId_, remoteInfo.data(), sizeof(int32_t));

  state_ = TransportState::Connected;
  UNIFLOW_LOG_INFO(
      "connect: device {} connected to peer device {}",
      deviceId_,
      peerDeviceId_);
  return Ok();
}

std::future<Status> NVLinkTransport::transfer(
    std::vector<CopyOp> ops,
    cudaStream_t cudaStream) {
  std::promise<Status> promise;
  auto future = promise.get_future();

  evb_->dispatch([evb = evb_,
                  cudaApi = cuda_api_,
                  deviceId = deviceId_,
                  promise = std::move(promise),
                  ops = std::move(ops),
                  cudaStream]() mutable noexcept {
    CudaDeviceGuard deviceGuard(*cudaApi, deviceId);

    for (auto& op : ops) {
      CHECK_SET_PROMISE(
          promise,
          cudaApi->memcpyAsync(
              op.dst, op.src, op.size, cudaMemcpyDeviceToDevice, cudaStream));
    }

    // Record a CUDA event after the last memcpy.
    cudaEvent_t event;
    CHECK_SET_PROMISE(promise, cudaApi->eventCreate(&event));

    CHECK_SET_PROMISE(
        promise,
        cudaApi->eventRecord(event, cudaStream),
        cudaApi->eventDestroy(event));

    // Poll the event until it completes, re-dispatching on EventBase.
    auto poll = [evb, promise = std::move(promise), event, cudaApi](
                    auto& self) mutable noexcept {
      auto res = cudaApi->eventQuery(event);
      CHECK_SET_PROMISE(promise, res, cudaApi->eventDestroy(event));

      if (res.value()) {
        CHECK_SET_PROMISE(promise, cudaApi->eventDestroy(event));
        promise.set_value(Ok());
        return;
      }

      evb->dispatch(
          [self = std::move(self)]() mutable noexcept { self(self); });
    };
    poll(poll);
  });

  return future;
}

Result<const NVLinkRemoteRegistrationHandle*> NVLinkTransport::findRemoteHandle(
    const RemoteRegisteredSegment::Span& span) const {
  for (const auto& h : span.handles_) {
    if (h->transportType() == TransportType::NVLink) {
      if (auto* nvh =
              static_cast<const NVLinkRemoteRegistrationHandle*>(h.get())) {
        return nvh;
      }
    }
  }
  return Err(
      ErrCode::InvalidArgument,
      "NVLink: no NVLink remote registration handle found");
}

std::future<Status> NVLinkTransport::put(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected) {
    UNIFLOW_LOG_ERROR("put: transport not connected");
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "NVLink put: not connected"));
  }

  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  std::vector<CopyOp> ops;
  ops.reserve(requests.size());
  for (auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      UNIFLOW_LOG_ERROR(
          "put: buffer size mismatch (local={}, remote={})",
          req.local.size(),
          req.remote.size());
      return make_ready_future<Status>(
          Err(ErrCode::InvalidArgument,
              "NVLink put: local and remote buffer sizes must match"));
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      UNIFLOW_LOG_ERROR("put: no NVLink remote handle found");
      return make_ready_future<Status>(std::move(remoteHandle).error());
    }
    auto* remoteDst = static_cast<uint8_t*>(remoteHandle.value()->mappedPtr()) +
        req.remote.nvlinkOffset_;
    ops.emplace_back(remoteDst, req.local.data(), req.local.size());
  }

  UNIFLOW_LOG_DEBUG(
      "put: {} requests, total {} bytes", requests.size(), ops[0].size);

  return transfer(
      std::move(ops),
      static_cast<cudaStream_t>(options.stream.value_or(nullptr)));
}

std::future<Status> NVLinkTransport::get(
    std::span<const TransferRequest> requests,
    const RequestOptions& options) {
  if (state_ != TransportState::Connected) {
    UNIFLOW_LOG_ERROR("get: transport not connected");
    return make_ready_future<Status>(
        Err(ErrCode::NotConnected, "NVLink get: not connected"));
  }

  if (requests.empty()) {
    return make_ready_future<Status>(Ok());
  }

  std::vector<CopyOp> ops;
  ops.reserve(requests.size());
  for (auto& req : requests) {
    if (req.local.size() != req.remote.size()) {
      UNIFLOW_LOG_ERROR(
          "get: buffer size mismatch (local={}, remote={})",
          req.local.size(),
          req.remote.size());
      return make_ready_future<Status>(
          Err(ErrCode::InvalidArgument,
              "NVLink get: local and remote buffer sizes must match"));
    }
    auto remoteHandle = findRemoteHandle(req.remote);
    if (!remoteHandle) {
      UNIFLOW_LOG_ERROR("get: no NVLink remote handle found");
      return make_ready_future<Status>(std::move(remoteHandle).error());
    }
    auto* remoteSrc = static_cast<uint8_t*>(remoteHandle.value()->mappedPtr()) +
        req.remote.nvlinkOffset_;
    ops.emplace_back(req.local.mutable_data(), remoteSrc, req.remote.size());
  }

  return transfer(
      std::move(ops),
      static_cast<cudaStream_t>(options.stream.value_or(nullptr)));
}

std::future<Status> NVLinkTransport::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  // TODO: Implement NVLink send (registered)
  std::promise<Status> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Status> NVLinkTransport::send(
    Segment::Span src,
    const RequestOptions& options) {
  // TODO: Implement NVLink send (unregistered)
  std::promise<Status> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Result<size_t>> NVLinkTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  // TODO: Implement NVLink recv (registered)
  std::promise<Result<size_t>> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

std::future<Result<size_t>> NVLinkTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  // TODO: Implement NVLink recv (unregistered)
  std::promise<Result<size_t>> promise;
  promise.set_value(ErrCode::NotImplemented);
  return promise.get_future();
}

void NVLinkTransport::shutdown() {
  UNIFLOW_LOG_INFO("shutdown: device {}", deviceId_);
  state_ = TransportState::Disconnected;
}

// ---------------------------------------------------------------------------
// NVLinkTransportFactory
// ---------------------------------------------------------------------------

NVLinkTransportFactory::NVLinkTransportFactory(
    int device,
    EventBase* evb,
    std::shared_ptr<NvmlApi> nvml_api,
    std::shared_ptr<CudaApi> cuda_api,
    std::shared_ptr<CudaDriverApi> cuda_driver_api)
    : TransportFactory(TransportType::NVLink),
      deviceId_(device),
      evb_(evb),
      nvmlApi_(std::move(nvml_api)),
      cuda_api_(std::move(cuda_api)),
      cuda_driver_api_(std::move(cuda_driver_api)) {
  CHECK_THROW_EXCEPTION(evb_ != nullptr, std::invalid_argument);
  if (nvmlApi_ == nullptr) {
    nvmlApi_ = std::make_shared<NvmlApi>();
  }
  if (cuda_api_ == nullptr) {
    cuda_api_ = std::make_shared<CudaApi>();
  }
  if (cuda_driver_api_ == nullptr) {
    cuda_driver_api_ = std::make_shared<CudaDriverApi>();
  }
  auto& cache = NVLinkTopologyCache::instance(nvmlApi_.get());
  CHECK_THROW_ERROR(cache.available());
  auto count = nvmlApi_->deviceCount();
  CHECK_THROW_ERROR(count);
  CHECK_THROW_EXCEPTION(
      deviceId_ >= 0 && deviceId_ < count.value(), std::out_of_range);
  CHECK_THROW_EXCEPTION(cache.getTopology(deviceId_), std::runtime_error);

  auto handleType = cuda_driver_api_->getCuMemHandleType();
  CHECK_THROW_ERROR(handleType);
  handleType_ = handleType.value();
}

Result<std::unique_ptr<RegistrationHandle>>
NVLinkTransportFactory::registerSegment(Segment& segment) {
  if (segment.memType() != MemoryType::VRAM) {
    UNIFLOW_LOG_ERROR("registerSegment: segment must be VRAM memory");
    return Err(
        ErrCode::InvalidArgument,
        "NVLink registerSegment: segment must be VRAM memory");
  }

  // Activate the CUDA context for the owning device — required by
  // cuMemGetAddressRange_v2 below.
  CudaDeviceGuard deviceGuard(*cuda_api_, deviceId_);

  // Get allocation handle from existing cuMem VMM allocation.
  CUmemGenericAllocationHandle allocHandle;
  CHECK_EXPR(cuda_driver_api_->cuMemRetainAllocationHandle(
      &allocHandle, segment.mutable_data()));

  // Query the actual VMM allocation size. The segment may be a sub-range
  // of a larger allocation (e.g., PyTorch caching allocator pools).
  // cuMemMap on the import side requires the full allocation size.
  CUdeviceptr allocBase = 0;
  size_t allocSize = 0;
  auto rangeStatus = cuda_driver_api_->cuMemGetAddressRange_v2(
      &allocBase,
      &allocSize,
      reinterpret_cast<CUdeviceptr>(segment.mutable_data()));
  if (rangeStatus.hasError()) {
    // Fall back to segment length if query fails (e.g., non-VMM memory).
    UNIFLOW_LOG_WARN(
        "registerSegment: cuMemGetAddressRange_v2 failed ({}), "
        "falling back to segment length {}",
        rangeStatus.error().message(),
        segment.len());
    allocSize = segment.len();
  }

  if (handleType_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // FD mode: export to POSIX file descriptor.
    int fd = -1;
    auto exportStatus = cuda_driver_api_->cuMemExportToShareableHandle(
        &fd, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    if (exportStatus.hasError()) {
      cuda_driver_api_->cuMemRelease(allocHandle);
      return std::move(exportStatus).error();
    }

    pid_t pid = ::getpid();
    UNIFLOW_LOG_INFO(
        "registerSegment: FD mode, device {}, fd={}, pid={}, "
        "segmentSize={}, allocSize={}",
        deviceId_,
        fd,
        pid,
        segment.len(),
        allocSize);
    return std::make_unique<NVLinkFdRegistrationHandle>(
        allocHandle, fd, pid, allocSize, cuda_driver_api_);
  } else if (handleType_ == CU_MEM_HANDLE_TYPE_FABRIC) {
    // Fabric mode: export to fabric handle for cross-node sharing.
    CUmemFabricHandle fabricHandle;
    auto exportStatus = cuda_driver_api_->cuMemExportToShareableHandle(
        &fabricHandle, allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (exportStatus.hasError()) {
      UNIFLOW_LOG_ERROR(
          "registerSegment: fabric export failed: {}",
          exportStatus.error().message());
      cuda_driver_api_->cuMemRelease(allocHandle);
      return std::move(exportStatus).error();
    }

    UNIFLOW_LOG_INFO("registerSegment: Fabric mode, device {}", deviceId_);
    return std::make_unique<NVLinkFabricRegistrationHandle>(
        allocSize, allocHandle, fabricHandle, cuda_driver_api_);
  }

  CHECK_EXPR(cuda_driver_api_->cuMemRelease(allocHandle));
  return Err(ErrCode::InvalidArgument, "NVLink registerSegment: unknown mode");
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
NVLinkTransportFactory::importSegment(
    size_t segmentLength,
    std::span<const uint8_t> payload) {
  if (payload.empty()) {
    UNIFLOW_LOG_ERROR("importSegment: empty payload");
    return Err(ErrCode::InvalidArgument, "NVLink importSegment: empty payload");
  }

  auto mode = static_cast<MemSharingMode>(payload[0]);
  UNIFLOW_LOG_INFO(
      "importSegment: device {}, mode={}, segmentLength={}, payloadSize={}",
      deviceId_,
      static_cast<int>(mode),
      segmentLength,
      payload.size());

  switch (mode) {
    case MemSharingMode::Fabric:
      return importSegmentFabric(payload);
    case MemSharingMode::PosixFd:
      return importSegmentFd(payload);
    default:
      return Err(
          ErrCode::InvalidArgument,
          "NVLink importSegment: unknown mode byte " +
              std::to_string(payload[0]));
  }
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
NVLinkTransportFactory::importSegmentFabric(std::span<const uint8_t> payload) {
  if (payload.size() != sizeof(NVLinkFabricRegistrationHandle::Payload)) {
    return Err(
        ErrCode::InvalidArgument,
        "NVLink importSegment (Fabric): expected " +
            std::to_string(sizeof(NVLinkFabricRegistrationHandle::Payload)) +
            " bytes, got " + std::to_string(payload.size()));
  }
  NVLinkFabricRegistrationHandle::Payload fabricPayload;
  std::memcpy(&fabricPayload, payload.data(), sizeof(fabricPayload));
  CUmemFabricHandle fabricHandle = fabricPayload.fabricHandle;

  // Import the fabric handle to get a local allocation handle.
  CUmemGenericAllocationHandle allocHandle;
  CHECK_EXPR(cuda_driver_api_->cuMemImportFromShareableHandle(
      &allocHandle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC));

  return mapImportedAllocation(
      allocHandle,
      static_cast<size_t>(fabricPayload.allocationSize),
      CU_MEM_HANDLE_TYPE_FABRIC);
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
NVLinkTransportFactory::importSegmentFd(std::span<const uint8_t> payload) {
  if (payload.size() != sizeof(NVLinkFdRegistrationHandle::Payload)) {
    return Err(
        ErrCode::InvalidArgument,
        "NVLink importSegment (FD): expected " +
            std::to_string(sizeof(NVLinkFdRegistrationHandle::Payload)) +
            " bytes, got " + std::to_string(payload.size()));
  }
  NVLinkFdRegistrationHandle::Payload fdPayload;
  std::memcpy(&fdPayload, payload.data(), sizeof(fdPayload));

  // 1. Open a pidfd for the remote process.
  int pidfd = static_cast<int>(
      syscall(SYS_pidfd_open, static_cast<pid_t>(fdPayload.pid), 0));
  if (pidfd < 0) {
    return Err(
        ErrCode::DriverError,
        std::string("pidfd_open failed: ") +
            std::system_category().message(errno));
  }

  // 2. Duplicate the remote FD into our process.
  int localFd =
      static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, fdPayload.fd, 0));
  ::close(pidfd);
  if (localFd < 0) {
    return Err(
        ErrCode::DriverError,
        std::string("pidfd_getfd failed: ") + std::strerror(errno));
  }

  // 3. Import the FD to get a local allocation handle.
  // CUDA expects the fd value itself cast to void*, not a pointer to the fd.
  CUmemGenericAllocationHandle allocHandle;
  auto importStatus = cuda_driver_api_->cuMemImportFromShareableHandle(
      &allocHandle,
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(localFd),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  ::close(localFd);
  if (importStatus.hasError()) {
    return std::move(importStatus).error();
  }

  return mapImportedAllocation(
      allocHandle,
      static_cast<size_t>(fdPayload.allocationSize),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

Result<std::unique_ptr<RemoteRegistrationHandle>>
NVLinkTransportFactory::mapImportedAllocation(
    CUmemGenericAllocationHandle allocHandle,
    size_t segmentLength,
    CUmemAllocationHandleType handleType) {
  // Query allocation granularity for VA reservation alignment.
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = deviceId_;
  prop.requestedHandleTypes = handleType;

  size_t granularity = 0;
  auto granStatus = cuda_driver_api_->cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (granStatus.hasError()) {
    cuda_driver_api_->cuMemRelease(allocHandle);
    return std::move(granStatus).error();
  }

  size_t alignedSize =
      ((segmentLength + granularity - 1) / granularity) * granularity;

  // Reserve virtual address space.
  CUdeviceptr mappedPtr = 0;
  auto reserveStatus = cuda_driver_api_->cuMemAddressReserve(
      &mappedPtr, alignedSize, granularity, 0, 0);
  if (reserveStatus.hasError()) {
    cuda_driver_api_->cuMemRelease(allocHandle);
    return std::move(reserveStatus).error();
  }

  // Map the imported allocation into the reserved VA range.
  auto mapStatus =
      cuda_driver_api_->cuMemMap(mappedPtr, alignedSize, 0, allocHandle, 0);
  if (mapStatus.hasError()) {
    cuda_driver_api_->cuMemAddressFree(mappedPtr, alignedSize);
    cuda_driver_api_->cuMemRelease(allocHandle);
    return std::move(mapStatus).error();
  }

  // Set read/write access for the local device.
  CUmemAccessDesc accessDesc{};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId_;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  auto accessStatus =
      cuda_driver_api_->cuMemSetAccess(mappedPtr, alignedSize, &accessDesc, 1);
  if (accessStatus.hasError()) {
    cuda_driver_api_->cuMemUnmap(mappedPtr, alignedSize);
    cuda_driver_api_->cuMemAddressFree(mappedPtr, alignedSize);
    cuda_driver_api_->cuMemRelease(allocHandle);
    return std::move(accessStatus).error();
  }

  UNIFLOW_LOG_INFO(
      "mapImportedAllocation: device {}, mappedPtr={:#x}, alignedSize={}",
      deviceId_,
      mappedPtr,
      alignedSize);
  return std::make_unique<NVLinkRemoteRegistrationHandle>(
      allocHandle, mappedPtr, alignedSize, cuda_driver_api_);
}

Result<std::unique_ptr<Transport>> NVLinkTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  CHECK_EXPR(canConnect(peerTopology));
  UNIFLOW_LOG_INFO("createTransport: device {}", deviceId_);
  return std::make_unique<NVLinkTransport>(
      deviceId_, evb_, cuda_api_, cuda_driver_api_);
}

std::vector<uint8_t> NVLinkTransportFactory::getTopology() {
  return NVLinkTopologyCache::instance(nvmlApi_.get())
      .getTopology(deviceId_)
      ->serialize();
}

Status NVLinkTransportFactory::canConnect(
    std::span<const uint8_t> peerTopology) {
  auto& topo =
      NVLinkTopologyCache::instance(nvmlApi_.get()).getTopology(deviceId_);

  auto peer = NVLinkTopology::deserialize(peerTopology);
  CHECK_RETURN(peer);

  // MNNVL fabric path: both have non-zero cluster UUIDs in the same domain.
  if (!isZeroUUid(topo->clusterId) && !isZeroUUid(peer.value().clusterId)) {
    if (topo->sameDomain(peer.value())) {
      UNIFLOW_LOG_INFO(
          "canConnect: MNNVL fabric path, device {} -> peer device {}",
          deviceId_,
          peer.value().cudaDeviceId);
      return Ok();
    }
    // Different MNNVL domains — fall through to intra-node check in case
    // the devices are on the same host and support P2P.
  }

  // Intra-node path: check same host via hostHash, then verify P2P access.
  if (!topo->sameHost(peer.value())) {
    UNIFLOW_LOG_WARN(
        "canConnect: peer device {} not on same host",
        peer.value().cudaDeviceId);
    return Err(
        ErrCode::TopologyDisconnect, "NVLink: peer is not on the same host");
  }

  auto canAccess =
      cuda_api_->deviceCanAccessPeer(deviceId_, peer.value().cudaDeviceId);
  if (canAccess.hasError()) {
    UNIFLOW_LOG_ERROR(
        "canConnect: deviceCanAccessPeer failed: {}",
        canAccess.error().message());
    return std::move(canAccess).error();
  }
  if (!canAccess.value()) {
    UNIFLOW_LOG_WARN(
        "canConnect: P2P not supported between device {} and {}",
        deviceId_,
        peer.value().cudaDeviceId);
    return Err(
        ErrCode::TopologyDisconnect,
        "NVLink: P2P access not supported between devices");
  }

  return Ok();
}

} // namespace uniflow
