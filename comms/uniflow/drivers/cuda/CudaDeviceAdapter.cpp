// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

#include <unistd.h>

#include <utility>

#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"

namespace uniflow {

namespace {

class DeviceRestorer {
 public:
  DeviceRestorer(CudaApi& api, int previous) : api_(api), previous_(previous) {}

  DeviceRestorer(const DeviceRestorer&) = delete;
  DeviceRestorer& operator=(const DeviceRestorer&) = delete;
  DeviceRestorer(DeviceRestorer&&) = delete;
  DeviceRestorer& operator=(DeviceRestorer&&) = delete;

  ~DeviceRestorer() noexcept {
    if (active_) {
      try {
        (void)api_.setDevice(previous_);
      } catch (...) {
      }
    }
  }

  Status restore() {
    active_ = false;
    return api_.setDevice(previous_);
  }

 private:
  CudaApi& api_;
  int previous_;
  bool active_{true};
};

template <class R>
R afterRestore(R result, Status restored) {
  if (restored.hasError()) {
    if (result.hasError()) {
      return Err(
          result.error().code(),
          result.error().message() +
              "; device restoration failed: " + restored.error().message());
    }
    return restored.error();
  }
  return result;
}

template <class Operation>
auto onDevice(CudaApi& api, int deviceId, Operation&& operation)
    -> decltype(operation()) {
  using R = decltype(operation());
  if (deviceId < 0) {
    return Err(ErrCode::InvalidArgument, "GPU operation requires a device ID");
  }
  auto previous = api.getDevice();
  CHECK_RETURN(previous);
  if (previous.value() < 0) {
    return Err(ErrCode::DriverError, "invalid current GPU device");
  }
  DeviceRestorer restorer(api, previous.value());
  auto selected = api.setDevice(deviceId);
  if (selected.hasError()) {
    return afterRestore(R(selected.error()), restorer.restore());
  }
  auto result = operation();
  return afterRestore(std::move(result), restorer.restore());
}

template <class Operation>
void cleanupOnDevice(
    CudaApi& api,
    int deviceId,
    Operation&& operation) noexcept {
  try {
    (void)onDevice(api, deviceId, std::forward<Operation>(operation));
  } catch (...) {
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct UnpublishedCudaStream {
  CudaApi& api;
  int deviceId{};
  cudaStream_t stream{nullptr};

  ~UnpublishedCudaStream() noexcept {
    if (stream) {
      cleanupOnDevice(api, deviceId, [&] { return api.streamDestroy(stream); });
    }
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct UnpublishedDmaBuff {
  DmaBuff buff;

  ~UnpublishedDmaBuff() noexcept {
    if (buff.fd >= 0) {
      ::close(buff.fd);
    }
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct UnpublishedCudaHost {
  CudaApi& api;
  int deviceId{};
  void* ptr{nullptr};

  ~UnpublishedCudaHost() noexcept {
    if (ptr) {
      cleanupOnDevice(api, deviceId, [&] { return api.hostFree(ptr); });
    }
  }
};

class CudaDeviceCompletion : public DeviceCompletion {
 public:
  CudaDeviceCompletion(std::shared_ptr<CudaApi> api, int deviceId)
      : api_(std::move(api)), deviceId_(deviceId) {}

  CudaDeviceCompletion(const CudaDeviceCompletion&) = delete;
  CudaDeviceCompletion& operator=(const CudaDeviceCompletion&) = delete;
  CudaDeviceCompletion(CudaDeviceCompletion&&) = delete;
  CudaDeviceCompletion& operator=(CudaDeviceCompletion&&) = delete;

  ~CudaDeviceCompletion() noexcept override {
    if (event_) {
      cleanupOnDevice(
          *api_, deviceId_, [&] { return api_->eventDestroy(event_); });
    }
  }

  Status record(cudaStream_t stream) {
    CHECK_EXPR(api_->eventCreate(&event_));
    if (!event_) {
      return Err(ErrCode::DriverError, "GPU event creation returned no event");
    }
    return api_->eventRecord(event_, stream);
  }

  Result<bool> ready() override {
    return onDevice(*api_, deviceId_, [&] { return api_->eventQuery(event_); });
  }

  Status wait() override {
    return onDevice(
        *api_, deviceId_, [&] { return api_->eventSynchronize(event_); });
  }

 private:
  std::shared_ptr<CudaApi> api_;
  int deviceId_;
  cudaEvent_t event_{nullptr};
};

} // namespace

CudaDeviceAdapter::CudaDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi)
    : cudaApi_(std::move(cudaApi)), cudaDriverApi_(std::move(cudaDriverApi)) {
  const long ps = ::sysconf(_SC_PAGESIZE);
  pageSize_ = ps > 0 ? static_cast<size_t>(ps) : size_t{4096};
}

DeviceCapabilities CudaDeviceAdapter::capabilities() const noexcept {
  const bool runtime = static_cast<bool>(cudaApi_);
  return {
      runtime, runtime && static_cast<bool>(cudaDriverApi_), runtime, runtime};
}

Status CudaDeviceAdapter::copyAsync(
    int deviceId,
    void* dst,
    const void* src,
    size_t count,
    DeviceCopyKind kind,
    DeviceStream stream) {
  if (!cudaApi_) {
    return Err(ErrCode::NotImplemented, "device asynchronous copy unavailable");
  }
  cudaMemcpyKind cudaKind;
  switch (kind) {
    case DeviceCopyKind::HostToDevice:
      cudaKind = cudaMemcpyHostToDevice;
      break;
    case DeviceCopyKind::DeviceToHost:
      cudaKind = cudaMemcpyDeviceToHost;
      break;
    case DeviceCopyKind::DeviceToDevice:
      cudaKind = cudaMemcpyDeviceToDevice;
      break;
    default:
      return Err(ErrCode::InvalidArgument, "invalid device copy kind");
  }
  return onDevice(*cudaApi_, deviceId, [&] {
    return cudaApi_->memcpyAsync(
        dst, src, count, cudaKind, static_cast<cudaStream_t>(stream));
  });
}

Status CudaDeviceAdapter::signal64(
    int deviceId,
    DeviceStream stream,
    uint64_t* address,
    uint64_t value) {
  if (!cudaApi_ || !cudaDriverApi_) {
    return Err(ErrCode::NotImplemented, "device stream signal unavailable");
  }
  if (!address) {
    return Err(
        ErrCode::InvalidArgument, "device stream signal requires an address");
  }
  return onDevice(*cudaApi_, deviceId, [&] {
    return cudaDriverApi_->streamWriteValue64(
        static_cast<CUstream>(stream),
        toDevicePtr(reinterpret_cast<uint64_t>(address)),
        value,
        0);
  });
}

Result<DeviceStream> CudaDeviceAdapter::createNonBlockingStream(int deviceId) {
  if (!cudaApi_) {
    return Err(ErrCode::NotImplemented, "device private streams unavailable");
  }
  UnpublishedCudaStream out{*cudaApi_, deviceId};
  CHECK_EXPR(onDevice(*cudaApi_, deviceId, [&] {
    return cudaApi_->streamCreateNonBlocking(&out.stream);
  }));
  if (!out.stream) {
    return Err(ErrCode::DriverError, "GPU stream creation returned no stream");
  }
  return static_cast<DeviceStream>(std::exchange(out.stream, nullptr));
}

Status CudaDeviceAdapter::synchronizeStream(int deviceId, DeviceStream stream) {
  if (!cudaApi_) {
    return Err(
        ErrCode::NotImplemented, "device stream synchronization unavailable");
  }
  return onDevice(*cudaApi_, deviceId, [&] {
    return cudaApi_->streamSynchronize(static_cast<cudaStream_t>(stream));
  });
}

Status CudaDeviceAdapter::destroyStream(int deviceId, DeviceStream stream) {
  if (!cudaApi_) {
    return Err(
        ErrCode::NotImplemented, "device stream destruction unavailable");
  }
  if (!stream) {
    return Err(ErrCode::InvalidArgument, "cannot destroy a default stream");
  }
  return onDevice(*cudaApi_, deviceId, [&] {
    return cudaApi_->streamDestroy(static_cast<cudaStream_t>(stream));
  });
}

Result<std::unique_ptr<DeviceCompletion>> CudaDeviceAdapter::recordCompletion(
    int deviceId,
    DeviceStream stream) {
  if (!cudaApi_) {
    return Err(ErrCode::NotImplemented, "device completion unavailable");
  }
  auto completion = std::make_unique<CudaDeviceCompletion>(cudaApi_, deviceId);
  CHECK_EXPR(onDevice(*cudaApi_, deviceId, [&] {
    return completion->record(static_cast<cudaStream_t>(stream));
  }));
  return std::unique_ptr<DeviceCompletion>(std::move(completion));
}

Result<void*> CudaDeviceAdapter::pinnedHostAlloc(size_t size) {
  return cudaApi_->hostAlloc(size, cudaHostAllocMapped | cudaHostAllocPortable);
}

Status CudaDeviceAdapter::pinnedHostFree(void* ptr) {
  return cudaApi_->hostFree(ptr);
}

Result<void*> CudaDeviceAdapter::hostGetDevicePointer(void* hostPtr) {
  return cudaApi_->hostGetDevicePointer(hostPtr);
}

Result<void*> CudaDeviceAdapter::pinnedHostAlloc(int deviceId, size_t size) {
  if (!cudaApi_) {
    return Err(
        ErrCode::NotImplemented, "device pinned host storage unavailable");
  }
  if (size == 0) {
    return Err(ErrCode::InvalidArgument, "pinnedHostAlloc: size must be > 0");
  }
  UnpublishedCudaHost out{*cudaApi_, deviceId};
  CHECK_EXPR(onDevice(*cudaApi_, deviceId, [&]() -> Status {
    auto result = pinnedHostAlloc(size);
    CHECK_RETURN(result);
    out.ptr = result.value();
    if (!out.ptr) {
      return Err(
          ErrCode::DriverError, "GPU host allocation returned no storage");
    }
    return Ok();
  }));
  return std::exchange(out.ptr, nullptr);
}

Status CudaDeviceAdapter::pinnedHostFree(int deviceId, void* ptr) {
  if (!cudaApi_) {
    return Err(
        ErrCode::NotImplemented, "device pinned host storage unavailable");
  }
  return onDevice(*cudaApi_, deviceId, [&] { return pinnedHostFree(ptr); });
}

Result<void*> CudaDeviceAdapter::hostGetDevicePointer(
    int deviceId,
    void* hostPtr) {
  if (!cudaApi_) {
    return Err(ErrCode::NotImplemented, "device host mapping unavailable");
  }
  return onDevice(
      *cudaApi_, deviceId, [&] { return hostGetDevicePointer(hostPtr); });
}

Result<bool> CudaDeviceAdapter::isDmaBuffSupported(int deviceId) {
  if (!cudaDriverApi_) {
    return Err(
        ErrCode::DriverError, "CudaDeviceAdapter: CudaDriverApi unavailable");
  }
  return cudaDriverApi_->isDmaBufSupported(deviceId);
}

Result<DmaBuff> CudaDeviceAdapter::exportDmaBuff(
    int deviceId,
    void* ptr,
    size_t len,
    DmaBufMapping mapping) {
  if (!cudaDriverApi_) {
    return Err(
        ErrCode::DriverError, "CudaDeviceAdapter: CudaDriverApi unavailable");
  }
  if (!cudaApi_) {
    return Err(ErrCode::DriverError, "CudaDeviceAdapter: CudaApi unavailable");
  }
  if (ptr == nullptr || len == 0) {
    return Err(ErrCode::InvalidArgument, "exportDmaBuff: bad ptr or len");
  }

  /*
   * Resolve the mapping flag. Data Direct maps the buffer over PCIe BAR1 so the
   * NIC reaches GPU HBM directly (mlx5 GDAKI / NCCL_IB_DATA_DIRECT); it needs
   * the CUDA 12.8 range flag, unavailable on older toolkits.
   */
  unsigned long long flags = 0;
  if (mapping == DmaBufMapping::DataDirect) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#else
    return Err(
        ErrCode::DriverError,
        "exportDmaBuff: Data Direct requires CUDA >= 12.8");
#endif
  }

  // dma-buf registration requires a page-aligned base address. Align down
  // and remember the offset so the caller can pass it to regDmabufMr.
  const auto addr = reinterpret_cast<uint64_t>(ptr);
  const uintptr_t alignedAddr = addr & ~(pageSize_ - 1);
  const uint64_t dmaBufOffset = addr - alignedAddr;
  const size_t dmaBufLen =
      (len + dmaBufOffset + pageSize_ - 1) & ~(pageSize_ - 1);

  UnpublishedDmaBuff out;
  CHECK_EXPR(onDevice(*cudaApi_, deviceId, [&] {
    return cudaDriverApi_->memGetHandleForAddressRange(
        &out.buff.fd,
        toDevicePtr(alignedAddr),
        dmaBufLen,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
        flags);
  }));

  out.buff.offset = dmaBufOffset;
  out.buff.len = len;
  out.buff.iova = static_cast<uint64_t>(addr);
  out.buff.dmaBufLen = dmaBufLen;
  return std::exchange(out.buff, DmaBuff{});
}

Status CudaDeviceAdapter::closeDmaBuff(DmaBuff& buff) {
  if (buff.fd >= 0) {
    ::close(buff.fd);
    buff.fd = -1;
  }
  return Ok();
}

} // namespace uniflow
