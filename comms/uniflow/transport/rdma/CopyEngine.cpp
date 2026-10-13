// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/CopyEngine.h"

#include <cstring>
#include <exception>
#include <utility>

#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

namespace uniflow {

CopyEngine::CopyEngine(
    MemoryType memType,
    int deviceId,
    std::optional<DeviceStream> stream,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : memType_(memType),
      deviceId_(deviceId),
      stream_(stream),
      deviceAdapter_(std::move(deviceAdapter)) {}

CopyEngine::~CopyEngine() {
  (void)drain();
}

Status CopyEngine::validate(const RdmaSlab& slab) const {
  if (memType_ != MemoryType::VRAM) {
    return Ok();
  }
  if (deviceId_ < 0 || !stream_.has_value() || !deviceAdapter_) {
    return Err(
        ErrCode::InvalidArgument,
        "VRAM copy requires a device adapter, device ID and explicit stream");
  }
  const auto capabilities = deviceAdapter_->capabilities();
  if (!capabilities.asyncCopy) {
    return Err(
        ErrCode::NotImplemented, "selected backend cannot stage VRAM copies");
  }
  if (!capabilities.completion &&
      !(capabilities.streamSignal64 && slab.mappedStateAddr(deviceId_))) {
    return Err(
        ErrCode::NotImplemented,
        "selected backend cannot complete a copy into this staging pool");
  }
  return Ok();
}

Status CopyEngine::copyToSlab(RdmaSlab& slab, const void* src, size_t len) {
  return copy(slab, slab.ptr(), src, len, DeviceCopyKind::DeviceToHost);
}

Status CopyEngine::copyFromSlab(void* dst, RdmaSlab& slab, size_t len) {
  return copy(slab, dst, slab.ptr(), len, DeviceCopyKind::HostToDevice);
}

Status CopyEngine::copy(
    RdmaSlab& slab,
    void* dst,
    const void* src,
    size_t len,
    DeviceCopyKind kind) {
  if (len > slab.size()) {
    return Err(ErrCode::InvalidArgument, "copy exceeds staging slab size");
  }
  CHECK_EXPR(validate(slab));
  if (memType_ != MemoryType::VRAM) {
    // NOLINTNEXTLINE(facebook-security-vulnerable-memcpy)
    std::memcpy(dst, src, len);
    return Ok();
  }
  if (pendingCopies_.contains(slab.addr())) {
    return Err(
        ErrCode::InvalidArgument, "staging slab still has a pending copy");
  }

  try {
    // Create the owner before an adapter callback can enqueue or throw.
    auto& pending = pendingCopies_.try_emplace(slab.addr()).first->second;
    const auto marker = slab.mappedStateAddr(deviceId_);
    pending.mappedMarker =
        deviceAdapter_->capabilities().streamSignal64 && marker;
    if (pending.mappedMarker) {
      slab.state().store(0, std::memory_order_release);
    }
    auto status =
        deviceAdapter_->copyAsync(deviceId_, dst, src, len, kind, *stream_);
    if (!status) {
      return failAndDrain(std::move(status).error());
    }

    if (pending.mappedMarker) {
      status = deviceAdapter_->signal64(
          deviceId_,
          *stream_,
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          reinterpret_cast<uint64_t*>(*marker),
          1);
      if (!status) {
        return failAndDrain(std::move(status).error());
      }
    } else {
      auto completion = deviceAdapter_->recordCompletion(deviceId_, *stream_);
      if (!completion) {
        return failAndDrain(std::move(completion).error());
      }
      if (!completion.value()) {
        return failAndDrain(Err(
            ErrCode::DriverError, "backend returned a null copy completion"));
      }
      pending.completion = std::move(completion).value();
    }
    return Ok();
  } catch (const std::exception& error) {
    return failAndDrain(
        Err(ErrCode::DriverError,
            "device copy failed: " + std::string(error.what())));
  } catch (...) {
    return failAndDrain(Err(ErrCode::DriverError, "device copy threw"));
  }
}

Result<bool> CopyEngine::isCopyDone(const RdmaSlab& slab) {
  if (memType_ != MemoryType::VRAM) {
    return true;
  }
  auto it = pendingCopies_.find(slab.addr());
  if (it == pendingCopies_.end()) {
    return true;
  }

  try {
    Result<bool> ready = it->second.mappedMarker
        ? Result<bool>(slab.state().load(std::memory_order_acquire) != 0)
        : it->second.completion->ready();
    if (!ready) {
      return failAndDrain(std::move(ready).error()).error();
    }
    if (ready.value()) {
      pendingCopies_.erase(it);
    }
    return ready;
  } catch (const std::exception& error) {
    return failAndDrain(
               Err(ErrCode::DriverError,
                   "copy completion failed: " + std::string(error.what())))
        .error();
  } catch (...) {
    return failAndDrain(
               Err(ErrCode::DriverError, "copy completion query threw"))
        .error();
  }
}

Status CopyEngine::resetState(RdmaSlab& slab) {
  if (pendingCopies_.contains(slab.addr())) {
    return Err(
        ErrCode::InvalidArgument, "cannot reset a slab before copy completion");
  }
  if (memType_ == MemoryType::VRAM) {
    slab.state().store(0, std::memory_order_release);
  }
  return Ok();
}

Status CopyEngine::drain() {
  if (pendingCopies_.empty()) {
    return Ok();
  }
  try {
    CHECK_EXPR(deviceAdapter_->synchronizeStream(deviceId_, *stream_));
    pendingCopies_.clear();
    return Ok();
  } catch (const std::exception& error) {
    return Err(
        ErrCode::DriverError,
        "copy stream drain failed: " + std::string(error.what()));
  } catch (...) {
    return Err(ErrCode::DriverError, "copy stream drain threw");
  }
}

Status CopyEngine::failAndDrain(Err error) {
  (void)drain();
  return error;
}

} // namespace uniflow
