// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"
#include "comms/uniflow/Result.h"

#include <unistd.h>

#include <cstring>

namespace uniflow {

// ---------------------------------------------------------------------------
// NVLinkFabricRegistrationHandle
// ---------------------------------------------------------------------------

NVLinkFabricRegistrationHandle::NVLinkFabricRegistrationHandle(
    uint64_t allocationSize,
    CUmemGenericAllocationHandle allocHandle,
    CUmemFabricHandle fabricHandle,
    std::shared_ptr<CudaDriverApi> cuda_driver_api)
    : allocationSize_(allocationSize),
      allocHandle_(allocHandle),
      fabricHandle_(fabricHandle),
      cuda_driver_api_(std::move(cuda_driver_api)) {
  if (!cuda_driver_api_) {
    cuda_driver_api_ = std::make_shared<CudaDriverApi>();
  }
}

NVLinkFabricRegistrationHandle::~NVLinkFabricRegistrationHandle() {
  CHECK_THROW_ERROR(cuda_driver_api_->cuMemRelease(allocHandle_));
}

std::vector<uint8_t> NVLinkFabricRegistrationHandle::serialize() const {
  Payload payload{MemSharingMode::Fabric, allocationSize_, fabricHandle_};
  std::vector<uint8_t> buf(sizeof(payload));
  std::memcpy(buf.data(), &payload, sizeof(payload));
  return buf;
}

// ---------------------------------------------------------------------------
// NVLinkFdRegistrationHandle
// ---------------------------------------------------------------------------

NVLinkFdRegistrationHandle::NVLinkFdRegistrationHandle(
    CUmemGenericAllocationHandle allocHandle,
    int exportedFd,
    pid_t ownerPid,
    size_t allocationSize,
    std::shared_ptr<CudaDriverApi> cuda_driver_api)
    : allocHandle_(allocHandle),
      exportedFd_(exportedFd),
      ownerPid_(ownerPid),
      allocationSize_(allocationSize),
      cuda_driver_api_(std::move(cuda_driver_api)) {
  if (!cuda_driver_api_) {
    cuda_driver_api_ = std::make_shared<CudaDriverApi>();
  }
}

NVLinkFdRegistrationHandle::~NVLinkFdRegistrationHandle() {
  if (exportedFd_ >= 0) {
    ::close(exportedFd_);
  }
  CHECK_THROW_ERROR(cuda_driver_api_->cuMemRelease(allocHandle_));
}

std::vector<uint8_t> NVLinkFdRegistrationHandle::serialize() const {
  Payload payload{
      MemSharingMode::PosixFd,
      static_cast<int32_t>(exportedFd_),
      static_cast<int32_t>(ownerPid_),
      static_cast<uint64_t>(allocationSize_),
  };
  std::vector<uint8_t> buf(sizeof(payload));
  std::memcpy(buf.data(), &payload, sizeof(payload));
  return buf;
}

// ---------------------------------------------------------------------------
// NVLinkRemoteRegistrationHandle
// ---------------------------------------------------------------------------

NVLinkRemoteRegistrationHandle::NVLinkRemoteRegistrationHandle(
    CUmemGenericAllocationHandle allocHandle,
    CUdeviceptr mappedPtr,
    size_t mappedSize,
    std::shared_ptr<CudaDriverApi> cuda_driver_api)
    : allocHandle_(allocHandle),
      mappedPtr_(mappedPtr),
      mappedSize_(mappedSize),
      cuda_driver_api_(std::move(cuda_driver_api)) {
  if (!cuda_driver_api_) {
    cuda_driver_api_ = std::make_shared<CudaDriverApi>();
  }
}

NVLinkRemoteRegistrationHandle::~NVLinkRemoteRegistrationHandle() {
  CHECK_THROW_ERROR(cuda_driver_api_->cuMemUnmap(mappedPtr_, mappedSize_));
  CHECK_THROW_ERROR(
      cuda_driver_api_->cuMemAddressFree(mappedPtr_, mappedSize_));
  CHECK_THROW_ERROR(cuda_driver_api_->cuMemRelease(allocHandle_));
}

} // namespace uniflow
