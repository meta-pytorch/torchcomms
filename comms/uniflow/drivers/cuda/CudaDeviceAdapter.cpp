// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

#include <unistd.h>

#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"

namespace uniflow {

CudaDeviceAdapter::CudaDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi)
    : cudaApi_(std::move(cudaApi)), cudaDriverApi_(std::move(cudaDriverApi)) {
  const long ps = ::sysconf(_SC_PAGESIZE);
  pageSize_ = ps > 0 ? static_cast<size_t>(ps) : size_t{4096};
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

Result<bool> CudaDeviceAdapter::isDmaBuffSupported(int deviceId) {
  if (!cudaDriverApi_) {
    return Err(
        ErrCode::DriverError, "CudaDeviceAdapter: CudaDriverApi unavailable");
  }
  return cudaDriverApi_->isDmaBufSupported(deviceId);
}

Result<DmaBuff>
CudaDeviceAdapter::exportDmaBuff(int /* deviceId */, void* ptr, size_t len) {
  if (!cudaDriverApi_) {
    return Err(
        ErrCode::DriverError, "CudaDeviceAdapter: CudaDriverApi unavailable");
  }
  if (ptr == nullptr || len == 0) {
    return Err(ErrCode::InvalidArgument, "exportDmaBuff: bad ptr or len");
  }

  // dma-buf registration requires a page-aligned base address. Align down
  // and remember the offset so the caller can pass it to regDmabufMr.
  const auto addr = reinterpret_cast<uint64_t>(ptr);
  const uintptr_t alignedAddr = addr & ~(pageSize_ - 1);
  const uint64_t dmaBufOffset = addr - alignedAddr;
  const size_t dmaBufLen =
      (len + dmaBufOffset + pageSize_ - 1) & ~(pageSize_ - 1);

  DmaBuff out;
  // TODO: set CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE if a data-direct
  // link is available.
  constexpr unsigned long long kFlags = 0;
  auto status = cudaDriverApi_->cuMemGetHandleForAddressRange(
      &out.fd,
      toDevicePtr(alignedAddr),
      dmaBufLen,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      kFlags);
  CHECK_RETURN(status);

  out.offset = dmaBufOffset;
  out.len = len;
  out.iova = static_cast<uint64_t>(addr);
  return out;
}

Status CudaDeviceAdapter::closeDmaBuff(DmaBuff& buff) {
  if (buff.fd >= 0) {
    ::close(buff.fd);
    buff.fd = -1;
  }
  return Ok();
}

std::shared_ptr<DeviceAdapter> createDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi) {
  if (!cudaApi) {
    cudaApi = std::make_shared<CudaApi>();
  }
  if (!cudaDriverApi) {
    cudaDriverApi = std::make_shared<CudaDriverApi>();
  }
  return std::make_shared<CudaDeviceAdapter>(
      std::move(cudaApi), std::move(cudaDriverApi));
}

} // namespace uniflow
