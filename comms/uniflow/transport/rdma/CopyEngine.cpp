// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/CopyEngine.h"

#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include <cstring>

namespace uniflow {

CopyEngine::CopyEngine(
    MemoryType memType,
    int deviceId,
    std::optional<cudaStream_t> stream,
    CudaApi* cudaApi,
    CudaDriverApi* cudaDriverApi)
    : memType_(memType),
      deviceId_(deviceId),
      stream_(stream),
      cudaApi_(cudaApi),
      cudaDriverApi_(cudaDriverApi) {}

void CopyEngine::copyToSlab(RdmaSlab& slab, const void* src, size_t len) {
  if (memType_ == MemoryType::VRAM) {
    CudaDeviceGuard guard(*cudaApi_, deviceId_);
    cudaApi_->memcpyAsync(
        slab.ptr(), src, len, cudaMemcpyDeviceToHost, *stream_);
    cudaDriverApi_->streamWriteValue64(
        *stream_,
        toDevicePtr(slab.stateDeviceAddr()),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT);
  } else {
    std::memcpy(slab.ptr(), src, len);
  }
}

void CopyEngine::copyFromSlab(void* dst, RdmaSlab& slab, size_t len) {
  if (memType_ == MemoryType::VRAM) {
    CudaDeviceGuard guard(*cudaApi_, deviceId_);
    cudaApi_->memcpyAsync(
        dst, slab.ptr(), len, cudaMemcpyHostToDevice, *stream_);
    cudaDriverApi_->streamWriteValue64(
        *stream_,
        toDevicePtr(slab.stateDeviceAddr()),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT);
  } else {
    std::memcpy(dst, slab.ptr(), len);
  }
}

bool CopyEngine::isCopyDone(RdmaSlab& slab) const {
  if (memType_ != MemoryType::VRAM) {
    return true;
  }
  return slab.state().load(std::memory_order_acquire) != 0;
}

void CopyEngine::resetState(RdmaSlab& slab) {
  if (memType_ != MemoryType::VRAM) {
    return;
  }
  slab.state().store(0, std::memory_order_release);
}

} // namespace uniflow
