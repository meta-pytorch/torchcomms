// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/CopyEngine.h"
#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"

#include <cstdint>
#include <cstring>

namespace uniflow {

namespace {
// On NVIDIA CUdeviceptr is an integer handle, so a device address (uint64_t)
// converts with static_cast. After hipification CUdeviceptr is hipDeviceptr_t
// (a pointer), which requires reinterpret_cast from the integer address.
inline CUdeviceptr toDevicePtr(uint64_t addr) {
#if defined(__HIP_PLATFORM_AMD__)
  return reinterpret_cast<CUdeviceptr>(addr);
#else
  return static_cast<CUdeviceptr>(addr);
#endif
}
} // namespace

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
