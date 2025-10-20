// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/memory/SlabAllocator.h"

#include "comms/ctran/memory/Utils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevUtils.cuh"

namespace ncclx::memory {

SlabAllocator::SlabAllocator() {
  if (!ctran::utils::getCuMemSysSupported()) {
    FB_ERRORTHROW(
        commInvalidUsage,
        "NCCLX slab allocator only works with low-level cuMem APIs. Make sure CUDA Toolkit is 11.3 or higher.");
  }
  FB_COMMCHECKTHROW(computeCumemGranualirity());
}

commResult_t SlabAllocator::cuCallocAsync(
    void** ptr,
    size_t numBytes,
    cudaStream_t stream,
    const char* callsite,
    const CommLogData* logMetaData) {
  // align allocation size to 16 bytes first, so we make sure startPtr_ is 16
  // bytes aligned
  size_t allocSize = ctran::utils::roundUp(numBytes, kMinAlignSize);
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  FB_COMMCHECK(
      allocateMem(ptr, allocSize, callsite, logMetaData, nullptr, nullptr));
  FB_CUDACHECK(cudaMemsetAsync(*ptr, 0, allocSize, stream));
  FB_CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return commSuccess;
}

commResult_t SlabAllocator::cuMalloc(
    void** ptr,
    size_t numBytes,
    const char* callsite,
    const CommLogData* logMetaData,
    CUmemGenericAllocationHandle* handlep,
    size_t* newSlabSize) {
  // align allocation size to 16 bytes first, so we make sure startPtr_ is 16
  // bytes aligned
  size_t allocSize = ctran::utils::roundUp(numBytes, kMinAlignSize);

  FB_COMMCHECK(
      allocateMem(ptr, allocSize, callsite, logMetaData, handlep, newSlabSize));
  return commSuccess;
}

commResult_t SlabAllocator::allocateMem(
    void** ptr,
    size_t numBytes,
    const char* callsite,
    const CommLogData* logMetaData,
    CUmemGenericAllocationHandle* handlep,
    size_t* newSlabSize) {
  if (freeSize_ < numBytes) {
    // No more free space on the last slab, allocate a new one
    // align size to CUMEM_GRANULARITY
    size_t slabSize = numBytes;
    slabSize = ctran::utils::roundUp(slabSize, granularity_);

    void* slabPtr = nullptr;
    FB_COMMCHECK(ctran::utils::commCuMemAlloc(
        &slabPtr,
        &slabHandle_,
        ctran::utils::getCuMemAllocHandleType(),
        slabSize,
        logMetaData,
        callsite));
    slabPtrs_.push_back(slabPtr);
    *ptr = slabPtr;
    freeSize_ = slabSize - numBytes;
    startPtr_ = (char*)slabPtr + numBytes;
    totalMemAllocated_ += slabSize;
    if (newSlabSize != nullptr) {
      *newSlabSize = slabSize;
    }
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "{}: allocate a slab with size {} (granularity_={}), freeSize_={}",
        __func__,
        slabSize,
        granularity_,
        freeSize_);
  } else {
    // still there is free space on the last slab, use it and reduce free space
    // count
    *ptr = startPtr_;
    startPtr_ = (char*)startPtr_ + numBytes;
    freeSize_ -= numBytes;
  }
  if (handlep) {
    *handlep = slabHandle_; // copy current slab handle by value
  }
  return commSuccess;
}

commResult_t SlabAllocator::computeCumemGranualirity() {
  CUdevice currentDev;
  int cudaDev;
  FB_CUDACHECK(cudaGetDevice(&cudaDev));
  FB_CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ctran::utils::setCuMemHandleTypeForProp(
      prop_, ctran::utils::getCuMemAllocHandleType());
  prop_.location.id = currentDev;
  // Query device to see if RDMA support is available
  if (ctran::utils::gpuDirectRdmaWithCudaVmmSupported(currentDev, cudaDev)) {
    prop_.allocFlags.gpuDirectRDMACapable = 1;
  }
  FB_CUCHECK(cuMemGetAllocationGranularity(
      &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return commSuccess;
}

SlabAllocator::~SlabAllocator() {
  for (auto slabPtr : slabPtrs_) {
    ctran::utils::commCuMemFree(slabPtr);
  }
}
} // namespace ncclx::memory
