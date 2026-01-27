// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/MemUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/utils/checks.h"

namespace comms::utils::cumem {

namespace {

bool isBackedByCuMem(const void* ptr, const int devId) {
  if (ctran::utils::commCudaLibraryInit() != commSuccess) {
    return false;
  }

  DevMemType memType{DevMemType::kCudaMalloc};
  FB_COMMCHECKTHROW(getDevMemType(ptr, devId, memType));
  return memType == DevMemType::kCumem;
}

} // namespace

bool isBackedByMultipleCuMemAllocations(
    const void* ptr,
    const int devId,
    const size_t len) {
  if (!isBackedByCuMem(ptr, devId)) {
    return false;
  }

  size_t curRange = 0;
  CUdeviceptr curPbase = 0;
  CUdeviceptr ptr_ = reinterpret_cast<CUdeviceptr>(const_cast<void*>(ptr));

  FB_CUCHECKTHROW(cuMemGetAddressRange(&curPbase, &curRange, ptr_));
  const size_t remaining_in_first_alloc = (size_t)ctran::utils::subDevicePtr(
      ctran::utils::addDevicePtr(curPbase, curRange), (void*)ptr_);
  if (len <= remaining_in_first_alloc) {
    return false;
  }
  return true;
}

} // namespace comms::utils::cumem
