// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevUtils.cuh"

namespace {
#if CUDART_VERSION >= 12040
constexpr size_t kCtranAllocMinSize = 2097152UL;
#endif
CUmemAllocationHandleType cuMemAllocHandleType =
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
std::once_flag initCuMemAllocHandleTypeFlag;

void initCuMemAllocHandleTypeOnce() {
#if CUDART_VERSION < 12040
  return;
#else
  if (ctran::utils::isCuMemFabricHandleSupported()) {
    cuMemAllocHandleType =
        (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR |
                                    CU_MEM_HANDLE_TYPE_FABRIC);
  }
#endif
}
} // namespace

namespace ctran::utils {

commResult_t commCuMemAlloc(
    void** ptr,
    CUmemGenericAllocationHandle* handlep,
    CUmemAllocationHandleType type,
    size_t size,
    const CommLogData* logMetaData,
    const char* callsite) {
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;

  FB_CUDACHECK(cudaGetDevice(&cudaDev));
  FB_CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ctran::utils::setCuMemHandleTypeForProp(prop, type);
  prop.location.id = currentDev;

  // Query device to see if RDMA support is available
  if (ctran::utils::gpuDirectRdmaWithCudaVmmSupported(currentDev, cudaDev)) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  FB_CUCHECK(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size = ctran::utils::roundUp(size, granularity);
  FB_CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  FB_CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  FB_CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FB_CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
  if (handlep) {
    *handlep = handle;
  }
  CLOGF_TRACE(
      ALLOC,
      "CuMem Alloc Size {} pointer {} handle {}",
      size,
      *ptr,
      ctran::utils::toFormattableHandle(handle));
  logMemoryEvent(
      logMetaData ? *logMetaData : CommLogData{},
      callsite,
      "commCuMemAlloc",
      reinterpret_cast<uintptr_t>(*ptr),
      size);
  return commSuccess;
}

commResult_t commCuMemFree(void* ptr, const CommLogData* logMetaData) {
  if (ptr == nullptr) {
    return commSuccess;
  }
  CUmemGenericAllocationHandle handle;
  CUdeviceptr basePtr;
  size_t size = 0;
  FB_CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  // Virtual Memory Management introduces handlers to manage memory at a more
  // granular level, enabling high-performance operations. Internally, it
  // maintains a counter for each memory handler. This counter increases each
  // time a handler is created/retained. To free memory, the following
  // conditions must be met:
  // 1) The handler's reference count must be reduced to zero.
  // 2) The physical memory must be unmapped.
  // Only when both conditions are satisfied the memory guaranteed to be
  // freed. In our case, we increased the handler's counter when we allocated
  // memory and when we retained it in the code line above. Therefore, we need
  // to release it twice. Note that the memory is not yet released here, as
  // there is still a physical address mapped.
  FB_CUCHECK(cuMemRelease(handle));
  FB_CUCHECK(cuMemRelease(handle));
  FB_CUCHECK(cuMemGetAddressRange(&basePtr, &size, (CUdeviceptr)ptr));
  // At this moment, the counter for the memory handler is zero, so when we
  // unmap the physical memory, it is freed. Refer to the descriptions of
  // `cuMemUnmap` and `cuMemRelease` in the CUDA programming guide for more
  // details.
  FB_CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  FB_CUCHECK(cuMemAddressFree(basePtr, size));
  CLOGF_TRACE(
      ALLOC,
      "CuMem Free Size {} pointer {} handle {}",
      size,
      ptr,
      ctran::utils::toFormattableHandle(handle));
  logMemoryEvent(
      logMetaData ? *logMetaData : CommLogData{},
      "",
      "commCuMemFree",
      reinterpret_cast<uintptr_t>(ptr));
  return commSuccess;
}

bool isCuMemFabricHandleSupported() {
#if CUDART_VERSION < 12040
  return false;
#else
  // 1: For safty, a manual ENV to gate all NVL FABRIC features
  if (!NCCL_CTRAN_NVL_FABRIC_ENABLE) {
    return false;
  }

  // 2: checking cumem support
  if (!isCuMemSupported()) {
    return false;
  }

  // 3: checking cuDeviceGetAttribute
  CUdevice currentDev;
  int cudaDev;
  int flag = 0;
  FB_CUDACHECK_RETURN(cudaGetDevice(&cudaDev), false);
  FB_CUCHECK_RETURN(cuDeviceGet(&currentDev, cudaDev), false);
  // Query device to see if CUMEM FABRIC support is available
  // Ignore error if such attribute is not supported on older drivers or GPU
  // architectures, where it returns invalid argument error but harmless
  CUresult err = FB_CUPFN(cuDeviceGetAttribute(
      &flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
  if (err != CUDA_SUCCESS || !flag) {
    return false;
  }

  // 4: checking if fabric handle type of memory can be allocated
  void* addr = nullptr;
  CUmemGenericAllocationHandle allocHandle;

  commResult_t result = commCuMemAlloc(
      &addr,
      &allocHandle,
      CU_MEM_HANDLE_TYPE_FABRIC,
      kCtranAllocMinSize,
      nullptr /* CommLogData */,
      "isCuMemFabricHandleSupported");
  if (result != commSuccess) {
    if (addr) {
      commCuMemFree(addr, nullptr /* CommLogData */);
    }
    return false;
  }

  // 5: checking if import/export fabric handle type is supported
  CUmemFabricHandle sharedHandle;
  if (FB_CUPFN(cuMemExportToShareableHandle(
          &sharedHandle, allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0)) !=
      CUDA_SUCCESS) {
    commCuMemFree(addr, nullptr /* CommLogData */);
    return false;
  }

  if (FB_CUPFN(cuMemImportFromShareableHandle(
          &allocHandle, &sharedHandle, CU_MEM_HANDLE_TYPE_FABRIC)) !=
      CUDA_SUCCESS) {
    commCuMemFree(addr, nullptr /* CommLogData */);
    return false;
  } else {
    // cuMemImportFromShareableHandle increases the reference count of the
    // memory handler, we need to release it here in addition to commCuMemFree.
    FB_CUCHECK(cuMemRelease(allocHandle));
  }
  // matching free call to commCuMemAlloc if all the above checks succeeded
  commCuMemFree(addr, nullptr /* CommLogData */);
  return true;
#endif
}

CUmemAllocationHandleType getCuMemAllocHandleType() {
  std::call_once(initCuMemAllocHandleTypeFlag, initCuMemAllocHandleTypeOnce);
  return cuMemAllocHandleType;
}

} // namespace ctran::utils
