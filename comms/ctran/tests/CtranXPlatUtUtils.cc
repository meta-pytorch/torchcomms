// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include <folly/logging/xlog.h>
#include "common/fbwhoami/FbWhoAmI.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/ctran/utils/Utils.h"
#include "comms/utils/logger/LoggingFormat.h"

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

void logGpuMemoryStats(int gpu) {
  size_t free, total;
  CUDACHECK_TEST(cudaMemGetInfo(&free, &total));
  LOG(INFO) << "GPU " << gpu << " memory: free=" << free << ", total=" << total
            << std::endl;
}

void commSetMyThreadLoggingName(std::string_view name) {
  meta::comms::logger::initThreadMetaData(name);
}

commResult_t commMemAllocDisjoint(
    void** ptr,
    std::vector<size_t>& disjointSegmentSizes,
    std::vector<TestMemSegment>& segments,
    bool setRdmaSupport,
    std::optional<CUmemAllocationHandleType> handleType) {
  commResult_t ret = commSuccess;

  size_t numSegments = disjointSegmentSizes.size();
  size_t size = 0;
  for (int i = 0; i < numSegments; ++i) {
    size += disjointSegmentSizes[i];
  }
  size_t vaSize = 0;
  size_t memGran = 0;
  CUdeviceptr curPtr;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemAccessDesc accessDesc = {};
  std::vector<CUmemGenericAllocationHandle> handles(numSegments);
  std::vector<CUmemGenericAllocationHandle> unusedHandles(numSegments);
  int cudaDev;

  if (ptr == NULL || size == 0) {
    return ErrorStackTraceUtil::log(commInvalidArgument);
  }

  if (ctran::utils::commCudaLibraryInit() != commSuccess) {
    return ErrorStackTraceUtil::log(commSystemError);
  }

  // Still allow cumem based allocation if cumem is supported.
  if (!ctran::utils::getCuMemSysSupported()) {
    return ErrorStackTraceUtil::log(commSystemError);
  }
  CUDACHECK_TEST(cudaGetDevice(&cudaDev));
  FB_CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  if (handleType) {
    ctran::utils::setCuMemHandleTypeForProp(memprop, handleType.value());
  }
  memprop.location.id = currentDev;
  if (setRdmaSupport) {
    // Query device to see if RDMA support is available
    if (ctran::utils::gpuDirectRdmaWithCudaVmmSupported(currentDev, cudaDev)) {
      memprop.allocFlags.gpuDirectRDMACapable = 1;
    }
  }
  FB_CUCHECK(cuMemGetAllocationGranularity(
      &memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  vaSize = 0;
  std::vector<size_t> alignedSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    alignedSizes[i] = disjointSegmentSizes[i];
    ALIGN_SIZE(alignedSizes[i], memGran);
    vaSize += alignedSizes[i];
  }

  for (int i = 0; i < numSegments; i++) {
    /* Allocate the physical memory on the device */
    FB_CUCHECK(cuMemCreate(&handles[i], alignedSizes[i], &memprop, 0));
    FB_CUCHECK(cuMemCreate(&unusedHandles[i], alignedSizes[i], &memprop, 0));
  }
  // Free unused handles
  for (int i = 0; i < unusedHandles.size(); i++) {
    FB_CUCHECK(cuMemRelease(unusedHandles[i]));
  }
  /* Reserve a virtual address range */
  FB_CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, vaSize, memGran, 0, 0));
  /* Map the virtual address range to the physical allocation */
  curPtr = (CUdeviceptr)*ptr;
  for (int i = 0; i < numSegments; i++) {
    FB_CUCHECK(cuMemMap(curPtr, alignedSizes[i], 0, handles[i], 0));
    segments.emplace_back(reinterpret_cast<void*>(curPtr), alignedSizes[i]);
    LOG(INFO) << "ncclMemAllocDisjoint maps segments[" << i << "] ptr "
              << reinterpret_cast<void*>(curPtr) << " size " << alignedSizes[i]
              << "/" << vaSize;

    curPtr = ctran::utils::addDevicePtr(curPtr, alignedSizes[i]);
  }
  // Now allow RW access to the newly mapped memory.
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FB_CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, vaSize, &accessDesc, 1));

  return ret;
}

commResult_t commMemFreeDisjoint(
    void* ptr,
    std::vector<size_t>& disjointSegmentSizes) {
  commResult_t ret = commSuccess;
  int saveDevice;
  CUmemGenericAllocationHandle handle;

  CUDACHECK_TEST(cudaGetDevice(&saveDevice));
  CUdevice ptrDev = 0;

  if (ptr == NULL) {
    cudaSetDevice(saveDevice);
    return ErrorStackTraceUtil::log(commInvalidArgument);
  }

  if (ctran::utils::commCudaLibraryInit() != commSuccess) {
    cudaSetDevice(saveDevice);
    return ErrorStackTraceUtil::log(commSystemError);
  }

  if (!ctran::utils::getCuMemSysSupported()) {
    cudaSetDevice(saveDevice);
    return ErrorStackTraceUtil::log(commSystemError);
  }

  FB_CUCHECK(cuPointerGetAttribute(
      (void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr));
  CUDACHECK_TEST(cudaSetDevice((int)ptrDev));

  size_t memGran = 0;
  CUmemAllocationProp memprop = {};
  FB_CUCHECK(cuMemGetAllocationGranularity(
      &memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t vaSize = 0;
  size_t numSegments = disjointSegmentSizes.size();
  std::vector<size_t> alignedSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    alignedSizes[i] = disjointSegmentSizes[i];
    ALIGN_SIZE(alignedSizes[i], memGran);
    vaSize += alignedSizes[i];
  }

  CUdeviceptr curPtr = (CUdeviceptr)ptr;
  for (int i = 0; i < alignedSizes.size(); i++) {
    FB_CUCHECK(cuMemRetainAllocationHandle(&handle, (void*)curPtr));
    LOG(INFO) << "ncclMemFreeDisjoint unmaps segments[" << i << "] ptr "
              << reinterpret_cast<void*>(curPtr) << " size " << alignedSizes[i]
              << "/" << vaSize;
    FB_CUCHECK(cuMemRelease(handle));
    FB_CUCHECK(cuMemUnmap(curPtr, alignedSizes[i]));
    // call to cuMemRetainAllocationHandle increments reference count, requires
    // double cuMemRelease
    FB_CUCHECK(cuMemRelease(handle));
    curPtr = ctran::utils::addDevicePtr(curPtr, alignedSizes[i]);
  }
  FB_CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, vaSize));
  cudaSetDevice(saveDevice);
  return ret;
}

namespace {
size_t getSegmentSize(const size_t bufSize, const size_t numSegments) {
  // commMemAllocDisjoint internally would align the size to 2MB per segment
  // (queried from cuMemGetAllocationGranularity)
  return ctran::utils::align(bufSize, numSegments) / numSegments;
}
} // namespace

// Wrapper for memory allocation in tests with different memory types
// - bufSize: size of the buffer to allocate
// - memType: memory type to allocate
// - segments: vector of underlying allocated segments. It can be two segments
//             with kCuMemAllocDisjoint type, which map to a single virtual
//             memory range. For other mem types, it should be 1 segment.
// - return: pointer to the allocated virtual memory range.
void* commMemAlloc(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments) {
  void* buf = nullptr;
  switch (memType) {
    case kMemCudaMalloc:
      CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
      segments.emplace_back(buf, bufSize);
      break;
    case kCuMemAllocDisjoint: {
      // Allocate disjoint segments mapping to a single virtual memory range;
      // it mimics the behavior of Pytorch CCA expandable segment mode where a
      // single tensor may be mapped by two disjoint segments.
      const auto segSize = getSegmentSize(bufSize, 2);
      std::vector<size_t> disjointSegSizes(2, segSize);
      COMMCHECK_TEST(commMemAllocDisjoint(&buf, disjointSegSizes, segments));
      break;
    }
    case kMemHostManaged:
      CUDACHECK_TEST(cudaMallocHost(&buf, bufSize));
      segments.emplace_back(buf, bufSize);
      break;
    default:
      XLOG(FATAL) << "Unsupported memType: " << memType;
      break;
  }
  return buf;
}

void commMemFree(void* buf, size_t bufSize, MemAllocType memType) {
  switch (memType) {
    case kMemCudaMalloc:
      CUDACHECK_TEST(cudaFree(buf));
      break;
    case kCuMemAllocDisjoint: {
      const auto segSize = getSegmentSize(bufSize, 2);
      std::vector<size_t> disjointSegSizes(2, segSize);
      commMemFreeDisjoint(buf, disjointSegSizes);
      break;
    }
    case kMemHostManaged:
      cudaFreeHost(buf);
      break;
    default:
      XLOG(FATAL) << "Unsupported memType: " << memType;
      break;
  }
}

TestCtranCommRAII::TestCtranCommRAII(std::unique_ptr<mccl::McclComm> mcclComm)
    : mcclComm_(std::move(mcclComm)) {
  ctranComm = mcclComm_->comm_.get();
}

std::unique_ptr<TestCtranCommRAII> createDummyCtranComm() {
  CHECK_EQ(ctran::utils::commCudaLibraryInit(), commSuccess);
  mccl::McclCommCreateOpts mcclCreateOpts{
      .rank = 0,
      .cudaDeviceId = 0,
      .enableFaultTolerance = false,
  };

  auto mcclComm = std::make_unique<mccl::McclComm>(mcclCreateOpts);
  auto initURL = mcclComm->getInitURL();
  std::string uuid{"0"};
  auto initWorkHandle = mcclComm->init(
      mccl::InitOpts{
          .uuid = uuid,
          .urls = {initURL},
      });
  initWorkHandle->waitCpu();
  auto initResult = initWorkHandle->getResult();
  CHECK_EQ(initResult->code, commSuccess)
      << "init failed with error: " << initResult->message;

  return std::make_unique<TestCtranCommRAII>(std::move(mcclComm));
}
