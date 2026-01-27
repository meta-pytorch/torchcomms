// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CtranNcclTestUtils.h"

// Note that we have to rely on CudaWrap's internal symbol load to access cuda
// driver APIs since libcuda.so is not exposed in fbcode
#include "comms/ctran/utils/CudaWrap.h"

#include <folly/logging/xlog.h>
#include "comms/testinfra/TestsCuUtils.h"
#include "meta/wrapper/MetaFactory.h"

#include "nccl.h" // @manual

namespace ctran {

#if !defined(USE_ROCM)

ncclResult_t ncclMemFreeWithRefCheck(void* ptr) {
  CUmemGenericAllocationHandle refHandle = 0;
  CUCHECK_TEST(cuMemRetainAllocationHandle(&refHandle, ptr));
  ncclMemFree(ptr);
  CUCHECK_TEST(cuMemRelease(refHandle));

  // At this point, all internal handles should have been released,
  // and the only ones left should be the ones explicitly obtained by the
  // test. Additional release should result in an error.
  EXPECT_EQ(CUPFN_TEST(cuMemRelease)(refHandle), CUDA_ERROR_INVALID_VALUE);
  return ncclSuccess;
}

#endif // !defined(USE_ROCM)

// ============================================================================
// CtranNcclTestHelpers Implementation
// ============================================================================

/* static */
void* CtranNcclTestHelpers::prepareBuf(
    size_t bufSize,
    MemAllocType memType,
    std::vector<TestMemSegment>& segments) {
  void* buf = nullptr;
  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    segments.emplace_back(buf, bufSize);
  } else if (memType == kMemNcclMemAlloc) {
#if !defined(USE_ROCM)
    NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
    segments.emplace_back(buf, bufSize);
#else
    XLOG(FATAL) << "kMemNcclMemAlloc is not supported on AMD/HIP";
#endif
  } else {
#if !defined(USE_ROCM)
    std::vector<size_t> disjointSegmentSizes(2);
    disjointSegmentSizes[0] = bufSize / 2;
    disjointSegmentSizes[1] = bufSize / 2;
    COMMCHECK_TEST(commMemAllocDisjoint(&buf, disjointSegmentSizes, segments));
#else
    XLOG(FATAL) << "kCuMemAllocDisjoint is not supported on AMD/HIP";
#endif
  }
  return buf;
}

/* static */
void CtranNcclTestHelpers::releaseBuf(
    void* buf,
    size_t bufSize,
    MemAllocType memType) {
  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaFree(buf));
  } else if (memType == kMemNcclMemAlloc) {
#if !defined(USE_ROCM)
    NCCLCHECK_TEST(ncclMemFreeWithRefCheck(buf));
#else
    XLOG(FATAL) << "kMemNcclMemAlloc is not supported on AMD/HIP";
#endif
  } else {
#if !defined(USE_ROCM)
    std::vector<size_t> disjointSegmentSizes(2);
    disjointSegmentSizes[0] = bufSize / 2;
    disjointSegmentSizes[1] = bufSize / 2;
    COMMCHECK_TEST(commMemFreeDisjoint(buf, disjointSegmentSizes));
#else
    XLOG(FATAL) << "kCuMemAllocDisjoint is not supported on AMD/HIP";
#endif
  }
}

} // namespace ctran
