// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CtranMulticast.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"

// Single-process unit tests for the NVL-multicast registration surface: the
// CtranMulticast support/granularity gates and its self-detecting
// retainSegments() (physical VMM segment enumeration + retention of its own
// handles, declining cleanly on non-cuMem memory). The full multicast object
// data-path (create/bind/map + the fabric handle exchange + NVSwitch fan-out)
// needs a real >=2-GPU NVL group and is covered by the distributed test
// CtranMulticastDistTest.cc (1 rank/GPU).

using namespace ctran::utils;

class CtranMulticastTest : public ::testing::Test {
 public:
  void SetUp() override {
    ncclCvarInit();
    COMMCHECK_TEST(commCudaLibraryInit());
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {}
};

// The unicast (non-cuMem) case: a cudaMalloc-backed buffer has no physical VMM
// segments, so retainSegments() detects the memory type and fails cleanly (the
// caller falls back to unicast) -- no CtranIpc registration required.
TEST_F(CtranMulticastTest, CudaMallocBufferCannotRetain) {
  constexpr size_t size = 2 * 1024 * 1024;
  void* ptr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&ptr, size));

  CtranMulticast mc(/*nvlLocalRank=*/0, /*nLocalRanks=*/2, /*cudaDev=*/0);
  // Reject cleanly with a descriptive reason -- the specific commInvalidUsage
  // from the up-front memory-type check plus a message naming the actual memory
  // type -- NOT an opaque CUDA error bubbled up from
  // cuMemRetainAllocationHandle deep inside enumerateCuMemSegments.
  testing::internal::CaptureStderr();
  const commResult_t rc = mc.retainSegments(ptr, size);
  const std::string logs = testing::internal::GetCapturedStderr();

  EXPECT_EQ(rc, commInvalidUsage)
      << "retainSegments must reject a non-cuMem (cudaMalloc) buffer cleanly";
  EXPECT_NE(logs.find("multicast requires a cuMem"), std::string::npos)
      << "expected a descriptive 'requires cuMem' reason; got:\n"
      << logs;
  EXPECT_EQ(logs.find("cuMemRetainAllocationHandle"), std::string::npos)
      << "must not leak the opaque cuMemRetainAllocationHandle CUDA error:\n"
      << logs;

  CUDACHECK_TEST(cudaFree(ptr));
}

// A cuMem (VMM) buffer's segments can be retained into a standalone
// CtranMulticast, which enumerates and RETAINS ITS OWN handle for each physical
// segment (released in the dtor at scope exit). retainedSize() must cover the
// whole allocation (what addDeviceAndBind would bind).
TEST_F(CtranMulticastTest, CuMemBufferRetainCoversAllocation) {
  if (!getCuMemSysSupported()) {
    GTEST_SKIP() << "cuMem (VMM) not supported on this device";
  }
  const CUmemAllocationHandleType htype = getCuMemAllocHandleType();
  if (htype == CU_MEM_HANDLE_TYPE_NONE) {
    GTEST_SKIP() << "no shareable cuMem handle type on this system";
  }

  constexpr size_t size = 2 * 1024 * 1024;
  std::vector<size_t> segmentSizes = {size};
  std::vector<TestMemSegment> segments;
  void* ptr = nullptr;
  auto allocRes =
      ctran::commMemAllocDisjoint(&ptr, segmentSizes, segments, true, htype);
  if (allocRes != commSuccess || ptr == nullptr) {
    GTEST_SKIP() << "failed to allocate cuMem buffer";
  }

  {
    CtranMulticast mc(/*nvlLocalRank=*/0, /*nLocalRanks=*/2, /*cudaDev=*/0);
    COMMCHECK_TEST(mc.retainSegments(ptr, size));
    EXPECT_GE(mc.retainedSize(), size)
        << "retained segments must cover the whole allocation";
  } // mc dtor releases its own retained segment handles here

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}

// Multi-segment coverage (the rework's capability over the old single-segment
// limit): retaining a cuMem buffer backed by multiple physical VMM segments
// enumerates + retains all of them, so retainedSize() covers the whole
// allocation (addDeviceAndBind binds each at running offsets).
TEST_F(CtranMulticastTest, CuMemMultiSegmentRetainCoversAllocation) {
  if (!getCuMemSysSupported()) {
    GTEST_SKIP() << "cuMem (VMM) not supported on this device";
  }
  const CUmemAllocationHandleType htype = getCuMemAllocHandleType();
  if (htype == CU_MEM_HANDLE_TYPE_NONE) {
    GTEST_SKIP() << "no shareable cuMem handle type on this system";
  }

  // Two disjoint 2MB (min cuMem granularity) segments.
  constexpr size_t kSegSize = 2 * 1024 * 1024;
  std::vector<size_t> segmentSizes = {kSegSize, kSegSize};
  const size_t total = kSegSize * segmentSizes.size();
  std::vector<TestMemSegment> segments;
  void* ptr = nullptr;
  auto allocRes =
      ctran::commMemAllocDisjoint(&ptr, segmentSizes, segments, true, htype);
  if (allocRes != commSuccess || ptr == nullptr) {
    GTEST_SKIP() << "failed to allocate multi-segment cuMem buffer";
  }

  {
    CtranMulticast mc(/*nvlLocalRank=*/0, /*nLocalRanks=*/2, /*cudaDev=*/0);
    COMMCHECK_TEST(mc.retainSegments(ptr, total));
    EXPECT_GE(mc.retainedSize(), total)
        << "all physical segments must be enumerated + retained";
  } // mc dtor releases its own retained segment handles here

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}

// retainSegments() is single-use: once an instance has loaded a buffer's
// segments, a second call must be rejected (commInvalidUsage) rather than
// silently dropping the first buffer's retained segment handles and
// re-enumerating.
TEST_F(CtranMulticastTest, RetainSegmentsIsSingleUse) {
  if (!getCuMemSysSupported()) {
    GTEST_SKIP() << "cuMem (VMM) not supported on this device";
  }
  const CUmemAllocationHandleType htype = getCuMemAllocHandleType();
  if (htype == CU_MEM_HANDLE_TYPE_NONE) {
    GTEST_SKIP() << "no shareable cuMem handle type on this system";
  }

  constexpr size_t size = 2 * 1024 * 1024;
  std::vector<size_t> segmentSizes = {size};
  std::vector<TestMemSegment> segments;
  void* ptr = nullptr;
  auto allocRes =
      ctran::commMemAllocDisjoint(&ptr, segmentSizes, segments, true, htype);
  if (allocRes != commSuccess || ptr == nullptr) {
    GTEST_SKIP() << "failed to allocate cuMem buffer";
  }

  {
    CtranMulticast mc(/*nvlLocalRank=*/0, /*nLocalRanks=*/2, /*cudaDev=*/0);
    COMMCHECK_TEST(mc.retainSegments(ptr, size));
    EXPECT_NE(mc.retainSegments(ptr, size), commSuccess)
        << "a second retainSegments() must be rejected, not silently re-loaded";
  }

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}
