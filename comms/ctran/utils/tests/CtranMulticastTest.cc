// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CtranMulticast.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"

// Single-process unit tests for the NVL-multicast registration surface: the
// CtranMulticast support/granularity gates and its self-detecting import()
// (physical VMM segment enumeration + retention of its own handles), plus the
// CtranIpcMem cuMem-backed check + overlay accessors. The full multicast object
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

 protected:
  const char* desc_ = "CtranMulticastTest";
};

// The unicast (non-cuMem) case: a cudaMalloc-backed buffer has no physical VMM
// segments, so a multicast object cannot be bound over it. CtranIpcMem reports
// it is not cuMem-backed, import() fails, and the overlay accessors report "no
// overlay" (the caller falls back to unicast).
TEST_F(CtranMulticastTest, CudaMallocBufferCannotImport) {
  constexpr size_t size = 2 * 1024 * 1024;
  void* ptr = nullptr;
  CUDACHECK_TEST(cudaMalloc(&ptr, size));

  auto ipcMem = std::make_unique<CtranIpcMem>(0, desc_);
  bool supported = false;
  (void)ipcMem->tryLoad(ptr, size, supported, /*shouldSupportCudaMalloc=*/true);

  EXPECT_FALSE(ipcMem->isCuMemBacked())
      << "cudaMalloc buffer must not be cuMem-backed for multicast binding";
  EXPECT_FALSE(ipcMem->hasMulticast());
  EXPECT_EQ(ipcMem->getMulticastPtr(), nullptr);

  CtranMulticast mc(/*nvlLocalRank=*/0, /*nLocalRanks=*/2, /*cudaDev=*/0);
  EXPECT_NE(mc.import(ptr, size), commSuccess)
      << "import must fail on a non-cuMem (cudaMalloc) buffer";

  CUDACHECK_TEST(cudaFree(ptr));
}

// A cuMem (VMM) buffer can be import()ed into a standalone CtranMulticast,
// which enumerates and RETAINS ITS OWN handle for each physical segment
// (released in the dtor at scope exit). importedSize() must cover the whole
// allocation (what addDeviceAndBind would bind).
TEST_F(CtranMulticastTest, CuMemBufferImportCoversAllocation) {
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
    COMMCHECK_TEST(mc.import(ptr, size));
    EXPECT_GE(mc.importedSize(), size)
        << "imported segments must cover the whole allocation";
  } // mc dtor releases its own retained segment handles here

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}

// Multi-segment coverage (the rework's capability over the old single-segment
// limit): importing a cuMem buffer backed by multiple physical VMM segments
// enumerates + retains all of them, so importedSize() covers the whole
// allocation (addDeviceAndBind binds each at running offsets).
TEST_F(CtranMulticastTest, CuMemMultiSegmentImportCoversAllocation) {
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
    COMMCHECK_TEST(mc.import(ptr, total));
    EXPECT_GE(mc.importedSize(), total)
        << "all physical segments must be enumerated + retained";
  } // mc dtor releases its own retained segment handles here

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}

// import() is single-use: once an instance has loaded a buffer's segments, a
// second import() must be rejected (commInvalidUsage) rather than silently
// dropping the first buffer's retained segment handles and re-enumerating.
TEST_F(CtranMulticastTest, ImportIsSingleUse) {
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
    COMMCHECK_TEST(mc.import(ptr, size));
    EXPECT_NE(mc.import(ptr, size), commSuccess)
        << "a second import() must be rejected, not silently re-loaded";
  }

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}
