// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/commSpecs.h"

using namespace ctran::utils;

class IpcUT : public ::testing::Test {
 public:
  void SetUp() override {
    ncclCvarInit();
    COMMCHECK_TEST(commCudaLibraryInit());
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {}

 protected:
  const char* dummyDesc_ = "IpcUT";
};

// Test Case 1: tryLoad should reject memory allocated with
// CU_MEM_HANDLE_TYPE_NONE Currently this test is expected to FAIL because
// tryLoad does not validate this
TEST_F(IpcUT, TryLoadCuMemRejectsNoneHandleType) {
  if (!ctran::utils::getCuMemSysSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping test";
  }

  constexpr size_t size = 2 * 1024 * 1024; // 2MB (minimum granularity)
  std::vector<size_t> segmentSizes = {size};
  std::vector<TestMemSegment> segments;
  void* ptr = nullptr;

  auto result = ctran::commMemAllocDisjoint(
      &ptr, segmentSizes, segments, true, CU_MEM_HANDLE_TYPE_NONE);
  if (result != commSuccess || ptr == nullptr) {
    GTEST_SKIP() << "Failed to allocate cuMem with NONE handle type";
  }

  // Create CtranIpcMem in LOAD mode
  auto ipcMem = std::make_unique<CtranIpcMem>(0, dummyDesc_);

  // tryLoad should fail for memory with no shareable handle type
  bool supported = false;
  (void)ipcMem->tryLoad(ptr, size, supported, false);

  // Expected: tryLoad should return error or set supported=false
  // because memory with NONE handle type cannot be exported
  EXPECT_FALSE(supported)
      << "tryLoad should reject memory with CU_MEM_HANDLE_TYPE_NONE";

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
}

// Test Case 2: tryLoad should reject FABRIC-only memory when FABRIC is disabled
// Currently this test is expected to FAIL because tryLoad does not validate
// this
TEST_F(IpcUT, TryLoadCuMemRejectsFabricOnlyWhenFabricDisabled) {
#if CUDART_VERSION < 12040
  GTEST_SKIP() << "CUDA < 12.04, FABRIC handle type not available";
#else
  if (!ctran::utils::getCuMemSysSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping test";
  }

  // Skip if system doesn't support FABRIC - allocation would silently downgrade
  // to NONE (see D90902516 for details on CUDA's silent downgrade behavior)
  {
    EnvRAII fabricDisabled(NCCL_CTRAN_NVL_FABRIC_ENABLE, true);
    CUmemAllocationHandleType systemSupported =
        ctran::utils::getCuMemAllocHandleType();
    if ((systemSupported & CU_MEM_HANDLE_TYPE_FABRIC) == 0) {
      GTEST_SKIP()
          << "System does not support FABRIC handle type, skipping test";
    }
  }

  // Disable FABRIC support in ctran
  EnvRAII fabricDisabled(NCCL_CTRAN_NVL_FABRIC_ENABLE, false);

  constexpr size_t size = 2 * 1024 * 1024; // 2MB
  std::vector<size_t> segmentSizes = {size};
  std::vector<TestMemSegment> segments;
  void* ptr = nullptr;

  // Allocate with FABRIC-only handle type
  auto result = ctran::commMemAllocDisjoint(
      &ptr, segmentSizes, segments, true, CU_MEM_HANDLE_TYPE_FABRIC);
  if (result != commSuccess || ptr == nullptr) {
    GTEST_SKIP() << "Failed to allocate cuMem with FABRIC-only handle type";
  }

  // Create CtranIpcMem in LOAD mode
  auto ipcMem = std::make_unique<CtranIpcMem>(0, dummyDesc_);

  // tryLoad should fail because ctran would try to export as POSIX
  // but the allocation only supports FABRIC
  bool supported = false;
  (void)ipcMem->tryLoad(ptr, size, supported, false);

  // Expected: tryLoad should return error or set supported=false
  // because FABRIC-only memory cannot be exported as POSIX
  EXPECT_FALSE(supported)
      << "tryLoad should reject FABRIC-only memory when FABRIC export is disabled";

  ctran::commMemFreeDisjoint(ptr, segmentSizes);
#endif
}
