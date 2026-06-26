// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstddef>

#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/platform/CudaDriverLazy.h"

namespace comms::prims::tests {
namespace {

// Resolves a CUdevice on device 0 and returns false (skip) if VMM / CUDA 12.3
// is unavailable.
bool vmmDevice(CUdevice& cuDev) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  if (cuda_driver_lazy_init() != 0) {
    return false;
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    return false;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    return false;
  }
  return pfn_cuDeviceGet(&cuDev, 0) == CUDA_SUCCESS;
#endif
}

unsigned int requestMask() {
  return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC;
}

bool isPowerOfTwo(std::size_t x) {
  return x != 0 && (x & (x - 1)) == 0;
}

} // namespace

TEST(CuMemAllocationTest, CreateProducesUsableHandle) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }

  auto alloc = CuMemAllocation::create(cuDev, /*size=*/4096, requestMask());
  EXPECT_NE(alloc->handle(), 0u);
  EXPECT_GE(alloc->size(), std::size_t{4096});
  EXPECT_TRUE(isPowerOfTwo(alloc->granularity()));
  EXPECT_EQ(alloc->device(), cuDev);
  // The surviving cuMemCreate always keeps POSIX FD; fabric is kept only when
  // the device/driver permit it.
  EXPECT_NE(
      alloc->supportedHandleTypes() &
          static_cast<unsigned int>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
      0u);
}

TEST(CuMemAllocationTest, RoundsSizeUpToAlignFloor) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }

  // A 2 MiB power-of-two alignFloor must round the allocation up to a multiple
  // of it (the granularity becomes max(driverGranularity, alignFloor)).
  constexpr std::size_t kAlignFloor = 2u * 1024u * 1024u;
  auto alloc =
      CuMemAllocation::create(cuDev, /*size=*/4096, requestMask(), kAlignFloor);
  EXPECT_GE(alloc->granularity(), kAlignFloor);
  EXPECT_EQ(alloc->size() % alloc->granularity(), 0u);
  EXPECT_GE(alloc->size(), kAlignFloor);
}

TEST(CuMemAllocationTest, RejectsNonPowerOfTwoAlignFloor) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  EXPECT_THROW(
      CuMemAllocation::create(cuDev, 4096, requestMask(), /*alignFloor=*/3),
      std::runtime_error);
}

TEST(CuMemAllocationTest, MoveLeavesSourceInert) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  // Move-construct out of the owned object (not the unique_ptr) to exercise
  // CuMemAllocation's own move ctor + moved-from inert state.
  auto a = CuMemAllocation::create(cuDev, 4096, requestMask());
  const auto handle = a->handle();
  CuMemAllocation b = std::move(*a);
  EXPECT_EQ(b.handle(), handle);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentionally checking moved-from
  EXPECT_EQ(a->handle(), 0u);
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
