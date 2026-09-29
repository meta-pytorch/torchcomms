// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/CuMemMapping.h"
#include "comms/prims/platform/CudaDriverLazy.h"

namespace comms::prims::tests {
namespace {

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

std::shared_ptr<CuMemAllocation> makeAlloc(CUdevice cuDev, std::size_t size) {
  return CuMemAllocation::create(
      cuDev,
      size,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
}

void* asPtr(CUdeviceptr p) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
  return reinterpret_cast<void*>(p);
}

} // namespace

TEST(CuMemMappingTest, MapsAndAccessesAllocation) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  auto alloc = makeAlloc(cuDev, 4096);
  auto mapping =
      CuMemMapping::overAllocation(alloc, alloc->size(), alloc->granularity());
  ASSERT_NE(mapping.devicePtr(), 0u);

  ASSERT_EQ(
      cudaMemset(asPtr(mapping.devicePtr()), 0xAB, alloc->size()), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> host(alloc->size(), 0);
  ASSERT_EQ(
      cudaMemcpy(
          host.data(),
          asPtr(mapping.devicePtr()),
          alloc->size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (auto b : host) {
    ASSERT_EQ(b, 0xAB);
  }
}

TEST(CuMemMappingTest, TwoMappingsShareSamePhysical) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  auto alloc = makeAlloc(cuDev, 4096);
  auto m1 =
      CuMemMapping::overAllocation(alloc, alloc->size(), alloc->granularity());
  auto m2 =
      CuMemMapping::overAllocation(alloc, alloc->size(), alloc->granularity());
  // Distinct virtual addresses onto the same physical allocation.
  EXPECT_NE(m1.devicePtr(), m2.devicePtr());

  ASSERT_EQ(
      cudaMemset(asPtr(m1.devicePtr()), 0x5C, alloc->size()), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> host(alloc->size(), 0);
  ASSERT_EQ(
      cudaMemcpy(
          host.data(),
          asPtr(m2.devicePtr()),
          alloc->size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (auto b : host) {
    ASSERT_EQ(b, 0x5C) << "write via mapping 1 must be visible via mapping 2";
  }
}

TEST(CuMemMappingTest, KeepAliveOutlivesAllocationOwner) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  auto alloc = makeAlloc(cuDev, 4096);
  std::weak_ptr<CuMemAllocation> weak = alloc;
  auto mapping = std::make_unique<CuMemMapping>(
      CuMemMapping::overAllocation(alloc, alloc->size(), alloc->granularity()));
  const std::size_t size = alloc->size();

  // Drop the caller's reference; the mapping's keepAlive must keep the physical
  // allocation alive so the VA stays valid.
  alloc.reset();
  EXPECT_FALSE(weak.expired());
  ASSERT_EQ(cudaMemset(asPtr(mapping->devicePtr()), 0, size), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Releasing the mapping drops the last reference to the allocation.
  mapping.reset();
  EXPECT_TRUE(weak.expired());
}

TEST(CuMemMappingTest, MoveLeavesSourceInert) {
  CUdevice cuDev = 0;
  if (!vmmDevice(cuDev)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  auto alloc = makeAlloc(cuDev, 4096);
  auto m1 =
      CuMemMapping::overAllocation(alloc, alloc->size(), alloc->granularity());
  const auto ptr = m1.devicePtr();
  CuMemMapping m2 = std::move(m1);
  EXPECT_EQ(m2.devicePtr(), ptr);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentionally checking moved-from
  EXPECT_EQ(m1.devicePtr(), 0u);
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
