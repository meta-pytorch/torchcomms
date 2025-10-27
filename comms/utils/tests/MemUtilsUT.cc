// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/MemUtils.h"
#include "comms/utils/commSpecs.h"

TEST(MemUtilsTest, SingleAllocationCudaMalloc) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "CUDA not available";
  }
  const int devId = 0;
  CUDACHECK_TEST(cudaSetDevice(devId));

  const size_t bufferSize = 8192;
  void* buffer = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buffer, bufferSize));

  EXPECT_FALSE(
      comms::utils::cumem::isBackedByMultipleCuMemAllocations(
          buffer, devId, bufferSize))
      << "Regular cudaMalloc should not span multiple allocations";

  CUDACHECK_TEST(cudaFree(buffer));
}

TEST(MemUtilsTest, MultipleSegmentAllocation) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "CUDA not available";
  }
  const int devId = 0;
  CUDACHECK_TEST(cudaSetDevice(devId));

  if (!ctran::utils::getCuMemSysSupported()) {
    GTEST_SKIP() << "CuMem not supported";
  }

  void* bufBase = nullptr;
  std::vector<TestMemSegment> segments;
  const size_t segmentSize = 2097152; // 2MB per segment
  std::vector<size_t> segSizes = {segmentSize, segmentSize};

  auto result = commMemAllocDisjoint(&bufBase, segSizes, segments);
  if (result != commSuccess) {
    GTEST_SKIP() << "Disjoint allocation failed";
  }

  const size_t totalSize = segmentSize * 2;
  EXPECT_TRUE(
      comms::utils::cumem::isBackedByMultipleCuMemAllocations(
          bufBase, devId, totalSize))
      << "Buffer spanning multiple segments should be detected as multiple allocations";

  // Clean up
  EXPECT_EQ(commMemFreeDisjoint(bufBase, segSizes), commSuccess);
}

TEST(MemUtilsTest, SingleAllocationManaged) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "CUDA not available";
  }
  const int devId = 0;
  CUDACHECK_TEST(cudaSetDevice(devId));

  const size_t bufferSize = 8192;
  void* buffer = nullptr;
  CUDACHECK_TEST(cudaMallocManaged(&buffer, bufferSize));

  EXPECT_FALSE(
      comms::utils::cumem::isBackedByMultipleCuMemAllocations(
          buffer, devId, bufferSize))
      << "Managed memory allocation should not span multiple allocations";

  CUDACHECK_TEST(cudaFree(buffer));
}

TEST(MemUtilsTest, SingleAllocationHostPinned) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "CUDA not available";
  }
  const int devId = 0;
  CUDACHECK_TEST(cudaSetDevice(devId));

  const size_t bufferSize = 8192;
  void* buffer = nullptr;
  CUDACHECK_TEST(cudaMallocHost(&buffer, bufferSize));

  EXPECT_FALSE(
      comms::utils::cumem::isBackedByMultipleCuMemAllocations(
          buffer, devId, bufferSize))
      << "Host pinned memory allocation should not span multiple allocations";

  CUDACHECK_TEST(cudaFreeHost(buffer));
}

TEST(MemUtilsTest, SingleAllocationHostUnregistered) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "CUDA not available";
  }
  const int devId = 0;
  CUDACHECK_TEST(cudaSetDevice(devId));

  const size_t bufferSize = 8192;
  void* buffer = malloc(bufferSize);
  ASSERT_NE(buffer, nullptr) << "Failed to allocate host memory";

  EXPECT_FALSE(
      comms::utils::cumem::isBackedByMultipleCuMemAllocations(
          buffer, devId, bufferSize))
      << "Unregistered host memory allocation should not span multiple allocations";

  free(buffer);
}
