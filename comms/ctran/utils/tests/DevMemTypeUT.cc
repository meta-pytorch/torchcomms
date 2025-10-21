// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <folly/logging/xlog.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/utils/commSpecs.h"

class DevMemTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctran::utils::commCudaLibraryInit();
    FB_CUDACHECKTHROW(cudaSetDevice(cudaDev_));
  }

  void TearDown() override {
    // Check that all host memory has been freed
    EXPECT_TRUE(hostMemPtrs_.empty()) << "Not all host memory was freed";
  }

  void* allocMem(
      DevMemType type,
      size_t size,
      CUmemGenericAllocationHandle* pHandle = nullptr) {
    void* ptr = nullptr;
    if (type == DevMemType::kCudaMalloc) {
      FB_CUDACHECKTHROW(cudaMalloc(&ptr, size));
    } else if (type == DevMemType::kHostPinned) {
      FB_CUDACHECKTHROW(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    } else if (type == DevMemType::kManaged) {
      FB_CUDACHECKTHROW(cudaMallocManaged(&ptr, size));
    } else if (type == DevMemType::kCumem) {
      CUmemAllocationHandleType handleType =
          ctran::utils::getCuMemAllocHandleType();
      FB_COMMCHECKTHROW(ctran::utils::commCuMemAlloc(
          &ptr, pHandle, handleType, size, nullptr, "DevMemTypeUT"));
    } else if (type == DevMemType::kHostUnregistered) {
      ptr = malloc(size);
      hostMemPtrs_.insert(ptr);
    }

    return ptr;
  }

  commResult_t freeMem(DevMemType type, void* ptr) {
    EXPECT_NE(ptr, nullptr);
    if (type == DevMemType::kCudaMalloc || type == DevMemType::kManaged) {
      FB_CUDACHECK(cudaFree(ptr));
    } else if (type == DevMemType::kHostPinned) {
      FB_CUDACHECK(cudaFreeHost(ptr));
    } else if (type == DevMemType::kCumem) {
      FB_COMMCHECK(ctran::utils::commCuMemFree(ptr));
    } else if (type == DevMemType::kHostUnregistered) {
      if (hostMemPtrs_.find(ptr) != hostMemPtrs_.end()) {
        hostMemPtrs_.erase(ptr);
        free(ptr);
      } else {
        return commInternalError;
      }
    }

    return commSuccess;
  }

  const int cudaDev_ = 0;
  bool gpuDirectRdmaWithCudaVmmSupported_ = false;
  bool cuMemFabricSupported_ = false;
  std::unordered_set<void*> hostMemPtrs_;
};

TEST_F(DevMemTypeTest, NullPointerReturnsInvalidUsage) {
  DevMemType memType;
  commResult_t result = getDevMemType(nullptr, 0, memType);
  EXPECT_EQ(result, commInvalidUsage);
}

TEST_F(DevMemTypeTest, NegativeDeviceReturnsInvalidUsage) {
  DevMemType memType;
  int dummy = 42;
  commResult_t result = getDevMemType(&dummy, -1, memType);
  EXPECT_EQ(result, commInvalidUsage);
}

class DevMemTypeSizeTest : public DevMemTypeTest,
                           public ::testing::WithParamInterface<DevMemType> {};

TEST_P(DevMemTypeSizeTest, GetCorrectDevMemType) {
  const auto allocType = GetParam();
  std::vector<size_t> sizes = {
      1024, // 1 KB
      1024 * 1024, // 1 MB
      32 * 1024 * 1024, // 32 MB
      256 * 1024 * 1024 // 256 MB
  };

  for (const auto& size : sizes) {
    void* ptr = allocMem(allocType, size);
    DevMemType expectedType;
    FB_COMMCHECKTHROW(getDevMemType(ptr, cudaDev_, expectedType));
    EXPECT_EQ(allocType, expectedType);
    EXPECT_EQ(freeMem(allocType, ptr), commSuccess);
    EXPECT_NE(freeMem(allocType, ptr), commSuccess);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MemTypes,
    DevMemTypeSizeTest,
    ::testing::Values(
        DevMemType::kCudaMalloc,
        DevMemType::kManaged,
        DevMemType::kHostPinned,
        DevMemType::kHostUnregistered,
        DevMemType::kCumem),
    [](const ::testing::TestParamInfo<DevMemType>& info) {
      return std::string(devMemTypeStr(info.param));
    });
