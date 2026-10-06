// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/nvml/NvmlApi.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace uniflow {

class NvmlApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    const auto cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus == cudaErrorNoDevice ||
        cudaStatus == cudaErrorInsufficientDriver ||
        cudaStatus == cudaErrorStubLibrary) {
      GTEST_SKIP() << "NVML device discovery is unavailable";
    }
    ASSERT_EQ(cudaStatus, cudaSuccess) << cudaGetErrorString(cudaStatus);
    if (deviceCount == 0) {
      GTEST_SKIP() << "NVML device discovery is unavailable";
    }

    auto count = api.deviceCount();
    ASSERT_TRUE(count.hasValue()) << count.error().message();
    ASSERT_GT(count.value(), 0);
  }

  NvmlApi api;
};

TEST_F(NvmlApiTest, DeviceCountAfterInit) {
  auto countResult = api.deviceCount();
  ASSERT_TRUE(countResult.hasValue()) << countResult.error().message();
  EXPECT_GT(countResult.value(), 0);
}

TEST_F(NvmlApiTest, DeviceInfoValid) {
  auto countResult = api.deviceCount();
  ASSERT_TRUE(countResult.hasValue()) << countResult.error().message();
  int count = countResult.value();

  for (int i = 0; i < count; i++) {
    auto infoResult = api.deviceInfo(i);
    ASSERT_TRUE(infoResult.hasValue()) << "deviceInfo(" << i << ") failed";
    auto& info = infoResult.value();
    EXPECT_NE(info.handle, nullptr) << "device " << i << " handle is null";
  }
}

TEST_F(NvmlApiTest, HandleByIndexConsistent) {
  auto countResult = api.deviceCount();
  ASSERT_TRUE(countResult.hasValue()) << countResult.error().message();
  int count = countResult.value();

  for (int i = 0; i < count; i++) {
    auto infoResult = api.deviceInfo(i);
    ASSERT_TRUE(infoResult.hasValue());

    nvmlDevice_t handle = nullptr;
    Status s = api.nvmlDeviceGetHandleByIndex(i, &handle);
    ASSERT_FALSE(s.hasError()) << s.error().message();
    EXPECT_EQ(handle, infoResult.value().handle)
        << "handle mismatch for device " << i;
  }
}

TEST_F(NvmlApiTest, ComputeCapabilityConsistent) {
  auto countResult = api.deviceCount();
  ASSERT_TRUE(countResult.hasValue()) << countResult.error().message();
  int count = countResult.value();

  for (int i = 0; i < count; i++) {
    auto infoResult = api.deviceInfo(i);
    ASSERT_TRUE(infoResult.hasValue());
    auto& info = infoResult.value();

    int major = -1, minor = -1;
    Status s =
        api.nvmlDeviceGetCudaComputeCapability(info.handle, &major, &minor);
    ASSERT_FALSE(s.hasError()) << s.error().message();
    EXPECT_EQ(major, info.computeCapabilityMajor) << "device " << i;
    EXPECT_EQ(minor, info.computeCapabilityMinor) << "device " << i;
  }
}

TEST_F(NvmlApiTest, P2PStatusConsistent) {
  auto countResult = api.deviceCount();
  ASSERT_TRUE(countResult.hasValue()) << countResult.error().message();
  int count = countResult.value();

  for (int a = 0; a < count; a++) {
    for (int b = 0; b < count; b++) {
      auto pairResult = api.devicePairInfo(a, b);
      ASSERT_TRUE(pairResult.hasValue())
          << "devicePairInfo(" << a << ", " << b << ") failed";

      auto infoA = api.deviceInfo(a);
      auto infoB = api.deviceInfo(b);
      ASSERT_TRUE(infoA.hasValue()) << "deviceInfo(" << a << ") failed";
      ASSERT_TRUE(infoB.hasValue()) << "deviceInfo(" << b << ") failed";

      nvmlGpuP2PStatus_t readStatus, writeStatus;
      Status sr = api.nvmlDeviceGetP2PStatus(
          infoA.value().handle,
          infoB.value().handle,
          NVML_P2P_CAPS_INDEX_READ,
          &readStatus);
      ASSERT_FALSE(sr.hasError()) << sr.error().message();
      EXPECT_EQ(readStatus, pairResult.value().p2pStatusRead)
          << "read mismatch for (" << a << ", " << b << ")";

      Status sw = api.nvmlDeviceGetP2PStatus(
          infoA.value().handle,
          infoB.value().handle,
          NVML_P2P_CAPS_INDEX_WRITE,
          &writeStatus);
      ASSERT_FALSE(sw.hasError()) << sw.error().message();
      EXPECT_EQ(writeStatus, pairResult.value().p2pStatusWrite)
          << "write mismatch for (" << a << ", " << b << ")";
    }
  }
}

} // namespace uniflow
