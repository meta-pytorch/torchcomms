// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <folly/Expected.h>
#include <gtest/gtest.h>
#include <string>

#include "comms/ctran/utils/CudaUtils.h"

using namespace ctran::utils;

TEST(BusIdTest, MakeFromInt) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "Skipping test because CUDA device is not available.";
  }
  int cudaDev = 0;
  ASSERT_LT(cudaDev, device_count);

  auto busId = BusId::makeFrom(cudaDev);
  EXPECT_EQ(busId.toStr().size(), 17);
  EXPECT_EQ(busId.toStr()[16], '\0');
}

TEST(BusIdTest, MakeFromString) {
  std::string validBusId = "0000:00:00.0";
  auto busId = BusId::makeFrom(validBusId);
  EXPECT_EQ(busId.toStr().substr(0, 12), validBusId);
}

TEST(BusIdTest, MakeFromInt64) {
  int64_t busIdInt64 = 0x000000000000;
  auto busId = BusId::makeFrom(busIdInt64);
  EXPECT_EQ(busId.toStr().substr(0, 12), "0000:00:00.0");
}

TEST(BusIdTest, ToStrAndToInt64RoundTrip) {
  std::string busIdStr = "000a:bc:de.f";
  auto busId = BusId::makeFrom(busIdStr);
  EXPECT_EQ(busId.toStr().substr(0, 12), busIdStr);
  int64_t intVal = busId.toInt64();
  auto busId2 = BusId::makeFrom(intVal);
  EXPECT_EQ(busId2.toStr().substr(0, 12), busIdStr);
}

TEST(BusIdTest, OperatorEqTrue) {
  auto busId1 = BusId::makeFrom("0000:00:00.0");
  auto busId2 = BusId::makeFrom("0000:00:00.0");
  EXPECT_TRUE(busId1 == busId2);
}

TEST(BusIdTest, OperatorEqFalse) {
  auto busId1 = BusId::makeFrom("0000:00:00.0");
  auto busId2 = BusId::makeFrom("0001:00:00.0");
  EXPECT_FALSE(busId1 == busId2);
}

TEST(BusIdTest, Int64ToStringEdgeCases) {
  auto busId = BusId::makeFrom(int64_t(0));
  EXPECT_EQ(busId.toStr().substr(0, 12), "0000:00:00.0");
  int64_t maxBusId = (0xffffL << 20) | (0xffL << 12) | (0xffL << 4) | 0xfL;
  auto busIdMax = BusId::makeFrom(maxBusId);
  EXPECT_EQ(busIdMax.toStr().substr(0, 12), "ffff:ff:ff.f");
}
