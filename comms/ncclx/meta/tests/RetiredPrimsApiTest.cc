// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <nccl.h>

TEST(RetiredPrimsApiTest, DeviceSurfaceReturnsInvalidUsage) {
  void* devicePtr = nullptr;
  ncclLkeyPerDevice lkeys{};
  lkeys.size = 1;
  lkeys.values[0] = 1;

  EXPECT_EQ(
      ncclWinCreateDeviceWin(nullptr, 0, 0, 0, &devicePtr), ncclInvalidUsage);
  EXPECT_EQ(ncclWinDestroyDeviceWin(nullptr), ncclInvalidUsage);
  EXPECT_EQ(
      ncclGetMultiPeerDeviceHandle(
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      ncclInvalidUsage);
  EXPECT_EQ(
      ncclWinLocalRegisterBuffer(nullptr, nullptr, 0, &lkeys),
      ncclInvalidUsage);
  EXPECT_EQ(lkeys.size, 0);
  EXPECT_EQ(lkeys.values[0], 0);
  EXPECT_EQ(ncclWinLocalDeregisterBuffer(nullptr, nullptr), ncclInvalidUsage);
}

TEST(RetiredPrimsApiTest, DeviceAllToAllvReturnsInvalidUsage) {
  EXPECT_EQ(
      ncclx::deviceAllToAllv(
          nullptr, nullptr, nullptr, nullptr, ncclInt, nullptr, nullptr),
      ncclInvalidUsage);
}
