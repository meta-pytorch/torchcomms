// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/ibverbs/IbvApi.h"

#include <gtest/gtest.h>

namespace uniflow {

class IbvApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto status = api_.init();
    ASSERT_FALSE(status.hasError()) << status.error().message();

    auto result = api_.getDeviceList(&numDevices_);
    ASSERT_TRUE(result.hasValue()) << result.error().message();
    ASSERT_GT(numDevices_, 0);
    deviceList_ = result.value();
  }

  void TearDown() override {
    if (deviceList_ != nullptr) {
      api_.freeDeviceList(deviceList_);
    }
  }

  IbvApi api_;
  ibv_device** deviceList_{nullptr};
  int numDevices_{0};
};

TEST_F(IbvApiTest, GetDeviceNameValid) {
  for (int i = 0; i < numDevices_; i++) {
    auto nameResult = api_.getDeviceName(deviceList_[i]);
    ASSERT_TRUE(nameResult.hasValue())
        << "getDeviceName(" << i
        << ") failed: " << nameResult.error().message();
    EXPECT_NE(nameResult.value(), nullptr) << "device " << i << " name is null";
  }
}

TEST_F(IbvApiTest, QueryDeviceSucceeds) {
  for (int i = 0; i < numDevices_; i++) {
    auto ctxResult = api_.openDevice(deviceList_[i]);
    ASSERT_TRUE(ctxResult.hasValue());

    ibv_device_attr attr{};
    auto status = api_.queryDevice(ctxResult.value(), &attr);
    EXPECT_FALSE(status.hasError())
        << "queryDevice(" << i << ") failed: " << status.error().message();

    status = api_.closeDevice(ctxResult.value());
    ASSERT_FALSE(status.hasError())
        << "closeDevice(" << i << ") failed: " << status.error().message();
  }
}

TEST_F(IbvApiTest, QueryPortSucceeds) {
  for (int i = 0; i < numDevices_; i++) {
    auto ctxResult = api_.openDevice(deviceList_[i]);
    ASSERT_TRUE(ctxResult.hasValue());

    ibv_port_attr portAttr{};
    auto status = api_.queryPort(ctxResult.value(), 1, &portAttr);
    EXPECT_FALSE(status.hasError())
        << "queryPort(" << i
        << ", port=1) failed: " << status.error().message();

    status = api_.closeDevice(ctxResult.value());
    ASSERT_FALSE(status.hasError())
        << "closeDevice(" << i << ") failed: " << status.error().message();
  }
}

TEST_F(IbvApiTest, QueryGidSucceeds) {
  for (int i = 0; i < numDevices_; i++) {
    auto ctxResult = api_.openDevice(deviceList_[i]);
    ASSERT_TRUE(ctxResult.hasValue());

    ibv_gid gid{};
    auto status = api_.queryGid(ctxResult.value(), 1, 0, &gid);
    EXPECT_FALSE(status.hasError())
        << "queryGid(" << i
        << ", port=1, index=0) failed: " << status.error().message();

    status = api_.closeDevice(ctxResult.value());
    ASSERT_FALSE(status.hasError())
        << "closeDevice(" << i << ") failed: " << status.error().message();
  }
}

TEST_F(IbvApiTest, Mlx5dvIsSupportedSucceeds) {
  for (int i = 0; i < numDevices_; i++) {
    auto result = api_.mlx5dvIsSupported(deviceList_[i]);
    ASSERT_TRUE(result.hasValue()) << "mlx5dvIsSupported(" << i
                                   << ") failed: " << result.error().message();
  }
}

TEST_F(IbvApiTest, Mlx5dvGetDataDirectSysfsPathSucceeds) {
  for (int i = 0; i < numDevices_; i++) {
    auto supported = api_.mlx5dvIsSupported(deviceList_[i]);
    ASSERT_TRUE(supported.hasValue());
    if (!supported.value()) {
      continue;
    }

    auto ctxResult = api_.openDevice(deviceList_[i]);
    ASSERT_TRUE(ctxResult.hasValue());

    char buf[256]{};
    auto status =
        api_.mlx5dvGetDataDirectSysfsPath(ctxResult.value(), buf, sizeof(buf));
    // This API may fail if the symbol is not available or the device
    // does not support data direct — that is acceptable.
    // We only verify that the call does not crash.
    if (!status.hasError()) {
      EXPECT_NE(buf[0], '\0') << "device " << i << " returned empty sysfs path";
    }

    status = api_.closeDevice(ctxResult.value());
    ASSERT_FALSE(status.hasError())
        << "closeDevice(" << i << ") failed: " << status.error().message();
  }
}

} // namespace uniflow
