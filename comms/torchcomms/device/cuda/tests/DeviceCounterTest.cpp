// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/device/cuda/DeviceCounter.h"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CudaMock.hpp"

using ::testing::_;
using ::testing::Return;

namespace torch::comms::test {

class DeviceCounterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_ = std::make_shared<::testing::NiceMock<CudaMock>>();
    mock_->setupDefaultBehaviors();
  }

  std::shared_ptr<::testing::NiceMock<CudaMock>> mock_;
};

TEST_F(DeviceCounterTest, CreateAllocatesAndZeros) {
  std::unique_ptr<DeviceCounter> counter;
  auto err = DeviceCounter::create(mock_.get(), counter);

  ASSERT_EQ(err, cudaSuccess);
  ASSERT_NE(counter, nullptr);
  ASSERT_NE(counter->ptr(), nullptr);

  EXPECT_EQ(counter->read(), 0);
}

TEST_F(DeviceCounterTest, IncrementAndRead) {
  std::unique_ptr<DeviceCounter> counter;
  ASSERT_EQ(DeviceCounter::create(mock_.get(), counter), cudaSuccess);

  ASSERT_EQ(counter->increment(nullptr), cudaSuccess);
  ASSERT_EQ(counter->increment(nullptr), cudaSuccess);

  EXPECT_EQ(counter->read(), 2);
}

TEST_F(DeviceCounterTest, IncrementWithCustomAmount) {
  std::unique_ptr<DeviceCounter> counter;
  ASSERT_EQ(DeviceCounter::create(mock_.get(), counter), cudaSuccess);

  ASSERT_EQ(counter->increment(nullptr, 5), cudaSuccess);
  ASSERT_EQ(counter->increment(nullptr, 10), cudaSuccess);

  EXPECT_EQ(counter->read(), 15);
}

TEST_F(DeviceCounterTest, CreateFailsOnHostAllocError) {
  ON_CALL(*mock_, hostAlloc(_, _, _))
      .WillByDefault(Return(cudaErrorMemoryAllocation));

  std::unique_ptr<DeviceCounter> counter;
  auto err = DeviceCounter::create(mock_.get(), counter);

  EXPECT_NE(err, cudaSuccess);
  EXPECT_EQ(counter, nullptr);
}

} // namespace torch::comms::test
