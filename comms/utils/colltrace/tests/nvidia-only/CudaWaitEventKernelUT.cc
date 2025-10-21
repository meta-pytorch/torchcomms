// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/colltrace/CudaWaitEvent.h"
#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.h"

using meta::comms::CudaStream;

class CudaWaitEventTest : public ::testing::Test {
 protected:
  CudaStream stream_;
};

TEST_F(CudaWaitEventTest, WaitCollStartTimeout) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Launch a long-running kernel to delay the event completion
  auto kernel = meta::comms::colltrace::CPUControlledKernel(stream_.get());
  kernel.launch();

  // Record start event
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  auto afterResult = event->afterCollKernelScheduled();
  if (afterResult.hasError()) {
    FAIL() << afterResult.error().message;
  }

  // Wait with a timeout, should time out
  auto waitResult = event->waitCollStart(std::chrono::milliseconds(100));
  if (waitResult.hasError()) {
    FAIL() << waitResult.error().message;
  }
  EXPECT_FALSE(waitResult.value());

  // Let the kernel destructor to do the cleanup. Do not wait for Event
  // synchronize!
}

TEST_F(CudaWaitEventTest, WaitCollEndTimeout) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Record start and end events
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  // Launch a long-running kernel to delay the event completion
  auto kernel = meta::comms::colltrace::CPUControlledKernel(stream_.get());
  kernel.launch();

  auto afterResult = event->afterCollKernelScheduled();
  if (afterResult.hasError()) {
    FAIL() << afterResult.error().message;
  }

  // Wait with a timeout, should time out
  auto waitResult = event->waitCollEnd(std::chrono::milliseconds(100));
  if (waitResult.hasError()) {
    FAIL() << waitResult.error().message;
  }
  EXPECT_FALSE(waitResult.value());

  // Let the kernel destructor to do the cleanup. Do not wait for Event
  // synchronize!
}

TEST_F(CudaWaitEventTest, GetCollEndTimeInMargin) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Record start and end events
  EXPECT_CHECK_TEST(event->beforeCollKernelScheduled());

  // Launch a long-running kernel to delay the event completion
  auto kernel = meta::comms::colltrace::CPUControlledKernel(stream_.get());
  kernel.launch();

  EXPECT_CHECK_TEST(event->afterCollKernelScheduled());

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  ASSERT_TRUE(kernel.endKernel());
  auto endTimeCPU = std::chrono::system_clock::now();

  auto waitRes =
      event->waitCollEnd(std::chrono::milliseconds(1000)).orElse([](auto err) {
        FAIL() << err.message;
      });
  EXPECT_TRUE(waitRes.value());

  auto endTimeFromEvent =
      event->getCollEndTime().orElse([](auto err) { FAIL() << err.message; });
  auto endTimeDiffUs =
      std::abs(std::chrono::duration_cast<std::chrono::microseconds>(
                   endTimeFromEvent.value() - endTimeCPU)
                   .count());
  // We should get reasonably close to the actual end time. Since we will see
  // error on both endTimeCPU (from GPU to CPU sync) and endTimeFromEvent (from
  // the error with Cuda Event itself), we now expect the error to be less
  // than 15ms. We shoule aim to get this down.
  EXPECT_LT(endTimeDiffUs, 20000);
}
