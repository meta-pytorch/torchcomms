// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/colltrace/CudaWaitEvent.h"

using meta::comms::CudaStream;

class CudaWaitEventTest : public ::testing::Test {
 protected:
  CudaStream stream_;
};

TEST_F(CudaWaitEventTest, Constructor) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Verify that the enqueue time is set during construction
  auto enqueueTimeResult = event->getCollEnqueueTime();
  if (enqueueTimeResult.hasError()) {
    FAIL() << enqueueTimeResult.error().message;
  }

  // Enqueue time should be close to now
  auto now = std::chrono::system_clock::now();
  auto enqueueTime = enqueueTimeResult.value();
  auto diff = now - enqueueTime;

  // The difference should be small (less than 1 second)
  EXPECT_LT(diff, std::chrono::milliseconds{1000});
}

TEST_F(CudaWaitEventTest, BeforeAndAfterCollKernelScheduled) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Call beforeCollKernelScheduled
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  // Call afterCollKernelScheduled
  auto afterResult = event->afterCollKernelScheduled();
  if (afterResult.hasError()) {
    FAIL() << afterResult.error().message;
  }

  // Synchronize the stream to ensure events are processed
  CUDACHECK_TEST(cudaStreamSynchronize(stream_.get()));

  // Wait for the start event with a timeout
  auto waitStartResult = event->waitCollStart(std::chrono::milliseconds(1000));
  if (waitStartResult.hasError()) {
    FAIL() << waitStartResult.error().message;
  }
  EXPECT_TRUE(waitStartResult.value());

  // Wait for the end event with a timeout
  auto waitEndResult = event->waitCollEnd(std::chrono::milliseconds(1000));
  if (waitEndResult.hasError()) {
    FAIL() << waitEndResult.error().message;
  }
  EXPECT_TRUE(waitEndResult.value());
}

TEST_F(CudaWaitEventTest, GetCollStartAndEndTime) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Record events
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  auto afterResult = event->afterCollKernelScheduled();
  if (afterResult.hasError()) {
    FAIL() << afterResult.error().message;
  }

  // Synchronize the stream to ensure events are processed
  CUDACHECK_TEST(cudaStreamSynchronize(stream_.get()));

  // Wait for events to complete
  auto waitStartResult = event->waitCollStart(std::chrono::milliseconds(1000));
  if (waitStartResult.hasError()) {
    FAIL() << waitStartResult.error().message;
  }
  EXPECT_TRUE(waitStartResult.value());

  auto waitEndResult = event->waitCollEnd(std::chrono::milliseconds(1000));
  if (waitEndResult.hasError()) {
    FAIL() << waitEndResult.error().message;
  }
  EXPECT_TRUE(waitEndResult.value());

  // Get timestamps
  auto enqueueTimeResult = event->getCollEnqueueTime();
  if (enqueueTimeResult.hasError()) {
    FAIL() << enqueueTimeResult.error().message;
  }
  auto enqueueTime = enqueueTimeResult.value();

  auto startTimeResult = event->getCollStartTime();
  if (startTimeResult.hasError()) {
    FAIL() << startTimeResult.error().message;
  }
  auto startTime = startTimeResult.value();

  auto endTimeResult = event->getCollEndTime();
  if (endTimeResult.hasError()) {
    FAIL() << endTimeResult.error().message;
  }
  auto endTime = endTimeResult.value();

  // Verify the sequence of timestamps
  EXPECT_LE(enqueueTime, startTime);
  EXPECT_LE(startTime, endTime);
}

TEST_F(CudaWaitEventTest, GetCollStartTimeBeforeReady) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Record start event but don't synchronize
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  // Trying to get start time before the event is ready should return an error
  auto startTimeResult = event->getCollStartTime();
  ASSERT_FALSE(startTimeResult.hasValue());
  EXPECT_EQ(startTimeResult.error().errorCode, commInternalError);

  // Synchronize the stream to clean up
  CUDACHECK_TEST(cudaStreamSynchronize(stream_.get()));
}

TEST_F(CudaWaitEventTest, GetCollEndTimeBeforeReady) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Record start and end events but don't synchronize
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  auto afterResult = event->afterCollKernelScheduled();
  if (afterResult.hasError()) {
    FAIL() << afterResult.error().message;
  }

  // Trying to get end time before the event is ready should return an error
  auto endTimeResult = event->getCollEndTime();
  ASSERT_FALSE(endTimeResult.hasValue());
  EXPECT_EQ(endTimeResult.error().errorCode, commInternalError);

  // Synchronize the stream to clean up
  CUDACHECK_TEST(cudaStreamSynchronize(stream_.get()));
}

TEST_F(CudaWaitEventTest, WaitCollStartBeforeRecorded) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Wait for start event before recording it should return an error
  auto waitResult = event->waitCollStart(std::chrono::milliseconds(1));
  ASSERT_FALSE(waitResult.hasValue());
  EXPECT_EQ(waitResult.error().errorCode, commInternalError);
}

TEST_F(CudaWaitEventTest, WaitCollEndBeforeRecorded) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Wait for end event before recording it should return an error
  auto waitResult = event->waitCollEnd(std::chrono::milliseconds(1));
  ASSERT_FALSE(waitResult.hasValue());
  EXPECT_EQ(waitResult.error().errorCode, commInternalError);
}

TEST_F(CudaWaitEventTest, SignalCollStartAndEnd) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // CudaWaitEvent ignores signalCollStart and
  // signalCollEnd
  auto signalStartResult = event->signalCollStart();
  if (signalStartResult.hasError()) {
    FAIL() << signalStartResult.error().message;
  }

  auto signalEndResult = event->signalCollEnd();
  if (signalEndResult.hasError()) {
    FAIL() << signalEndResult.error().message;
  }
}

TEST_F(CudaWaitEventTest, CompleteSequence) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  // Test the complete sequence of operations

  // 1. Get enqueue time
  auto enqueueTimeResult = event->getCollEnqueueTime();
  if (enqueueTimeResult.hasError()) {
    FAIL() << enqueueTimeResult.error().message;
  }
  auto enqueueTime = enqueueTimeResult.value();

  // 2. Record start event
  auto beforeResult = event->beforeCollKernelScheduled();
  if (beforeResult.hasError()) {
    FAIL() << beforeResult.error().message;
  }

  // 3. Record end event
  auto afterResult = event->afterCollKernelScheduled();
  if (afterResult.hasError()) {
    FAIL() << afterResult.error().message;
  }

  // 4. Synchronize the stream to ensure events are processed
  CUDACHECK_TEST(cudaStreamSynchronize(stream_.get()));

  // 5. Wait for start event
  auto waitStartResult = event->waitCollStart(std::chrono::milliseconds(1000));
  if (waitStartResult.hasError()) {
    FAIL() << waitStartResult.error().message;
  }
  EXPECT_TRUE(waitStartResult.value());

  // 6. Wait for end event
  auto waitEndResult = event->waitCollEnd(std::chrono::milliseconds(1000));
  if (waitEndResult.hasError()) {
    FAIL() << waitEndResult.error().message;
  }
  EXPECT_TRUE(waitEndResult.value());

  // 7. Get start time
  auto startTimeResult = event->getCollStartTime();
  if (startTimeResult.hasError()) {
    FAIL() << startTimeResult.error().message;
  }
  auto startTime = startTimeResult.value();

  // 8. Get end time
  auto endTimeResult = event->getCollEndTime();
  if (endTimeResult.hasError()) {
    FAIL() << endTimeResult.error().message;
  }
  auto endTime = endTimeResult.value();

  // 9. Verify the sequence of timestamps
  EXPECT_LE(enqueueTime, startTime);
  EXPECT_LE(startTime, endTime);
}

TEST_F(CudaWaitEventTest, GetCollStartTimeInMargin) {
  auto event =
      std::make_unique<meta::comms::colltrace::CudaWaitEvent>(stream_.get());

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Record start and end events
  EXPECT_CHECK_TEST(event->beforeCollKernelScheduled());
  auto startTimeCPU = std::chrono::system_clock::now();

  EXPECT_CHECK_TEST(event->afterCollKernelScheduled());

  auto waitRes = event->waitCollStart(std::chrono::milliseconds(1000))
                     .orElse([](auto err) { FAIL() << err.message; });
  ASSERT_TRUE(waitRes.value());

  auto startTimeFromEvent =
      event->getCollStartTime().orElse([](auto err) { FAIL() << err.message; });
  auto startTimeDiffUs = std::abs(
      std::chrono::duration_cast<std::chrono::microseconds>(
          startTimeFromEvent.value() - startTimeCPU)
          .count());
  // We should get reasonably close to the actual start time. Since we will see
  // error on both endTimeCPU (from GPU to CPU sync) and endTimeFromEvent (from
  // the error with Cuda Event itself), we now expect the error to be less
  // than 15ms. We shoule aim to get this down.
  EXPECT_LT(startTimeDiffUs, 15000);
}
