// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/mccl/McclTypes.h"
#include "comms/mccl/tests/MockWorkHandle.h"
#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"

namespace torch::comms::test {

class TorchWorkMCCLQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_ = std::make_unique<TorchWorkMCCLQueue>();
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<mccl::testing::MockWorkHandle> createMockWorkHandle() {
    return std::make_unique<mccl::testing::MockWorkHandle>();
  }

  c10::intrusive_ptr<TorchWorkMCCL> createWork(
      std::unique_ptr<mccl::testing::MockWorkHandle> mockHandle) {
    return c10::make_intrusive<TorchWorkMCCL>(
        /*comm=*/nullptr,
        /*stream=*/nullptr,
        /*inputTensors=*/at::Tensor{},
        /*outputTensors=*/at::Tensor{},
        /*workHandle=*/std::move(mockHandle),
        /*timeout=*/std::chrono::milliseconds(600000));
  }

  std::unique_ptr<TorchWorkMCCLQueue> queue_;

  // Use nullptr as a placeholder stream for testing.
  // Note: in real usage, this would be an actual CUDA stream.
  static constexpr cudaStream_t kTestStream1 = nullptr;
};

// Test: Empty queue returns COMPLETED status on garbageCollect
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectOnEmptyQueueReturnsCompleted) {
  // An empty queue should return COMPLETED since there's nothing to process
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: Empty queue returns COMPLETED status on finalize
TEST_F(TorchWorkMCCLQueueTest, FinalizeOnEmptyQueueReturnsCompleted) {
  // Finalizing an empty queue should return COMPLETED
  auto status = queue_->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: garbageCollect removes completed work from queue
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectRemovesCompletedWork) {
  auto mockHandle = createMockWorkHandle();
  mccl::Result successResult{.code = commSuccess, .message = ""};

  EXPECT_CALL(*mockHandle, getResult())
      .WillOnce(::testing::Return(successResult));

  auto work = createWork(std::move(mockHandle));
  queue_->enqueueWork(work, kTestStream1);

  // garbageCollect should process and remove the completed work
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);

  // Now queue is empty, subsequent garbageCollects should return COMPLETED
  status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: garbageCollect stops at in-progress work
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectStopsAtInProgressWork) {
  auto mockHandle = createMockWorkHandle();

  // Work returns nullopt (indicating still in progress)
  EXPECT_CALL(*mockHandle, getResult())
      .WillRepeatedly(::testing::Return(std::nullopt));

  auto work = createWork(std::move(mockHandle));
  queue_->enqueueWork(work, kTestStream1);

  // garbageCollect should return INPROGRESS since work is not done
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::INPROGRESS);
}

// Test: garbageCollect returns ERROR immediately when work fails
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectReturnsErrorOnFailedWork) {
  auto mockHandle = createMockWorkHandle();
  mccl::Result errorResult{.code = commInternalError, .message = "test error"};

  EXPECT_CALL(*mockHandle, getResult())
      .WillOnce(::testing::Return(errorResult));

  auto work = createWork(std::move(mockHandle));
  queue_->enqueueWork(work, kTestStream1);

  // garbageCollect should return ERROR immediately
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::ERROR);
}

// Test: terminal work is not popped, so every later garbageCollect() re-reports
// its latched status without re-querying the handle.
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectKeepsReportingTerminalStatus) {
  auto mockHandle = createMockWorkHandle();
  mccl::Result errorResult{.code = commInternalError, .message = "test error"};

  EXPECT_CALL(*mockHandle, getResult())
      .Times(1)
      .WillOnce(::testing::Return(errorResult));

  auto work = createWork(std::move(mockHandle));
  queue_->enqueueWork(work, kTestStream1);

  EXPECT_EQ(queue_->garbageCollect(), TorchWork::WorkStatus::ERROR);
  EXPECT_EQ(queue_->garbageCollect(), TorchWork::WorkStatus::ERROR);
  EXPECT_EQ(queue_->garbageCollect(), TorchWork::WorkStatus::ERROR);
}

// Test: Multiple completed works are all removed
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectRemovesMultipleCompletedWorks) {
  mccl::Result successResult{.code = commSuccess, .message = ""};

  // Create three work items, all will complete successfully
  auto mockHandle1 = createMockWorkHandle();
  auto mockHandle2 = createMockWorkHandle();
  auto mockHandle3 = createMockWorkHandle();

  EXPECT_CALL(*mockHandle1, getResult())
      .WillOnce(::testing::Return(successResult));
  EXPECT_CALL(*mockHandle2, getResult())
      .WillOnce(::testing::Return(successResult));
  EXPECT_CALL(*mockHandle3, getResult())
      .WillOnce(::testing::Return(successResult));

  queue_->enqueueWork(createWork(std::move(mockHandle1)), kTestStream1);
  queue_->enqueueWork(createWork(std::move(mockHandle2)), kTestStream1);
  queue_->enqueueWork(createWork(std::move(mockHandle3)), kTestStream1);

  // All works should be processed and removed
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: garbageCollect processes works in order (FIFO)
TEST_F(TorchWorkMCCLQueueTest, GarbageCollectProcessesWorksInOrder) {
  mccl::Result successResult{.code = commSuccess, .message = ""};

  // First work completes, second is in progress
  auto mockHandle1 = createMockWorkHandle();
  auto mockHandle2 = createMockWorkHandle();

  EXPECT_CALL(*mockHandle1, getResult())
      .WillOnce(::testing::Return(successResult));
  EXPECT_CALL(*mockHandle2, getResult())
      .WillOnce(::testing::Return(std::nullopt))
      .WillOnce(::testing::Return(successResult));

  queue_->enqueueWork(createWork(std::move(mockHandle1)), kTestStream1);
  queue_->enqueueWork(createWork(std::move(mockHandle2)), kTestStream1);

  // First call: removes work1, stops at work2 (in progress)
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::INPROGRESS);

  // Second call: work2 now completes
  status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: Works on different streams are processed independently
TEST_F(TorchWorkMCCLQueueTest, WorksOnDifferentStreamsProcessedIndependently) {
  mccl::Result successResult{.code = commSuccess, .message = ""};

  // Work on stream1 is in progress, work on stream2 completes
  auto mockHandle1 = createMockWorkHandle();
  auto mockHandle2 = createMockWorkHandle();

  EXPECT_CALL(*mockHandle1, getResult())
      .WillOnce(::testing::Return(std::nullopt))
      .WillOnce(::testing::Return(successResult));
  EXPECT_CALL(*mockHandle2, getResult())
      .WillOnce(::testing::Return(successResult));

  static const cudaStream_t testStream2 = reinterpret_cast<cudaStream_t>(0x1);

  queue_->enqueueWork(createWork(std::move(mockHandle1)), kTestStream1);
  queue_->enqueueWork(createWork(std::move(mockHandle2)), testStream2);

  // First garbageCollect: stream2's work completes, stream1's is in progress
  // The last status seen will be INPROGRESS (from stream1)
  auto status = queue_->garbageCollect();
  // Status depends on iteration order, but should not be ERROR
  EXPECT_NE(status, TorchWork::WorkStatus::ERROR);
  EXPECT_TRUE(
      status == TorchWork::WorkStatus::INPROGRESS ||
      status == TorchWork::WorkStatus::COMPLETED);

  // Second garbageCollect: stream1's work now completes
  status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: finalize returns ERROR if any work has error status
TEST_F(TorchWorkMCCLQueueTest, FinalizeReturnsErrorOnFailedWork) {
  mccl::Result errorResult{.code = commInternalError, .message = "test error"};

  auto mockHandle = createMockWorkHandle();
  EXPECT_CALL(*mockHandle, getResult())
      .WillOnce(::testing::Return(errorResult));

  queue_->enqueueWork(createWork(std::move(mockHandle)), kTestStream1);

  // finalize should detect the error and return ERROR status
  auto status = queue_->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::ERROR);
}

// Test: finalize processes work until completion or error
TEST_F(TorchWorkMCCLQueueTest, FinalizeProcessesUntilCompletionOrError) {
  mccl::Result successResult{.code = commSuccess, .message = ""};

  auto mockHandle = createMockWorkHandle();

  // Work transitions from in-progress to completed
  EXPECT_CALL(*mockHandle, getResult())
      .WillOnce(::testing::Return(std::nullopt))
      .WillOnce(::testing::Return(successResult));

  queue_->enqueueWork(createWork(std::move(mockHandle)), kTestStream1);

  // finalize should wait for work to complete
  auto status = queue_->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

// Test: finalize returns ERROR when called after garbageCollect already
// detected error
TEST_F(
    TorchWorkMCCLQueueTest,
    FinalizeReturnsErrorAfterGarbageCollectDetectedError) {
  mccl::Result errorResult{.code = commInternalError, .message = "test error"};

  auto mockHandle = createMockWorkHandle();
  // The work will return error on both calls (from garbageCollect and finalize)
  EXPECT_CALL(*mockHandle, getResult())
      .WillRepeatedly(::testing::Return(errorResult));

  queue_->enqueueWork(createWork(std::move(mockHandle)), kTestStream1);

  // First call garbageCollect - it should return ERROR
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::ERROR);

  // Now call finalize - it should still return ERROR
  // The failed work remains in queue and finalize should detect it
  status = queue_->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::ERROR);
}

// Test: destroying the queue with in-flight work closes it out (TIMEDOUT) and
// fires its end hook (clog "E"). On PAFT fault recovery the comm is destroyed
// with a wedged collective still pending; without this the work would be
// dropped without a terminal transition and the clog would show a bare "Q"/"S"
// with no matching "E".
TEST_F(TorchWorkMCCLQueueTest, DestructorClosesInProgressWorkAndFiresEndHook) {
  auto mockHandle = createMockWorkHandle();

  // Work is INPROGRESS from construction and never completes on its own.
  auto work = createWork(std::move(mockHandle));
  ASSERT_EQ(work->status(), TorchWork::WorkStatus::INPROGRESS);

  int end_count = 0;
  work->registerWorkEndHook([&end_count]() { end_count++; });

  queue_->enqueueWork(work, kTestStream1);
  EXPECT_EQ(end_count, 0);

  // Tear down the queue while the work is still in flight (fault-recovery
  // teardown). We still hold `work`, so its hooks are intact when the queue
  // marks it terminal.
  queue_.reset();

  EXPECT_EQ(work->status(), TorchWork::WorkStatus::TIMEDOUT);
  EXPECT_EQ(end_count, 1);
}

} // namespace torch::comms::test
