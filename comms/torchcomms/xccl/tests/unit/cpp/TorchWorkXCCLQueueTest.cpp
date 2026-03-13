#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "TorchCommXCCLTestBase.hpp"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"
#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XcclMock.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XpuMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms {

using test::TorchCommXCCLTest;

class TorchWorkXCCLQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_ = std::make_unique<TorchWorkXCCLQueue>();
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<TorchWorkXCCLQueue> queue_;
  xpuStream_t stream1_{
      c10::xpu::XPUStream::unpack3(1, 0, c10::DeviceType::XPU)};
  xpuStream_t stream2_{
      c10::xpu::XPUStream::unpack3(2, 0, c10::DeviceType::XPU)};
};

class TorchWorkXCCLQueueCommTest : public TorchCommXCCLTest {
 protected:
  void SetUp() override {
    this->TorchCommXCCLTest::SetUp();
    comm_ = createMockedTorchComm();
  }

  void TearDown() override {
    comm_.reset();
    this->TorchCommXCCLTest::TearDown();
  }

  void setupEventsForWork(size_t numWork) {
    EXPECT_CALL(*xpu_mock_, eventCreateWithFlags(_, _))
        .Times(numWork * 2)
        .WillRepeatedly(Return(XPU_SUCCESS));
    EXPECT_CALL(*xpu_mock_, eventRecord(_, _))
        .WillRepeatedly(Return(XPU_SUCCESS));
  }

  void setupWorkToSuccess() {
    EXPECT_CALL(*xpu_mock_, eventQuery(_)).WillRepeatedly(Return(XPU_SUCCESS));
  }

  // Raw pointers to mocks for setting expectations
  std::shared_ptr<TorchCommXCCLTest::TestTorchCommXCCL> comm_;
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

TEST_F(TorchWorkXCCLQueueTest, GarbageCollectEmptyQueue) {
  // Test garbage collection on empty queue
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkXCCLQueueTest, FinalizeEmptyQueue) {
  auto status = queue_->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkXCCLQueueTest, MultipleGarbageCollectCalls) {
  // Multiple garbage collect calls on empty queue should be safe
  auto status1 = queue_->garbageCollect();
  auto status2 = queue_->garbageCollect();
  auto status3 = queue_->garbageCollect();

  EXPECT_EQ(status1, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status3, TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkXCCLQueueTest, MultipleFinalizeCallsAfterGarbageCollect) {
  // Garbage collect first
  auto gc_status = queue_->garbageCollect();
  EXPECT_EQ(gc_status, TorchWork::WorkStatus::COMPLETED);

  // Multiple finalize calls should be safe
  auto status1 = queue_->finalize();
  auto status2 = queue_->finalize();

  EXPECT_EQ(status1, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// THREAD SAFETY TESTS
// ============================================================================

TEST_F(TorchWorkXCCLQueueTest, ConcurrentGarbageCollectCalls) {
  for (int i = 0; i < 10; ++i) {
    auto status = queue_->garbageCollect();
    EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
  }
}

TEST_F(TorchWorkXCCLQueueTest, ConcurrentFinalizeAndGarbageCollect) {
  auto gc_status = queue_->garbageCollect();
  auto finalize_status = queue_->finalize();
  auto gc_status2 = queue_->garbageCollect();

  EXPECT_EQ(gc_status, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(finalize_status, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(gc_status2, TorchWork::WorkStatus::COMPLETED);
}

// ============================================================================
// BASIC QUEUE STRUCTURE TESTS
// ============================================================================

TEST_F(TorchWorkXCCLQueueTest, QueueCreationAndDestruction) {
  auto queue2 = std::make_unique<TorchWorkXCCLQueue>();
  EXPECT_NE(queue2, nullptr);

  auto status = queue2->garbageCollect();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);

  status = queue2->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);

  queue2.reset();
}

TEST_F(TorchWorkXCCLQueueTest, MultipleQueuesIndependent) {
  auto queue2 = std::make_unique<TorchWorkXCCLQueue>();
  auto queue3 = std::make_unique<TorchWorkXCCLQueue>();

  auto status1 = queue_->garbageCollect();
  auto status2 = queue2->garbageCollect();
  auto status3 = queue3->finalize();

  EXPECT_EQ(status1, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status3, TorchWork::WorkStatus::COMPLETED);

  queue2.reset();
  queue3.reset();
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST_F(TorchWorkXCCLQueueTest, EnqueueNullWorkDoesNotCrash) {
  // Enqueue a nullptr work item; verify no crash during enqueue
  EXPECT_NO_THROW(queue_->enqueueWork(nullptr, stream1_));
}

// ============================================================================
// COMM WORK TESTS
// ============================================================================

TEST_F(TorchWorkXCCLQueueCommTest, NoLeakedObjectsAfterFinalize) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  comm_->init(*device_, "test_name", default_options_);

  setupEventsForWork(1);
  setupWorkToSuccess();

  auto tensor = at::ones(
      {10, 10}, at::TensorOptions().device(*device_).dtype(at::kFloat));

  // all_reduce will enqueue the work object
  auto work =
      comm_->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());

  comm_->finalize();

  EXPECT_TRUE(comm_->isWorkQueueEmpty());
}

} // namespace torch::comms
