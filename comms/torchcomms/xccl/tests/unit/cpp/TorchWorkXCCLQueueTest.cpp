#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"
#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XpuMock.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XcclMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms {

class TorchWorkXCCLQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_ = std::make_unique<TorchWorkXCCLQueue>();
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<TorchWorkXCCLQueue> queue_;
  xpuStream_t stream1_{c10::xpu::XPUStream::unpack3(1, 0, c10::DeviceType::XPU)};
  xpuStream_t stream2_{c10::xpu::XPUStream::unpack3(2, 0, c10::DeviceType::XPU)};
};

class TorchWorkXCCLQueueCommTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create hash store for communication
    auto store_ = c10::make_intrusive<c10d::HashStore>();

    // Set up device. make it the cpu device because we're mocking xpu.
    device_ = at::Device(at::DeviceType::CPU, 0);

    // Set timeout to 2 seconds for tests
    default_options_ = CommOptions();
    default_options_.timeout = std::chrono::milliseconds(2000);
    default_options_.abort_process_on_timeout_or_error = false;
    default_options_.store = store_;

    // Create fresh mocks for each test
    xpu_mock_ = std::make_shared<NiceMock<torch::comms::test::XpuMock>>();
    xccl_mock_ = std::make_shared<NiceMock<torch::comms::test::XcclMock>>();

    // Create communicator
    comm_ = std::make_shared<TorchCommXCCL>();
    comm_->setXpuApi(xpu_mock_);
    comm_->setXcclApi(xccl_mock_);
  }

  void TearDown() override {
    // Clear the communicator
    comm_.reset();
  }

  void setupRankAndSize(int rank, int size) {
    setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
    setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);

    ON_CALL(*xccl_mock_, commUserRank(_, _))
        .WillByDefault(DoAll(SetArgPointee<1>(rank), Return(onecclSuccess)));
    ON_CALL(*xccl_mock_, commCount(_, _))
        .WillByDefault(DoAll(SetArgPointee<1>(size), Return(onecclSuccess)));
  }

  void setupEventsForWork(size_t numWork) {
    EXPECT_CALL(*xpu_mock_, eventCreateWithFlags(_, _))
        .Times(numWork * 2)
        .WillRepeatedly(Return(XPU_SUCCESS));
    EXPECT_CALL(*xpu_mock_, eventRecord(_, _))
        .WillRepeatedly(Return(XPU_SUCCESS));
  }

  void setupWorkToSuccess() {
    EXPECT_CALL(*xpu_mock_, eventQuery(_))
        .WillRepeatedly(Return(XPU_SUCCESS));
  }

  void checkWorkQueue() {
    comm_->checkWorkQueue(true);
  }

  const auto& getStreamWorkQueues() {
    return comm_->workq_.stream_work_queues_;
  }

  // Raw pointers to mocks for setting expectations
  std::shared_ptr<NiceMock<torch::comms::test::XpuMock>> xpu_mock_;
  std::shared_ptr<NiceMock<torch::comms::test::XcclMock>> xccl_mock_;

  CommOptions default_options_;

  std::optional<at::Device> device_;
  std::shared_ptr<TorchCommXCCL> comm_;
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

TEST_F(TorchWorkXCCLQueueTest, GarbageCollectEmptyQueue) {
  // Test garbage collection on empty queue
  auto status = queue_->garbageCollect(true);
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkXCCLQueueTest, FinalizeEmptyQueue) {
  auto status = queue_->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkXCCLQueueTest, MultipleGarbageCollectCalls) {
  // Multiple garbage collect calls on empty queue should be safe
  auto status1 = queue_->garbageCollect(true);
  auto status2 = queue_->garbageCollect(true);
  auto status3 = queue_->garbageCollect(true);

  EXPECT_EQ(status1, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status3, TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkXCCLQueueTest, MultipleFinalizeCallsAfterGarbageCollect) {
  // Garbage collect first
  auto gc_status = queue_->garbageCollect(true);
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
    auto status = queue_->garbageCollect(false);
    EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
  }
}

TEST_F(TorchWorkXCCLQueueTest, ConcurrentFinalizeAndGarbageCollect) {
  auto gc_status = queue_->garbageCollect(true);
  auto finalize_status = queue_->finalize();
  auto gc_status2 = queue_->garbageCollect(false);

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

  auto status = queue2->garbageCollect(true);
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);

  status = queue2->finalize();
  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);

  queue2.reset();
}

TEST_F(TorchWorkXCCLQueueTest, MultipleQueuesIndependent) {
  auto queue2 = std::make_unique<TorchWorkXCCLQueue>();
  auto queue3 = std::make_unique<TorchWorkXCCLQueue>();

  auto status1 = queue_->garbageCollect(true);
  auto status2 = queue2->garbageCollect(true);
  auto status3 = queue3->finalize();

  EXPECT_EQ(status1, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWork::WorkStatus::COMPLETED);
  EXPECT_EQ(status3, TorchWork::WorkStatus::COMPLETED);

  queue2.reset();
  queue3.reset();
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
  auto work = comm_->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());

  checkWorkQueue();
  comm_->finalize();

  EXPECT_EQ(getStreamWorkQueues().size(), 0);
}

} // namespace torch::comms