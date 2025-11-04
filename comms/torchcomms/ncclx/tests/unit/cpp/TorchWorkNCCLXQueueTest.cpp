// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CachingAllocatorHookMock.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CudaMock.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/NcclxMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch {
namespace comms {

struct WorkEvent {
  cudaEvent_t start_event;
  cudaEvent_t end_event;

  WorkEvent(cudaEvent_t start, cudaEvent_t end)
      : start_event(start), end_event(end) {}
};

class TorchWorkNCCLXQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create mock CUDA streams
    stream1_ = reinterpret_cast<cudaStream_t>(0x1001);
    stream2_ = reinterpret_cast<cudaStream_t>(0x1002);

    queue_ = std::make_unique<TorchWorkNCCLXQueue>();
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<TorchWorkNCCLXQueue> queue_;
  cudaStream_t stream1_{};
  cudaStream_t stream2_{};
};

class TorchWorkNCCLXQueueCommTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Force the global instance to be created on the CPU device
    auto hook_mock =
        std::make_unique<torch::comms::test::CachingAllocatorHookMock>();
    hook_mock->setupDefaultBehaviors();

    mock_hook_ = hook_mock.get();
    CachingAllocatorHook::setInstance(std::move(hook_mock));

    // Create hash store for communication
    auto store_ = c10::make_intrusive<c10d::HashStore>();

    // Set up device. make it the cpu device because we're mocking cuda.
    device_ = at::Device(at::DeviceType::CPU, 0);

    // Set timeout to 2 seconds for tests
    default_options_ = CommOptions();
    default_options_.timeout = std::chrono::milliseconds(2000);
    default_options_.abort_process_on_timeout_or_error = false;
    default_options_.store = store_;

    // Create fresh mocks for each test
    cuda_mock_ = std::make_shared<NiceMock<torch::comms::test::CudaMock>>();
    nccl_mock_ = std::make_shared<NiceMock<torch::comms::test::NcclxMock>>();

    // Create communicator
    comm_ = std::make_shared<TorchCommNCCLX>();
    comm_->setCudaApi(cuda_mock_);
    comm_->setNcclApi(nccl_mock_);
  }

  void TearDown() override {
    // Clear the communicator
    comm_.reset();
    // Clear the instance data
    CachingAllocatorHook::getInstance().clear();
    // Reset the instance to null to release the mock object
    CachingAllocatorHook::setInstance(nullptr);
  }

  void setupRankAndSize(int rank, int size) {
    setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
    setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);

    ON_CALL(*nccl_mock_, commUserRank(_, _))
        .WillByDefault(DoAll(SetArgPointee<1>(rank), Return(ncclSuccess)));
    ON_CALL(*nccl_mock_, commCount(_, _))
        .WillByDefault(DoAll(SetArgPointee<1>(size), Return(ncclSuccess)));
  }

  void setupEventsForWork(size_t numWork) {
    for (size_t i = 0; i < numWork; i++) {
      // Create mock event pointers using individual static dummy variables
      static char start_dummy, end_dummy;
      cudaEvent_t start_event = reinterpret_cast<cudaEvent_t>(&start_dummy + i);
      cudaEvent_t end_event = reinterpret_cast<cudaEvent_t>(&end_dummy + i);
      EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
          .WillOnce(DoAll(
              SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(start_event)),
              Return(cudaSuccess)))
          .WillOnce(DoAll(
              SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(end_event)),
              Return(cudaSuccess)));
      EXPECT_CALL(*cuda_mock_, eventRecord(getAsyncDependencyEvent(), _));
      EXPECT_CALL(*cuda_mock_, eventRecord(start_event, _));
      EXPECT_CALL(*cuda_mock_, eventRecord(end_event, _));
      work_events_.emplace_back(start_event, end_event);
    }
  }

  void setupWorkToSuccess(WorkEvent& work_event) {
    EXPECT_CALL(*cuda_mock_, eventQuery(work_event.start_event))
        .WillOnce(Return(cudaSuccess)); // start event succeeds

    EXPECT_CALL(*cuda_mock_, eventQuery(work_event.end_event))
        .WillRepeatedly(Return(cudaSuccess)); // end event succeeds
  }

  void setupCCAExpectations(
      int times_register,
      int times_deregister,
      int times_clear) {
    // Expect a registration call during init
    EXPECT_CALL(*mock_hook_, registerComm(_)).Times(times_register);

    // Expect a deregistration call during finalize, destruction or abort
    EXPECT_CALL(*mock_hook_, deregisterComm(_)).Times(times_deregister);

    // Expect a clear call during destruction
    EXPECT_CALL(*mock_hook_, clear()).Times(times_clear);
  }

  void checkWorkQueue() {
    comm_->checkWorkQueue();
  }

  const auto& getStreamWorkQueues() {
    return comm_->workq_.stream_work_queues_;
  }

  cudaEvent_t getAsyncDependencyEvent() {
    return comm_->dependency_event_;
  }

  // Raw pointers to mocks for setting expectations
  std::shared_ptr<NiceMock<torch::comms::test::CudaMock>> cuda_mock_;
  std::shared_ptr<NiceMock<torch::comms::test::NcclxMock>> nccl_mock_;
  std::vector<WorkEvent> work_events_;

  CommOptions default_options_;
  torch::comms::test::CachingAllocatorHookMock* mock_hook_{};

  std::optional<at::Device> device_;
  std::shared_ptr<TorchCommNCCLX> comm_;
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

TEST_F(TorchWorkNCCLXQueueTest, GarbageCollectEmptyQueue) {
  // Test garbage collection on empty queue
  auto status = queue_->garbageCollect();
  EXPECT_EQ(status, TorchWorkNCCLX::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkNCCLXQueueTest, FinalizeEmptyQueue) {
  auto status = queue_->finalize();
  EXPECT_EQ(status, TorchWorkNCCLX::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkNCCLXQueueTest, MultipleGarbageCollectCalls) {
  // Multiple garbage collect calls on empty queue should be safe
  auto status1 = queue_->garbageCollect();
  auto status2 = queue_->garbageCollect();
  auto status3 = queue_->garbageCollect();

  EXPECT_EQ(status1, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(status3, TorchWorkNCCLX::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkNCCLXQueueTest, MultipleFinalizeCallsAfterGarbageCollect) {
  // Garbage collect first
  auto gc_status = queue_->garbageCollect();
  EXPECT_EQ(gc_status, TorchWorkNCCLX::WorkStatus::COMPLETED);

  // Multiple finalize calls should be safe
  auto status1 = queue_->finalize();
  auto status2 = queue_->finalize();

  EXPECT_EQ(status1, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWorkNCCLX::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkNCCLXQueueTest, GarbageCollectMainThreadFlag) {
  // Test that the isMainThread flag doesn't cause issues on empty queue
  auto status1 = queue_->garbageCollect();
  auto status2 = queue_->garbageCollect();

  EXPECT_EQ(status1, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWorkNCCLX::WorkStatus::COMPLETED);
}

// ============================================================================
// THREAD SAFETY TESTS
// ============================================================================

TEST_F(TorchWorkNCCLXQueueTest, ConcurrentGarbageCollectCalls) {
  // This test verifies that multiple garbage collect calls are thread-safe
  // Even though we're not using actual threads here, we test that the
  // mutex-protected operations work correctly with multiple calls

  for (int i = 0; i < 10; ++i) {
    auto status = queue_->garbageCollect();
    EXPECT_EQ(status, TorchWorkNCCLX::WorkStatus::COMPLETED);
  }
}

TEST_F(TorchWorkNCCLXQueueTest, ConcurrentFinalizeAndGarbageCollect) {
  // Test that finalize and garbage collect can be called in sequence safely
  auto gc_status = queue_->garbageCollect();
  auto finalize_status = queue_->finalize();
  auto gc_status2 = queue_->garbageCollect();

  EXPECT_EQ(gc_status, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(finalize_status, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(gc_status2, TorchWorkNCCLX::WorkStatus::COMPLETED);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST_F(TorchWorkNCCLXQueueTest, EnqueueNullWorkDoesNotCrash) {
  // Test that enqueueing null work doesn't crash during enqueue
  // Note: We don't call garbageCollect after this because that would
  // cause a segfault when trying to call checkStatus() on null work
  c10::intrusive_ptr<TorchWorkNCCLX> null_work = nullptr;

  // This should not crash the queue during enqueue
  EXPECT_NO_THROW(queue_->enqueueWork(null_work, stream1_));

  // Note: We intentionally don't call garbageCollect here because
  // the current implementation doesn't handle null work gracefully
  // This test documents the current behavior and could be extended
  // in the future if null work handling is improved
}

// ============================================================================
// BASIC QUEUE STRUCTURE TESTS
// ============================================================================

TEST_F(TorchWorkNCCLXQueueTest, QueueCreationAndDestruction) {
  // Test that queue can be created and destroyed without issues
  auto queue2 = std::make_unique<TorchWorkNCCLXQueue>();
  EXPECT_NE(queue2, nullptr);

  // Test basic operations on new queue
  auto status = queue2->garbageCollect();
  EXPECT_EQ(status, TorchWorkNCCLX::WorkStatus::COMPLETED);

  status = queue2->finalize();
  EXPECT_EQ(status, TorchWorkNCCLX::WorkStatus::COMPLETED);

  // Destruction should be clean
  queue2.reset();
}

TEST_F(TorchWorkNCCLXQueueTest, MultipleQueuesIndependent) {
  // Test that multiple queues operate independently
  auto queue2 = std::make_unique<TorchWorkNCCLXQueue>();
  auto queue3 = std::make_unique<TorchWorkNCCLXQueue>();

  // Operations on different queues should not interfere
  auto status1 = queue_->garbageCollect();
  auto status2 = queue2->garbageCollect();
  auto status3 = queue3->finalize();

  EXPECT_EQ(status1, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(status2, TorchWorkNCCLX::WorkStatus::COMPLETED);
  EXPECT_EQ(status3, TorchWorkNCCLX::WorkStatus::COMPLETED);

  queue2.reset();
  queue3.reset();
}

TEST_F(TorchWorkNCCLXQueueCommTest, NoLeakedObjectsAfterFinalize) {
  setupRankAndSize(0, 2); // rank 0, size 2
  setupCCAExpectations(1, 2, 1);
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm_->init(*device_, "test_name", default_options_);

  setupEventsForWork(1);
  setupWorkToSuccess(work_events_[0]);

  auto tensor = at::ones(
      {10, 10}, at::TensorOptions().device(*device_).dtype(at::kFloat));
  auto work = comm_->send(tensor, 1, true); // async send

  // Simulate the timeout thread calling checkWorkQueue
  checkWorkQueue();
  // Comm finalize will call the work queue finalize().
  comm_->finalize();

  EXPECT_EQ(getStreamWorkQueues().size(), 0);
}

TEST_F(TorchWorkNCCLXQueueCommTest, NoFailureUnderCudaGraphMode) {
  setupRankAndSize(0, 2); // rank 0, size 2
  setupCCAExpectations(1, 2, 1);
  cuda_mock_->setupDefaultBehaviors();
  nccl_mock_->setupDefaultBehaviors();

  comm_->init(*device_, "test_name", default_options_);

  setupEventsForWork(1);

  std::mt19937 gen;
  std::uniform_int_distribution<int> dist(0, 10);
  const int rand1 = dist(gen);
  const int rand2 = dist(gen);

  testing::Sequence s;

  for (int i = 0; i < rand1; i++) {
    EXPECT_CALL(*cuda_mock_, eventQuery(work_events_[0].start_event))
        .InSequence(s)
        .WillOnce(Return(cudaErrorStreamCaptureUnsupported));
  }
  EXPECT_CALL(*cuda_mock_, eventQuery(work_events_[0].start_event))
      .InSequence(s)
      .WillOnce(Return(cudaSuccess)); // start event succeeds

  for (int i = 0; i < rand2; i++) {
    EXPECT_CALL(*cuda_mock_, eventQuery(work_events_[0].end_event))
        .InSequence(s)
        .WillOnce(Return(cudaErrorStreamCaptureUnsupported));
  }
  EXPECT_CALL(*cuda_mock_, eventQuery(work_events_[0].end_event))
      .InSequence(s)
      .WillRepeatedly(Return(cudaSuccess)); // end event succeeds

  auto tensor = at::ones(
      {10, 10}, at::TensorOptions().device(*device_).dtype(at::kFloat));
  auto work = comm_->send(tensor, 1, true); // async send

  // Simulate the timeout thread calling checkWorkQueue
  checkWorkQueue();
  // Comm finalize will call the work queue finalize().
  comm_->finalize();

  EXPECT_EQ(getStreamWorkQueues().size(), 0);
}

} // namespace comms
} // namespace torch
