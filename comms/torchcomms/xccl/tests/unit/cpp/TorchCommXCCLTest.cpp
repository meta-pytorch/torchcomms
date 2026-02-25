#include <gtest/gtest.h>
#include "TorchCommXCCLTestBase.hpp"

namespace torch::comms::test {

// ============================================================================
// 1. INITIALIZATION TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, TestOptionsEnvironmentVariables) {
  setOptionsEnvironmentVariables(false, 1); // false abort, 1 second

  CommOptions options1;
  EXPECT_EQ(options1.abort_process_on_timeout_or_error, false);
  EXPECT_EQ(options1.timeout, std::chrono::milliseconds(1000));

  setOptionsEnvironmentVariables(true, 2); // true abort, 2 seconds
  CommOptions options2;
  EXPECT_EQ(options2.abort_process_on_timeout_or_error, true);
  EXPECT_EQ(options2.timeout, std::chrono::milliseconds(2000));
}

TEST_F(TorchCommXCCLTest, UniqueCommDesc) {
  setupRankAndSize(0, 4); // rank 0, size 4
  xpu_mock_->setupDefaultBehaviors();

  ON_CALL(*xccl_mock_, getUniqueId(_))
      .WillByDefault(
          DoAll(SetArgPointee<0>(onecclUniqueId{}), Return(onecclSuccess)));

  ON_CALL(*xccl_mock_, commInitRankConfig(_, _, _, 0, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<onecclComm_t>(0x1000)),
          Return(onecclSuccess)));

  auto comm = createMockedTorchComm();

  EXPECT_CALL(*xccl_mock_, commDestroy(reinterpret_cast<onecclComm_t>(0x1000)))
      .WillOnce(Return(onecclSuccess));

  comm->init(*device_, "test_comm", default_options_);

  comm->finalize();
}

// ============================================================================
// 2. ALL_REDUCE TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, AllReduce_Success) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({10, 10});

  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  EXPECT_CALL(*xpu_mock_, eventQuery(_))
      .WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());
  work->wait();

  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
  comm->finalize();
}

// ============================================================================
// 3. BROADCAST TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, DISABLED_Broadcast_Success) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({5, 5});

  EXPECT_CALL(*xccl_mock_, broadcast(_, _, _, _, 0, _, _))
      .WillOnce(Return(onecclSuccess));

  EXPECT_CALL(*xpu_mock_, eventQuery(_))
      .WillRepeatedly(Return(XPU_SUCCESS));

  auto work = comm->broadcast(tensor, 0, true, BroadcastOptions());
  work->wait();

  EXPECT_EQ(work->status(), TorchWork::WorkStatus::COMPLETED);
  comm->finalize();
}

// ============================================================================
// 4. TIMEOUT / ABORT TESTS
// ============================================================================

TEST_F(TorchCommXCCLTest, AsyncErrorAbortsCommunicator) {
  setupRankAndSize(0, 2);
  xpu_mock_->setupDefaultBehaviors();
  xccl_mock_->setupDefaultBehaviors();
  
  // Set the abort on timeout/error option so watchdog can abort
  default_options_.abort_process_on_timeout_or_error = false;
  default_options_.timeout = std::chrono::milliseconds(200);

  auto comm = createMockedTorchComm();
  comm->init(*device_, "test_comm", default_options_);

  setupEventsForWork(*comm, 1);

  auto tensor = createTestTensor({10, 10});

  // Make collective succeed initially
  EXPECT_CALL(*xccl_mock_, allReduce(_, _, _, _, _, _, _))
      .WillOnce(Return(onecclSuccess));

  // The watchdog thread runs commGetAsyncError. Mock it to report an error eventually
  EXPECT_CALL(*xccl_mock_, commGetAsyncError(_, _))
      .WillOnce(DoAll(SetArgPointee<1>(onecclInternalError), Return(onecclSuccess)))
      .WillRepeatedly(Return(onecclNotImplemented)); // Fallback

  // Expect event to never be ready to force watchdog to look at async error
  EXPECT_CALL(*xpu_mock_, eventQuery(_))
      .WillOnce(Return(XPU_SUCCESS))
      .WillRepeatedly(Return(XPU_ERROR_NOT_READY));

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, true, AllReduceOptions());
  
  // Wait for the comm state to enter ERROR
  comm->waitTillError();
  
  EXPECT_EQ(work->status(), TorchWork::WorkStatus::INPROGRESS);
  EXPECT_THROW(comm->finalize(), std::runtime_error);
}

} // namespace torch::comms::test
