// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/mccl/McclTypes.h"
#include "comms/mccl/tests/MockMcclComm.h"
#include "comms/mccl/tests/MockWorkHandle.h"
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"
#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"
#include "comms/torchcomms/mccl/tests/unit/cpp/MockCudaApi.h"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

namespace torch::comms::test {

enum class TorchWorkCreationMode {
  SINGLE_TENSOR,
  VECTOR_TENSOR,
  MOVE_SINGLE_TENSOR,
  MOVE_VECTOR_TENSOR,
};

class TorchWorkMCCLTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  std::unique_ptr<mccl::testing::MockWorkHandle> createMockWorkHandle() {
    return std::make_unique<mccl::testing::MockWorkHandle>();
  }

  TorchWorkMCCL testCheckStatusSetsCompletedOnSuccess(
      TorchWorkCreationMode mode);
  TorchWorkMCCL testCheckStatusReturnsInProgressWhenNotReady(
      TorchWorkCreationMode mode);
  TorchWorkMCCL testCheckStatusSetsErrorOnFailure(TorchWorkCreationMode mode);
  TorchWorkMCCL testCheckStatusSetsTimedOutOnTimeout(
      TorchWorkCreationMode mode);

  void testMoveCtorTransfersCompletedStatus(TorchWorkCreationMode mode);
  void testMoveCtorTransfersErrorStatus(TorchWorkCreationMode mode);
};

TorchWorkMCCL createWork(
    TorchWorkCreationMode mode,
    std::unique_ptr<mccl::IWorkHandle> workHandle) {
  // Use a large timeout so timeout detection doesn't interfere with tests
  auto timeout = std::chrono::milliseconds(600000);
  if (mode == TorchWorkCreationMode::SINGLE_TENSOR) {
    return TorchWorkMCCL(
        nullptr,
        nullptr,
        at::Tensor{},
        at::Tensor{},
        std::move(workHandle),
        timeout);
  } else if (mode == TorchWorkCreationMode::VECTOR_TENSOR) {
    std::vector<at::Tensor> inputTensors;
    std::vector<at::Tensor> outputTensors;
    return TorchWorkMCCL(
        nullptr,
        nullptr,
        inputTensors,
        outputTensors,
        std::move(workHandle),
        timeout);
  } else if (mode == TorchWorkCreationMode::MOVE_SINGLE_TENSOR) {
    auto originalWork = TorchWorkMCCL(
        nullptr,
        nullptr,
        at::Tensor{},
        at::Tensor{},
        std::move(workHandle),
        timeout);
    TorchWorkMCCL movedWork(std::move(originalWork));
    return movedWork;
  } else if (mode == TorchWorkCreationMode::MOVE_VECTOR_TENSOR) {
    std::vector<at::Tensor> inputTensors;
    std::vector<at::Tensor> outputTensors;
    auto originalWork = TorchWorkMCCL(
        nullptr,
        nullptr,
        inputTensors,
        outputTensors,
        std::move(workHandle),
        timeout);
    TorchWorkMCCL movedWork(std::move(originalWork));
    return movedWork;
  } else {
    throw std::runtime_error(
        fmt::format(
            "Invalid TorchWorkCreationMode {}", static_cast<int>(mode)));
  }
}

torch::comms::ReconfigureOptions makeDefaultOpts(uint64_t uuid = 42) {
  torch::comms::ReconfigureOptions opts;
  opts.uuid = uuid;
  opts.handles = std::vector<torch::comms::InitHandle>{"url0", "url1"};
  opts.timeout = std::chrono::milliseconds(1000);
  return opts;
}

std::shared_ptr<TorchCommMCCL> createInitializedComm() {
  auto mockComm = std::make_unique<mccl::testing::MockMcclComm>();
  auto* mockCommPtr = mockComm.get();

  auto mockWork = std::make_unique<mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mockWork, waitCpu());
  EXPECT_CALL(*mockWork, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  EXPECT_CALL(*mockCommPtr, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(mockWork))));
  EXPECT_CALL(*mockCommPtr, getInitURL()).WillOnce(Return("url0"));
  EXPECT_CALL(*mockCommPtr, getRankAssignment())
      .WillOnce(Return(mccl::RankAssignment({"url0", "url1"})));

  auto comm = std::make_shared<TorchCommMCCL>(
      std::move(mockComm), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  comm->reconfigure(makeDefaultOpts());

  return comm;
}

TorchWorkMCCL TorchWorkMCCLTest::testCheckStatusSetsCompletedOnSuccess(
    TorchWorkCreationMode mode) {
  auto mockWorkHandle = createMockWorkHandle();

  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult())
      .WillOnce(::testing::Return(successResult));

  TorchWorkMCCL work = createWork(mode, std::move(mockWorkHandle));

  auto status = work.checkStatus();

  EXPECT_EQ(status, TorchWork::WorkStatus::COMPLETED);
  EXPECT_TRUE(work.isCompleted());

  return work;
}

TorchWorkMCCL TorchWorkMCCLTest::testCheckStatusSetsErrorOnFailure(
    TorchWorkCreationMode mode) {
  auto mockWorkHandle = createMockWorkHandle();

  mccl::Result errorResult{.code = commInternalError, .message = "test error"};
  EXPECT_CALL(*mockWorkHandle, getResult())
      .WillOnce(::testing::Return(errorResult));

  TorchWorkMCCL work = createWork(mode, std::move(mockWorkHandle));

  auto status = work.checkStatus();

  EXPECT_EQ(status, TorchWork::WorkStatus::ERROR);
  EXPECT_FALSE(work.isCompleted());

  return work;
}

TorchWorkMCCL TorchWorkMCCLTest::testCheckStatusSetsTimedOutOnTimeout(
    TorchWorkCreationMode mode) {
  auto mockWorkHandle = createMockWorkHandle();

  mccl::Result timeoutResult{
      .code = commTimeout, .message = "operation timed out"};
  EXPECT_CALL(*mockWorkHandle, getResult())
      .WillOnce(::testing::Return(timeoutResult));

  TorchWorkMCCL work = createWork(mode, std::move(mockWorkHandle));

  EXPECT_EQ(work.checkStatus(), TorchWork::WorkStatus::TIMEDOUT);
  EXPECT_FALSE(work.isCompleted());

  return work;
}

TorchWorkMCCL TorchWorkMCCLTest::testCheckStatusReturnsInProgressWhenNotReady(
    TorchWorkCreationMode mode) {
  auto mockWorkHandle = createMockWorkHandle();

  EXPECT_CALL(*mockWorkHandle, getResult())
      .WillOnce(::testing::Return(std::nullopt));

  TorchWorkMCCL work = createWork(mode, std::move(mockWorkHandle));

  auto status = work.checkStatus();

  EXPECT_EQ(status, TorchWork::WorkStatus::INPROGRESS);
  EXPECT_FALSE(work.isCompleted());

  return work;
}

void TorchWorkMCCLTest::testMoveCtorTransfersCompletedStatus(
    TorchWorkCreationMode mode) {
  auto originalWork = testCheckStatusSetsCompletedOnSuccess(mode);

  TorchWorkMCCL movedWork(std::move(originalWork));

  // The moved work should have COMPLETED status
  EXPECT_EQ(movedWork.status(), TorchWork::WorkStatus::COMPLETED);
  EXPECT_TRUE(movedWork.isCompleted());
}

void TorchWorkMCCLTest::testMoveCtorTransfersErrorStatus(
    TorchWorkCreationMode mode) {
  auto originalWork = testCheckStatusSetsErrorOnFailure(mode);

  TorchWorkMCCL movedWork(std::move(originalWork));

  // The moved work should have ERROR status
  EXPECT_EQ(movedWork.status(), TorchWork::WorkStatus::ERROR);
  EXPECT_FALSE(movedWork.isCompleted());
}

class TorchWorkMCCLStateTransitionTest
    : public TorchWorkMCCLTest,
      public ::testing::WithParamInterface<TorchWorkCreationMode> {};
TEST_P(TorchWorkMCCLStateTransitionTest, CheckStatusSetsCompletedOnSuccess) {
  auto mode = GetParam();
  this->testCheckStatusSetsCompletedOnSuccess(mode);
}

TEST_P(TorchWorkMCCLStateTransitionTest, CheckStatusSetsErrorOnFailure) {
  auto mode = GetParam();
  this->testCheckStatusSetsErrorOnFailure(mode);
}

TEST_P(TorchWorkMCCLStateTransitionTest, CheckStatusSetsTimedOutOnTimeout) {
  auto mode = GetParam();
  this->testCheckStatusSetsTimedOutOnTimeout(mode);
}

TEST_P(
    TorchWorkMCCLStateTransitionTest,
    CheckStatusReturnsInProgressWhenNotReady) {
  auto mode = GetParam();
  this->testCheckStatusReturnsInProgressWhenNotReady(mode);
}

TEST_P(TorchWorkMCCLStateTransitionTest, MoveCtorTransfersCompletedStatus) {
  auto mode = GetParam();
  this->testMoveCtorTransfersCompletedStatus(mode);
}

TEST_P(TorchWorkMCCLStateTransitionTest, MoveCtorTransfersErrorStatus) {
  auto mode = GetParam();
  this->testMoveCtorTransfersErrorStatus(mode);
}

INSTANTIATE_TEST_SUITE_P(
    BasicTransitions,
    TorchWorkMCCLStateTransitionTest,
    ::testing::Values(
        TorchWorkCreationMode::SINGLE_TENSOR,
        TorchWorkCreationMode::VECTOR_TENSOR,
        TorchWorkCreationMode::MOVE_SINGLE_TENSOR,
        TorchWorkCreationMode::MOVE_VECTOR_TENSOR),
    [](const ::testing::TestParamInfo<
        TorchWorkMCCLStateTransitionTest::ParamType>& info) {
      switch (info.param) {
        case TorchWorkCreationMode::SINGLE_TENSOR:
          return "SingleTensor";
        case TorchWorkCreationMode::VECTOR_TENSOR:
          return "VectorTensor";
        case TorchWorkCreationMode::MOVE_SINGLE_TENSOR:
          return "MoveSingleTensor";
        case TorchWorkCreationMode::MOVE_VECTOR_TENSOR:
          return "MoveVectorTensor";
        default:
          throw std::runtime_error(
              fmt::format(
                  "Invalid TorchWorkCreationMode {}",
                  static_cast<int>(info.param)));
      }
    });

TEST_P(TorchWorkMCCLStateTransitionTest, WaitCPUSucceedsWithNullComm) {
  auto mockWorkHandle = createMockWorkHandle();

  // waitCPU() calls mcclWork_->waitCpu(), then checkStatus() which calls
  // getResult()
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult())
      .WillOnce(::testing::Return(successResult));

  auto mode = GetParam();
  TorchWorkMCCL work = createWork(mode, std::move(mockWorkHandle));

  EXPECT_NO_THROW(work.waitCPU());
  EXPECT_EQ(work.status(), TorchWork::WorkStatus::COMPLETED);
}

// Validate correct behavior in scenario in which a TorchWorkMCCL is created
// when the associated comm is INITIALIZED, then a failed re-reconfigure
// transitions the comm to UNINITIALIZED, then waitCPU() is called.
TEST_F(TorchWorkMCCLTest, WaitCPUSucceedsAfterCommBecomesUninitialized) {
  auto mockComm = std::make_unique<mccl::testing::MockMcclComm>();
  auto* mockCommPtr = mockComm.get();

  auto mockWorkSuccess = std::make_unique<mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mockWorkSuccess, waitCpu());
  EXPECT_CALL(*mockWorkSuccess, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{.code = commSuccess})));

  auto mockWorkFail = std::make_unique<mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mockWorkFail, waitCpu());
  EXPECT_CALL(*mockWorkFail, getResult())
      .WillRepeatedly(Return(
          std::optional<mccl::Result>(mccl::Result{
              .code = commInternalError, .message = "test failure"})));

  EXPECT_CALL(*mockCommPtr, reconfigure(_))
      .WillOnce(Return(::testing::ByMove(std::move(mockWorkSuccess))))
      .WillOnce(Return(::testing::ByMove(std::move(mockWorkFail))));
  EXPECT_CALL(*mockCommPtr, getInitURL()).WillRepeatedly(Return("url0"));
  EXPECT_CALL(*mockCommPtr, getRankAssignment())
      .WillRepeatedly(Return(mccl::RankAssignment({"url0", "url1"})));

  auto comm = std::make_shared<TorchCommMCCL>(
      std::move(mockComm), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  // Step 1: Reconfigure successfully
  comm->reconfigure(makeDefaultOpts());
  ASSERT_TRUE(comm->isInitialized());

  // Step 2: Create TorchWorkMCCL while comm is INITIALIZED
  auto mockWorkHandle = createMockWorkHandle();
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult()).WillOnce(Return(successResult));

  TorchWorkMCCL work(
      comm,
      nullptr,
      at::Tensor{},
      at::Tensor{},
      std::move(mockWorkHandle),
      std::chrono::milliseconds(600000));

  // Step 3: Failed reconfigure makes comm UNINITIALIZED
  comm->reconfigure(makeDefaultOpts(43));
  ASSERT_FALSE(comm->isInitialized());

  // Step 4: waitCPU() should succeed because tracingInfo_
  // was captured at construction.
  EXPECT_NO_THROW(work.waitCPU());
  EXPECT_EQ(work.status(), TorchWork::WorkStatus::COMPLETED);
}

// Validates the happy path: tracingInfo_ is populated from a real
// initialized comm with actual values.
TEST_F(TorchWorkMCCLTest, WaitCPUSucceedsWithInitializedComm) {
  auto comm = createInitializedComm();

  auto mockWorkHandle = createMockWorkHandle();
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult()).WillOnce(Return(successResult));

  TorchWorkMCCL work(
      comm,
      nullptr,
      at::Tensor{},
      at::Tensor{},
      std::move(mockWorkHandle),
      std::chrono::milliseconds(600000));

  EXPECT_NO_THROW(work.waitCPU());
  EXPECT_EQ(work.status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkMCCLTest, WaitCPUSucceedsWithUninitializedNonNullComm) {
  auto mockComm = std::make_unique<mccl::testing::MockMcclComm>();
  auto comm = std::make_shared<TorchCommMCCL>(
      std::move(mockComm), 0, 2, std::make_shared<NiceMock<MockCudaApi>>());

  auto mockWorkHandle = createMockWorkHandle();
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult()).WillOnce(Return(successResult));

  TorchWorkMCCL work(
      comm,
      nullptr,
      at::Tensor{},
      at::Tensor{},
      std::move(mockWorkHandle),
      std::chrono::milliseconds(600000));

  EXPECT_NO_THROW(work.waitCPU());
  EXPECT_EQ(work.status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkMCCLTest, WaitCpuWorkDelegatesToWaitCPU) {
  auto mockWorkHandle = createMockWorkHandle();
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult()).WillOnce(Return(successResult));

  TorchWorkMCCL work = createWork(
      TorchWorkCreationMode::SINGLE_TENSOR, std::move(mockWorkHandle));
  work.setCpuWork(true);

  EXPECT_NO_THROW(work.wait());
  EXPECT_EQ(work.status(), TorchWork::WorkStatus::COMPLETED);
}

TEST_F(TorchWorkMCCLTest, MoveCtorPreservesWaitCPUWithInitializedComm) {
  auto comm = createInitializedComm();

  auto mockWorkHandle = createMockWorkHandle();
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult()).WillOnce(Return(successResult));

  TorchWorkMCCL original(
      comm,
      nullptr,
      at::Tensor{},
      at::Tensor{},
      std::move(mockWorkHandle),
      std::chrono::milliseconds(600000));

  TorchWorkMCCL moved(std::move(original));
  EXPECT_NO_THROW(moved.waitCPU());
  EXPECT_EQ(moved.status(), TorchWork::WorkStatus::COMPLETED);
}

// Verifies TorchWorkMCCL uses cached comm metadata (not live accessors)
// when the communicator is non-null but UNINITIALIZED -- the production
// failure scenario after a failed reconfigure().
TEST(TorchWorkMCCLUninitializedCommTest, WaitCPUSucceedsWithUninitializedComm) {
  using ::testing::NiceMock;
  using ::testing::Return;

  // Create a TorchCommMCCL without calling init() -- initState_ stays
  // UNINITIALIZED, which is the state after a failed reconfigure().
  auto mockMcclComm = std::make_unique<NiceMock<mccl::testing::MockMcclComm>>();
  auto comm = std::make_shared<TorchCommMCCL>(std::move(mockMcclComm), 0, 2);

  // comm->isInitialized() is false; cached values should be sentinels
  ASSERT_FALSE(comm->isInitialized());

  auto mockWorkHandle = std::make_unique<mccl::testing::MockWorkHandle>();
  EXPECT_CALL(*mockWorkHandle, waitCpu());
  mccl::Result successResult{.code = commSuccess, .message = ""};
  EXPECT_CALL(*mockWorkHandle, getResult()).WillOnce(Return(successResult));

  // Construct TorchWorkMCCL with the uninitialized (but non-null) comm.
  // The constructor caches commSize_=0 and rank_=-1 because
  // comm->isInitialized() is false.
  auto timeout = std::chrono::milliseconds(600000);
  TorchWorkMCCL work(
      comm,
      cudaStream_t{0},
      at::Tensor{},
      at::Tensor{},
      std::move(mockWorkHandle),
      timeout);

  // waitCPU() creates a TracingGuard with the cached values -- must not
  // throw even though the comm is UNINITIALIZED.
  EXPECT_NO_THROW(work.waitCPU());
  EXPECT_EQ(work.status(), TorchWork::WorkStatus::COMPLETED);
}

// TODO(T249225560): add tests for tensor lifecycle checks

} // namespace torch::comms::test
