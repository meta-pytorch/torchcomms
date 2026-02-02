// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"
#include "comms/torchcomms/rcclx/TorchCommRCCLXBootstrap.hpp"
#include "comms/torchcomms/rcclx/tests/unit/cpp/mocks/HipMock.hpp"
#include "comms/torchcomms/rcclx/tests/unit/cpp/mocks/RcclxMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch::comms::test {

constexpr std::chrono::seconds kTimeout{60};

// Testable subclass of TorchCommRCCLX that tracks createWork calls.
// This allows us to verify that createWork is called with the correct
// tensor parameters based on async_op value.
class TestableTorchCommRCCLX : public TorchCommRCCLX {
 public:
  // Track the last createWork call parameters
  struct CreateWorkCall {
    bool called = false;
    bool hasTensors = false;
    size_t tensorCount = 0;
    bool usedSingleTensorOverload = false;
  };

  CreateWorkCall lastCreateWorkCall;

  void resetTracking() {
    lastCreateWorkCall = CreateWorkCall{};
  }

  c10::intrusive_ptr<TorchWorkRCCLX> createWork(
      hipStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors = {}) override {
    lastCreateWorkCall.called = true;
    lastCreateWorkCall.hasTensors = !inputTensors.empty();
    lastCreateWorkCall.tensorCount = inputTensors.size();
    lastCreateWorkCall.usedSingleTensorOverload = false;
    return TorchCommRCCLX::createWork(stream, timeout, inputTensors);
  }

  c10::intrusive_ptr<TorchWorkRCCLX> createWork(
      hipStream_t stream,
      std::chrono::milliseconds timeout,
      const at::Tensor& inputTensor) override {
    lastCreateWorkCall.called = true;
    lastCreateWorkCall.hasTensors = inputTensor.defined();
    lastCreateWorkCall.tensorCount = inputTensor.defined() ? 1 : 0;
    lastCreateWorkCall.usedSingleTensorOverload = true;
    return TorchCommRCCLX::createWork(stream, timeout, inputTensor);
  }
};

// Verify the following:
// 1. async_op=true: Collective calls use internal stream for RCCLX operations
// 2. async_op=false: Collective calls use current stream for RCCLX operations
// 3. Work is properly created and returned from collectives
class TorchCommRCCLXWorkQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create hash store for communication
    store_ = c10::make_intrusive<c10d::HashStore>();

    // Set up device
    device_ = at::Device(at::DeviceType::CPU, 0);

    // Create mocks
    rcclx_mock_ = std::make_shared<NiceMock<RcclxMock>>();
    hip_mock_ = std::make_shared<NiceMock<HipMock>>();

    // Set environment variables for rank and size
    setenv("TORCHCOMM_RANK", "0", 1);
    setenv("TORCHCOMM_SIZE", "2", 1);

    setupDefaultMockBehaviors();
  }

  void TearDown() override {
    unsetenv("TORCHCOMM_RANK");
    unsetenv("TORCHCOMM_SIZE");
  }

  void setupDefaultMockBehaviors() {
    // HIP mock behaviors
    ON_CALL(*hip_mock_, setDevice(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, getDeviceCount(_))
        .WillByDefault(DoAll(SetArgPointee<0>(1), Return(hipSuccess)));
    ON_CALL(*hip_mock_, streamCreateWithPriority(_, _, _))
        .WillByDefault(
            DoAll(SetArgPointee<0>(internal_stream_), Return(hipSuccess)));
    ON_CALL(*hip_mock_, streamDestroy(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, eventCreate(_))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<hipEvent_t>(0x2000)),
            Return(hipSuccess)));
    ON_CALL(*hip_mock_, eventDestroy(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, eventRecord(_, _)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, eventQuery(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, streamSynchronize(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, streamWaitEvent(_, _, _))
        .WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, getCurrentHIPStreamMasqueradingAsCUDA(_))
        .WillByDefault(Return(current_stream_));
    ON_CALL(*hip_mock_, getStreamPriorityRange(_, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(0), SetArgPointee<1>(-1), Return(hipSuccess)));
    ON_CALL(*hip_mock_, malloc(_, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<void*>(0x4000)),
            Return(hipSuccess)));
    ON_CALL(*hip_mock_, free(_)).WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, getErrorString(_))
        .WillByDefault(Return("mock hip error"));
    ON_CALL(*hip_mock_, getDeviceProperties(_, _))
        .WillByDefault(Return(hipSuccess));
    ON_CALL(*hip_mock_, memGetInfo(_, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(1024 * 1024 * 1024),
            SetArgPointee<1>(2UL * 1024 * 1024 * 1024),
            Return(hipSuccess)));

    // RCCLX mock behaviors
    ncclUniqueId mock_id{};
    memset(&mock_id, 0x42, sizeof(mock_id));
    ON_CALL(*rcclx_mock_, getUniqueId(_))
        .WillByDefault(DoAll(SetArgPointee<0>(mock_id), Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, commInitRankConfig(_, _, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x5000)),
            Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, commDestroy(_)).WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, commAbort(_)).WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, commCount(_, _))
        .WillByDefault(DoAll(SetArgPointee<1>(2), Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, commUserRank(_, _))
        .WillByDefault(DoAll(SetArgPointee<1>(0), Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, commGetAsyncError(_, _))
        .WillByDefault(
            DoAll(SetArgPointee<1>(ncclSuccess), Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, groupStart()).WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, groupEnd()).WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, getErrorString(_))
        .WillByDefault(Return("mock nccl error"));
    ON_CALL(*rcclx_mock_, getLastError(_)).WillByDefault(Return(""));

    // Collective operation mocks
    ON_CALL(*rcclx_mock_, allReduce(_, _, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, bcast(_, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, reduce(_, _, _, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, allGather(_, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, reduceScatter(_, _, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, allToAll(_, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, send(_, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, recv(_, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
  }

  std::shared_ptr<TorchCommRCCLX> createAndInitComm() {
    // Store unique ID for bootstrap
    ncclUniqueId expected_id{};
    memset(&expected_id, 0x42, sizeof(expected_id));
    std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
    memcpy(id_vec.data(), &expected_id, sizeof(expected_id));
    std::string store_key = TorchCommRCCLXBootstrap::getRCCLXStoreKeyPrefix() +
        std::to_string(TorchCommRCCLXBootstrap::getRCCLXStoreKeyCounter());
    store_->set(store_key, id_vec);

    auto comm = std::make_shared<TorchCommRCCLX>();
    comm->setRcclxApi(rcclx_mock_);
    comm->setHipApi(hip_mock_);

    CommOptions options;
    options.store = store_;
    options.timeout = kTimeout;
    comm->init(device_, "test_comm", options);
    return comm;
  }

  // Create a testable comm that tracks createWork calls
  std::shared_ptr<TestableTorchCommRCCLX> createAndInitTestableComm() {
    ncclUniqueId expected_id{};
    memset(&expected_id, 0x42, sizeof(expected_id));
    std::vector<uint8_t> id_vec(sizeof(ncclUniqueId));
    memcpy(id_vec.data(), &expected_id, sizeof(expected_id));
    std::string store_key = TorchCommRCCLXBootstrap::getRCCLXStoreKeyPrefix() +
        std::to_string(TorchCommRCCLXBootstrap::getRCCLXStoreKeyCounter());
    store_->set(store_key, id_vec);

    auto comm = std::make_shared<TestableTorchCommRCCLX>();
    comm->setRcclxApi(rcclx_mock_);
    comm->setHipApi(hip_mock_);

    CommOptions options;
    options.store = store_;
    options.timeout = kTimeout;
    comm->init(device_, "test_comm", options);
    return comm;
  }

  // Different streams for async vs sync operations
  hipStream_t internal_stream_ = reinterpret_cast<hipStream_t>(0x1000);
  hipStream_t current_stream_ = reinterpret_cast<hipStream_t>(0x3000);

  c10::intrusive_ptr<c10d::Store> store_;
  at::Device device_{at::DeviceType::CPU, 0};
  std::shared_ptr<NiceMock<RcclxMock>> rcclx_mock_;
  std::shared_ptr<NiceMock<HipMock>> hip_mock_;
};

// Test that all_reduce with async_op=true uses internal stream
TEST_F(TorchCommRCCLXWorkQueueTest, AllReduceAsyncOpTrueUsesInternalStream) {
  // For async_op=true, allReduce should be called with internal_stream_
  EXPECT_CALL(*rcclx_mock_, allReduce(_, _, _, _, _, _, internal_stream_))
      .WillOnce(Return(ncclSuccess));

  auto comm = createAndInitComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/true);
  EXPECT_NE(work, nullptr);

  comm->finalize();
}

// Test that all_reduce with async_op=false uses current stream
TEST_F(TorchCommRCCLXWorkQueueTest, AllReduceAsyncOpFalseUsesCurrentStream) {
  // For async_op=false, allReduce should be called with current_stream_
  EXPECT_CALL(*rcclx_mock_, allReduce(_, _, _, _, _, _, current_stream_))
      .WillOnce(Return(ncclSuccess));

  auto comm = createAndInitComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);
  EXPECT_NE(work, nullptr);

  comm->finalize();
}

// Test that work returned from async operation can be waited on
TEST_F(TorchCommRCCLXWorkQueueTest, AsyncOpWorkCanBeWaited) {
  auto comm = createAndInitComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/true);
  EXPECT_NE(work, nullptr);

  // Wait should complete without errors
  EXPECT_NO_THROW(work->wait());

  comm->finalize();
}

// Test that work returned from sync operation can be waited on
TEST_F(TorchCommRCCLXWorkQueueTest, SyncOpWorkCanBeWaited) {
  auto comm = createAndInitComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);
  EXPECT_NE(work, nullptr);

  // Wait should complete without errors
  EXPECT_NO_THROW(work->wait());

  comm->finalize();
}

// ============================================================================
// Verify that tensors are conditionally passed to createWork()
// When async_op=true: tensors ARE stored (to keep them alive during async GPU
// ops) When async_op=false: tensors are NOT stored (caller waits inline, no
// need)
// ============================================================================

// Test that async_op=true stores tensors in the work object
TEST_F(TorchCommRCCLXWorkQueueTest, AsyncOpTrueStoresTensors) {
  auto comm = createAndInitComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/true);
  EXPECT_NE(work, nullptr);

  // Cast to TorchWorkRCCLX to access hasTensorsStored()
  auto* rcclx_work = static_cast<TorchWorkRCCLX*>(work.get());
  EXPECT_NE(rcclx_work, nullptr);

  //  async_op=true should store tensors to keep them alive
  EXPECT_TRUE(rcclx_work->hasTensorsStored())
      << "async_op=true should store tensors in work object";

  comm->finalize();
}

// Test that async_op=false does NOT store tensors in the work object
TEST_F(TorchCommRCCLXWorkQueueTest, AsyncOpFalseDoesNotStoreTensors) {
  auto comm = createAndInitComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);
  EXPECT_NE(work, nullptr);

  // Cast to TorchWorkRCCLX to access hasTensorsStored()
  auto* rcclx_work = static_cast<TorchWorkRCCLX*>(work.get());
  EXPECT_NE(rcclx_work, nullptr);

  //  async_op=false should NOT store tensors (unnecessary refs)
  EXPECT_FALSE(rcclx_work->hasTensorsStored())
      << "async_op=false should NOT store tensors in work object";

  comm->finalize();
}

// ============================================================================
// Verify createWork is called with correct tensor parameters
// These tests use TestableTorchCommRCCLX to intercept and verify createWork
// calls
// ============================================================================

// Test that createWork is called WITH tensor when async_op=true
TEST_F(TorchCommRCCLXWorkQueueTest, CreateWorkCalledWithTensorForAsyncOpTrue) {
  auto comm = createAndInitTestableComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  comm->resetTracking();
  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/true);
  EXPECT_NE(work, nullptr);

  // Verify createWork was called with tensor
  EXPECT_TRUE(comm->lastCreateWorkCall.called) << "createWork should be called";
  EXPECT_TRUE(comm->lastCreateWorkCall.hasTensors)
      << "async_op=true: createWork should receive tensor";
  EXPECT_EQ(comm->lastCreateWorkCall.tensorCount, 1)
      << "async_op=true: createWork should receive exactly 1 tensor";

  comm->finalize();
}

// Test that createWork is called WITHOUT tensor when async_op=false
TEST_F(
    TorchCommRCCLXWorkQueueTest,
    CreateWorkCalledWithoutTensorForAsyncOpFalse) {
  auto comm = createAndInitTestableComm();
  at::Tensor tensor = at::ones({4, 4}, at::kFloat);

  comm->resetTracking();
  auto work = comm->all_reduce(tensor, ReduceOp::SUM, /*async_op=*/false);
  EXPECT_NE(work, nullptr);

  // Verify createWork was called without tensor
  EXPECT_TRUE(comm->lastCreateWorkCall.called) << "createWork should be called";
  EXPECT_FALSE(comm->lastCreateWorkCall.hasTensors)
      << "async_op=false: createWork should NOT receive tensor";
  EXPECT_EQ(comm->lastCreateWorkCall.tensorCount, 0)
      << "async_op=false: createWork should receive 0 tensors";

  comm->finalize();
}

} // namespace torch::comms::test
