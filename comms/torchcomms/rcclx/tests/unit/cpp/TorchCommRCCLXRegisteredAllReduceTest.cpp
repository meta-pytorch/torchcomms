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
constexpr size_t kHalfMiB = 512 * 1024;
constexpr int64_t kBFloat16Elements = kHalfMiB / 2;

class TestableTorchCommRCCLX : public TorchCommRCCLX {
 public:
  int createWorkCalls() const {
    return create_work_calls_;
  }

 protected:
  c10::intrusive_ptr<TorchWorkRCCLX> createWork(
      hipStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors = {}) override {
    ++create_work_calls_;
    return TorchCommRCCLX::createWork(stream, timeout, inputTensors);
  }

  c10::intrusive_ptr<TorchWorkRCCLX> createWork(
      hipStream_t stream,
      std::chrono::milliseconds timeout,
      const at::Tensor& inputTensor) override {
    ++create_work_calls_;
    return TorchCommRCCLX::createWork(stream, timeout, inputTensor);
  }

 private:
  int create_work_calls_{0};
};

class TorchCommRCCLXRegisteredAllReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<c10d::HashStore>();
    device_ = at::Device(at::DeviceType::CPU, 0);

    rcclx_mock_ = std::make_shared<NiceMock<RcclxMock>>();
    hip_mock_ = std::make_shared<NiceMock<HipMock>>();

    setenv("TORCHCOMM_RANK", "0", 1);
    setenv("TORCHCOMM_SIZE", "4", 1);

    setupDefaultMockBehaviors();
  }

  void TearDown() override {
    unsetenv("TORCHCOMM_RANK");
    unsetenv("TORCHCOMM_SIZE");
  }

  void setupDefaultMockBehaviors() {
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
    ON_CALL(*hip_mock_, getCurrentCUDAStream(_))
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
        .WillByDefault(DoAll(SetArgPointee<1>(4), Return(ncclSuccess)));
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

    ON_CALL(*rcclx_mock_, registeredAllReduceAbiVersion())
        .WillByDefault(Return(1));
    ON_CALL(*rcclx_mock_, registeredAllReduceInit(_, _, _, _, _))
        .WillByDefault(
            DoAll(SetArgPointee<4>(registered_request_), Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, registeredAllReduceExec(_, _, _, _, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, registeredAllReduceFinalize(_, _))
        .WillByDefault(Return(ncclSuccess));
  }

  std::shared_ptr<TestableTorchCommRCCLX> createAndInitComm() {
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

  at::Tensor makeRegisteredTensor(at::ScalarType dtype = at::kBFloat16) {
    return at::empty({kBFloat16Elements}, at::TensorOptions().dtype(dtype));
  }

  hipStream_t internal_stream_ = reinterpret_cast<hipStream_t>(0x1000);
  hipStream_t current_stream_ = reinterpret_cast<hipStream_t>(0x3000);
  void* registered_request_ = reinterpret_cast<void*>(0x8000);

  c10::intrusive_ptr<c10d::Store> store_;
  at::Device device_{at::DeviceType::CPU, 0};
  std::shared_ptr<NiceMock<RcclxMock>> rcclx_mock_;
  std::shared_ptr<NiceMock<HipMock>> hip_mock_;
};

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, InitCallsRegisteredAllReduceApi) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();

  EXPECT_CALL(*rcclx_mock_, registeredAllReduceAbiVersion())
      .WillOnce(Return(1));
  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceInit(
          input.data_ptr(), output.data_ptr(), kHalfMiB, _, _))
      .WillOnce(
          DoAll(SetArgPointee<4>(registered_request_), Return(ncclSuccess)));

  auto request = comm->registered_all_reduce(input, output, kHalfMiB);
  EXPECT_FALSE(request->isClosed());
  EXPECT_EQ(request->capacityBytes(), kHalfMiB);
  EXPECT_TRUE(request->input().is_same(input));
  EXPECT_TRUE(request->output().is_same(output));
  request->close();
  comm->finalize();
}

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, StrongTensorLifetime) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  const void* inputPtr = input.data_ptr();
  void* outputPtr = output.data_ptr();

  auto request = comm->registered_all_reduce(input, output, kHalfMiB);
  input = at::Tensor();
  output = at::Tensor();

  EXPECT_TRUE(request->input().defined());
  EXPECT_TRUE(request->output().defined());
  EXPECT_EQ(request->input().data_ptr(), inputPtr);
  EXPECT_EQ(request->output().data_ptr(), outputPtr);
  request->close();
  comm->finalize();
}

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, RejectsInvalidInitArgs) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  auto floatOutput = makeRegisteredTensor(at::kFloat);
  auto sharedStorage = at::empty(
      {kBFloat16Elements * 2}, at::TensorOptions().dtype(at::kBFloat16));
  auto overlappingInput = sharedStorage.narrow(0, 0, kBFloat16Elements);
  auto overlappingOutput =
      sharedStorage.narrow(0, kBFloat16Elements / 2, kBFloat16Elements);

  EXPECT_THROW(comm->registered_all_reduce(input, input, kHalfMiB), c10::Error);
  EXPECT_THROW(
      comm->registered_all_reduce(
          overlappingInput, overlappingOutput, kHalfMiB),
      c10::Error);
  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB - 1), c10::Error);
  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB + 1), c10::Error);
  EXPECT_THROW(
      comm->registered_all_reduce(input, floatOutput, kHalfMiB), c10::Error);

  RegisteredAllReduceOptions hintOptions;
  hintOptions.hints.emplace("unsupported", "true");
  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB, hintOptions),
      c10::Error);

  RegisteredAllReduceOptions timeoutOptions;
  timeoutOptions.timeout = std::chrono::milliseconds(1);
  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB, timeoutOptions),
      c10::Error);

  comm->finalize();
}

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, ExecUsesCurrentStreamAndNoWork) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  auto request = comm->registered_all_reduce(input, output, kHalfMiB);

  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceExec(
          input.data_ptr(),
          output.data_ptr(),
          kBFloat16Elements,
          ncclBfloat16,
          ncclSum,
          current_stream_,
          registered_request_))
      .Times(2)
      .WillRepeatedly(Return(ncclSuccess));

  request->all_reduce(input, ReduceOp::SUM, output);
  request->all_reduce(input, ReduceOp::SUM, output);

  EXPECT_EQ(comm->createWorkCalls(), 0);
  request->close();
  comm->finalize();
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    PinsExecutionAndFinalizationToOneStream) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  auto request = comm->registered_all_reduce(input, output, kHalfMiB);
  const hipStream_t otherStream = reinterpret_cast<hipStream_t>(0x4000);

  EXPECT_CALL(*hip_mock_, getCurrentCUDAStream(_))
      .WillOnce(Return(current_stream_))
      .WillOnce(Return(otherStream));
  EXPECT_CALL(
      *rcclx_mock_, registeredAllReduceExec(_, _, _, _, _, current_stream_, _))
      .Times(1)
      .WillOnce(Return(ncclSuccess));
  EXPECT_CALL(
      *rcclx_mock_, registeredAllReduceExec(_, _, _, _, _, otherStream, _))
      .Times(0);
  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceFinalize(registered_request_, current_stream_))
      .Times(1)
      .WillOnce(Return(ncclSuccess));

  request->all_reduce(input, ReduceOp::SUM, output);
  EXPECT_THROW(request->all_reduce(input, ReduceOp::SUM, output), c10::Error);
  request->close();
  comm->finalize();
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    RejectsUnregisteredInputOrOutput) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  auto otherInput = makeRegisteredTensor();
  auto otherOutput = makeRegisteredTensor();
  auto request = comm->registered_all_reduce(input, output, kHalfMiB);

  EXPECT_THROW(
      request->all_reduce(otherInput, ReduceOp::SUM, output), c10::Error);
  EXPECT_THROW(
      request->all_reduce(input, ReduceOp::SUM, otherOutput), c10::Error);

  request->close();
  comm->finalize();
}

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, PropagatesExecRejections) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor(at::kFloat);
  auto output = makeRegisteredTensor(at::kFloat);
  auto request = comm->registered_all_reduce(input, output, kHalfMiB * 2);

  EXPECT_CALL(
      *rcclx_mock_, registeredAllReduceExec(_, _, _, ncclFloat32, _, _, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_THROW(
      request->all_reduce(input, ReduceOp::SUM, output), RCCLXException);

  request->close();
  comm->finalize();
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    RejectsPremulSumWithoutAllocatingOp) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  auto request = comm->registered_all_reduce(input, output, kHalfMiB);

  EXPECT_CALL(*rcclx_mock_, redOpCreatePreMulSum(_, _, _, _, _)).Times(0);
  EXPECT_THROW(
      request->all_reduce(input, ReduceOp::make_nccl_premul_sum(1.0), output),
      c10::Error);

  request->close();
  comm->finalize();
}

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, CloseFinalizesOnCurrentStream) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  auto request = comm->registered_all_reduce(input, output, kHalfMiB);

  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceFinalize(registered_request_, current_stream_))
      .Times(1)
      .WillOnce(Return(ncclSuccess));

  request->close();
  EXPECT_TRUE(request->isClosed());
  request->close();

  EXPECT_THROW(request->all_reduce(input, ReduceOp::SUM, output), c10::Error);
  comm->finalize();
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    InitFailureFinalizesReturnedCleanupHandle) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();

  EXPECT_CALL(*rcclx_mock_, registeredAllReduceInit(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<4>(registered_request_), Return(ncclInternalError)));
  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceFinalize(registered_request_, current_stream_))
      .Times(1)
      .WillOnce(Return(ncclSuccess));

  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB), RCCLXException);
  comm->finalize();
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    InitCleanupRetriesBeforePropagatingError) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();

  EXPECT_CALL(*rcclx_mock_, registeredAllReduceInit(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<4>(registered_request_), Return(ncclInternalError)));
  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceFinalize(registered_request_, current_stream_))
      .Times(2)
      .WillOnce(Return(ncclInternalError))
      .WillOnce(Return(ncclSuccess));

  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB), RCCLXException);
  comm->finalize();
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    InitCleanupFailureAbortsCommunicator) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();

  EXPECT_CALL(*rcclx_mock_, registeredAllReduceInit(_, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<4>(registered_request_), Return(ncclInternalError)));
  EXPECT_CALL(
      *rcclx_mock_,
      registeredAllReduceFinalize(registered_request_, current_stream_))
      .Times(3)
      .WillRepeatedly(Return(ncclInternalError));
  EXPECT_CALL(*rcclx_mock_, commAbort(_))
      .Times(1)
      .WillOnce(Return(ncclSuccess));

  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB), RCCLXException);
}

TEST_F(
    TorchCommRCCLXRegisteredAllReduceTest,
    RejectsReconfigurableCommunicator) {
  auto comm = std::make_shared<TestableTorchCommRCCLX>();
  comm->setRcclxApi(rcclx_mock_);
  comm->setHipApi(hip_mock_);

  CommOptions options;
  options.store = store_;
  options.enable_reconfigure = true;
  comm->init(device_, "dynamic_test_comm", options);

  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();
  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB), c10::Error);
}

TEST_F(TorchCommRCCLXRegisteredAllReduceTest, InitPropagatesRcclxErrors) {
  auto comm = createAndInitComm();
  auto input = makeRegisteredTensor();
  auto output = makeRegisteredTensor();

  EXPECT_CALL(*rcclx_mock_, registeredAllReduceInit(_, _, _, _, _))
      .WillOnce(Return(ncclInvalidArgument));

  EXPECT_THROW(
      comm->registered_all_reduce(input, output, kHalfMiB), RCCLXException);
  comm->finalize();
}

} // namespace torch::comms::test
