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

class TorchCommRCCLXAllGatherPTest : public ::testing::Test {
 protected:
  void SetUp() override {
    store_ = c10::make_intrusive<c10d::HashStore>();
    device_ = at::Device(at::DeviceType::CPU, 0);

    rcclx_mock_ = std::make_shared<NiceMock<RcclxMock>>();
    hip_mock_ = std::make_shared<NiceMock<HipMock>>();

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

    // Memory registration
    ON_CALL(*rcclx_mock_, commRegister(_, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<3>(reinterpret_cast<void*>(0x6000)),
            Return(ncclSuccess)));

    // Persistent AllGather defaults
    ON_CALL(*rcclx_mock_, allGatherInit(_, _, _, _, _, _, _))
        .WillByDefault(DoAll(
            SetArgPointee<6>(reinterpret_cast<void*>(0x7000)),
            Return(ncclSuccess)));
    ON_CALL(*rcclx_mock_, allGatherExec(_, _, _, _))
        .WillByDefault(Return(ncclSuccess));
    ON_CALL(*rcclx_mock_, pFree(_)).WillByDefault(Return(ncclSuccess));
  }

  std::shared_ptr<TorchCommRCCLX> createAndInitComm() {
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

  hipStream_t internal_stream_ = reinterpret_cast<hipStream_t>(0x1000);
  hipStream_t current_stream_ = reinterpret_cast<hipStream_t>(0x3000);

  c10::intrusive_ptr<c10d::Store> store_;
  at::Device device_{at::DeviceType::CPU, 0};
  std::shared_ptr<NiceMock<RcclxMock>> rcclx_mock_;
  std::shared_ptr<NiceMock<HipMock>> hip_mock_;
};

TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherInitCallsRcclxApi) {
  void* fake_request = reinterpret_cast<void*>(0x7000);

  EXPECT_CALL(*rcclx_mock_, allGatherInit(_, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgPointee<6>(fake_request), Return(ncclSuccess)));

  auto comm = createAndInitComm();
  at::Tensor output = at::ones({8}, at::kFloat);

  auto handle = comm->all_gather_p_init(output);
  EXPECT_EQ(handle, fake_request);

  comm->finalize();
}

TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherExecCallsRcclxApi) {
  auto comm = createAndInitComm();
  at::Tensor output = at::ones({8}, at::kFloat);

  auto handle = comm->all_gather_p_init(output);

  EXPECT_CALL(*rcclx_mock_, allGatherExec(_, _, _, handle))
      .WillOnce(Return(ncclSuccess));

  at::Tensor input = at::ones({4}, at::kFloat);
  auto work = comm->all_gather_p_exec(handle, input, /*async_op=*/true);
  EXPECT_NE(work, nullptr);

  comm->finalize();
}

TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherInitPassesHints) {
  AllGatherPInitOptions options;
  options.hints["key1"] = "value1";
  options.hints["key2"] = "value2";

  const RcclxHints expected_hints{{"key1", "value1"}, {"key2", "value2"}};

  EXPECT_CALL(*rcclx_mock_, allGatherInit(_, _, expected_hints, _, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<6>(reinterpret_cast<void*>(0x7000)),
          Return(ncclSuccess)));

  auto comm = createAndInitComm();
  at::Tensor output = at::ones({8}, at::kFloat);

  comm->all_gather_p_init(output, options);

  comm->finalize();
}

TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherInitHandlesError) {
  EXPECT_CALL(*rcclx_mock_, allGatherInit(_, _, _, _, _, _, _))
      .WillOnce(Return(ncclInternalError));

  auto comm = createAndInitComm();
  at::Tensor output = at::ones({8}, at::kFloat);

  EXPECT_THROW(comm->all_gather_p_init(output), RCCLXException);

  comm->finalize();
}

TEST_F(TorchCommRCCLXAllGatherPTest, AllGatherExecHandlesError) {
  auto comm = createAndInitComm();
  at::Tensor output = at::ones({8}, at::kFloat);
  auto handle = comm->all_gather_p_init(output);

  EXPECT_CALL(*rcclx_mock_, allGatherExec(_, _, _, handle))
      .WillOnce(Return(ncclInternalError));

  at::Tensor input = at::ones({4}, at::kFloat);

  EXPECT_THROW(
      comm->all_gather_p_exec(handle, input, /*async_op=*/true),
      RCCLXException);

  comm->finalize();
}

} // namespace torch::comms::test
