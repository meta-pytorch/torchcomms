#include "TorchCommXCCLTestBase.hpp"

namespace torch::comms::test {

void TorchCommXCCLTest::SetUp() {
  xpu_mock_ = std::make_shared<NiceMock<XpuMock>>();
  xccl_mock_ = std::make_shared<NiceMock<XcclMock>>();

  store_ = c10::make_intrusive<c10d::HashStore>();

  device_ = at::Device(at::DeviceType::CPU, 0);

  default_options_.store = store_;
  default_options_.timeout = std::chrono::milliseconds(2000);
  default_options_.abort_process_on_timeout_or_error = false;
}

void TorchCommXCCLTest::TearDown() {
  unsetenv("TORCHCOMM_RANK");
  unsetenv("TORCHCOMM_SIZE");
  unsetenv("TORCHCOMM_ABORT_ON_ERROR");
  unsetenv("TORCHCOMM_TIMEOUT_SECONDS");
  unsetenv("TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD");
}

void TorchCommXCCLTest::setupRankAndSize(int rank, int size) {
  setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
  setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);

  ON_CALL(*xccl_mock_, commUserRank(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(rank), Return(onecclSuccess)));
  ON_CALL(*xccl_mock_, commCount(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(size), Return(onecclSuccess)));
}

void TorchCommXCCLTest::setOptionsEnvironmentVariables(
    bool abort_on_error,
    uint64_t timeout_secs) {
  if (abort_on_error) {
    setenv("TORCHCOMM_ABORT_ON_ERROR", "1", 1);
  } else {
    setenv("TORCHCOMM_ABORT_ON_ERROR", "0", 1);
  }
  setenv("TORCHCOMM_TIMEOUT_SECONDS", std::to_string(timeout_secs).c_str(), 1);
}

void TorchCommXCCLTest::setupEventsForWork(
    TestTorchCommXCCL& torchcomm,
    size_t numWork) {
  for (size_t i = 0; i < numWork; ++i) {
    EXPECT_CALL(*xpu_mock_, eventCreateWithFlags(_, _))
        .WillOnce(Return(XPU_SUCCESS))
        .WillOnce(Return(XPU_SUCCESS));
    EXPECT_CALL(*xpu_mock_, eventRecord(_, _))
        .Times(3)
        .WillRepeatedly(Return(XPU_SUCCESS));
  }

  (void)torchcomm;
}

std::shared_ptr<TorchCommXCCLTest::TestTorchCommXCCL>
TorchCommXCCLTest::createMockedTorchComm() {
  auto comm = std::make_shared<TestTorchCommXCCL>();
  comm->setXpuApi(xpu_mock_);
  comm->setXcclApi(xccl_mock_);
  return comm;
}

at::Tensor TorchCommXCCLTest::createTestTensor(
    const std::vector<int64_t>& sizes,
    const at::ScalarType type) {
  return at::ones(
      sizes,
      at::TensorOptions().device(*device_).dtype(type).requires_grad(false));
}

void TorchCommXCCLTest::setupWorkToTimeout() {
  EXPECT_CALL(*xpu_mock_, eventQuery(_))
      .WillOnce(Return(XPU_SUCCESS))
      .WillRepeatedly(Return(XPU_ERROR_NOT_READY));
}

void TorchCommXCCLTest::setupWorkToError() {
  EXPECT_CALL(*xpu_mock_, eventQuery(_))
      .WillOnce(Return(XPU_SUCCESS))
      .WillOnce(Return(XPU_ERROR_INVALID_VALUE));
}

} // namespace torch::comms::test
