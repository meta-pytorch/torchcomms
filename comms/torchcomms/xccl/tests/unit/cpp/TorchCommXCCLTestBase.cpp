#include "TorchCommXCCLTestBase.hpp"

namespace google {
  bool IsGoogleLoggingInitialized();
}

// Satisfy linker for old glog namespace expected by TorchCommLogging.hpp
// since we reverted the change that used the new namespace to keep non-XCCL files untouched.
namespace google::glog_internal_namespace_ {
  bool IsGoogleLoggingInitialized();
  bool IsGoogleLoggingInitialized() {
    return ::google::IsGoogleLoggingInitialized();
  }
}

namespace torch::comms::test {

void TorchCommXCCLTest::SetUp() {
  xpu_mock_ = std::make_shared<NiceMock<XpuMock>>();
  xccl_mock_ = std::make_shared<NiceMock<XcclMock>>();

  store_ = c10::make_intrusive<c10d::HashStore>();

  device_ = at::Device(at::DeviceType::XPU, 0);

  default_options_.store = store_;
  default_options_.timeout = std::chrono::milliseconds(2000);
}

void TorchCommXCCLTest::TearDown() {
  unsetenv("TORCHCOMM_TIMEOUT_SECS");
  unsetenv("TORCHCOMM_ABORT_PROCESS_ON_TIMEOUT_OR_ERROR");
}

void TorchCommXCCLTest::setupRankAndSize(int rank, int size) {
  setenv("RANK", std::to_string(rank).c_str(), 1);
  setenv("WORLD_SIZE", std::to_string(size).c_str(), 1);
}

void TorchCommXCCLTest::setOptionsEnvironmentVariables(
    bool abort_on_error,
    uint64_t timeout_secs) {
  if (abort_on_error) {
    setenv("TORCHCOMM_ABORT_ON_ERROR", "1", 1);
  } else {
    setenv("TORCHCOMM_ABORT_ON_ERROR", "0", 1);
  }
  setenv(
      "TORCHCOMM_TIMEOUT_SECONDS", std::to_string(timeout_secs).c_str(), 1);
}

void TorchCommXCCLTest::setupEventsForWork(
    TestTorchCommXCCL& torchcomm,
    size_t numWork) {
  work_events_.clear();

  // Create start and end events for each work item
  for (size_t i = 0; i < numWork; ++i) {
    auto start = std::make_shared<xpuEvent_t>();
    auto end = std::make_shared<xpuEvent_t>();
    work_events_.push_back(WorkEventXCCL(start, end));
  }

  // Set up expectation for dependency_event creation
  EXPECT_CALL(*xpu_mock_, eventCreateWithFlags(_, _))
      .WillRepeatedly(DoAll(
          Invoke([](xpuEvent_t& event, unsigned int flags) {
             return XPU_SUCCESS;
          }),
          Return(XPU_SUCCESS)));

  size_t work_index = 0;

  // Set up expectations for start and end event creation per work
  EXPECT_CALL(*xpu_mock_, eventCreateWithFlags(_, _))
      .WillRepeatedly(DoAll(
          Invoke([&work_index, this](xpuEvent_t& event, unsigned int flags) {
             return XPU_SUCCESS;
          }),
          Return(XPU_SUCCESS)));
}

std::shared_ptr<TorchCommXCCLTest::TestTorchCommXCCL>
TorchCommXCCLTest::createMockedTorchComm() {
  auto comm = std::make_shared<TestTorchCommXCCL>();
  comm->setXpuApi(xpu_mock_);
  comm->setXcclApi(xccl_mock_);
  return comm;
}

void TorchCommXCCLTest::setupNormalDestruction(
    TestTorchCommXCCL& torchcomm,
    int times) {
  // Wait for queue to be empty
  while (!torchcomm.getStreamWorkQueues().empty()) {
    std::this_thread::yield();
  }

  EXPECT_CALL(*xccl_mock_, commDestroy(_))
      .Times(times)
      .WillRepeatedly(Return(onecclSuccess));
}

at::Tensor TorchCommXCCLTest::createTestTensor(
    const std::vector<int64_t>& sizes,
    const at::ScalarType type) {
  return at::ones(
      sizes,
      at::TensorOptions().device(*device_).dtype(type).requires_grad(
          false));
}

void TorchCommXCCLTest::setupWorkToTimeout(WorkEventXCCL& work_event) {
  EXPECT_CALL(*xpu_mock_, eventQuery(testing::Ref(*work_event.end_event)))
      .WillRepeatedly(Return(XPU_ERROR_NOT_READY));
}

void TorchCommXCCLTest::setupWorkToError(WorkEventXCCL& work_event) {
  EXPECT_CALL(*xpu_mock_, eventQuery(testing::Ref(*work_event.end_event)))
      .WillRepeatedly(Return(XPU_ERROR_INVALID_VALUE));
}

} // namespace torch::comms::test
