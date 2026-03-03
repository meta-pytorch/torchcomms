#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp>

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XpuMock.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XcclMock.hpp"

namespace torch::comms::test {

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SetArgPointee;

struct WorkEventXCCL {
  std::shared_ptr<xpuEvent_t> start_event;
  std::shared_ptr<xpuEvent_t> end_event;

  WorkEventXCCL(std::shared_ptr<xpuEvent_t> start, std::shared_ptr<xpuEvent_t> end)
      : start_event(start), end_event(end) {}
};

class TorchCommXCCLTest : public ::testing::Test {
 public:
  std::chrono::milliseconds getWorkTimeout(TorchWorkXCCL* work) {
    return work->getTimeout();
  }

 protected:
  void SetUp() override;

  void TearDown() override;

  void setupRankAndSize(int rank, int size);

  void setOptionsEnvironmentVariables(
      bool abort_on_error,
      uint64_t timeout_secs);

  class TestTorchCommXCCL : public TorchCommXCCL {
   public:
    virtual ~TestTorchCommXCCL() = default;

    TestTorchCommXCCL() : TorchCommXCCL() {}

    void waitTillCommState(CommState expectedState) {
      while (comm_state_ != expectedState) {
        std::this_thread::yield();
      }
    }

    void waitTillError() {
      while (comm_state_ != CommState::ERROR) {
      }
    }

    void waitTillTimeout() {
      while (comm_state_ != CommState::TIMEOUT) {
      }
    }

    xpuEvent_t& getAsyncDependencyEvent() {
      return dependency_event_.value();
    }
    
    std::unordered_map<xpuStream_t, std::queue<c10::intrusive_ptr<TorchWorkXCCL>>>& getStreamWorkQueues() {
      return workq_.getStreamWorkQueues();
    }
  };

  void setupEventsForWork(TestTorchCommXCCL& torchcomm, size_t numWork);

  std::shared_ptr<TestTorchCommXCCL> createMockedTorchComm();

  void setupNormalDestruction(TestTorchCommXCCL& torchcomm, int times = 1);

  at::Tensor createTestTensor(
      const std::vector<int64_t>& sizes,
      const at::ScalarType type = at::kFloat);

  void setupWorkToTimeout(WorkEventXCCL& work_event);

  void setupWorkToError(WorkEventXCCL& work_event);

  c10::intrusive_ptr<c10d::Store> store_;
  std::optional<at::Device> device_;

  std::shared_ptr<NiceMock<XpuMock>> xpu_mock_;
  std::shared_ptr<NiceMock<XcclMock>> xccl_mock_;
  std::vector<WorkEventXCCL> work_events_;

  CommOptions default_options_;
};

} // namespace torch::comms::test
