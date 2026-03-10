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
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XcclMock.hpp"
#include "comms/torchcomms/xccl/tests/unit/cpp/mocks/XpuMock.hpp"

namespace torch::comms::test {

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SetArgPointee;

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

    std::atomic<CommState>& commStateForTests() {
      return comm_state_;
    }

    bool isWorkQueueEmpty() {
      return workq_.empty();
    }

    xpuEvent_t& getAsyncDependencyEvent() {
      return dependency_event_.value();
    }

    bool testGetHighPriorityStream() const {
      return high_priority_stream_;
    }

    size_t testGetMaxEventPoolSize() const {
      return max_event_pool_size_;
    }
  };

  void setupEventsForWork(TestTorchCommXCCL& torchcomm, size_t numWork);

  std::shared_ptr<TestTorchCommXCCL> createMockedTorchComm();

  at::Tensor createTestTensor(
      const std::vector<int64_t>& sizes,
      const at::ScalarType type = at::kFloat);

  void setupWorkToTimeout();

  void setupWorkToError();

  c10::intrusive_ptr<c10d::Store> store_;
  std::optional<at::Device> device_;

  std::shared_ptr<NiceMock<XpuMock>> xpu_mock_;
  std::shared_ptr<NiceMock<XcclMock>> xccl_mock_;

  CommOptions default_options_;
};

} // namespace torch::comms::test
