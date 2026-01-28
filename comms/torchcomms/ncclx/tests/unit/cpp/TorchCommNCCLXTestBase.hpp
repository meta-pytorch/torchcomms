// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CachingAllocatorHookMock.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/CudaMock.hpp"
#include "comms/torchcomms/ncclx/tests/unit/cpp/mocks/NcclxMock.hpp"

namespace torch::comms::test {

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SetArgPointee;

struct WorkEvent {
  cudaEvent_t start_event;
  cudaEvent_t end_event;

  WorkEvent(cudaEvent_t start, cudaEvent_t end)
      : start_event(start), end_event(end) {}
};

class TorchCommNCCLXTest : public ::testing::Test {
 public:
  // Wrapper function to access private getTimeout() method
  // Since TorchCommNCCLXTest is a friend class of TorchWorkNCCLX,
  // it can access private members
  std::chrono::milliseconds getWorkTimeout(TorchWorkNCCLX* work) {
    return work->getTimeout();
  }

 protected:
  void SetUp() override;

  void TearDown() override;

  void setupRankAndSize(int rank, int size);

  void setOptionsEnvironmentVariables(
      bool abort_on_error,
      uint64_t timeout_secs);

  class TestTorchCommNCCLX : public TorchCommNCCLX {
   public:
    virtual ~TestTorchCommNCCLX() = default;

    TestTorchCommNCCLX() : TorchCommNCCLX() {}

    void waitTillCommState(CommState expectedState) {
      while (comm_state_ != expectedState) {
        // Use a very short yield instead of sleep to avoid test flakiness
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

    cudaEvent_t getAsyncDependencyEvent() const {
      return dependency_event_;
    }
  };

  void setupEventsForWork(TestTorchCommNCCLX& torchcomm, size_t numWork);

  // Helper method to create a TorchCommNCCLX with mocked APIs
  std::shared_ptr<TestTorchCommNCCLX> createMockedTorchComm();

  void setupCCAExpectations(
      int times_register,
      int times_deregister,
      int times_clear);

  void setupNormalDestruction(TestTorchCommNCCLX& torchcomm, int times = 1);

  // Helper method to create a tensor for testing
  at::Tensor createTestTensor(
      const std::vector<int64_t>& sizes,
      const at::ScalarType type = at::kFloat);

  void setupWorkToTimeout(WorkEvent& work_event);

  void setupWorkToError(WorkEvent& work_event);
  c10::cuda::CUDACachingAllocator::TraceEntry createAllocation(uintptr_t addr);
  c10::cuda::CUDACachingAllocator::TraceEntry createDeallocation(
      uintptr_t addr);

  c10::intrusive_ptr<c10d::Store> store_;
  std::optional<at::Device> device_;

  // Raw pointers to mocks for setting expectations
  std::shared_ptr<NiceMock<CudaMock>> cuda_mock_;
  std::shared_ptr<NiceMock<NcclxMock>> nccl_mock_;
  std::vector<WorkEvent> work_events_;

  CommOptions default_options_;
  CachingAllocatorHookMock* mock_hook_{};
};

} // namespace torch::comms::test
