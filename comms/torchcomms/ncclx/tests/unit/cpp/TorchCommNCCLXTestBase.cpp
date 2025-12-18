// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/StoreManager.hpp"

#include "TorchCommNCCLXTestBase.hpp"

namespace torch {
namespace comms {
namespace test {

void TorchCommNCCLXTest::SetUp() {
  // Force the global instance to be created on the CPU device
  auto hook_mock = std::make_unique<CachingAllocatorHookMock>();
  hook_mock->setupDefaultBehaviors();

  mock_hook_ = hook_mock.get();
  CachingAllocatorHook::setInstance(std::move(hook_mock));

  // Create hash store for communication
  store_ = c10::make_intrusive<c10d::HashStore>();
  StoreManager::get().injectMockStore(store_);

  // Set up device. make it the cpu device because we're mocking cuda.
  device_ = at::Device(at::DeviceType::CPU, 0);

  // Set timeout to 2 seconds for tests
  default_options_ = CommOptions();
  default_options_.timeout = std::chrono::milliseconds(2000);
  default_options_.abort_process_on_timeout_or_error = false;
  default_options_.store = store_;

  // Create fresh mocks for each test
  cuda_mock_ = std::make_shared<NiceMock<CudaMock>>();
  nccl_mock_ = std::make_shared<NiceMock<NcclxMock>>();

  // Set up environment variables for different test scenarios
  // Default: rank 0, size 2
  setupRankAndSize(0, 2);
}

void TorchCommNCCLXTest::TearDown() {
  // Clean up environment variables
  unsetenv("TORCHCOMM_RANK");
  unsetenv("TORCHCOMM_SIZE");
  unsetenv("TORCHCOMM_ABORT_ON_ERROR");
  unsetenv("TORCHCOMM_TIMEOUT_SECONDS");

  // Clear the instance data
  CachingAllocatorHook::getInstance().clear();

  // Reset the instance to null to release the mock object
  CachingAllocatorHook::setInstance(nullptr);
}

void TorchCommNCCLXTest::setupRankAndSize(int rank, int size) {
  setenv("TORCHCOMM_RANK", std::to_string(rank).c_str(), 1);
  setenv("TORCHCOMM_SIZE", std::to_string(size).c_str(), 1);

  ON_CALL(*nccl_mock_, commUserRank(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(rank), Return(ncclSuccess)));
  ON_CALL(*nccl_mock_, commCount(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(size), Return(ncclSuccess)));
}

void TorchCommNCCLXTest::setOptionsEnvironmentVariables(
    bool abort_on_error,
    float timeout_secs) {
  setenv("TORCHCOMM_ABORT_ON_ERROR", abort_on_error ? "1" : "0", 1);
  setenv("TORCHCOMM_TIMEOUT_SECONDS", std::to_string(timeout_secs).c_str(), 1);
}

void TorchCommNCCLXTest::setupEventsForWork(
    TorchCommNCCLXTest::TestTorchCommNCCLX& torchcomm,
    size_t numWork) {
  for (size_t i = 0; i < numWork; i++) {
    // Create mock event pointers using individual static dummy variables
    static uintptr_t start_dummy{0x5000}, end_dummy{0x6000};
    cudaEvent_t start_event = reinterpret_cast<cudaEvent_t>(&start_dummy + i);
    cudaEvent_t end_event = reinterpret_cast<cudaEvent_t>(&end_dummy + i);
    EXPECT_CALL(*cuda_mock_, eventCreateWithFlags(_, _))
        .WillOnce(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(start_event)),
            Return(cudaSuccess)))
        .WillOnce(DoAll(
            SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(end_event)),
            Return(cudaSuccess)));
    EXPECT_CALL(
        *cuda_mock_, eventRecord(torchcomm.getAsyncDependencyEvent(), _));
    EXPECT_CALL(*cuda_mock_, eventRecord(start_event, _));
    EXPECT_CALL(*cuda_mock_, eventRecord(end_event, _));
    work_events_.emplace_back(start_event, end_event);
  }
}

// Helper method to create a TorchCommNCCLX with mocked APIs
std::shared_ptr<TorchCommNCCLXTest::TestTorchCommNCCLX>
TorchCommNCCLXTest::createMockedTorchComm() {
  auto comm = std::make_shared<TestTorchCommNCCLX>();

  // Inject the mocks
  comm->setCudaApi(cuda_mock_);
  comm->setNcclApi(nccl_mock_);

  return comm;
}

void TorchCommNCCLXTest::setupCCAExpectations(
    int times_register,
    int times_deregister,
    int times_clear) {
  // Expect a registration call during init
  EXPECT_CALL(*mock_hook_, registerComm(_)).Times(times_register);

  // Expect a deregistration call during finalize, destruction or abort
  EXPECT_CALL(*mock_hook_, deregisterComm(_)).Times(times_deregister);

  // Expect a clear call during destruction
  EXPECT_CALL(*mock_hook_, clear()).Times(times_clear);
}

void TorchCommNCCLXTest::setupNormalDestruction(
    TorchCommNCCLXTest::TestTorchCommNCCLX& torchcomm,
    int times) {
  // Expect finalize to clean up resources
  EXPECT_CALL(*cuda_mock_, eventDestroy(_))
      .Times(0); // No events in pool initially

  // free the barrier buffer memory
  EXPECT_CALL(*cuda_mock_, free(_)).WillOnce(Return(cudaSuccess));

  EXPECT_CALL(*cuda_mock_, eventDestroy(torchcomm.getAsyncDependencyEvent()));

  // Destroy the internal stream
  EXPECT_CALL(*cuda_mock_, streamDestroy(_))
      .Times(times)
      .WillRepeatedly(Return(cudaSuccess));

  // Destroy the NCCL communicator
  EXPECT_CALL(*nccl_mock_, commDestroy(_))
      .Times(times)
      .WillRepeatedly(Return(ncclSuccess));
}

// Helper method to create a tensor for testing
at::Tensor TorchCommNCCLXTest::createTestTensor(
    const std::vector<int64_t>& sizes,
    const at::ScalarType type) {
  return at::ones(sizes, at::TensorOptions().device(*device_).dtype(type));
}

void TorchCommNCCLXTest::setupWorkToTimeout(WorkEvent& work_event) {
  EXPECT_CALL(*cuda_mock_, eventQuery(work_event.start_event))
      .WillOnce(Return(cudaSuccess)); // start event succeeds

  EXPECT_CALL(*cuda_mock_, eventQuery(work_event.end_event))
      .WillRepeatedly(Return(cudaErrorNotReady)); // end event fails

  EXPECT_CALL(*nccl_mock_, commAbort(reinterpret_cast<ncclComm_t>(0x3000)))
      .WillOnce(Return(ncclSuccess));
}

void TorchCommNCCLXTest::setupWorkToError(WorkEvent& work_event) {
  EXPECT_CALL(*cuda_mock_, eventQuery(work_event.start_event))
      .WillOnce(Return(cudaSuccess)); // start event succeeds

  EXPECT_CALL(*cuda_mock_, eventQuery(work_event.end_event))
      .WillOnce(Return(cudaErrorInvalidValue)); // end event fails

  // Second send operation should detect error and call commAbort
  EXPECT_CALL(*nccl_mock_, commGetAsyncError(_, _))
      .WillRepeatedly(
          DoAll(SetArgPointee<1>(ncclInternalError), Return(ncclSuccess)));

  EXPECT_CALL(*nccl_mock_, getErrorString(ncclInternalError))
      .WillRepeatedly(Return("internal error"));

  EXPECT_CALL(*nccl_mock_, commAbort(reinterpret_cast<ncclComm_t>(0x3000)))
      .WillOnce(Return(ncclSuccess));
}

c10::cuda::CUDACachingAllocator::TraceEntry
TorchCommNCCLXTest::createAllocation(uintptr_t addr) {
  return c10::cuda::CUDACachingAllocator::TraceEntry(
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC,
      0, // device
      static_cast<size_t>(addr), // addr
      1024, // size
      nullptr, // stream
      c10::cuda::MempoolId_t{0, 0}, // mempool
      c10::approx_time_t{0} // time
  );
}

c10::cuda::CUDACachingAllocator::TraceEntry
TorchCommNCCLXTest::createDeallocation(uintptr_t addr) {
  return c10::cuda::CUDACachingAllocator::TraceEntry(
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE,
      0, // device
      static_cast<size_t>(addr), // addr
      1024, // size
      nullptr, // stream
      c10::cuda::MempoolId_t{0, 0}, // mempool
      c10::approx_time_t{0} // time
  );
}

} // namespace test
} // namespace comms
} // namespace torch
