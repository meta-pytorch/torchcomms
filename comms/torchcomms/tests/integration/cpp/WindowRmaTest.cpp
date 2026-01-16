// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "WindowRmaTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"

std::unique_ptr<TorchCommTestWrapper> WindowRmaTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

// Called before EACH test run - create fresh instance per test
void WindowRmaTest::SetUp() {
  // Check skip condition FIRST, before any initialization
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping RMA tests (RUN_RMA_TEST not set)";
  }

  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();

  // Get allocator using global function - obtained once and reused
  allocator_ = torch::comms::get_mem_allocator(torchcomm_->getBackend());
}

// Called after EACH test run - destroy instance
void WindowRmaTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

bool WindowRmaTest::checkIfSkip() {
  // Check RUN_RMA_TEST env var (set by BUCK for NCCLX + CTran configurations)
  const char* rma_env = getenv("RUN_RMA_TEST");
  if (!rma_env) {
    return true; // skip if not set
  }
  std::string rma_val(rma_env);
  std::transform(rma_val.begin(), rma_val.end(), rma_val.begin(), ::tolower);
  if (rma_val != "1" && rma_val != "true") {
    return true; // skip if not enabled
  }

  // RUN_RMA_TEST is set, don't skip
  return false;
}

// Test function for basic window allocation & put
void WindowRmaTest::testWindowPutBasic(
    int count,
    at::ScalarType dtype,
    bool async_op,
    bool async_signal) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Window Put with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create separate streams for put and wait operations
  auto put_stream = at::cuda::getStreamFromPool(false, device_index_);
  auto wait_stream = at::cuda::getStreamFromPool(false, device_index_);

  // Create tensor with different values based on rank
  at::Tensor input_tensor = createWindowRmaTensor(rank_, count, dtype);

  // Get global allocator for the backend and create MemPool for RDMA-compatible
  // memory
  auto cuda_allocator =
      torch::comms::get_mem_allocator(torchcomm_->getBackend());
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      static_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator*>(
          cuda_allocator.get()));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Allocate tensor from pool
  at::Tensor tensor;
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  tensor = at::ones({count * num_ranks_}, options);

  // End pool context immediately after allocation (Python: context exits here)
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Collective call to register tensor with window
  torchcomm_->barrier(false);
  auto win = torchcomm_->new_window();
  win->tensor_register(tensor);
  torchcomm_->barrier(false);

  auto dst_rank = (rank_ + 1) % num_ranks_;
  auto src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  // Perform multiple put operations to test repeated usage
  const int num_iterations = 10;
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    performWindowPutIteration(
        win,
        input_tensor,
        dst_rank,
        src_rank,
        count,
        async_op,
        async_signal,
        put_stream,
        wait_stream);
  }

  // Cleanup sequence
  // 1. Sync streams before deregistering
  put_stream.synchronize();
  wait_stream.synchronize();

  // 2. Deregister tensor from window (collective operation with internal
  // barriers)
  win->tensor_deregister();

  // 3. Explicitly destroy the window object
  win.reset();

  // 4. Reset memory pool (matching Python's del pool)
  mem_pool.reset();
}

// Helper function to create tensor for window RMA
at::Tensor WindowRmaTest::createWindowRmaTensor(
    int value,
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor tensor;

  // Initialize tensor based on dtype
  if (dtype == at::kFloat) {
    tensor = at::ones({count}, options) * static_cast<float>(value);
  } else if (dtype == at::kInt) {
    tensor = at::ones({count}, options) * static_cast<int>(value);
  } else if (dtype == at::kChar) {
    tensor = at::ones({count}, options) * static_cast<signed char>(value);
  }

  return tensor;
}

// Helper function to verify results
void WindowRmaTest::verifyWindowRmaResults(
    const at::Tensor& tensor,
    int value) {
  // Use verifyTensorEquality to compare tensor with expected tensor
  std::string description = "Window RMA with value " + std::to_string(value);
  verifyTensorEquality(tensor, value, description);
}

// Helper function to perform one iteration of window put/signal/wait/verify
void WindowRmaTest::performWindowPutIteration(
    std::shared_ptr<torch::comms::TorchCommWindow> win,
    const at::Tensor& input_tensor,
    int dst_rank,
    int src_rank,
    int count,
    bool async_op,
    bool async_signal,
    at::cuda::CUDAStream put_stream,
    at::cuda::CUDAStream wait_stream) {
  // Put the tensor to the Window of the next rank using put_stream
  {
    at::cuda::CUDAStreamGuard guard(put_stream);
    auto work = win->put(input_tensor, dst_rank, dst_rank * count, async_op);
    if (async_op) {
      work->wait();
    }
  }

  // Signal the next rank to proceed on put_stream
  {
    at::cuda::CUDAStreamGuard guard(put_stream);
    auto signal_work = win->signal(dst_rank, async_signal);
    if (async_signal) {
      signal_work->wait();
    }
  }

  // Wait for signal from the previous rank on wait_stream
  at::Tensor local_tensor;
  {
    at::cuda::CUDAStreamGuard guard(wait_stream);
    auto wait_signal_work = win->wait_signal(src_rank, async_signal);
    if (async_signal) {
      wait_signal_work->wait();
    }

    local_tensor = win->map_remote_tensor(rank_);
  }

  // Wait for wait_stream to complete before slicing
  wait_stream.synchronize();

  at::Tensor result_tensor = local_tensor.index(
      {at::indexing::Slice(rank_ * count, (rank_ + 1) * count)});

  // Verify results
  verifyWindowRmaResults(result_tensor.cpu(), src_rank);
}

TEST_P(WindowRmaTest, WindowPutBasic) {
  int count = std::get<0>(GetParam());
  at::ScalarType dtype = std::get<1>(GetParam());
  bool async_op = std::get<2>(GetParam());
  bool async_signal = std::get<3>(GetParam());
  testWindowPutBasic(count, dtype, async_op, async_signal);
}

INSTANTIATE_TEST_SUITE_P(
    WindowRmaTestParams,
    WindowRmaTest,
    ::testing::Combine(
        // count, dtype, async_op, async_signal
        ::testing::Values(4, 1024, 1024 * 1024),
        ::testing::Values(at::kFloat, at::kInt, at::kChar),
        ::testing::Values(true, false),
        ::testing::Values(true, false)),

    [](const ::testing::TestParamInfo<
        std::tuple<int, at::ScalarType, bool, bool>>& info) {
      int count = std::get<0>(info.param);
      at::ScalarType dtype = std::get<1>(info.param);
      std::string async_op = std::get<2>(info.param) ? "asyncOp" : "syncOp";
      std::string async_signal =
          std::get<3>(info.param) ? "asyncSignal" : "syncSignal";
      return "Count_" + std::to_string(count) + "_" + getDtypeName(dtype) +
          "_" + async_op + "_" + async_signal;
    });

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
