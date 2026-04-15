// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDAGuard.h>

#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"

class WindowAllGatherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto skipReason = shouldSkipRmaTest();
    if (!skipReason.empty()) {
      GTEST_SKIP() << skipReason;
    }

    wrapper_ = std::make_unique<TorchCommTestWrapper>();
    torchcomm_ = wrapper_->getTorchComm();
    rank_ = torchcomm_->getRank();
    num_ranks_ = torchcomm_->getSize();
    device_index_ = rank_ % at::cuda::device_count();
  }

  void TearDown() override {
    torchcomm_.reset();
    wrapper_.reset();
  }

  // Create a MemPool-allocated tensor for window registration
  std::pair<at::Tensor, std::unique_ptr<at::cuda::MemPool>> allocWinBuffer(
      int64_t numel,
      at::ScalarType dtype) {
    auto cuda_allocator =
        torch::comms::get_mem_allocator(torchcomm_->getBackend());
    auto mem_pool = std::make_unique<at::cuda::MemPool>(
        std::static_pointer_cast<
            c10::cuda::CUDACachingAllocator::CUDAAllocator>(cuda_allocator));
    c10::cuda::CUDACachingAllocator::beginAllocateToPool(
        mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

    auto options =
        at::TensorOptions().dtype(dtype).device(c10::kCUDA, device_index_);
    auto tensor = at::zeros({numel}, options);

    c10::cuda::CUDACachingAllocator::endAllocateToPool(
        mem_pool->device(), mem_pool->id());
    return {tensor, std::move(mem_pool)};
  }

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_{0};
  int num_ranks_{0};
  int device_index_{0};
};

TEST_F(WindowAllGatherTest, Basic) {
  const int64_t count = 1024;
  auto dtype = at::kFloat;
  auto options =
      at::TensorOptions().dtype(dtype).device(c10::kCUDA, device_index_);

  // Allocate window buffer: count * num_ranks elements
  auto [win_buf, mem_pool] = allocWinBuffer(count * num_ranks_, dtype);

  // Create window and register buffer
  torchcomm_->barrier(false);
  auto win = torchcomm_->new_window();
  win->tensor_register(win_buf);
  torchcomm_->barrier(false);

  // Initialize persistent allgather
  win->allgather_init();

  // Create send tensor: this rank's chunk filled with (rank + 1) to avoid
  // confusion with the zero-initialized window buffer
  at::Tensor sendbuf =
      at::ones({count}, options) * static_cast<float>(rank_ + 1);

  // Execute allgather multiple times
  const int num_iters = 5;
  for (int iter = 0; iter < num_iters; ++iter) {
    win->allgather(sendbuf, /*asyncOp=*/false);
  }

  // Barrier to ensure all ranks have completed
  torchcomm_->barrier(false);

  // Verify: each rank's chunk in win_buf should contain (r + 1)
  at::Tensor result = win_buf.cpu();
  for (int r = 0; r < num_ranks_; ++r) {
    at::Tensor chunk =
        result.index({at::indexing::Slice(r * count, (r + 1) * count)});
    at::Tensor expected = at::ones({count}) * static_cast<float>(r + 1);
    verifyTensorEquality(
        chunk, expected, "AllGather chunk for rank " + std::to_string(r));
  }

  // Cleanup
  win->allgather_destroy();
  win->tensor_deregister();
  win.reset();
  mem_pool.reset();
}

TEST_F(WindowAllGatherTest, MultipleIterationsWithChangingData) {
  const int64_t count = 512;
  auto dtype = at::kFloat;
  auto options =
      at::TensorOptions().dtype(dtype).device(c10::kCUDA, device_index_);

  auto [win_buf, mem_pool] = allocWinBuffer(count * num_ranks_, dtype);

  torchcomm_->barrier(false);
  auto win = torchcomm_->new_window();
  win->tensor_register(win_buf);
  torchcomm_->barrier(false);

  win->allgather_init();

  // Each iteration sends different data
  const int num_iters = 3;
  for (int iter = 0; iter < num_iters; ++iter) {
    float val = static_cast<float>(rank_ * 100 + iter);
    at::Tensor sendbuf = at::ones({count}, options) * val;
    win->allgather(sendbuf, /*asyncOp=*/false);

    torchcomm_->barrier(false);

    // Verify this iteration's results
    at::Tensor result = win_buf.cpu();
    for (int r = 0; r < num_ranks_; ++r) {
      float expected_val = static_cast<float>(r * 100 + iter);
      at::Tensor chunk =
          result.index({at::indexing::Slice(r * count, (r + 1) * count)});
      at::Tensor expected = at::ones({count}) * expected_val;
      verifyTensorEquality(
          chunk,
          expected,
          "AllGather iter " + std::to_string(iter) + " rank " +
              std::to_string(r));
    }
  }

  win->allgather_destroy();
  win->tensor_deregister();
  win.reset();
  mem_pool.reset();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
