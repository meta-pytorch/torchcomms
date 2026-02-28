// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "WindowRmaTimeoutTest.hpp"

#include <functional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommOptions.hpp"

using torch::comms::test::RankExpectation;
using torch::comms::test::TimeoutTestHelper;
using ExecMode = TimeoutTestHelper::ExecMode;

std::string WindowRmaTimeoutTest::rmaTypeName(const RmaType type) {
  switch (type) {
    case RmaType::kWaitSignal:
      return "WaitSignal";
    case RmaType::kPutWaitSignal:
      return "PutWaitSignal";
  }
  return "Unknown";
}

namespace {
at::Tensor makeTestTensor(
    const std::vector<int64_t>& sizes,
    const c10::DeviceType device_type,
    const int rank) {
  const auto device_index = rank % at::cuda::device_count();
  return at::ones(
      sizes,
      at::TensorOptions().dtype(at::kFloat).device(device_type, device_index));
}
} // namespace

void WindowRmaTimeoutTest::childSetUp() {
  wrapper_ = std::make_unique<TorchCommTestWrapper>();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  if (isRunningOnCPU()) {
    device_type_ = c10::DeviceType::CPU;
  } else {
    device_type_ = c10::DeviceType::CUDA;
  }

  // Initialize window with memory pool allocation
  const auto cuda_allocator =
      torch::comms::get_mem_allocator(torchcomm_->getBackend());
  mem_pool_ = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          cuda_allocator));
  const auto device_index = rank_ % at::cuda::device_count();
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      device_index, mem_pool_->id(), [](cudaStream_t) { return true; });
  constexpr int kCount = 1024;
  window_tensor_ = at::ones(
      {kCount * num_ranks_},
      at::TensorOptions().dtype(at::kFloat).device(device_type_, device_index));
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      device_index, mem_pool_->id());

  torchcomm_->barrier(false);
  window_ = torchcomm_->new_window();
  window_->tensor_register(window_tensor_);
  torchcomm_->barrier(false);
}

void WindowRmaTimeoutTest::childTearDown() {
  // Tear down window before communicator
  if (window_) {
    try {
      window_->tensor_deregister();
    } catch (...) {
    }
    window_.reset();
  }
  window_tensor_ = at::Tensor();
  mem_pool_.reset();

  if (torchcomm_) {
    try {
      torchcomm_->finalize();
    } catch (...) {
    }
    torchcomm_.reset();
  }
}

void WindowRmaTimeoutTest::execute(
    const RmaType type,
    const bool asyncOp,
    const std::chrono::milliseconds timeout) {
  constexpr int kCount = 1024;
  const int dst = (rank_ + 1) % num_ranks_;
  const int src = (rank_ + num_ranks_ - 1) % num_ranks_;

  c10::intrusive_ptr<torch::comms::TorchWork> work;
  torch::comms::WaitSignalOptions opts;
  opts.timeout = timeout;

  switch (type) {
    case RmaType::kWaitSignal: {
      window_->signal(dst, false);
      work = window_->wait_signal(src, asyncOp, opts);
      break;
    }
    case RmaType::kPutWaitSignal: {
      auto input = makeTestTensor({kCount}, device_type_, rank_);
      window_->put(input, dst, rank_ * kCount, false);
      window_->signal(dst, false);
      work = window_->wait_signal(src, asyncOp, opts);
      break;
    }
  }
  if (asyncOp) {
    work->wait();
  }
}

void WindowRmaTimeoutTest::testTimeout(
    const RmaType type,
    const ExecMode mode) {
  // RMA window ops require ncclx backend with ctran enabled
  const auto skipReason = shouldSkipRmaTest();
  if (!skipReason.empty()) {
    GTEST_SKIP() << skipReason;
  }

  if (mode != ExecMode::kEager && isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph timeout tests not supported on CPU";
  }

  // Expected exit behavior per rank:
  // - Rank 0 exits cleanly or gets aborted
  // - Rank 1+ must be aborted with timeout log
  const std::vector<RankExpectation> expectations = {
      {.exitCode = 0, .signal = SIGABRT},
      {.signal = SIGABRT,
       .logMustContain = {"Aborting process due to timeout"}},
  };

  helper_.launch(
      rmaTypeName(type),
      num_ranks_,
      [&](int /*rank*/) {
        childSetUp();

        std::vector<std::function<void()>> ops;
        ops.reserve(kNumWarmup + 1);
        for (int i = 0; i < kNumWarmup; i++) {
          ops.emplace_back([&] { execute(type); });
        }
        if (rank_ != 0) {
          ops.emplace_back([&] { execute(type, true, kTimeout); });
        }
        helper_.exec(mode, ops);

        // rank 0 skips the last op so other ranks can timeout
        if (rank_ == 0) {
          std::this_thread::sleep_for(
              kRank0Sleep); // NOLINT(facebook-hte-BadCall-sleep_for)
          _exit(0);
        } else {
          childTearDown();
        }
      },
      expectations);
}
