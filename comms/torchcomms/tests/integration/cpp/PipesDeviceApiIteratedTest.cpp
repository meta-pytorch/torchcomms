// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Iterated functional tests for TorchComm Device API — Pipes (IBGDA+NVLink).
// Key difference from NCCLx: uses DeviceWindowPipes type and monotonic signals
// only (no reset_signal).

#include "PipesDeviceApiIteratedTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "IteratedTestHelpers.hpp"
#include "PipesDeviceApiIteratedTestKernels.cuh"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"

using namespace torchcomms::device;
using namespace torchcomms::device::test;

// =============================================================================
// Setup / Teardown
// =============================================================================

void PipesDeviceApiIteratedTest::SetUp() {
  if (!shouldRunIteratedTest()) {
    GTEST_SKIP()
        << "Skipping iterated tests (RUN_DEVICE_ITERATED_TEST not set)";
  }
  const char* pipes_env = getenv("RUN_PIPES_DEVICE_API_TEST");
  if (!pipes_env) {
    GTEST_SKIP()
        << "Skipping Pipes iterated tests (RUN_PIPES_DEVICE_API_TEST not set)";
  }

  config_ = parseIteratedTestConfig();
  wrapper_ = std::make_unique<TorchCommTestWrapper>();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();
  allocator_ = torch::comms::get_mem_allocator(torchcomm_->getBackend());
}

void PipesDeviceApiIteratedTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

// =============================================================================
// Helpers (Pipes-specific: uses DeviceWindowPipes)
// =============================================================================

namespace {

struct PipesWindowSetup {
  std::unique_ptr<at::cuda::MemPool> mem_pool;
  at::Tensor win_tensor;
  at::Tensor src_tensor;
  std::shared_ptr<torch::comms::TorchCommWindow> win;
  DeviceWindowPipes* dev_win{nullptr};
  RegisteredBufferPipes src_buf{};
};

PipesWindowSetup createPipesWindowSetup(
    std::shared_ptr<torch::comms::TorchComm>& torchcomm,
    std::shared_ptr<c10::Allocator>& allocator,
    int device_index,
    int num_ranks,
    size_t count,
    int signal_count,
    int counter_count,
    int barrier_count) {
  PipesWindowSetup s;

  s.mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      s.mem_pool->device(), s.mem_pool->id(), [](cudaStream_t) {
        return true;
      });

  auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
  s.win_tensor = at::zeros({static_cast<int64_t>(count * num_ranks)}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      s.mem_pool->device(), s.mem_pool->id());

  // Allocate src_tensor OUTSIDE the pool to ensure it gets its own cuMem
  // allocation. When both tensors share the same cuMem block and the src_tensor
  // is not 4096-aligned within that block, NCCL LOCAL_ONLY window registration
  // truncates ginOffset4K, causing put failures with P2P disabled.
  s.src_tensor = at::zeros({static_cast<int64_t>(count)}, options);

  torchcomm->barrier(false);
  s.win = torchcomm->new_window();
  s.win->tensor_register(s.win_tensor);
  torchcomm->barrier(false);

  s.dev_win = static_cast<DeviceWindowPipes*>(
      s.win->get_device_window(signal_count, counter_count, barrier_count));

  s.src_buf = s.win->register_local_buffer(s.src_tensor);

  // Ensure both ranks have completed all registration before kernels launch
  torchcomm->barrier(false);

  // Ensure all GPU work (tensor zeroing, registration) is complete before
  // kernels launch on a different stream
  cudaDeviceSynchronize();

  return s;
}

void teardownPipesWindow(
    PipesWindowSetup& s,
    std::shared_ptr<torch::comms::TorchComm>& torchcomm) {
  s.win->deregister_local_buffer(s.src_buf);
  s.win->tensor_deregister();
  s.win.reset();
  s.mem_pool.reset();
  torchcomm->barrier(false);
}

void checkKernelResults(
    int* d_results,
    int iterations,
    const std::string& tag) {
  std::vector<int> h_results(iterations);
  cudaMemcpy(
      h_results.data(),
      d_results,
      iterations * sizeof(int),
      cudaMemcpyDeviceToHost);
  for (int i = 0; i < iterations; i++) {
    ASSERT_EQ(h_results[i], 1)
        << tag << ": verification failed at iteration " << i;
  }
}

} // namespace

// =============================================================================
// Test Implementations
// =============================================================================

void PipesDeviceApiIteratedTest::testIteratedPut(
    size_t msg_bytes,
    CoopScope scope) {
  size_t count = msg_bytes / sizeof(float);
  if (count == 0) {
    count = 1;
  }
  int num_threads = threadsForScope(scope);
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "PipesIteratedPut msg=" << formatBytes(msg_bytes)
                           << " scope=" << scopeName(scope)
                           << " iters=" << iterations);

  // Need at least 2 signals: signal_id=0 for put, signal_id=1 for read-ack
  int signal_count = std::max(num_ranks_, 2);
  auto s = createPipesWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      signal_count,
      -1,
      -1);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  size_t bytes = count * sizeof(float);

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchPipesIteratedPutKernel(
        s.dev_win,
        s.src_buf,
        s.src_tensor.data_ptr<float>(),
        s.win_tensor.data_ptr<float>(),
        0,
        rank_ * bytes,
        bytes,
        count,
        dst_rank,
        src_rank,
        0,
        iterations,
        scope,
        num_threads,
        d_results,
        stream.stream());
  }
  stream.synchronize();

  checkKernelResults(
      d_results,
      iterations,
      "PipesIteratedPut(" + formatBytes(msg_bytes) + "," + scopeName(scope) +
          ")");
  cudaFree(d_results);
  teardownPipesWindow(s, torchcomm_);
}

void PipesDeviceApiIteratedTest::testIteratedSignal(CoopScope scope) {
  int iterations = config_.num_iterations;
  int num_threads = threadsForScope(scope);

  SCOPED_TRACE(
      ::testing::Message() << "PipesIteratedSignal scope=" << scopeName(scope));

  auto s = createPipesWindowSetup(
      torchcomm_, allocator_, device_index_, num_ranks_, 1, num_ranks_, -1, 1);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchPipesIteratedSignalKernel(
        s.dev_win,
        dst_rank,
        src_rank,
        0,
        iterations,
        scope,
        num_threads,
        stream.stream());
  }
  stream.synchronize();
  teardownPipesWindow(s, torchcomm_);
}

void PipesDeviceApiIteratedTest::testIteratedBarrier(CoopScope scope) {
  int iterations = config_.num_iterations;
  int num_threads = threadsForScope(scope);

  SCOPED_TRACE(
      ::testing::Message() << "PipesIteratedBarrier scope="
                           << scopeName(scope));

  auto s = createPipesWindowSetup(
      torchcomm_, allocator_, device_index_, num_ranks_, 1, -1, -1, 1);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchPipesIteratedBarrierKernel(
        s.dev_win, iterations, scope, num_threads, stream.stream());
  }
  stream.synchronize();
  teardownPipesWindow(s, torchcomm_);
}

void PipesDeviceApiIteratedTest::testIteratedCombined(size_t msg_bytes) {
  size_t count = msg_bytes / sizeof(float);
  if (count == 0) {
    count = 1;
  }
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "PipesIteratedCombined msg="
                           << formatBytes(msg_bytes));

  auto s = createPipesWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      num_ranks_,
      -1,
      4);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  size_t bytes = count * sizeof(float);

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchPipesIteratedCombinedKernel(
        s.dev_win,
        s.src_buf,
        s.src_tensor.data_ptr<float>(),
        s.win_tensor.data_ptr<float>(),
        0,
        rank_ * bytes,
        bytes,
        count,
        dst_rank,
        src_rank,
        0,
        0,
        iterations,
        d_results,
        stream.stream());
  }
  stream.synchronize();

  checkKernelResults(
      d_results,
      iterations,
      "PipesIteratedCombined(" + formatBytes(msg_bytes) + ")");
  cudaFree(d_results);
  teardownPipesWindow(s, torchcomm_);
}

void PipesDeviceApiIteratedTest::testMultiWindow() {
  int num_windows = config_.window_count;
  int iterations = config_.num_iterations / 2;
  size_t count = 1024;

  SCOPED_TRACE(
      ::testing::Message() << "PipesMultiWindow windows=" << num_windows);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  std::vector<PipesWindowSetup> windows;
  windows.reserve(num_windows);
  for (int w = 0; w < num_windows; w++) {
    windows.push_back(createPipesWindowSetup(
        torchcomm_,
        allocator_,
        device_index_,
        num_ranks_,
        count,
        std::max(num_ranks_, 2),
        -1,
        -1));
  }

  std::vector<int*> d_results_vec(num_windows, nullptr);
  std::vector<at::cuda::CUDAStream> streams;
  streams.reserve(num_windows);

  for (int w = 0; w < num_windows; w++) {
    ASSERT_EQ(
        cudaMalloc(&d_results_vec[w], iterations * sizeof(int)), cudaSuccess);
    ASSERT_EQ(
        cudaMemset(d_results_vec[w], 0, iterations * sizeof(int)), cudaSuccess);

    auto stream = at::cuda::getStreamFromPool(false, device_index_);
    streams.push_back(stream);

    size_t bytes = count * sizeof(float);
    c10::cuda::CUDAStreamGuard guard(stream);
    launchPipesIteratedPutKernel(
        windows[w].dev_win,
        windows[w].src_buf,
        windows[w].src_tensor.data_ptr<float>(),
        windows[w].win_tensor.data_ptr<float>(),
        0,
        rank_ * bytes,
        bytes,
        count,
        dst_rank,
        src_rank,
        0,
        iterations,
        CoopScope::THREAD,
        1,
        d_results_vec[w],
        stream.stream());
  }

  for (auto& stream : streams) {
    stream.synchronize();
  }

  for (int w = 0; w < num_windows; w++) {
    checkKernelResults(
        d_results_vec[w],
        iterations,
        "PipesMultiWindow[" + std::to_string(w) + "]");
    cudaFree(d_results_vec[w]);
  }

  for (auto& ws : windows) {
    teardownPipesWindow(ws, torchcomm_);
  }
}

void PipesDeviceApiIteratedTest::testWindowLifecycle() {
  int cycles = config_.lifecycle_cycles;
  size_t count = 256;

  SCOPED_TRACE(
      ::testing::Message() << "PipesWindowLifecycle cycles=" << cycles);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  for (int cycle = 0; cycle < cycles; cycle++) {
    auto s = createPipesWindowSetup(
        torchcomm_,
        allocator_,
        device_index_,
        num_ranks_,
        count,
        std::max(num_ranks_, 2),
        -1,
        -1);

    int* d_result = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_result, 0, sizeof(int)), cudaSuccess);

    size_t bytes = count * sizeof(float);
    auto stream = at::cuda::getStreamFromPool(false, device_index_);
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      launchPipesIteratedPutKernel(
          s.dev_win,
          s.src_buf,
          s.src_tensor.data_ptr<float>(),
          s.win_tensor.data_ptr<float>(),
          0,
          rank_ * bytes,
          bytes,
          count,
          dst_rank,
          src_rank,
          0,
          1,
          CoopScope::THREAD,
          1,
          d_result,
          stream.stream());
    }
    stream.synchronize();

    checkKernelResults(
        d_result, 1, "PipesWindowLifecycle[" + std::to_string(cycle) + "]");
    cudaFree(d_result);
    teardownPipesWindow(s, torchcomm_);
  }
}

void PipesDeviceApiIteratedTest::testMultiComm() {
  int num_comms = config_.comm_count;
  int iterations = config_.num_iterations / 2;
  size_t count = 1024;

  SCOPED_TRACE(
      ::testing::Message() << "PipesMultiComm comms=" << num_comms
                           << " iters=" << iterations);

  // Create multiple communicators
  std::vector<std::unique_ptr<TorchCommTestWrapper>> wrappers;
  wrappers.reserve(num_comms);
  std::vector<std::shared_ptr<torch::comms::TorchComm>> comms;
  comms.reserve(num_comms);
  for (int c = 0; c < num_comms; c++) {
    auto w = std::make_unique<TorchCommTestWrapper>();
    comms.push_back(w->getTorchComm());
    wrappers.push_back(std::move(w));
  }

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  // Create a window per communicator
  std::vector<PipesWindowSetup> windows;
  windows.reserve(num_comms);
  for (int c = 0; c < num_comms; c++) {
    windows.push_back(createPipesWindowSetup(
        comms[c],
        allocator_,
        device_index_,
        num_ranks_,
        count,
        std::max(num_ranks_, 2),
        -1,
        -1));
  }

  // Run iterated put on each comm's window
  std::vector<int*> d_results_vec(num_comms, nullptr);
  std::vector<at::cuda::CUDAStream> streams;
  streams.reserve(num_comms);

  for (int c = 0; c < num_comms; c++) {
    ASSERT_EQ(
        cudaMalloc(&d_results_vec[c], iterations * sizeof(int)), cudaSuccess);
    ASSERT_EQ(
        cudaMemset(d_results_vec[c], 0, iterations * sizeof(int)), cudaSuccess);

    auto stream = at::cuda::getStreamFromPool(false, device_index_);
    streams.push_back(stream);

    size_t bytes = count * sizeof(float);
    c10::cuda::CUDAStreamGuard guard(stream);
    launchPipesIteratedPutKernel(
        windows[c].dev_win,
        windows[c].src_buf,
        windows[c].src_tensor.data_ptr<float>(),
        windows[c].win_tensor.data_ptr<float>(),
        0,
        rank_ * bytes,
        bytes,
        count,
        dst_rank,
        src_rank,
        0,
        iterations,
        CoopScope::THREAD,
        1,
        d_results_vec[c],
        stream.stream());
  }

  for (auto& stream : streams) {
    stream.synchronize();
  }

  for (int c = 0; c < num_comms; c++) {
    checkKernelResults(
        d_results_vec[c],
        iterations,
        "PipesMultiComm[" + std::to_string(c) + "]");
    cudaFree(d_results_vec[c]);
  }

  for (int c = 0; c < num_comms; c++) {
    teardownPipesWindow(windows[c], comms[c]);
  }

  comms.clear();
  wrappers.clear();
}

// =============================================================================
// Parameterized Test Registrations
// =============================================================================

// --- Put: parameterized by (msg_bytes, scope) ---

struct PipesPutParam {
  size_t msg_bytes;
  CoopScope scope;
};

class PipesDeviceApiIteratedPutTest
    : public PipesDeviceApiIteratedTest,
      public ::testing::WithParamInterface<PipesPutParam> {};

TEST_P(PipesDeviceApiIteratedPutTest, Put) {
  testIteratedPut(GetParam().msg_bytes, GetParam().scope);
}

INSTANTIATE_TEST_SUITE_P(
    IteratedPut,
    PipesDeviceApiIteratedPutTest,
    ::testing::Values(
        PipesPutParam{4, CoopScope::THREAD},
        PipesPutParam{1024, CoopScope::THREAD},
        PipesPutParam{1048576, CoopScope::THREAD},
        PipesPutParam{16777216, CoopScope::THREAD},
        PipesPutParam{1024, CoopScope::WARP},
        PipesPutParam{1024, CoopScope::BLOCK}),
    [](const ::testing::TestParamInfo<PipesPutParam>& info) {
      return std::to_string(info.param.msg_bytes) + "B_" +
          scopeName(info.param.scope);
    });

// --- Signal: parameterized by scope ---

class PipesDeviceApiIteratedSignalTest
    : public PipesDeviceApiIteratedTest,
      public ::testing::WithParamInterface<CoopScope> {};

TEST_P(PipesDeviceApiIteratedSignalTest, Signal) {
  testIteratedSignal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    IteratedSignal,
    PipesDeviceApiIteratedSignalTest,
    ::testing::Values(CoopScope::THREAD, CoopScope::WARP, CoopScope::BLOCK),
    [](const ::testing::TestParamInfo<CoopScope>& info) {
      return std::string(scopeName(info.param));
    });

// --- Barrier: parameterized by scope ---

class PipesDeviceApiIteratedBarrierTest
    : public PipesDeviceApiIteratedTest,
      public ::testing::WithParamInterface<CoopScope> {};

TEST_P(PipesDeviceApiIteratedBarrierTest, Barrier) {
  testIteratedBarrier(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    IteratedBarrier,
    PipesDeviceApiIteratedBarrierTest,
    ::testing::Values(CoopScope::THREAD, CoopScope::WARP, CoopScope::BLOCK),
    [](const ::testing::TestParamInfo<CoopScope>& info) {
      return std::string(scopeName(info.param));
    });

// --- Combined: parameterized by msg_bytes ---

class PipesDeviceApiIteratedCombinedTest
    : public PipesDeviceApiIteratedTest,
      public ::testing::WithParamInterface<size_t> {};

TEST_P(PipesDeviceApiIteratedCombinedTest, Combined) {
  testIteratedCombined(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    IteratedCombined,
    PipesDeviceApiIteratedCombinedTest,
    ::testing::Values(static_cast<size_t>(1024), static_cast<size_t>(1048576)),
    [](const ::testing::TestParamInfo<size_t>& info) {
      return std::to_string(info.param) + "B";
    });

// --- Non-parameterized tests ---

TEST_F(PipesDeviceApiIteratedTest, MultiWindow) {
  testMultiWindow();
}

TEST_F(PipesDeviceApiIteratedTest, MultiComm) {
  testMultiComm();
}

TEST_F(PipesDeviceApiIteratedTest, WindowLifecycle) {
  testWindowLifecycle();
}
