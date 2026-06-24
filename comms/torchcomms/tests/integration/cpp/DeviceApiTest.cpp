// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Stress functional tests for TorchComm Device API — NCCLx (GIN+LSA) backend.

#include "DeviceApiTest.hpp"

#include <gtest/gtest.h>
#include "DeviceApiTestKernels.cuh"
#include "StressTestHelpers.hpp"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"

using namespace torchcomms::device;
using namespace torchcomms::device::test;

// =============================================================================
// Setup / Teardown
// =============================================================================

void DeviceApiTest::SetUp() {
  if (!shouldRunStressTest()) {
    GTEST_SKIP() << "Skipping stress tests (RUN_DEVICE_STRESS_TEST not set)";
  }

  config_ = parseStressTestConfig();
  wrapper_ = std::make_unique<TorchCommTestWrapper>();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();
  allocator_ = torch::comms::get_mem_allocator(torchcomm_->getBackend());
}

void DeviceApiTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

// =============================================================================
// Helper: create MemPool, allocate tensors, create window, get device window
// =============================================================================

namespace {

struct WindowSetup {
  std::unique_ptr<at::cuda::MemPool> mem_pool;
  at::Tensor win_tensor;
  at::Tensor src_tensor;
  std::shared_ptr<torch::comms::TorchCommWindow> win;
  DeviceWindowNCCL* dev_win{nullptr};
  RegisteredBufferNCCL src_buf{};
};

WindowSetup createWindowSetup(
    std::shared_ptr<torch::comms::TorchComm>& torchcomm,
    std::shared_ptr<c10::Allocator>& allocator,
    int device_index,
    int num_ranks,
    size_t count,
    int signal_count,
    int counter_count,
    int barrier_count) {
  WindowSetup s;

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
  // truncates ginOffset4K, and NVLink put with P2P disabled fails to deliver
  // data. Separate allocations avoid this issue.
  s.src_tensor = at::zeros({static_cast<int64_t>(count)}, options);

  torchcomm->barrier(false);
  s.win = torchcomm->new_window();
  s.win->tensor_register(s.win_tensor);
  torchcomm->barrier(false);

  s.dev_win = static_cast<DeviceWindowNCCL*>(
      s.win->get_device_window(signal_count, counter_count, barrier_count));

  s.src_buf = s.win->register_local_buffer(s.src_tensor);

  // Ensure both ranks have completed all registration before kernels launch
  torchcomm->barrier(false);

  // Ensure all GPU work (tensor zeroing, registration) is complete before
  // kernels launch on a different stream
  cudaDeviceSynchronize();

  return s;
}

// Overload that accepts a custom dtype for the window and source tensors.
WindowSetup createWindowSetupWithDtype(
    std::shared_ptr<torch::comms::TorchComm>& torchcomm,
    std::shared_ptr<c10::Allocator>& allocator,
    int device_index,
    int num_ranks,
    size_t count,
    int signal_count,
    int counter_count,
    int barrier_count,
    at::ScalarType dtype) {
  WindowSetup s;

  s.mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      s.mem_pool->device(), s.mem_pool->id(), [](cudaStream_t) {
        return true;
      });

  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index);
  s.win_tensor = at::zeros({static_cast<int64_t>(count * num_ranks)}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      s.mem_pool->device(), s.mem_pool->id());

  // Allocate src_tensor OUTSIDE the pool to ensure it gets its own cuMem
  // allocation. When both tensors share the same cuMem block and the
  // src_tensor is not 4096-aligned within that block, NCCL LOCAL_ONLY window
  // registration truncates ginOffset4K, and NVLink put with P2P disabled
  // fails to deliver data. Separate allocations avoid this issue.
  s.src_tensor = at::zeros({static_cast<int64_t>(count)}, options);

  torchcomm->barrier(false);
  s.win = torchcomm->new_window();
  s.win->tensor_register(s.win_tensor);
  torchcomm->barrier(false);

  s.dev_win = static_cast<DeviceWindowNCCL*>(
      s.win->get_device_window(signal_count, counter_count, barrier_count));

  s.src_buf = s.win->register_local_buffer(s.src_tensor);

  torchcomm->barrier(false);
  cudaDeviceSynchronize();

  return s;
}

void teardownWindow(
    WindowSetup& s,
    std::shared_ptr<torch::comms::TorchComm>& torchcomm) {
  s.win->deregister_local_buffer(s.src_buf);
  s.win->tensor_deregister();
  s.win.reset();
  s.mem_pool.reset();
  torchcomm->barrier(false);
}

// Allocate device int array, check results on host after kernel completion.
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
// Category 1: Stress Correctness
// =============================================================================

void DeviceApiTest::testStressPut(size_t msg_bytes, CoopScope scope) {
  size_t count = msg_bytes / sizeof(float);
  if (count == 0) {
    count = 1;
  }
  int num_threads = threadsForScope(scope);
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "StressPut msg=" << formatBytes(msg_bytes)
                           << " scope=" << scopeName(scope)
                           << " iters=" << iterations);

  auto s = createWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/num_ranks_,
      /*counter_count=*/-1,
      /*barrier_count=*/2);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  size_t bytes = count * sizeof(float);
  size_t src_offset = 0;
  size_t dst_offset = rank_ * bytes;

  // Allocate results buffer on device
  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchStressPutKernel(
        s.dev_win,
        s.src_buf,
        s.src_tensor.data_ptr<float>(),
        s.win_tensor.data_ptr<float>(),
        src_offset,
        dst_offset,
        bytes,
        count,
        dst_rank,
        src_rank,
        /*signal_id=*/0,
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
      "StressPut(" + formatBytes(msg_bytes) + "," + scopeName(scope) + ")");

  cudaFree(d_results);
  teardownWindow(s, torchcomm_);
}

void DeviceApiTest::testStressSignal(CoopScope scope) {
  int iterations = config_.num_iterations;
  int num_threads = threadsForScope(scope);

  SCOPED_TRACE(
      ::testing::Message() << "StressSignal scope=" << scopeName(scope)
                           << " iters=" << iterations);

  // Minimal window — only need signal infrastructure
  size_t count = 1;
  auto s = createWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/num_ranks_,
      /*counter_count=*/-1,
      /*barrier_count=*/1);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchStressSignalKernel(
        s.dev_win,
        dst_rank,
        src_rank,
        /*signal_id=*/0,
        iterations,
        scope,
        num_threads,
        stream.stream());
  }
  // If signal is broken, this hangs — timeout is the failure signal.
  stream.synchronize();

  teardownWindow(s, torchcomm_);
}

void DeviceApiTest::testStressBarrier(CoopScope scope) {
  int iterations = config_.num_iterations;
  int num_threads = threadsForScope(scope);

  SCOPED_TRACE(
      ::testing::Message() << "StressBarrier scope=" << scopeName(scope)
                           << " iters=" << iterations);

  size_t count = 1;
  auto s = createWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/-1,
      /*counter_count=*/-1,
      /*barrier_count=*/2);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchStressBarrierKernel(
        s.dev_win, iterations, scope, num_threads, stream.stream());
  }
  stream.synchronize();

  teardownWindow(s, torchcomm_);
}

void DeviceApiTest::testStressCombined(size_t msg_bytes) {
  size_t count = msg_bytes / sizeof(float);
  if (count == 0) {
    count = 1;
  }
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "StressCombined msg=" << formatBytes(msg_bytes)
                           << " iters=" << iterations);

  auto s = createWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/num_ranks_,
      /*counter_count=*/-1,
      /*barrier_count=*/4);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  size_t bytes = count * sizeof(float);
  size_t src_offset = 0;
  size_t dst_offset = rank_ * bytes;

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchStressCombinedKernel(
        s.dev_win,
        s.src_buf,
        s.src_tensor.data_ptr<float>(),
        s.win_tensor.data_ptr<float>(),
        src_offset,
        dst_offset,
        bytes,
        count,
        dst_rank,
        src_rank,
        /*signal_id=*/0,
        /*barrier_id_base=*/0,
        iterations,
        d_results,
        stream.stream());
  }
  stream.synchronize();

  checkKernelResults(
      d_results, iterations, "StressCombined(" + formatBytes(msg_bytes) + ")");

  cudaFree(d_results);
  teardownWindow(s, torchcomm_);
}

// =============================================================================
// Category 2: Concurrency
// =============================================================================

void DeviceApiTest::testMultiWindow() {
  int num_windows = config_.window_count;
  int iterations = config_.num_iterations / 2; // fewer iters per window
  size_t count = 1024; // 4KB per window

  SCOPED_TRACE(
      ::testing::Message() << "MultiWindow windows=" << num_windows
                           << " iters=" << iterations);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  // Create multiple windows on the same communicator
  std::vector<WindowSetup> windows;
  windows.reserve(num_windows);
  for (int w = 0; w < num_windows; w++) {
    windows.push_back(createWindowSetup(
        torchcomm_,
        allocator_,
        device_index_,
        num_ranks_,
        count,
        /*signal_count=*/num_ranks_,
        /*counter_count=*/-1,
        /*barrier_count=*/2));
  }

  // Run put iterations on each window sequentially (different streams)
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
    launchStressPutKernel(
        windows[w].dev_win,
        windows[w].src_buf,
        windows[w].src_tensor.data_ptr<float>(),
        windows[w].win_tensor.data_ptr<float>(),
        /*src_offset=*/0,
        /*dst_offset=*/rank_ * bytes,
        bytes,
        count,
        dst_rank,
        src_rank,
        /*signal_id=*/0,
        iterations,
        CoopScope::THREAD,
        1,
        d_results_vec[w],
        stream.stream());
  }

  // Wait for all
  for (auto& stream : streams) {
    stream.synchronize();
  }

  // Verify all windows
  for (int w = 0; w < num_windows; w++) {
    checkKernelResults(
        d_results_vec[w], iterations, "MultiWindow[" + std::to_string(w) + "]");
    cudaFree(d_results_vec[w]);
  }

  for (auto& ws : windows) {
    teardownWindow(ws, torchcomm_);
  }
}

void DeviceApiTest::testMultiComm() {
  int num_comms = config_.comm_count;
  int iterations = config_.num_iterations / 2;
  size_t count = 1024;

  SCOPED_TRACE(
      ::testing::Message() << "MultiComm comms=" << num_comms
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
  std::vector<WindowSetup> windows;
  windows.reserve(num_comms);
  for (int c = 0; c < num_comms; c++) {
    windows.push_back(createWindowSetup(
        comms[c],
        allocator_,
        device_index_,
        num_ranks_,
        count,
        /*signal_count=*/num_ranks_,
        /*counter_count=*/-1,
        /*barrier_count=*/2));
  }

  // Run stress put on each comm's window
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
    launchStressPutKernel(
        windows[c].dev_win,
        windows[c].src_buf,
        windows[c].src_tensor.data_ptr<float>(),
        windows[c].win_tensor.data_ptr<float>(),
        /*src_offset=*/0,
        /*dst_offset=*/rank_ * bytes,
        bytes,
        count,
        dst_rank,
        src_rank,
        /*signal_id=*/0,
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
        d_results_vec[c], iterations, "MultiComm[" + std::to_string(c) + "]");
    cudaFree(d_results_vec[c]);
  }

  for (int c = 0; c < num_comms; c++) {
    teardownWindow(windows[c], comms[c]);
  }

  comms.clear();
  wrappers.clear();
}

// =============================================================================
// Category 3: Resource Exhaustion
// =============================================================================

void DeviceApiTest::testWindowLifecycle() {
  int cycles = config_.lifecycle_cycles;
  size_t count = 256; // Small window per cycle

  SCOPED_TRACE(::testing::Message() << "WindowLifecycle cycles=" << cycles);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  for (int cycle = 0; cycle < cycles; cycle++) {
    auto s = createWindowSetup(
        torchcomm_,
        allocator_,
        device_index_,
        num_ranks_,
        count,
        /*signal_count=*/num_ranks_,
        /*counter_count=*/-1,
        /*barrier_count=*/2);

    // Do one put+verify per cycle
    int* d_result = nullptr;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_result, 0, sizeof(int)), cudaSuccess);

    size_t bytes = count * sizeof(float);
    auto stream = at::cuda::getStreamFromPool(false, device_index_);
    {
      c10::cuda::CUDAStreamGuard guard(stream);
      launchStressPutKernel(
          s.dev_win,
          s.src_buf,
          s.src_tensor.data_ptr<float>(),
          s.win_tensor.data_ptr<float>(),
          /*src_offset=*/0,
          /*dst_offset=*/rank_ * bytes,
          bytes,
          count,
          dst_rank,
          src_rank,
          /*signal_id=*/0,
          /*iterations=*/1,
          CoopScope::THREAD,
          1,
          d_result,
          stream.stream());
    }
    stream.synchronize();

    checkKernelResults(
        d_result, 1, "WindowLifecycle[" + std::to_string(cycle) + "]");
    cudaFree(d_result);

    teardownWindow(s, torchcomm_);
  }
}

// =============================================================================
// Aggregated wait_signal + read_signal + reset
// =============================================================================

void DeviceApiTest::testStressAggregatedSignal() {
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "StressAggregatedSignal iters=" << iterations);

  size_t count = 1;
  auto s = createWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/num_ranks_,
      /*counter_count=*/-1,
      /*barrier_count=*/2);

  int dst_rank = (rank_ + 1) % num_ranks_;

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchStressAggregatedSignalKernel(
        s.dev_win,
        dst_rank,
        /*signal_id=*/0,
        iterations,
        d_results,
        stream.stream());
  }
  stream.synchronize();

  checkKernelResults(d_results, iterations, "StressAggregatedSignal");

  cudaFree(d_results);
  teardownWindow(s, torchcomm_);
}

// =============================================================================
// Half-precision put
// =============================================================================

void DeviceApiTest::testStressPutHalf(size_t msg_bytes, CoopScope scope) {
  size_t count = msg_bytes / sizeof(at::Half);
  if (count == 0) {
    count = 1;
  }
  int num_threads = threadsForScope(scope);
  int iterations = config_.num_iterations;

  SCOPED_TRACE(
      ::testing::Message() << "StressPutHalf msg=" << formatBytes(msg_bytes)
                           << " scope=" << scopeName(scope)
                           << " iters=" << iterations);

  auto s = createWindowSetupWithDtype(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/num_ranks_,
      /*counter_count=*/-1,
      /*barrier_count=*/2,
      at::kHalf);

  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  size_t bytes = count * sizeof(at::Half);
  size_t src_offset = 0;
  size_t dst_offset = rank_ * bytes;

  int* d_results = nullptr;
  ASSERT_EQ(cudaMalloc(&d_results, iterations * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_results, 0, iterations * sizeof(int)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchStressPutHalfKernel(
        s.dev_win,
        s.src_buf,
        s.src_tensor.data_ptr(),
        s.win_tensor.data_ptr(),
        src_offset,
        dst_offset,
        bytes,
        count,
        dst_rank,
        src_rank,
        /*signal_id=*/0,
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
      "StressPutHalf(" + formatBytes(msg_bytes) + "," + scopeName(scope) + ")");

  cudaFree(d_results);
  teardownWindow(s, torchcomm_);
}

// =============================================================================
// Address Query Tests
// =============================================================================

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
void DeviceApiTest::testGetMultimemAddress() {
  SCOPED_TRACE(::testing::Message() << "GetMultimemAddress");

  size_t count = 1024;
  auto s = createWindowSetup(
      torchcomm_,
      allocator_,
      device_index_,
      num_ranks_,
      count,
      /*signal_count=*/num_ranks_,
      /*counter_count=*/-1,
      /*barrier_count=*/1);

  // Get host-side multimem address via the typed window
  auto typed_win =
      std::dynamic_pointer_cast<torch::comms::TorchCommWindowNCCLXGin>(s.win);
  ASSERT_NE(typed_win, nullptr) << "Window is not TorchCommWindowNCCLXGin";

  void* host_ptr = typed_win->get_multimem_address(0);
  int64_t host_addr = reinterpret_cast<int64_t>(host_ptr);

  // Get device-side multimem address via kernel
  int64_t* d_result = nullptr;
  ASSERT_EQ(cudaMalloc(&d_result, sizeof(int64_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d_result, 0, sizeof(int64_t)), cudaSuccess);

  auto stream = at::cuda::getStreamFromPool(false, device_index_);
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    launchGetMultimemAddressKernel(
        s.dev_win, /*offset=*/0, d_result, stream.stream());
  }
  stream.synchronize();

  int64_t device_addr = 0;
  cudaMemcpy(&device_addr, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  // Both should match — both can be nullptr if NVLS is not supported on
  // this hardware, but they should be consistent.
  ASSERT_EQ(device_addr, host_addr)
      << "get_multimem_address mismatch: device=" << device_addr
      << " host=" << host_addr;

  teardownWindow(s, torchcomm_);
}
#endif

// =============================================================================
// Parameterized Test Registrations
// =============================================================================

// --- Put: parameterized by (msg_bytes, scope) ---

struct PutParam {
  size_t msg_bytes;
  CoopScope scope;
};

class DeviceApiPutTest : public DeviceApiTest,
                         public ::testing::WithParamInterface<PutParam> {};

TEST_P(DeviceApiPutTest, Put) {
  testStressPut(GetParam().msg_bytes, GetParam().scope);
}

INSTANTIATE_TEST_SUITE_P(
    StressPut,
    DeviceApiPutTest,
    ::testing::Values(
        PutParam{4, CoopScope::THREAD},
        PutParam{1024, CoopScope::THREAD},
        PutParam{1048576, CoopScope::THREAD},
        PutParam{16777216, CoopScope::THREAD},
        PutParam{1024, CoopScope::WARP},
        PutParam{1024, CoopScope::BLOCK},
        PutParam{1048576, CoopScope::WARP},
        PutParam{1048576, CoopScope::BLOCK}),
    [](const auto& info) {
      return std::to_string(info.param.msg_bytes) + "B_" +
          scopeName(info.param.scope);
    });

// --- Signal: parameterized by scope ---

class DeviceApiSignalTest : public DeviceApiTest,
                            public ::testing::WithParamInterface<CoopScope> {};

TEST_P(DeviceApiSignalTest, Signal) {
  testStressSignal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    StressSignal,
    DeviceApiSignalTest,
    ::testing::Values(CoopScope::THREAD, CoopScope::WARP, CoopScope::BLOCK),
    [](const ::testing::TestParamInfo<CoopScope>& info) {
      return std::string(scopeName(info.param));
    });

// --- Barrier: parameterized by scope ---

class DeviceApiBarrierTest : public DeviceApiTest,
                             public ::testing::WithParamInterface<CoopScope> {};

TEST_P(DeviceApiBarrierTest, Barrier) {
  testStressBarrier(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    StressBarrier,
    DeviceApiBarrierTest,
    ::testing::Values(CoopScope::THREAD, CoopScope::WARP, CoopScope::BLOCK),
    [](const ::testing::TestParamInfo<CoopScope>& info) {
      return std::string(scopeName(info.param));
    });

// --- Combined: parameterized by msg_bytes ---

class DeviceApiCombinedTest : public DeviceApiTest,
                              public ::testing::WithParamInterface<size_t> {};

TEST_P(DeviceApiCombinedTest, Combined) {
  testStressCombined(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    StressCombined,
    DeviceApiCombinedTest,
    ::testing::Values(static_cast<size_t>(1024), static_cast<size_t>(1048576)),
    [](const auto& info) { return std::to_string(info.param) + "B"; });

// --- Non-parameterized tests ---

TEST_F(DeviceApiTest, MultiWindow) {
  testMultiWindow();
}

TEST_F(DeviceApiTest, MultiComm) {
  testMultiComm();
}

TEST_F(DeviceApiTest, WindowLifecycle) {
  testWindowLifecycle();
}

// --- Aggregated signal + read_signal + reset ---

TEST_F(DeviceApiTest, AggregatedSignal) {
  testStressAggregatedSignal();
}

// --- Half-precision put: 1KB THREAD only ---
// Put is dtype-agnostic (operates on bytes). Float16 only tests tensor
// allocation and element_size calculation, so one test suffices.

TEST_F(DeviceApiTest, PutHalf) {
  testStressPutHalf(1024, CoopScope::THREAD);
}

// --- Address query tests ---

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
TEST_F(DeviceApiTest, DISABLED_GetMultimemAddress) {
  testGetMultimemAddress();
}
#endif
