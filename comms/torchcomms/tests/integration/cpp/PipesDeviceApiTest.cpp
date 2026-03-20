// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test - Pipes Backend

#include "PipesDeviceApiTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include "PipesDeviceApiTestKernels.cuh"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

std::unique_ptr<TorchCommTestWrapper> PipesDeviceApiTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void PipesDeviceApiTest::SetUp() {
  // Check skip condition FIRST, before any initialization
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping Pipes Device API tests "
                    "(RUN_PIPES_DEVICE_API_TEST not set or "
                    "NCCL_CTRAN_USE_PIPES not enabled)";
  }

  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();

  // Get allocator using global function - obtained once and reused
  allocator_ = torch::comms::get_mem_allocator(torchcomm_->getBackend());
}

void PipesDeviceApiTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

bool PipesDeviceApiTest::checkIfSkip() {
  // Check RUN_PIPES_DEVICE_API_TEST env var
  const char* run_env = getenv("RUN_PIPES_DEVICE_API_TEST");
  if (!run_env) {
    return true; // skip if not set
  }
  std::string val(run_env);
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  if (val != "1" && val != "true") {
    return true; // skip if not enabled
  }

  // Also check NCCL_CTRAN_USE_PIPES=1 is set.
  // Without this, new_window() returns TorchCommWindowNCCLXGin instead of
  // Pipes.
  const char* pipes_env = getenv("NCCL_CTRAN_USE_PIPES");
  if (!pipes_env || std::string(pipes_env) != "1") {
    return true;
  }

  return false;
}

at::Tensor PipesDeviceApiTest::createTestTensor(
    int64_t count,
    at::ScalarType dtype) {
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  return at::ones({count}, options) * (rank_ + 1);
}

std::string PipesDeviceApiTest::getDtypeName(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return "float32";
    case at::kDouble:
      return "float64";
    case at::kHalf:
      return "float16";
    case at::kBFloat16:
      return "bfloat16";
    case at::kInt:
      return "int32";
    case at::kLong:
      return "int64";
    default:
      return "unknown";
  }
}

// =============================================================================
// Pipes Device Window Creation Test
// =============================================================================
// Validates the Pipes device window creation flow:
//   1. All ranks register the same-sized tensor in a symmetric window
//   2. Barrier ensures all registrations complete before device window creation
//   3. get_device_window() triggers ctran_win->get_device_win() (allGather):
//      - IBGDA path: exchanges remote buffer registration info
//      - NVLink path: exchanges NVLink-mapped remote pointers
//   4. Verify the returned device pointer is non-null
//
// NOTE: TEST_F macros MUST be in this file (compiled with
// TORCHCOMMS_HAS_NCCL_DEVICE_API and ENABLE_PIPES) so that
// TorchCommWindowNCCLXPipes resolves to the correct type (PipesDeviceBackend).

void PipesDeviceApiTest::testPipesDeviceWindowCreation(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Pipes Device Window Creation with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create MemPool for RDMA-compatible memory allocation (cuMem-based).
  // ctran's window registration uses cuMem-allocated buffers for IBGDA rkey
  // and NVLink mapping. Regular cudaMalloc may not support this.
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Window layout: [rank0_slot | rank1_slot | ... | rankN-1_slot]
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // End pool context immediately after allocation
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // All ranks must register the tensor collectively
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to Pipes window to access device API.
  // NCCL_CTRAN_USE_PIPES=1 makes new_window() return Pipes.
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXPipes*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes. "
                             "Is NCCL_CTRAN_USE_PIPES=1 set?";

  // get_device_window() is COLLECTIVE: internally calls
  // ctran_win->get_device_win() which does an allGather to exchange IBGDA
  // buffer registration info and NVLink-mapped remote buffer pointers.
  // All ranks must call this simultaneously.
  decltype(win->get_device_window()) dev_win = nullptr;
  try {
    dev_win = win->get_device_window();
  } catch (const std::runtime_error& e) {
    // Gracefully skip if IBGDA hardware is not available.
    // Both ranks hit this simultaneously (multiPeerTransport is null on both),
    // so no deadlock: both will skip cleanly.
    base_win->tensor_deregister();
    base_win.reset();
    mem_pool.reset();
    GTEST_SKIP() << "Skipping: IBGDA/Pipes hardware not available: "
                 << e.what();
  }
  EXPECT_NE(dev_win, nullptr)
      << "Pipes device window pointer should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(PipesDeviceApiTest, PipesDeviceWindowCreationFloat) {
  testPipesDeviceWindowCreation(1024, at::kFloat);
}

// =============================================================================
// Local Buffer Registration Test (Pipes)
// =============================================================================
// Validates register_local_buffer() for the Pipes (IBGDA) backend:
//   1. Create window, register tensor, create device window
//   2. Register a separate source tensor as a local buffer
//   3. Verify RegisteredBuffer has valid lkey (used for IBGDA WQE construction)
//   4. Verify backend_window is null (only GIN uses backend_window)
//   5. Deregister and verify cleanup

void PipesDeviceApiTest::testLocalBufferRegistration(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing Pipes Local Buffer Registration with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create MemPool for RDMA-compatible memory allocation (cuMem-based).
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  at::Tensor src_tensor = at::zeros({count}, options);
  src_tensor.fill_(static_cast<float>(rank_ + 1));

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // All ranks must register the tensor collectively
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXPipes*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes. "
                             "Is NCCL_CTRAN_USE_PIPES=1 set?";

  // get_device_window() is COLLECTIVE — must be called before
  // register_local_buffer() to initialize the Pipes device window.
  decltype(win->get_device_window()) dev_win = nullptr;
  try {
    dev_win = win->get_device_window();
  } catch (const std::runtime_error& e) {
    base_win->tensor_deregister();
    base_win.reset();
    mem_pool.reset();
    GTEST_SKIP() << "Skipping: IBGDA/Pipes hardware not available: "
                 << e.what();
  }
  ASSERT_NE(dev_win, nullptr) << "Device window should not be null";

  auto src_buf = win->register_local_buffer(src_tensor);

  // Pipes backend: lkey is set (used for IBGDA WQE construction),
  // backend_window is null (only GIN uses backend_window).
  ASSERT_NE(src_buf.base_ptr, nullptr) << "Buffer base_ptr should not be null";
  ASSERT_GT(src_buf.size, 0u) << "Buffer size should be positive";
  EXPECT_NE(src_buf.lkey, 0u) << "Pipes backend should set lkey for IBGDA put";
  EXPECT_EQ(src_buf.backend_window, nullptr)
      << "Pipes backend should not set backend_window";

  // Deregister and verify cleanup
  win->deregister_local_buffer(src_buf);

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(PipesDeviceApiTest, LocalBufferRegistrationFloat) {
  testLocalBufferRegistration(1024, at::kFloat);
}

// =============================================================================
// Per-Peer Signal Test (Pipes)
// =============================================================================
// Ring pattern: rank i signals rank (i+1) % num_ranks
//   1. Each rank sends ADD 1 to the next rank via device kernel
//   2. Each rank waits for the aggregated signal to reach expected value
//   3. Verifies read_signal returns the correct aggregated value

void PipesDeviceApiTest::testPerPeerSignal() {
  SCOPED_TRACE(::testing::Message() << "Testing per-peer signal slots (Pipes)");

  auto op_stream = at::cuda::getStreamFromPool(false, device_index_);
  auto wait_stream = at::cuda::getStreamFromPool(false, device_index_);

  // Create MemPool for RDMA-compatible memory allocation
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Minimal window — we only need the signal infrastructure, not data
  auto options =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({1}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to Pipes window to access device API.
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXPipes*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes. "
                             "Is NCCL_CTRAN_USE_PIPES=1 set?";

  int signal_count = num_ranks_;
  decltype(win->get_device_window()) dev_win = nullptr;
  try {
    dev_win = win->get_device_window(signal_count, -1, 1);
  } catch (const std::runtime_error& e) {
    base_win->tensor_deregister();
    base_win.reset();
    mem_pool.reset();
    GTEST_SKIP() << "Skipping: IBGDA/Pipes hardware not available: "
                 << e.what();
  }
  ASSERT_NE(dev_win, nullptr);

  // Ring pattern: rank i signals rank (i+1) % num_ranks
  int dst_rank = (rank_ + 1) % num_ranks_;
  constexpr int kSignalId = 0;

  // Each rank sends ADD 1 to the next rank
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchPipesSignalKernel(
        dev_win,
        dst_rank,
        kSignalId,
        torchcomms::device::SignalOp::ADD,
        1,
        op_stream.stream());
  }

  // Each rank receives one signal, so aggregate sum should be >= 1
  {
    c10::cuda::CUDAStreamGuard guard(wait_stream);
    torchcomms::device::test::launchPipesWaitSignalKernel(
        dev_win, kSignalId, 1, wait_stream.stream());
  }

  op_stream.synchronize();
  wait_stream.synchronize();

  // Read signal value via kernel and verify on host
  uint64_t* d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_out, sizeof(uint64_t)), cudaSuccess);
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchPipesReadSignalKernel(
        dev_win, kSignalId, d_out, op_stream.stream());
  }
  op_stream.synchronize();

  uint64_t h_out = 0;
  cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  ASSERT_GE(h_out, 1u) << "Expected aggregated signal >= 1, got " << h_out;

  // NOTE: Pipes DeviceWindow does not support device-side signal reset.
  // Use monotonically increasing signal values or host-side cudaMemset
  // between kernel launches. No reset needed here since the window is
  // being destroyed.

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(PipesDeviceApiTest, PerPeerSignal) {
  testPerPeerSignal();
}

// =============================================================================
// Wait Signal From Specific Peer Test (Pipes)
// =============================================================================
// Ring pattern: rank i signals rank (i+1) % num_ranks
//   1. Each rank sends ADD 1 to the next rank
//   2. Receiver uses wait_signal_from(src_rank, ...) to wait for the
//      specific sender's slot (not aggregated)
//   3. Verifies completion without deadlock

void PipesDeviceApiTest::testWaitSignalFrom() {
  SCOPED_TRACE(::testing::Message() << "Testing wait_signal_from (Pipes)");

  auto op_stream = at::cuda::getStreamFromPool(false, device_index_);
  auto wait_stream = at::cuda::getStreamFromPool(false, device_index_);

  // Create MemPool for RDMA-compatible memory allocation
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  auto options =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({1}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXPipes*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes. "
                             "Is NCCL_CTRAN_USE_PIPES=1 set?";

  int signal_count = num_ranks_;
  decltype(win->get_device_window()) dev_win = nullptr;
  try {
    dev_win = win->get_device_window(signal_count, -1, 1);
  } catch (const std::runtime_error& e) {
    base_win->tensor_deregister();
    base_win.reset();
    mem_pool.reset();
    GTEST_SKIP() << "Skipping: IBGDA/Pipes hardware not available: "
                 << e.what();
  }
  ASSERT_NE(dev_win, nullptr);

  // Ring: rank i signals rank (i+1), receiver expects signal from (i-1)
  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  constexpr int kSignalId = 0;

  // Send signal to next rank
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchPipesSignalKernel(
        dev_win,
        dst_rank,
        kSignalId,
        torchcomms::device::SignalOp::ADD,
        1,
        op_stream.stream());
  }

  // Wait for signal from previous rank (point-to-point)
  {
    c10::cuda::CUDAStreamGuard guard(wait_stream);
    torchcomms::device::test::launchPipesWaitSignalFromKernel(
        dev_win,
        src_rank,
        kSignalId,
        torchcomms::device::CmpOp::GE,
        1,
        wait_stream.stream());
  }

  op_stream.synchronize();
  wait_stream.synchronize();

  // No device-side signal reset for Pipes. Window destruction handles cleanup.

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(PipesDeviceApiTest, WaitSignalFrom) {
  testWaitSignalFrom();
}

// =============================================================================
// Device Barrier Test (Pipes)
// =============================================================================
// Validates the device-side barrier:
//   1. All ranks launch a barrier kernel
//   2. Barrier synchronizes all ranks via DeviceWindow::barrier()
//   3. All ranks complete successfully
//   4. Second barrier verifies reusability

void PipesDeviceApiTest::testDeviceBarrier() {
  SCOPED_TRACE(::testing::Message() << "Testing device barrier (Pipes)");

  auto op_stream = at::cuda::getStreamFromPool(false, device_index_);

  // Create MemPool for RDMA-compatible memory allocation
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Minimal window for barrier infrastructure
  auto options =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({1}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXPipes*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXPipes. "
                             "Is NCCL_CTRAN_USE_PIPES=1 set?";

  // Get device window with barrier support
  int barrier_count = 2;
  decltype(win->get_device_window()) dev_win = nullptr;
  try {
    dev_win = win->get_device_window(-1, -1, barrier_count);
  } catch (const std::runtime_error& e) {
    base_win->tensor_deregister();
    base_win.reset();
    mem_pool.reset();
    GTEST_SKIP() << "Skipping: IBGDA/Pipes hardware not available: "
                 << e.what();
  }
  ASSERT_NE(dev_win, nullptr);

  constexpr int kBarrierId = 0;

  // Launch barrier kernel - all ranks must participate
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchPipesBarrierKernel(
        dev_win, kBarrierId, op_stream.stream());
  }

  // If barrier works, all ranks complete together
  op_stream.synchronize();

  // Second barrier to verify reusability via monotonic counter pattern.
  // Pipes barrier uses monotonic counters — reuse the same barrier_id.
  // The inbox accumulates (1, 2, ...) and barrierExpected_ tracks in lockstep.
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchPipesBarrierKernel(
        dev_win, kBarrierId, op_stream.stream());
  }
  op_stream.synchronize();

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(PipesDeviceApiTest, DeviceBarrier) {
  testDeviceBarrier();
}
