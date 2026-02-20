// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API Integration Test - NCCL GIN Backend

#include "DeviceApiTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include "DeviceApiTestKernels.cuh"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"

std::unique_ptr<TorchCommTestWrapper> DeviceApiTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void DeviceApiTest::SetUp() {
  // Check skip condition FIRST, before any initialization
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping Device API tests (RUN_DEVICE_API_TEST not set)";
  }

  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_index_ = rank_ % at::cuda::device_count();

  // Get allocator using global function - obtained once and reused
  allocator_ = torch::comms::get_mem_allocator(torchcomm_->getBackend());
}

void DeviceApiTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

bool DeviceApiTest::checkIfSkip() {
  // Check RUN_DEVICE_API_TEST env var
  const char* device_api_env = getenv("RUN_DEVICE_API_TEST");
  if (!device_api_env) {
    return true; // skip if not set
  }
  std::string val(device_api_env);
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  if (val != "1" && val != "true") {
    return true; // skip if not enabled
  }
  return false;
}

at::Tensor DeviceApiTest::createTestTensor(
    int64_t count,
    at::ScalarType dtype) {
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  return at::ones({count}, options) * (rank_ + 1);
}

std::string DeviceApiTest::getDtypeName(at::ScalarType dtype) {
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

// Test device window creation and basic properties
void DeviceApiTest::testDeviceWindowCreation(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Window Creation with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create MemPool for RDMA-compatible memory allocation (cuMem-based)
  // This is required for NCCL orig path (symmetric windows) which calls
  // cuMemRetainAllocationHandle - only works with cuMem-allocated memory.
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Allocate window buffer from the RDMA-compatible pool
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // End pool context immediately after allocation
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // Get device window (returns device pointer for use in CUDA/Triton kernels)
  auto* dev_win = win->get_device_window();
  EXPECT_NE(dev_win, nullptr) << "Device window pointer should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

void DeviceApiTest::testLocalBufferRegistration(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Local Buffer Registration with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create MemPool for RDMA-compatible memory allocation (cuMem-based)
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Allocate window buffer from the RDMA-compatible pool
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  at::Tensor src_tensor = at::zeros({count}, options);
  src_tensor.fill_(static_cast<float>(rank_ + 1));

  // End pool context immediately after allocation
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // Get device window first to ensure GIN is enabled.
  // GIN is only enabled when ncclDevCommCreate is called (inside
  // get_device_window), not during window registration.
  auto* dev_win = win->get_device_window();
  ASSERT_NE(dev_win, nullptr) << "Device window should not be null";

  auto src_buf = win->register_local_buffer(src_tensor);

  // Verify buffer properties
  ASSERT_NE(src_buf.base_ptr, nullptr) << "Buffer base_ptr should not be null";
  ASSERT_GT(src_buf.size, 0) << "Buffer size should be positive";
  ASSERT_NE(src_buf.backend_window, nullptr)
      << "Buffer backend_window should not be null";

  // Cleanup
  win->deregister_local_buffer(src_buf);
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

void DeviceApiTest::testDeviceWindowWithSignals(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Window with Signals count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create MemPool for RDMA-compatible memory allocation (cuMem-based)
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Allocate window buffer from the RDMA-compatible pool
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // End pool context immediately after allocation
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // Get device window with signals (returns device pointer for use in kernels)
  int signal_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, -1, 1);
  EXPECT_NE(dev_win, nullptr) << "Device window pointer should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

void DeviceApiTest::testDeviceWindowWithCounters(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Window with Counters count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create MemPool for RDMA-compatible memory allocation (cuMem-based)
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Allocate window buffer from the RDMA-compatible pool
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // End pool context immediately after allocation
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // Get device window with signals and counters (returns device pointer)
  int signal_count = num_ranks_;
  int counter_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, counter_count, 1);
  EXPECT_NE(dev_win, nullptr) << "Device window pointer should not be null";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

// =============================================================================
// Device Put Test - Uses CUDA kernel to perform device-initiated RMA
// =============================================================================
// This test validates the full device-side put flow:
//   1. Create window and get device window handle
//   2. Register local source buffer
//   3. Launch CUDA kernel that performs put to next rank
//   4. Use signals to synchronize sender/receiver
//   5. Verify data arrived correctly

void DeviceApiTest::testDevicePut(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing Device Put with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create streams for put and wait operations
  auto put_stream = at::cuda::getStreamFromPool(false, device_index_);
  auto wait_stream = at::cuda::getStreamFromPool(false, device_index_);

  // Create MemPool for RDMA-compatible memory allocation
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Window layout: [rank0_slot | rank1_slot | ... | rankN-1_slot]
  // Each slot has 'count' elements.
  auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({count * num_ranks_}, options);

  // Allocate separate source buffer for the put operation
  at::Tensor src_tensor = at::zeros({count}, options);
  src_tensor.fill_(static_cast<float>(rank_ + 1));

  // End pool context
  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Create destination window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  // Cast to NCCLX window to access device API
  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // Get device window with signals (returns device pointer for use in kernels)
  int signal_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, -1, 1);
  EXPECT_NE(dev_win, nullptr) << "Device window pointer should not be null";

  // Register source buffer as a local-only window using NCCL_WIN_LOCAL_ONLY.
  // This is NON-COLLECTIVE - uses the parent comm's PD but skips rkey
  // allGather. The resulting window can only be used as a source buffer for put
  // operations.
  auto src_buf = win->register_local_buffer(src_tensor);
  ASSERT_NE(src_buf.base_ptr, nullptr)
      << "Source buffer base_ptr should not be null";
  ASSERT_NE(src_buf.backend_window, nullptr)
      << "Source buffer backend_window should not be null";

  // Calculate ranks for ring pattern
  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  // Signal semantics: put() with signal_id increments that signal on the
  // DESTINATION rank. All ranks use signal 0. Each rank receives one put,
  // so waits for signal 0 >= 1.
  constexpr int kSignalId = 0;

  // Calculate offsets and bytes
  size_t elem_size = win_tensor.element_size();
  size_t bytes = count * elem_size;
  // Source: start of src_tensor (offset 0 within src_buf)
  size_t src_offset = 0;
  // Destination: our slot on the remote rank (rank i puts to slot i on dst)
  size_t dst_offset = rank_ * bytes;

  // Launch put kernel on put_stream
  {
    c10::cuda::CUDAStreamGuard guard(put_stream);
    torchcomms::device::test::launchDevicePutKernelWithOffsets(
        dev_win,
        src_buf,
        src_offset,
        dst_offset,
        bytes,
        dst_rank,
        kSignalId,
        put_stream.stream());
  }

  // Launch wait signal kernel on wait_stream
  // Wait for signal 0 to be incremented (indicating put completed to us)
  {
    c10::cuda::CUDAStreamGuard guard(wait_stream);
    torchcomms::device::test::launchDeviceWaitSignalKernel(
        dev_win, kSignalId, 1, wait_stream.stream());
  }

  // Synchronize streams
  put_stream.synchronize();
  wait_stream.synchronize();

  // Verify: the slot at src_rank's index should now contain src_rank's data
  // src_rank wrote (src_rank+1) to slot[src_rank]
  at::Tensor result_slice = win_tensor.index(
      {at::indexing::Slice(src_rank * count, (src_rank + 1) * count)});

  // Copy result to CPU for comparison
  at::Tensor result_cpu = result_slice.cpu();

  // Create expected tensor on CPU to avoid CUDA memory conflicts
  auto cpu_options = at::TensorOptions().dtype(dtype).device(at::kCPU);
  at::Tensor expected_cpu = at::zeros({count}, cpu_options);
  expected_cpu.fill_(static_cast<float>(src_rank + 1));

  bool equal = at::allclose(result_cpu, expected_cpu);
  ASSERT_TRUE(equal) << "Device put data mismatch: expected value "
                     << (src_rank + 1) << " from rank " << src_rank
                     << ", got first element: " << result_cpu[0].item<float>();

  // Reset signals for next iteration (if any)
  {
    c10::cuda::CUDAStreamGuard guard(put_stream);
    torchcomms::device::test::launchDeviceResetSignalKernel(
        dev_win, kSignalId, put_stream.stream());
  }
  put_stream.synchronize();

  // Cleanup - deregister local source buffer (non-collective), then destination
  win->deregister_local_buffer(src_buf);

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

// =============================================================================
// GTest Test Cases
// =============================================================================
// TEST_F macros MUST be in this file (compiled with
// TORCHCOMMS_HAS_NCCL_DEVICE_API) to ensure TorchCommWindowNCCLXGin resolves to
// the correct type (NCCLDeviceBackend).

TEST_F(DeviceApiTest, DeviceWindowCreationFloat) {
  testDeviceWindowCreation(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowCreationHalf) {
  testDeviceWindowCreation(1024, at::kHalf);
}

TEST_F(DeviceApiTest, LocalBufferRegistrationFloat) {
  // Test non-collective local buffer registration using NCCL_WIN_LOCAL_ONLY.
  // This uses the parent comm's PD but skips the rkey allGather, making
  // registration truly non-collective. The resulting window can only be used
  // as a source buffer for put operations.
  testLocalBufferRegistration(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowWithSignalsFloat) {
  testDeviceWindowWithSignals(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowWithCountersFloat) {
  testDeviceWindowWithCounters(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DevicePutFloat) {
  testDevicePut(1024, at::kFloat);
}

// =============================================================================
// GIN atomicAdd Test
// =============================================================================
// This test validates the gin.atomicAdd() API at the NCCLx layer:
//   1. Each rank creates a window with uint64_t slots
//   2. In a ring pattern, rank i atomically adds (rank+1) to slot[rank] on
//      the next rank's window
//   3. Signal the next rank after atomicAdd completes
//   4. Wait for signal from previous rank
//   5. Verify the atomically-added value matches expected

void DeviceApiTest::testGinAtomicAdd() {
  SCOPED_TRACE(::testing::Message() << "Testing GIN atomicAdd");

  auto op_stream = at::cuda::getStreamFromPool(false, device_index_);
  auto wait_stream = at::cuda::getStreamFromPool(false, device_index_);

  // Create MemPool for RDMA-compatible memory allocation
  auto mem_pool = std::make_unique<at::cuda::MemPool>(
      std::static_pointer_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator>(
          allocator_));
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      mem_pool->device(), mem_pool->id(), [](cudaStream_t) { return true; });

  // Window layout: num_ranks uint64_t slots, one per sender rank
  // Each slot is 8 bytes (sizeof(uint64_t))
  auto options =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_index_);
  at::Tensor win_tensor = at::zeros({num_ranks_}, options);

  c10::cuda::CUDACachingAllocator::endAllocateToPool(
      mem_pool->device(), mem_pool->id());

  // Create window and register tensor
  torchcomm_->barrier(false);
  auto base_win = torchcomm_->new_window();
  base_win->tensor_register(win_tensor);
  torchcomm_->barrier(false);

  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr) << "Window should be TorchCommWindowNCCLXGin";

  // Get device window with signals for synchronization
  int signal_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, -1, 1);
  ASSERT_NE(dev_win, nullptr) << "Device window pointer should not be null";

  // Ring pattern: rank i atomicAdds to rank (i+1) % num_ranks
  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;

  // Each rank writes to its own slot on the destination.
  // dst_offset = rank_ * sizeof(uint64_t) bytes into the window.
  size_t dst_offset = rank_ * sizeof(uint64_t);
  uint64_t add_value = static_cast<uint64_t>(rank_ + 1);
  constexpr int kSignalId = 0;

  // Launch atomicAdd kernel
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceGinAtomicAddKernel(
        dev_win,
        dst_offset,
        add_value,
        dst_rank,
        kSignalId,
        op_stream.stream());
  }

  // Wait for signal from src_rank indicating its atomicAdd completed to us
  {
    c10::cuda::CUDAStreamGuard guard(wait_stream);
    torchcomms::device::test::launchDeviceWaitSignalKernel(
        dev_win, kSignalId, 1, wait_stream.stream());
  }

  op_stream.synchronize();
  wait_stream.synchronize();

  // Verify: slot[src_rank] should have value (src_rank + 1) from the sender
  at::Tensor result_cpu = win_tensor.cpu();
  int64_t got = result_cpu[src_rank].item<int64_t>();
  int64_t expected = static_cast<int64_t>(src_rank + 1);
  ASSERT_EQ(got, expected) << "atomicAdd mismatch at slot[" << src_rank
                           << "]: expected " << expected << ", got " << got;

  // Reset signal for cleanup
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceResetSignalKernel(
        dev_win, kSignalId, op_stream.stream());
  }
  op_stream.synchronize();

  // Verify signal was reset to 0
  uint64_t* d_signal_out = nullptr;
  cudaMalloc(&d_signal_out, sizeof(uint64_t));
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceReadSignalKernel(
        dev_win, kSignalId, d_signal_out, op_stream.stream());
  }
  op_stream.synchronize();

  uint64_t signal_value = 0;
  cudaMemcpy(
      &signal_value, d_signal_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_signal_out);
  ASSERT_EQ(signal_value, 0) << "Signal should be reset to 0 after reset";

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(DeviceApiTest, GinAtomicAdd) {
  testGinAtomicAdd();
}

// =============================================================================
// Per-Peer Signal Test
// =============================================================================
// Validates the resource buffer per-peer signal slot model:
//   1. Each rank signals the next rank in a ring pattern (standalone signal,
//      no data transfer)
//   2. Each rank waits for the aggregated signal to reach expected value
//   3. Verifies read_signal returns the correct aggregated value

void DeviceApiTest::testPerPeerSignal() {
  SCOPED_TRACE(::testing::Message() << "Testing per-peer signal slots");

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

  auto* win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr);

  int signal_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, -1, 1);
  ASSERT_NE(dev_win, nullptr);

  // Ring pattern: rank i signals rank (i+1) % num_ranks
  int dst_rank = (rank_ + 1) % num_ranks_;
  constexpr int kSignalId = 0;

  // Each rank sends ADD 1 to the next rank
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceSignalKernel(
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
    torchcomms::device::test::launchDeviceWaitSignalKernel(
        dev_win, kSignalId, 1, wait_stream.stream());
  }

  op_stream.synchronize();
  wait_stream.synchronize();

  // Read signal value via kernel and verify on host
  uint64_t* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(uint64_t));
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceReadSignalKernel(
        dev_win, kSignalId, d_out, op_stream.stream());
  }
  op_stream.synchronize();

  uint64_t h_out = 0;
  cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  ASSERT_GE(h_out, 1u) << "Expected aggregated signal >= 1, got " << h_out;

  // Reset signals
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceResetSignalKernel(
        dev_win, kSignalId, op_stream.stream());
  }
  op_stream.synchronize();

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(DeviceApiTest, PerPeerSignal) {
  testPerPeerSignal();
}

// =============================================================================
// Wait Signal From Specific Peer Test
// =============================================================================
// Validates point-to-point signal synchronization via wait_signal_from():
//   1. Each rank signals the next rank in a ring
//   2. Receiver uses wait_signal_from(src_rank, ...) to wait for the
//      specific sender's slot (not aggregated)
//   3. Verifies that only the expected sender's slot was written

void DeviceApiTest::testWaitSignalFrom() {
  SCOPED_TRACE(::testing::Message() << "Testing wait_signal_from");

  auto op_stream = at::cuda::getStreamFromPool(false, device_index_);
  auto wait_stream = at::cuda::getStreamFromPool(false, device_index_);

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
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr);

  int signal_count = num_ranks_;
  auto* dev_win = win->get_device_window(signal_count, -1, 1);
  ASSERT_NE(dev_win, nullptr);

  // Ring: rank i signals rank (i+1), receiver expects signal from (i-1)
  int dst_rank = (rank_ + 1) % num_ranks_;
  int src_rank = (rank_ - 1 + num_ranks_) % num_ranks_;
  constexpr int kSignalId = 0;

  // Send signal
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceSignalKernel(
        dev_win,
        dst_rank,
        kSignalId,
        torchcomms::device::SignalOp::ADD,
        1,
        op_stream.stream());
  }

  // Wait for signal from specific peer (not aggregated)
  {
    c10::cuda::CUDAStreamGuard guard(wait_stream);
    torchcomms::device::test::launchDeviceWaitSignalFromKernel(
        dev_win,
        src_rank,
        kSignalId,
        torchcomms::device::CmpOp::GE,
        1,
        wait_stream.stream());
  }

  op_stream.synchronize();
  wait_stream.synchronize();

  // Reset signals
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceResetSignalKernel(
        dev_win, kSignalId, op_stream.stream());
  }
  op_stream.synchronize();

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(DeviceApiTest, WaitSignalFrom) {
  testWaitSignalFrom();
}

// =============================================================================
// Device Barrier Test
// =============================================================================
// Validates the device-side world barrier (LSA + GIN hierarchical):
//   1. All ranks launch a barrier kernel
//   2. Barrier synchronizes all ranks (LSA first, then GIN Rail)
//   3. All ranks complete successfully

void DeviceApiTest::testDeviceBarrier() {
  SCOPED_TRACE(::testing::Message() << "Testing device barrier");

  auto op_stream = at::cuda::getStreamFromPool(false, device_index_);

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
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(base_win.get());
  ASSERT_NE(win, nullptr);

  // Get device window with barrier support
  int barrier_count = 2;
  auto* dev_win = win->get_device_window(-1, -1, barrier_count);
  ASSERT_NE(dev_win, nullptr);

  constexpr int kBarrierId = 0;

  // Launch barrier kernel - all ranks must participate
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceBarrierKernel(
        dev_win, kBarrierId, op_stream.stream());
  }

  // Synchronize - if barrier works, all ranks complete together
  op_stream.synchronize();

  // Second barrier to verify reusability
  {
    c10::cuda::CUDAStreamGuard guard(op_stream);
    torchcomms::device::test::launchDeviceBarrierKernel(
        dev_win, kBarrierId + 1, op_stream.stream());
  }
  op_stream.synchronize();

  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(DeviceApiTest, DeviceBarrier) {
  testDeviceBarrier();
}
