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

  // Create and register local source buffer
  at::Tensor src_tensor = createTestTensor(count, dtype);
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

  // TODO: The current NCCL GIN implementation
  // and destination windows are registered with the same communicator. Windows
  // registered via ncclCommSplit (local_comm_) have separate window tables and
  // cannot be used with the parent comm's ncclDevComm. As a temporary
  // workaround, we register the source buffer as a separate collective window.
  // In the future, we should either:
  //   1. Implement proper non-collective local buffer registration for GIN
  //   2. Use a different approach like LSA (Load-Store Access) for local
  //   buffers
  //   3. Work with NCCL team to enable cross-comm window access for split comms
  //
  // Create source window (COLLECTIVE - all ranks must participate)
  torchcomm_->barrier(false);
  auto src_base_win = torchcomm_->new_window();
  src_base_win->tensor_register(src_tensor);
  torchcomm_->barrier(false);

  auto* src_win =
      dynamic_cast<torch::comms::TorchCommWindowNCCLXGin*>(src_base_win.get());
  ASSERT_NE(src_win, nullptr)
      << "Source window should be TorchCommWindowNCCLXGin";

  // Get the source device window to access its nccl_orig_win_ via window_ field
  auto* src_dev_win = src_win->get_device_window(signal_count, -1, 1);
  ASSERT_NE(src_dev_win, nullptr)
      << "Source device window pointer should not be null";

  // Create RegisteredBuffer pointing to the source window's nccl_orig_win_
  // Note: We need to copy window_ from device memory to access it on host.
  // For now, we use the host-side NCCL window from the TorchCommWindow.
  torchcomms::device::RegisteredBufferNCCL src_buf;
  src_buf.base_ptr = src_tensor.data_ptr();
  src_buf.size = count * src_tensor.element_size();
  src_buf.backend_window = src_win->get_nccl_window();
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

  // Cleanup - deregister source window first (collective), then destination
  src_base_win->tensor_deregister();
  src_base_win.reset();

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
// the correct type (NCCLGinBackend).

TEST_F(DeviceApiTest, DeviceWindowCreationFloat) {
  testDeviceWindowCreation(1024, at::kFloat);
}

TEST_F(DeviceApiTest, DeviceWindowCreationHalf) {
  testDeviceWindowCreation(1024, at::kHalf);
}

TEST_F(DeviceApiTest, LocalBufferRegistrationFloat) {
  // TODO(T123456789): Skip this test until initLocalComm() is fixed.
  // The register_local_buffer() API requires a local split communicator
  // (ncclCommSplit), but this is currently disabled because split-comm
  // windows have separate window tables from parent ncclDevComm.
  // The DevicePutFloat test uses collective window registration as a
  // workaround.
  GTEST_SKIP() << "Skipping: register_local_buffer() requires initLocalComm() "
                  "which is currently disabled";
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

  // Cleanup
  base_win->tensor_deregister();
  base_win.reset();
  mem_pool.reset();

  torchcomm_->barrier(false);
}

TEST_F(DeviceApiTest, GinAtomicAdd) {
  testGinAtomicAdd();
}
