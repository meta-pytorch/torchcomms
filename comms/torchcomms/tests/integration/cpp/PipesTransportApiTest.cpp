// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Pipes Transport API Integration Test

#include "PipesTransportApiTest.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include "PipesTransportApiTestKernels.cuh"
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

#include "comms/pipes/MultiPeerDeviceHandle.cuh"

#include <cstring>
#include <vector>

std::unique_ptr<TorchCommTestWrapper> PipesTransportApiTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void PipesTransportApiTest::SetUp() {
  // Check skip condition FIRST, before any initialization
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping Pipes Transport API tests "
                    "(RUN_PIPES_DEVICE_API_TEST not set)";
  }

  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess)
      << "cudaGetDeviceCount failed";
  device_index_ = (device_count > 0) ? rank_ % device_count : 0;
}

void PipesTransportApiTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

bool PipesTransportApiTest::checkIfSkip() {
  // Check RUN_PIPES_DEVICE_API_TEST env var (unified gate for all pipes tests)
  const char* run_env = getenv("RUN_PIPES_DEVICE_API_TEST");
  if (!run_env) {
    return true; // skip if not set
  }
  std::string val(run_env);
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  if (val != "1" && val != "true") {
    return true; // skip if not enabled
  }

  return false;
}

// =============================================================================
// Helper: fetch transport handle
// =============================================================================

bool PipesTransportApiTest::getTransportHandle(
    comms::pipes::MultiPeerDeviceHandle& handle) {
  auto ncclx = std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(
      torchcomm_->getBackendImpl());
  EXPECT_NE(ncclx, nullptr) << "Backend is not TorchCommNCCLX";
  if (!ncclx) {
    return false;
  }

  auto handle_ptr = ncclx->get_device_transport();
  EXPECT_NE(handle_ptr, 0) << "get_device_transport returned null";
  if (handle_ptr == 0) {
    return false;
  }

  auto copy_err = cudaMemcpy(
      &handle,
      reinterpret_cast<void*>(handle_ptr),
      sizeof(comms::pipes::MultiPeerDeviceHandle),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(copy_err, cudaSuccess) << "Failed to copy transport handle";
  return copy_err == cudaSuccess;
}

// =============================================================================
// Get Device Transport Test
// =============================================================================
// Validates that TorchCommNCCLX::get_device_transport() returns a valid
// MultiPeerDeviceHandle with correct rank info and non-null transport array.

void PipesTransportApiTest::testGetDeviceTransport() {
  SCOPED_TRACE(::testing::Message() << "Testing get_device_transport()");

  auto ncclx = std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(
      torchcomm_->getBackendImpl());
  ASSERT_NE(ncclx, nullptr) << "Backend is not TorchCommNCCLX";

  try {
    auto handle_ptr = ncclx->get_device_transport();
    ASSERT_NE(handle_ptr, 0) << "get_device_transport returned null";

    // Copy the device-allocated handle back to host for validation
    comms::pipes::MultiPeerDeviceHandle handle{};
    auto copy_err = cudaMemcpy(
        &handle,
        reinterpret_cast<void*>(handle_ptr),
        sizeof(comms::pipes::MultiPeerDeviceHandle),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(copy_err, cudaSuccess)
        << "Failed to copy transport handle to host";

    EXPECT_EQ(handle.myRank, rank_);
    EXPECT_EQ(handle.nRanks, num_ranks_);
    EXPECT_NE(handle.transports.data(), nullptr);
    EXPECT_GE(handle.numNvlPeers, 0);
    EXPECT_GE(handle.numIbPeers, 0);
  } catch (const std::runtime_error& e) {
    GTEST_SKIP() << "Pipes transport not available: " << e.what();
  }
}

TEST_F(PipesTransportApiTest, GetDeviceTransport) {
  testGetDeviceTransport();
}

// =============================================================================
// NVL Send/Recv Test
// =============================================================================
// Rank 0 sends a buffer filled with 0x42 to rank 1 via NVLink transport.
// Rank 1 verifies the received data matches.
// Requires at least 2 ranks and NVL peers.

void PipesTransportApiTest::testNvlSendRecv(size_t nbytes) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing NVL send/recv with nbytes=" << nbytes);

  ASSERT_GE(num_ranks_, 2) << "Need at least 2 ranks for send/recv test";

  auto ncclx = std::dynamic_pointer_cast<torch::comms::TorchCommNCCLX>(
      torchcomm_->getBackendImpl());
  ASSERT_NE(ncclx, nullptr) << "Backend is not TorchCommNCCLX";

  auto handle_ptr = ncclx->get_device_transport();
  ASSERT_NE(handle_ptr, 0) << "get_device_transport returned null";

  // Copy the device-allocated handle back to host for kernel launch
  comms::pipes::MultiPeerDeviceHandle handle{};
  auto copy_err = cudaMemcpy(
      &handle,
      reinterpret_cast<void*>(handle_ptr),
      sizeof(comms::pipes::MultiPeerDeviceHandle),
      cudaMemcpyDeviceToHost);
  ASSERT_EQ(copy_err, cudaSuccess) << "Failed to copy transport handle to host";

  if (handle.numNvlPeers == 0) {
    GTEST_SKIP() << "No NVL peers available — send/recv requires NVLink";
  }

  void* buf_d = nullptr;
  auto cuda_err = cudaMalloc(&buf_d, nbytes);
  ASSERT_EQ(cuda_err, cudaSuccess) << "cudaMalloc failed";

  int peer = (rank_ == 0) ? 1 : 0;

  if (rank_ == 0) {
    cuda_err = cudaMemset(buf_d, 0x42, nbytes);
    ASSERT_EQ(cuda_err, cudaSuccess);
  } else {
    cuda_err = cudaMemset(buf_d, 0x00, nbytes);
    ASSERT_EQ(cuda_err, cudaSuccess);
  }

  torchcomm_->barrier(false);

  if (rank_ == 0) {
    torchcomms::device::test::launchNvlSendKernel(handle, peer, buf_d, nbytes);
  } else if (rank_ == 1) {
    torchcomms::device::test::launchNvlRecvKernel(handle, peer, buf_d, nbytes);
  }

  cuda_err = cudaDeviceSynchronize();
  ASSERT_EQ(cuda_err, cudaSuccess) << "Kernel execution failed";

  if (rank_ == 1) {
    std::vector<uint8_t> host_buf(nbytes);
    cuda_err =
        cudaMemcpy(host_buf.data(), buf_d, nbytes, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cuda_err, cudaSuccess);

    std::vector<uint8_t> expected(nbytes, 0x42);
    ASSERT_EQ(memcmp(host_buf.data(), expected.data(), nbytes), 0)
        << "Data mismatch in received buffer";
  }

  torchcomm_->barrier(false);
  ASSERT_EQ(cudaFree(buf_d), cudaSuccess) << "cudaFree failed";
}

TEST_F(PipesTransportApiTest, NvlSendRecvSmall) {
  testNvlSendRecv(4096);
}

TEST_F(PipesTransportApiTest, NvlSendRecvLarge) {
  testNvlSendRecv(1024 * 1024);
}

// =============================================================================
// NVL Signal Test
// =============================================================================
// Both ranks signal each other (ADD 1 on signal_id 0) and wait (GE 1).
// Validates cross-GPU signal/wait via the transport handle.

void PipesTransportApiTest::testNvlSignal() {
  SCOPED_TRACE(::testing::Message() << "Testing NVL signal");

  ASSERT_GE(num_ranks_, 2) << "Need at least 2 ranks for signal test";

  comms::pipes::MultiPeerDeviceHandle handle{};
  ASSERT_TRUE(getTransportHandle(handle));

  if (handle.numNvlPeers == 0) {
    GTEST_SKIP() << "No NVL peers available — signal requires NVLink";
  }

  int peer = (rank_ == 0) ? 1 : 0;

  torchcomm_->barrier(false);

  torchcomms::device::test::launchNvlSignalKernel(handle, peer);

  auto cuda_err = cudaDeviceSynchronize();
  ASSERT_EQ(cuda_err, cudaSuccess) << "Signal kernel execution failed";

  torchcomm_->barrier(false);
}

TEST_F(PipesTransportApiTest, NvlSignal) {
  testNvlSignal();
}

// =============================================================================
// NVL LL128 Send/Recv Test
// =============================================================================
// Rank 0 sends a buffer filled with 0xAB to rank 1 via LL128 protocol.
// Rank 1 receives and verifies the data.
// LL128 requires 16-byte alignment and nbytes multiple of 16.

void PipesTransportApiTest::testNvlLl128SendRecv(size_t nbytes) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing NVL LL128 send/recv nbytes=" << nbytes);

  ASSERT_GE(num_ranks_, 2) << "Need at least 2 ranks for LL128 test";
  ASSERT_EQ(nbytes % 16, 0u) << "LL128 requires nbytes multiple of 16";

  comms::pipes::MultiPeerDeviceHandle handle{};
  ASSERT_TRUE(getTransportHandle(handle));

  if (handle.numNvlPeers == 0) {
    GTEST_SKIP() << "No NVL peers available — LL128 requires NVLink";
  }

  // Check LL128 availability on host side BEFORE launching kernels.
  // Both ranks must agree on skip/run to avoid one side hanging.
  int peer = (rank_ == 0) ? 1 : 0;
  int ll128_available =
      torchcomms::device::test::checkLl128Available(handle, peer);
  if (!ll128_available) {
    GTEST_SKIP() << "LL128 not configured on this hardware";
  }

  void* buf_d = nullptr;
  auto cuda_err = cudaMalloc(&buf_d, nbytes);
  ASSERT_EQ(cuda_err, cudaSuccess) << "cudaMalloc failed";

  if (rank_ == 0) {
    cuda_err = cudaMemset(buf_d, 0xAB, nbytes);
    ASSERT_EQ(cuda_err, cudaSuccess);
  } else {
    cuda_err = cudaMemset(buf_d, 0x00, nbytes);
    ASSERT_EQ(cuda_err, cudaSuccess);
  }

  torchcomm_->barrier(false);

  if (rank_ == 0) {
    torchcomms::device::test::launchNvlLl128SendKernel(
        handle, peer, buf_d, nbytes);
  } else if (rank_ == 1) {
    torchcomms::device::test::launchNvlLl128RecvKernel(
        handle, peer, buf_d, nbytes);
  }

  cuda_err = cudaDeviceSynchronize();
  ASSERT_EQ(cuda_err, cudaSuccess) << "LL128 kernel execution failed";

  if (rank_ == 1) {
    std::vector<uint8_t> host_buf(nbytes);
    cuda_err =
        cudaMemcpy(host_buf.data(), buf_d, nbytes, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cuda_err, cudaSuccess);

    std::vector<uint8_t> expected(nbytes, 0xAB);
    ASSERT_EQ(memcmp(host_buf.data(), expected.data(), nbytes), 0)
        << "Data mismatch in LL128 received buffer";
  }

  torchcomm_->barrier(false);
  ASSERT_EQ(cudaFree(buf_d), cudaSuccess) << "cudaFree failed";
}

TEST_F(PipesTransportApiTest, NvlLl128SendRecvSmall) {
  testNvlLl128SendRecv(1024);
}

TEST_F(PipesTransportApiTest, NvlLl128SendRecvLarge) {
  testNvlLl128SendRecv(65536);
}
