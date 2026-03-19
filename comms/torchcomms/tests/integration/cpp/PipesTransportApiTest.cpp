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

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

std::unique_ptr<TorchCommTestWrapper> PipesTransportApiTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void PipesTransportApiTest::SetUp() {
  // Check skip condition FIRST, before any initialization
  if (checkIfSkip()) {
    GTEST_SKIP() << "Skipping Pipes Transport API tests "
                    "(RUN_PIPES_TRANSPORT_API_TEST not set)";
  }

  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  device_index_ = (device_count > 0) ? rank_ % device_count : 0;
}

void PipesTransportApiTest::TearDown() {
  torchcomm_.reset();
  wrapper_.reset();
}

bool PipesTransportApiTest::checkIfSkip() {
  // Check RUN_PIPES_TRANSPORT_API_TEST env var
  const char* run_env = getenv("RUN_PIPES_TRANSPORT_API_TEST");
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
