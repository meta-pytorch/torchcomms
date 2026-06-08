// Copyright (c) Meta Platforms, Inc. and affiliates.
// Confidential and proprietary.
// XGMI Peer-to-Peer Integration Test for AMD GPU intra-node transfers
//
// This test verifies P2P transfers over XGMI (AMD) or NVLink (NVIDIA) using
// the platform-agnostic CudaApi abstraction. The test works on both platforms
// by using CudaApi methods which have HIP implementations for AMD.

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h> // @manual=third-party//rocm:hip
#else
#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy
#endif

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace uniflow {

namespace {

// XGMI Peer-to-Peer test fixture
// Tests intra-node GPU-to-GPU transfers over XGMI (AMD) or NVLink (NVIDIA)
// using the platform-agnostic CudaApi abstraction.
class XGMIPeerToPeerTest : public ::testing::Test {
 protected:
  static constexpr int kDeviceA = 0;
  static constexpr int kDeviceB = 1;
  static constexpr size_t kTransferSize = 1 << 20; // 1MB

  void SetUp() override {
    // Check if we have at least 2 GPUs
    auto deviceCount = cudaApi_.getDeviceCount();
    if (!deviceCount.hasValue() || deviceCount.value() < 2) {
      GTEST_SKIP() << "XGMI P2P test requires at least 2 GPUs";
    }

    // Check if P2P is supported between devices
    auto canAccess = cudaApi_.deviceCanAccessPeer(kDeviceA, kDeviceB);
    if (!canAccess.hasValue() || !canAccess.value()) {
      GTEST_SKIP() << "P2P not supported between GPU " << kDeviceA
                   << " and GPU " << kDeviceB;
    }
  }

  void TearDown() override {}

  // Fill buffer with a deterministic pattern for verification
  void fillPattern(std::vector<uint8_t>& buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
      buf[i] = static_cast<uint8_t>((i * 37) & 0xFF);
    }
  }

  // Allocate pinned host memory using CudaApi abstraction
  // This memory is accessible from GPUs and suitable for P2P transfers.
  // Works on both CUDA (NVIDIA) and HIP (AMD) platforms.
  // On AMD: Uses hipHostMalloc which is P2P-capable over XGMI
  // On NVIDIA: Uses cudaHostAlloc which is P2P-capable over NVLink
  void* allocatePinnedHostMemory(int deviceId, size_t size) {
    cudaApi_.setDevice(deviceId);
    auto result = cudaApi_.hostAlloc(size, 0);
    if (result.hasValue()) {
      return result.value();
    }
    return nullptr;
  }

  void freePinnedHostMemory(void* ptr) {
    if (ptr) {
      cudaApi_.hostFree(ptr);
    }
  }

  CudaApi cudaApi_;
  CudaDriverApi driverApi_;
  ScopedEventBaseThread evbThread_{"XGMIPeerToPeerTest"};
};

// Test basic P2P transfer from Device A to Device B
// This verifies the XGMI/NVLink data path works correctly
TEST_F(XGMIPeerToPeerTest, BasicP2PTransfer) {
  // Allocate memory on both devices
  void* devA = allocatePinnedHostMemory(kDeviceA, kTransferSize);
  void* devB = allocatePinnedHostMemory(kDeviceB, kTransferSize);
  ASSERT_NE(devA, nullptr) << "Failed to allocate memory on device A";
  ASSERT_NE(devB, nullptr) << "Failed to allocate memory on device B";

  // Fill device A with pattern
  std::vector<uint8_t> srcHost(kTransferSize);
  fillPattern(srcHost);

  cudaApi_.setDevice(kDeviceA);
#ifdef __HIP_PLATFORM_AMD__
  auto status = cudaApi_.memcpyAsync(
      devA, srcHost.data(), kTransferSize, hipMemcpyHostToDevice, nullptr);
#else
  auto status = cudaApi_.memcpyAsync(
      devA, srcHost.data(), kTransferSize, cudaMemcpyHostToDevice, nullptr);
#endif
  ASSERT_TRUE(status.hasValue()) << "Failed to copy to device A";
  cudaApi_.streamSynchronize(nullptr);

  // Perform P2P transfer: Device A -> Device B
  // This uses the underlying interconnect:
  // - On AMD: XGMI via hipMemcpyPeerAsync
  // - On NVIDIA: NVLink via cudaMemcpyPeerAsync
  // The CudaApi abstraction handles platform differences internally.
  cudaApi_.setDevice(kDeviceA);
  status = cudaApi_.memcpyPeerAsync(
      devB, kDeviceB, devA, kDeviceA, kTransferSize, nullptr);
  ASSERT_TRUE(status.hasValue())
      << "P2P transfer failed: " << status.error().message();
  cudaApi_.streamSynchronize(nullptr);

  // Copy device B back to host and verify
  std::vector<uint8_t> dstHost(kTransferSize, 0);
  cudaApi_.setDevice(kDeviceB);
#ifdef __HIP_PLATFORM_AMD__
  status = cudaApi_.memcpyAsync(
      dstHost.data(), devB, kTransferSize, hipMemcpyDeviceToHost, nullptr);
#else
  status = cudaApi_.memcpyAsync(
      dstHost.data(), devB, kTransferSize, cudaMemcpyDeviceToHost, nullptr);
#endif
  ASSERT_TRUE(status.hasValue()) << "Failed to copy from device B";
  cudaApi_.streamSynchronize(nullptr);

  // Verify data integrity
  EXPECT_EQ(srcHost, dstHost) << "P2P transfer data mismatch";

  // Cleanup
  freePinnedHostMemory(devA);
  freePinnedHostMemory(devB);
}

// Test multiple small P2P transfers (simulates real-world usage)
TEST_F(XGMIPeerToPeerTest, MultipleSmallP2PTransfers) {
  constexpr size_t kNumTransfers = 10;
  constexpr size_t kSmallSize = 4096; // 4KB each

  void* devA = allocatePinnedHostMemory(kDeviceA, kTransferSize);
  void* devB = allocatePinnedHostMemory(kDeviceB, kTransferSize);
  ASSERT_NE(devA, nullptr);
  ASSERT_NE(devB, nullptr);

  // Perform multiple small transfers
  for (size_t i = 0; i < kNumTransfers; ++i) {
    std::vector<uint8_t> src(kSmallSize, static_cast<uint8_t>(i & 0xFF));
    size_t offset = i * kSmallSize;

    // Copy to device A
    cudaApi_.setDevice(kDeviceA);
#ifdef __HIP_PLATFORM_AMD__
    cudaApi_.memcpyAsync(
        static_cast<uint8_t*>(devA) + offset,
        src.data(),
        kSmallSize,
        hipMemcpyHostToDevice,
        nullptr);
    cudaApi_.streamSynchronize(nullptr);

    // P2P transfer A->B
    auto status = cudaApi_.memcpyPeerAsync(
        static_cast<uint8_t*>(devB) + offset,
        kDeviceB,
        static_cast<uint8_t*>(devA) + offset,
        kDeviceA,
        kSmallSize,
        nullptr);
#else
    cudaApi_.memcpyAsync(
        static_cast<uint8_t*>(devA) + offset,
        src.data(),
        kSmallSize,
        cudaMemcpyHostToDevice,
        nullptr);
    cudaApi_.streamSynchronize(nullptr);

    // P2P transfer A->B
    auto status = cudaApi_.memcpyPeerAsync(
        static_cast<uint8_t*>(devB) + offset,
        kDeviceB,
        static_cast<uint8_t*>(devA) + offset,
        kDeviceA,
        kSmallSize,
        nullptr);
#endif
    ASSERT_TRUE(status.hasValue()) << "Transfer " << i << " failed";
    cudaApi_.streamSynchronize(nullptr);
  }

  // Verify all transfers
  std::vector<uint8_t> dst(kTransferSize);
  cudaApi_.setDevice(kDeviceB);
#ifdef __HIP_PLATFORM_AMD__
  cudaApi_.memcpyAsync(
      dst.data(), devB, kTransferSize, hipMemcpyDeviceToHost, nullptr);
#else
  cudaApi_.memcpyAsync(
      dst.data(), devB, kTransferSize, cudaMemcpyDeviceToHost, nullptr);
#endif
  cudaApi_.streamSynchronize(nullptr);

  for (size_t i = 0; i < kNumTransfers; ++i) {
    size_t offset = i * kSmallSize;
    uint8_t expected = static_cast<uint8_t>(i & 0xFF);
    for (size_t j = 0; j < kSmallSize; ++j) {
      EXPECT_EQ(dst[offset + j], expected)
          << "Mismatch at transfer " << i << " byte " << j;
    }
  }

  freePinnedHostMemory(devA);
  freePinnedHostMemory(devB);
}

} // namespace
} // namespace uniflow
