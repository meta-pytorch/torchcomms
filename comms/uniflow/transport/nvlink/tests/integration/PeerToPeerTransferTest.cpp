// Copyright (c) Meta Platforms, Inc. and affiliates.
// Confidential and proprietary.
// Peer-to-Peer GPU transfer integration test for intra-node device-to-device
// transfers.
//
// This test verifies device-to-device P2P transfers over the GPU interconnect
// (XGMI on AMD, NVLink on NVIDIA) using the platform-agnostic CudaApi
// abstraction. Buffers are allocated in device memory (cudaMalloc / hipMalloc)
// and peer access is enabled in both directions, so memcpyPeerAsync actually
// traverses the GPU-to-GPU interconnect rather than staging through host
// memory; a host staging buffer is used only to seed device A and to read back
// device B for verification.

#include "comms/uniflow/drivers/cuda/CudaApi.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace uniflow {

namespace {

// Peer-to-peer GPU transfer test fixture.
// Tests intra-node GPU-to-GPU transfers over the GPU interconnect (XGMI on AMD,
// NVLink on NVIDIA) using the platform-agnostic CudaApi abstraction.
class PeerToPeerTransferTest : public ::testing::Test {
 protected:
  static constexpr int kDeviceA = 0;
  static constexpr int kDeviceB = 1;
  static constexpr size_t kTransferSize = 1 << 20; // 1MB

  void SetUp() override {
    // Check if we have at least 2 GPUs.
    auto deviceCount = cudaApi_->getDeviceCount();
    if (!deviceCount.hasValue() || deviceCount.value() < 2) {
      GTEST_SKIP() << "P2P transfer test requires at least 2 GPUs";
    }

    // Check if P2P is supported between devices.
    auto canAccess = cudaApi_->deviceCanAccessPeer(kDeviceA, kDeviceB);
    if (!canAccess.hasValue() || !canAccess.value()) {
      GTEST_SKIP() << "P2P not supported between GPU " << kDeviceA
                   << " and GPU " << kDeviceB;
    }

    // deviceCanAccessPeer only reports capability; peer access must actually be
    // enabled or memcpyPeerAsync may stage through host memory instead of going
    // over the GPU interconnect (XGMI/NVLink) -- which would make this test
    // pass without exercising the interconnect it is meant to validate. Enable
    // in both directions, each from its own device context
    // (deviceEnablePeerAccess acts on the current device). The seam treats
    // "already enabled" as success, so this is safe across both test cases.
    ASSERT_TRUE(cudaApi_->setDevice(kDeviceA).hasValue());
    ASSERT_TRUE(cudaApi_->deviceEnablePeerAccess(kDeviceB).hasValue())
        << "Failed to enable peer access " << kDeviceA << " -> " << kDeviceB;
    ASSERT_TRUE(cudaApi_->setDevice(kDeviceB).hasValue());
    ASSERT_TRUE(cudaApi_->deviceEnablePeerAccess(kDeviceA).hasValue())
        << "Failed to enable peer access " << kDeviceB << " -> " << kDeviceA;
  }

  // Release device buffers unconditionally, so a failing ASSERT_* in a test
  // body does not leak GPU memory.
  void TearDown() override {
    freeDeviceMemory(kDeviceA, devA_);
    freeDeviceMemory(kDeviceB, devB_);
    devA_ = nullptr;
    devB_ = nullptr;
  }

  // Build a host buffer of `size` bytes with a deterministic pattern.
  static std::vector<uint8_t> makePattern(size_t size) {
    std::vector<uint8_t> buf(size);
    for (size_t i = 0; i < size; ++i) {
      buf[i] = static_cast<uint8_t>((i * 37) & 0xFF);
    }
    return buf;
  }

  // Allocate device memory on `deviceId` via cudaMalloc (hipMalloc on AMD), so
  // P2P transfers traverse the GPU-to-GPU interconnect rather than host memory.
  void* allocateDeviceMemory(int deviceId, size_t size) {
    if (!cudaApi_->setDevice(deviceId).hasValue()) {
      return nullptr;
    }
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
      return nullptr;
    }
    return ptr;
  }

  void freeDeviceMemory(int deviceId, void* ptr) {
    if (ptr) {
      cudaApi_->setDevice(deviceId);
      // hipFree is [[nodiscard]] on HIP (cudaFree is not on CUDA); discard
      // explicitly so the single source compiles under -Werror on both.
      (void)cudaFree(ptr);
    }
  }

  void* devA_ = nullptr;
  void* devB_ = nullptr;
  std::shared_ptr<CudaApi> cudaApi_{std::make_shared<CudaApi>()};
};

// Test basic P2P transfer from Device A to Device B.
// Seeds device A from a host buffer, copies A->B over the interconnect, reads
// device B back, and verifies the round trip preserved the data.
TEST_F(PeerToPeerTransferTest, BasicP2PTransfer) {
  devA_ = allocateDeviceMemory(kDeviceA, kTransferSize);
  devB_ = allocateDeviceMemory(kDeviceB, kTransferSize);
  ASSERT_NE(devA_, nullptr) << "Failed to allocate device memory on device A";
  ASSERT_NE(devB_, nullptr) << "Failed to allocate device memory on device B";

  // Seed device A with a known pattern from host.
  const std::vector<uint8_t> srcHost = makePattern(kTransferSize);
  ASSERT_TRUE(cudaApi_->setDevice(kDeviceA).hasValue());
  ASSERT_TRUE(cudaApi_
                  ->memcpyAsync(
                      devA_,
                      srcHost.data(),
                      kTransferSize,
                      cudaMemcpyHostToDevice,
                      nullptr)
                  .hasValue())
      << "Failed to copy to device A";
  ASSERT_TRUE(cudaApi_->streamSynchronize(nullptr).hasValue());

  // Device-to-device P2P transfer: Device A -> Device B over the interconnect:
  // - On AMD: XGMI via hipMemcpyPeerAsync
  // - On NVIDIA: NVLink via cudaMemcpyPeerAsync
  auto status = cudaApi_->memcpyPeerAsync(
      devB_, kDeviceB, devA_, kDeviceA, kTransferSize, nullptr);
  ASSERT_TRUE(status.hasValue())
      << "P2P transfer failed: " << status.error().message();
  ASSERT_TRUE(cudaApi_->streamSynchronize(nullptr).hasValue());

  // Read device B back to host and verify against the original pattern.
  std::vector<uint8_t> dstHost(kTransferSize, 0);
  ASSERT_TRUE(cudaApi_->setDevice(kDeviceB).hasValue());
  ASSERT_TRUE(cudaApi_
                  ->memcpyAsync(
                      dstHost.data(),
                      devB_,
                      kTransferSize,
                      cudaMemcpyDeviceToHost,
                      nullptr)
                  .hasValue())
      << "Failed to copy from device B";
  ASSERT_TRUE(cudaApi_->streamSynchronize(nullptr).hasValue());

  EXPECT_EQ(srcHost, dstHost) << "P2P transfer data mismatch";
}

// Test multiple small P2P transfers (simulates real-world usage). Each chunk is
// staged into device A then copied A->B device-to-device; the whole device B
// buffer is read back once and compared against the constructed expected
// buffer.
TEST_F(PeerToPeerTransferTest, MultipleSmallP2PTransfers) {
  constexpr size_t kNumTransfers = 10;
  constexpr size_t kSmallSize = 4096; // 4KB each
  constexpr size_t kTotalSize = kNumTransfers * kSmallSize;
  static_assert(kTotalSize <= kTransferSize, "transfers exceed buffer size");

  devA_ = allocateDeviceMemory(kDeviceA, kTransferSize);
  devB_ = allocateDeviceMemory(kDeviceB, kTransferSize);
  ASSERT_NE(devA_, nullptr);
  ASSERT_NE(devB_, nullptr);

  // Construct the expected buffer host-side: chunk i carries the byte value i.
  std::vector<uint8_t> expected(kTotalSize);
  for (size_t i = 0; i < kNumTransfers; ++i) {
    std::fill_n(
        expected.begin() + i * kSmallSize,
        kSmallSize,
        static_cast<uint8_t>(i & 0xFF));
  }

  for (size_t i = 0; i < kNumTransfers; ++i) {
    const size_t offset = i * kSmallSize;

    // Stage chunk i into device A from host.
    ASSERT_TRUE(cudaApi_->setDevice(kDeviceA).hasValue());
    ASSERT_TRUE(cudaApi_
                    ->memcpyAsync(
                        static_cast<uint8_t*>(devA_) + offset,
                        expected.data() + offset,
                        kSmallSize,
                        cudaMemcpyHostToDevice,
                        nullptr)
                    .hasValue())
        << "H2D staging for transfer " << i << " failed";
    ASSERT_TRUE(cudaApi_->streamSynchronize(nullptr).hasValue());

    // Device-to-device P2P transfer for chunk i.
    auto status = cudaApi_->memcpyPeerAsync(
        static_cast<uint8_t*>(devB_) + offset,
        kDeviceB,
        static_cast<uint8_t*>(devA_) + offset,
        kDeviceA,
        kSmallSize,
        nullptr);
    ASSERT_TRUE(status.hasValue())
        << "P2P transfer " << i << " failed: " << status.error().message();
    ASSERT_TRUE(cudaApi_->streamSynchronize(nullptr).hasValue());
  }

  // Read the whole device B buffer back and compare in one shot.
  std::vector<uint8_t> actual(kTotalSize, 0);
  ASSERT_TRUE(cudaApi_->setDevice(kDeviceB).hasValue());
  ASSERT_TRUE(
      cudaApi_
          ->memcpyAsync(
              actual.data(), devB_, kTotalSize, cudaMemcpyDeviceToHost, nullptr)
          .hasValue())
      << "Failed to copy from device B";
  ASSERT_TRUE(cudaApi_->streamSynchronize(nullptr).hasValue());

  EXPECT_EQ(actual, expected) << "P2P transfer data mismatch";
}

} // namespace
} // namespace uniflow
