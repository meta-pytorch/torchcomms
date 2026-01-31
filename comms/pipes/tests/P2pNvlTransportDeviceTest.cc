// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <vector>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/tests/P2pNvlTransportDeviceTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes {

// =============================================================================
// Single-GPU Test Fixture for Signal Struct Tests
// =============================================================================

class P2pNvlTransportDeviceTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {}
};

// =============================================================================
// Two-GPU Test Fixture for P2pNvlTransportDevice Tests
// Requires 2 GPUs with P2P access enabled
// =============================================================================

class P2pNvlTransportDeviceTwoGpuFixture : public ::testing::Test {
 protected:
  static constexpr int kGpu0 = 0;
  static constexpr int kGpu1 = 1;

  cudaStream_t stream0_;
  cudaStream_t stream1_;

  void SetUp() override {
    int deviceCount = 0;
    CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      GTEST_SKIP() << "Test requires at least 2 GPUs";
    }

    // Check P2P access capability
    int canAccessPeer01 = 0;
    int canAccessPeer10 = 0;
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccessPeer01, kGpu0, kGpu1));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccessPeer10, kGpu1, kGpu0));
    if (!canAccessPeer01 || !canAccessPeer10) {
      GTEST_SKIP() << "Test requires P2P access between GPU 0 and GPU 1";
    }

    // Enable bidirectional P2P access
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    auto err0 = cudaDeviceEnablePeerAccess(kGpu1, 0);
    if (err0 == cudaErrorPeerAccessAlreadyEnabled) {
      // Clear the error from the runtime state
      cudaGetLastError();
    } else if (err0 != cudaSuccess) {
      CUDACHECK_TEST(err0);
    }
    CUDACHECK_TEST(cudaStreamCreate(&stream0_));

    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    auto err1 = cudaDeviceEnablePeerAccess(kGpu0, 0);
    if (err1 == cudaErrorPeerAccessAlreadyEnabled) {
      // Clear the error from the runtime state
      cudaGetLastError();
    } else if (err1 != cudaSuccess) {
      CUDACHECK_TEST(err1);
    }
    CUDACHECK_TEST(cudaStreamCreate(&stream1_));
  }

  void TearDown() override {
    // Cleanup streams
    cudaSetDevice(kGpu0);
    cudaStreamDestroy(stream0_);
    cudaSetDevice(kGpu1);
    cudaStreamDestroy(stream1_);
  }
};

// =============================================================================
// Signal Struct Unit Tests
// These test the Signal struct directly without P2pNvlTransportDevice
// =============================================================================

TEST_F(P2pNvlTransportDeviceTestFixture, SignalBasicSet) {
  // Allocate a single Signal on device
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Signal with SIGNAL_SET to value 42
  test::testRawSignal(signal_d, SignalOp::SIGNAL_SET, 42, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Read back the signal value
  test::testReadSignal(signal_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  ASSERT_EQ(result_h, 42) << "Signal value should be 42 after SIGNAL_SET";

  CUDACHECK_TEST(cudaFree(signal_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalBasicAdd) {
  // Allocate a single Signal on device
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Signal with SIGNAL_ADD to add 10
  test::testRawSignal(signal_d, SignalOp::SIGNAL_ADD, 10, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Signal with SIGNAL_ADD to add 5 more
  test::testRawSignal(signal_d, SignalOp::SIGNAL_ADD, 5, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Read back the signal value
  test::testReadSignal(signal_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  ASSERT_EQ(result_h, 15) << "Signal value should be 15 after two SIGNAL_ADDs";

  CUDACHECK_TEST(cudaFree(signal_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalWaitCmpEq) {
  // Allocate a single Signal on device
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // First set the signal to expected value
  test::testRawSignal(
      signal_d, SignalOp::SIGNAL_SET, 100, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_EQ should complete immediately since value is already 100
  test::testRawWaitSignal(signal_d, CmpOp::CMP_EQ, 100, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // If we get here without hanging, the test passed
  SUCCEED();

  CUDACHECK_TEST(cudaFree(signal_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalWaitCmpGe) {
  // Allocate a single Signal on device
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Set the signal to 50
  test::testRawSignal(signal_d, SignalOp::SIGNAL_SET, 50, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_GE 40 should complete immediately since 50 >= 40
  test::testRawWaitSignal(signal_d, CmpOp::CMP_GE, 40, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_GE 50 should also complete since 50 >= 50
  test::testRawWaitSignal(signal_d, CmpOp::CMP_GE, 50, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaFree(signal_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalWaitCmpGt) {
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Set the signal to 50
  test::testRawSignal(signal_d, SignalOp::SIGNAL_SET, 50, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_GT 40 should complete since 50 > 40
  test::testRawWaitSignal(signal_d, CmpOp::CMP_GT, 40, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaFree(signal_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalWaitCmpLe) {
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Set the signal to 30
  test::testRawSignal(signal_d, SignalOp::SIGNAL_SET, 30, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_LE 50 should complete since 30 <= 50
  test::testRawWaitSignal(signal_d, CmpOp::CMP_LE, 50, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_LE 30 should also complete since 30 <= 30
  test::testRawWaitSignal(signal_d, CmpOp::CMP_LE, 30, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaFree(signal_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalWaitCmpLt) {
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Set the signal to 30
  test::testRawSignal(signal_d, SignalOp::SIGNAL_SET, 30, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_LT 50 should complete since 30 < 50
  test::testRawWaitSignal(signal_d, CmpOp::CMP_LT, 50, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaFree(signal_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalWaitCmpNe) {
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Set the signal to 42
  test::testRawSignal(signal_d, SignalOp::SIGNAL_SET, 42, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Wait for CMP_NE 0 should complete since 42 != 0
  test::testRawWaitSignal(signal_d, CmpOp::CMP_NE, 0, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaFree(signal_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalBlockGroups) {
  // Test with block-level thread groups
  SignalState* signal_d;
  CUDACHECK_TEST(cudaMalloc(&signal_d, sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signal_d, 0, sizeof(SignalState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 256;

  // Signal with SIGNAL_SET using block groups
  test::testRawSignal(
      signal_d,
      SignalOp::SIGNAL_SET,
      123,
      numBlocks,
      blockSize,
      test::GroupType::BLOCK);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Read back the signal value
  test::testReadSignal(signal_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  ASSERT_EQ(result_h, 123)
      << "Signal value should be 123 after SIGNAL_SET with block groups";

  // Wait with block groups
  test::testRawWaitSignal(
      signal_d,
      CmpOp::CMP_EQ,
      123,
      numBlocks,
      blockSize,
      test::GroupType::BLOCK);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaFree(signal_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(P2pNvlTransportDeviceTestFixture, SignalMultipleSignals) {
  // Test with multiple Signal objects in an array
  const int numSignals = 8;
  SignalState* signals_d;
  CUDACHECK_TEST(cudaMalloc(&signals_d, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(cudaMemset(signals_d, 0, numSignals * sizeof(SignalState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Signal each signal with a different value
  for (int i = 0; i < numSignals; ++i) {
    test::testRawSignal(
        &signals_d[i], SignalOp::SIGNAL_SET, i * 10, numBlocks, blockSize);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify each signal has the correct value
  for (int i = 0; i < numSignals; ++i) {
    test::testReadSignal(&signals_d[i], result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    uint64_t result_h;
    CUDACHECK_TEST(cudaMemcpy(
        &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    ASSERT_EQ(result_h, static_cast<uint64_t>(i * 10))
        << "Signal " << i << " should have value " << (i * 10);
  }

  CUDACHECK_TEST(cudaFree(signals_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

// =============================================================================
// P2pNvlTransportDevice Signal API Tests (Two-GPU Configuration)
// These tests use 2 GPUs with P2P access to test cross-GPU signaling
// =============================================================================

TEST_F(P2pNvlTransportDeviceTwoGpuFixture, DeviceSignalTwoGpu) {
  // Allocate signal buffers on each GPU
  const int numSignals = 8;

  // GPU 0's signal buffer
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  SignalState* signalBuffer0;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer0, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer0, 0, numSignals * sizeof(SignalState)));

  // GPU 1's signal buffer
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  SignalState* signalBuffer1;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer1, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer1, 0, numSignals * sizeof(SignalState)));

  // Create transport options
  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  // Transport on GPU 0: signals to GPU 1's buffer, waits on GPU 0's buffer
  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  P2pNvlTransportDevice transport0(
      kGpu0, kGpu1, options, localState0, remoteState0);

  // Allocate device copy of transport0
  P2pNvlTransportDevice* transport0_d;
  CUDACHECK_TEST(cudaMalloc(&transport0_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport0_d,
      &transport0,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  // Transport on GPU 1: signals to GPU 0's buffer, waits on GPU 1's buffer
  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  P2pNvlTransportDevice transport1(
      kGpu1, kGpu0, options, localState1, remoteState1);

  // Allocate device copy of transport1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  P2pNvlTransportDevice* transport1_d;
  CUDACHECK_TEST(cudaMalloc(&transport1_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport1_d,
      &transport1,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  const int numBlocks = 1;
  const int blockSize = 32;
  const uint64_t signalId = 0;

  // GPU 0 signals to GPU 1's buffer
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceSignal(
      transport0_d, signalId, SignalOp::SIGNAL_SET, 42, numBlocks, blockSize);

  // GPU 1 waits on its local buffer - should complete immediately
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testDeviceWaitSignal(
      transport1_d, signalId, CmpOp::CMP_EQ, 42, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Now GPU 1 signals to GPU 0's buffer
  test::testDeviceSignal(
      transport1_d, signalId, SignalOp::SIGNAL_SET, 100, numBlocks, blockSize);

  // GPU 0 waits on its local buffer - should complete immediately
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceWaitSignal(
      transport0_d, signalId, CmpOp::CMP_EQ, 100, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(signalBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(signalBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(P2pNvlTransportDeviceTwoGpuFixture, DeviceSignalTwoGpuMultipleIds) {
  const int numSignals = 8;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  SignalState* signalBuffer0;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer0, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer0, 0, numSignals * sizeof(SignalState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  SignalState* signalBuffer1;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer1, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer1, 0, numSignals * sizeof(SignalState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  P2pNvlTransportDevice transport0(
      kGpu0, kGpu1, options, localState0, remoteState0);

  // Allocate device copy of transport0
  P2pNvlTransportDevice* transport0_d;
  CUDACHECK_TEST(cudaMalloc(&transport0_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport0_d,
      &transport0,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  P2pNvlTransportDevice transport1(
      kGpu1, kGpu0, options, localState1, remoteState1);

  // Allocate device copy of transport1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  P2pNvlTransportDevice* transport1_d;
  CUDACHECK_TEST(cudaMalloc(&transport1_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport1_d,
      &transport1,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Test multiple signal IDs
  std::vector<uint64_t> testSignalIds = {0, 1, 3, 7};

  for (uint64_t signalId : testSignalIds) {
    // GPU 0 signals to GPU 1 with signalId + 1 as value
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    test::testDeviceSignal(
        transport0_d,
        signalId,
        SignalOp::SIGNAL_SET,
        signalId + 1,
        numBlocks,
        blockSize);

    // GPU 1 waits for the expected value
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    test::testDeviceWaitSignal(
        transport1_d,
        signalId,
        CmpOp::CMP_EQ,
        signalId + 1,
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(signalBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(signalBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(P2pNvlTransportDeviceTwoGpuFixture, DeviceSignalTwoGpuAdd) {
  const int numSignals = 8;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  SignalState* signalBuffer0;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer0, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer0, 0, numSignals * sizeof(SignalState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  SignalState* signalBuffer1;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer1, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer1, 0, numSignals * sizeof(SignalState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  P2pNvlTransportDevice transport0(
      kGpu0, kGpu1, options, localState0, remoteState0);

  // Allocate device copy of transport0
  P2pNvlTransportDevice* transport0_d;
  CUDACHECK_TEST(cudaMalloc(&transport0_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport0_d,
      &transport0,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  P2pNvlTransportDevice transport1(
      kGpu1, kGpu0, options, localState1, remoteState1);

  // Allocate device copy of transport1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  P2pNvlTransportDevice* transport1_d;
  CUDACHECK_TEST(cudaMalloc(&transport1_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport1_d,
      &transport1,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  const int numBlocks = 1;
  const int blockSize = 32;
  const uint64_t signalId = 0;

  // GPU 0 adds 5 to GPU 1's signal
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceSignal(
      transport0_d, signalId, SignalOp::SIGNAL_ADD, 5, numBlocks, blockSize);

  // GPU 0 adds 10 more to GPU 1's signal
  test::testDeviceSignal(
      transport0_d, signalId, SignalOp::SIGNAL_ADD, 10, numBlocks, blockSize);

  // GPU 1 waits for >= 15
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testDeviceWaitSignal(
      transport1_d, signalId, CmpOp::CMP_GE, 15, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(signalBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(signalBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(P2pNvlTransportDeviceTwoGpuFixture, DeviceSignalTwoGpuBlockGroups) {
  const int numSignals = 8;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  SignalState* signalBuffer0;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer0, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer0, 0, numSignals * sizeof(SignalState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  SignalState* signalBuffer1;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer1, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer1, 0, numSignals * sizeof(SignalState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  P2pNvlTransportDevice transport0(
      kGpu0, kGpu1, options, localState0, remoteState0);

  // Allocate device copy of transport0
  P2pNvlTransportDevice* transport0_d;
  CUDACHECK_TEST(cudaMalloc(&transport0_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport0_d,
      &transport0,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  P2pNvlTransportDevice transport1(
      kGpu1, kGpu0, options, localState1, remoteState1);

  // Allocate device copy of transport1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  P2pNvlTransportDevice* transport1_d;
  CUDACHECK_TEST(cudaMalloc(&transport1_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport1_d,
      &transport1,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  const int numBlocks = 1;
  const int blockSize = 256;
  const uint64_t signalId = 0;

  // GPU 0 signals with block groups
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceSignal(
      transport0_d,
      signalId,
      SignalOp::SIGNAL_SET,
      999,
      numBlocks,
      blockSize,
      test::GroupType::BLOCK);

  // GPU 1 waits with block groups
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testDeviceWaitSignal(
      transport1_d,
      signalId,
      CmpOp::CMP_EQ,
      999,
      numBlocks,
      blockSize,
      test::GroupType::BLOCK);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(signalBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(signalBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(P2pNvlTransportDeviceTwoGpuFixture, DeviceSignalTwoGpuPingPong) {
  // Test ping-pong signaling between 2 GPUs
  const int numSignals = 8;
  const int numSteps = 10;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  SignalState* signalBuffer0;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer0, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer0, 0, numSignals * sizeof(SignalState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  SignalState* signalBuffer1;
  CUDACHECK_TEST(cudaMalloc(&signalBuffer1, numSignals * sizeof(SignalState)));
  CUDACHECK_TEST(
      cudaMemset(signalBuffer1, 0, numSignals * sizeof(SignalState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  P2pNvlTransportDevice transport0(
      kGpu0, kGpu1, options, localState0, remoteState0);

  // Allocate device copy of transport0
  P2pNvlTransportDevice* transport0_d;
  CUDACHECK_TEST(cudaMalloc(&transport0_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport0_d,
      &transport0,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer1, numSignals),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(signalBuffer0, numSignals),
  };
  P2pNvlTransportDevice transport1(
      kGpu1, kGpu0, options, localState1, remoteState1);

  // Allocate device copy of transport1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  P2pNvlTransportDevice* transport1_d;
  CUDACHECK_TEST(cudaMalloc(&transport1_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport1_d,
      &transport1,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  const int numBlocks = 1;
  const int blockSize = 32;
  const uint64_t signalId = 0;

  // Ping-pong: GPU 0 signals, GPU 1 waits, GPU 1 signals, GPU 0 waits
  for (int step = 1; step <= numSteps; ++step) {
    // GPU 0 signals step value to GPU 1
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    test::testDeviceSignal(
        transport0_d,
        signalId,
        SignalOp::SIGNAL_SET,
        step,
        numBlocks,
        blockSize);

    // GPU 1 waits for step value
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    test::testDeviceWaitSignal(
        transport1_d, signalId, CmpOp::CMP_EQ, step, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // GPU 1 signals step * 10 back to GPU 0
    test::testDeviceSignal(
        transport1_d,
        signalId,
        SignalOp::SIGNAL_SET,
        step * 10,
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // GPU 0 waits for step * 10
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    test::testDeviceWaitSignal(
        transport0_d, signalId, CmpOp::CMP_EQ, step * 10, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(signalBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(signalBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
