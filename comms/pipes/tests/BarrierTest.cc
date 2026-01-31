// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <vector>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/BarrierTest.cuh"
#include "comms/pipes/tests/P2pNvlTransportDeviceTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes {

// =============================================================================
// Single-GPU Test Fixture for Barrier Struct Tests
// =============================================================================

class BarrierTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {}
};

// =============================================================================
// Two-GPU Test Fixture for P2pNvlTransportDevice Barrier Tests
// Requires 2 GPUs with P2P access enabled
// =============================================================================

class BarrierTwoGpuFixture : public ::testing::Test {
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
// Barrier Struct Unit Tests
// These test the Barrier struct directly without P2pNvlTransportDevice
// =============================================================================

TEST_F(BarrierTestFixture, BarrierBasicArrive) {
  // Allocate a single Barrier on device
  BarrierState* barrier_d;
  CUDACHECK_TEST(cudaMalloc(&barrier_d, sizeof(BarrierState)));
  CUDACHECK_TEST(cudaMemset(barrier_d, 0, sizeof(BarrierState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Arrive should increment the current counter
  test::testRawBarrierArrive(
      barrier_d, numBlocks, blockSize, test::GroupType::WARP);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Read back the current counter value
  test::testReadBarrierCurrentCounter(barrier_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  ASSERT_EQ(result_h, 1)
      << "Current counter should be 1 after one arrive operation";

  CUDACHECK_TEST(cudaFree(barrier_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(BarrierTestFixture, BarrierMultipleArrives) {
  BarrierState* barrier_d;
  CUDACHECK_TEST(cudaMalloc(&barrier_d, sizeof(BarrierState)));
  CUDACHECK_TEST(cudaMemset(barrier_d, 0, sizeof(BarrierState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Multiple arrives should increment the counter each time
  for (int i = 0; i < 5; ++i) {
    test::testRawBarrierArrive(
        barrier_d, numBlocks, blockSize, test::GroupType::WARP);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Read back the current counter value
  test::testReadBarrierCurrentCounter(barrier_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  ASSERT_EQ(result_h, 5)
      << "Current counter should be 5 after five arrive operations";

  CUDACHECK_TEST(cudaFree(barrier_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(BarrierTestFixture, BarrierArriveWait) {
  BarrierState* barrier_d;
  CUDACHECK_TEST(cudaMalloc(&barrier_d, sizeof(BarrierState)));
  CUDACHECK_TEST(cudaMemset(barrier_d, 0, sizeof(BarrierState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Arrive then wait in the same kernel should complete successfully
  test::testRawBarrierArriveWait(
      barrier_d, numBlocks, blockSize, test::GroupType::WARP);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify current counter is 1 (one arrive)
  test::testReadBarrierCurrentCounter(barrier_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t current_h;
  CUDACHECK_TEST(cudaMemcpy(
      &current_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  ASSERT_EQ(current_h, 1) << "Current counter should be 1";

  // Verify expected counter is 1 (one wait completed)
  test::testReadBarrierExpectedCounter(barrier_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t expected_h;
  CUDACHECK_TEST(cudaMemcpy(
      &expected_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  ASSERT_EQ(expected_h, 1) << "Expected counter should be 1";

  CUDACHECK_TEST(cudaFree(barrier_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(BarrierTestFixture, BarrierBlockGroups) {
  BarrierState* barrier_d;
  CUDACHECK_TEST(cudaMalloc(&barrier_d, sizeof(BarrierState)));
  CUDACHECK_TEST(cudaMemset(barrier_d, 0, sizeof(BarrierState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 256;

  // Test with block-level thread groups
  test::testRawBarrierArriveWait(
      barrier_d, numBlocks, blockSize, test::GroupType::BLOCK);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify current counter is 1
  test::testReadBarrierCurrentCounter(barrier_d, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint64_t result_h;
  CUDACHECK_TEST(cudaMemcpy(
      &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  ASSERT_EQ(result_h, 1) << "Current counter should be 1 with block groups";

  CUDACHECK_TEST(cudaFree(barrier_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

TEST_F(BarrierTestFixture, BarrierMultipleBarriers) {
  // Test with multiple Barrier objects in an array
  const int numBarriers = 8;
  BarrierState* barriers_d;
  CUDACHECK_TEST(cudaMalloc(&barriers_d, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(cudaMemset(barriers_d, 0, numBarriers * sizeof(BarrierState)));

  uint64_t* result_d;
  CUDACHECK_TEST(cudaMalloc(&result_d, sizeof(uint64_t)));

  const int numBlocks = 1;
  const int blockSize = 32;

  // Arrive and wait on each barrier
  for (int i = 0; i < numBarriers; ++i) {
    test::testRawBarrierArriveWait(
        &barriers_d[i], numBlocks, blockSize, test::GroupType::WARP);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify each barrier has the correct counter values
  for (int i = 0; i < numBarriers; ++i) {
    test::testReadBarrierCurrentCounter(&barriers_d[i], result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    uint64_t result_h;
    CUDACHECK_TEST(cudaMemcpy(
        &result_h, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    ASSERT_EQ(result_h, 1) << "Barrier " << i << " current counter should be 1";
  }

  CUDACHECK_TEST(cudaFree(barriers_d));
  CUDACHECK_TEST(cudaFree(result_d));
}

// =============================================================================
// P2pNvlTransportDevice Barrier API Tests (Two-GPU Configuration)
// These tests use 2 GPUs with P2P access to test cross-GPU barrier sync
// =============================================================================

TEST_F(BarrierTwoGpuFixture, DeviceBarrierSyncTwoGpu) {
  // Allocate barrier buffers on each GPU
  const int numBarriers = 8;

  // GPU 0's barrier buffer
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  BarrierState* barrierBuffer0;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer0, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer0, 0, numBarriers * sizeof(BarrierState)));

  // GPU 1's barrier buffer
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  BarrierState* barrierBuffer1;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer1, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer1, 0, numBarriers * sizeof(BarrierState)));

  // Create transport options
  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  // Transport on GPU 0: writes to GPU 1's barrier, waits on GPU 0's barrier
  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
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

  // Transport on GPU 1: writes to GPU 0's barrier, waits on GPU 1's barrier
  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
  };
  P2pNvlTransportDevice transport1(
      kGpu1, kGpu0, options, localState1, remoteState1);

  // Allocate device copy of transport1
  P2pNvlTransportDevice* transport1_d;
  CUDACHECK_TEST(cudaMalloc(&transport1_d, sizeof(P2pNvlTransportDevice)));
  CUDACHECK_TEST(cudaMemcpy(
      transport1_d,
      &transport1,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  const int numBlocks = 1;
  const int blockSize = 32;
  const uint64_t barrierId = 0;

  // Launch barrier sync on both GPUs concurrently
  // They should synchronize with each other
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceBarrierSync(
      transport0_d, barrierId, numBlocks, blockSize, test::GroupType::WARP);

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testDeviceBarrierSync(
      transport1_d, barrierId, numBlocks, blockSize, test::GroupType::WARP);

  // Wait for both to complete
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(barrierBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(barrierBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(BarrierTwoGpuFixture, DeviceBarrierSyncTwoGpuMultipleIds) {
  const int numBarriers = 8;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  BarrierState* barrierBuffer0;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer0, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer0, 0, numBarriers * sizeof(BarrierState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  BarrierState* barrierBuffer1;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer1, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer1, 0, numBarriers * sizeof(BarrierState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
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
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
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

  // Test multiple barrier IDs
  std::vector<uint64_t> testBarrierIds = {0, 1, 3, 7};

  for (uint64_t barrierId : testBarrierIds) {
    // Reset barrier buffers
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    CUDACHECK_TEST(
        cudaMemset(barrierBuffer0, 0, numBarriers * sizeof(BarrierState)));
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(
        cudaMemset(barrierBuffer1, 0, numBarriers * sizeof(BarrierState)));

    // Launch barrier sync on both GPUs
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    test::testDeviceBarrierSync(
        transport0_d, barrierId, numBlocks, blockSize, test::GroupType::WARP);

    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    test::testDeviceBarrierSync(
        transport1_d, barrierId, numBlocks, blockSize, test::GroupType::WARP);

    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(barrierBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(barrierBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(BarrierTwoGpuFixture, DeviceBarrierSyncTwoGpuMultipleSyncs) {
  const int numBarriers = 8;
  const int numSyncs = 10;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  BarrierState* barrierBuffer0;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer0, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer0, 0, numBarriers * sizeof(BarrierState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  BarrierState* barrierBuffer1;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer1, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer1, 0, numBarriers * sizeof(BarrierState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
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
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
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
  const uint64_t barrierId = 0;

  // Multiple syncs in the same kernel
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceBarrierSyncMultiple(
      transport0_d,
      barrierId,
      numSyncs,
      numBlocks,
      blockSize,
      test::GroupType::WARP);

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testDeviceBarrierSyncMultiple(
      transport1_d,
      barrierId,
      numSyncs,
      numBlocks,
      blockSize,
      test::GroupType::WARP);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(barrierBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(barrierBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(BarrierTwoGpuFixture, DeviceBarrierSyncTwoGpuBlockGroups) {
  const int numBarriers = 8;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  BarrierState* barrierBuffer0;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer0, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer0, 0, numBarriers * sizeof(BarrierState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  BarrierState* barrierBuffer1;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer1, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer1, 0, numBarriers * sizeof(BarrierState)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
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
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
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
  const uint64_t barrierId = 0;

  // Test with block groups
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testDeviceBarrierSync(
      transport0_d, barrierId, numBlocks, blockSize, test::GroupType::BLOCK);

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testDeviceBarrierSync(
      transport1_d, barrierId, numBlocks, blockSize, test::GroupType::BLOCK);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  SUCCEED();

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(barrierBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(barrierBuffer1));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

TEST_F(BarrierTwoGpuFixture, DeviceBarrierDataTransferVerification) {
  // Test that data written by GPU 0 to GPU 1's buffer is visible to GPU 1
  // after barrier sync. Uses put() API for data transfer.
  // Each thread group uses its own barrier id.
  const int numBarriers = 8; // Enough for multiple thread groups
  const size_t dataSize = 64 * 1024; // 64KB of data
  const uint8_t fillValue = 0x42;

  // Allocate barrier buffers on each GPU
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  BarrierState* barrierBuffer0;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer0, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer0, 0, numBarriers * sizeof(BarrierState)));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  BarrierState* barrierBuffer1;
  CUDACHECK_TEST(
      cudaMalloc(&barrierBuffer1, numBarriers * sizeof(BarrierState)));
  CUDACHECK_TEST(
      cudaMemset(barrierBuffer1, 0, numBarriers * sizeof(BarrierState)));

  // Allocate data buffers on each GPU
  // GPU 0's buffer is the source, GPU 1's buffer is the destination
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  uint8_t* dataBuffer0;
  CUDACHECK_TEST(cudaMalloc(&dataBuffer0, dataSize));
  CUDACHECK_TEST(cudaMemset(dataBuffer0, fillValue, dataSize)); // Fill source

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  uint8_t* dataBuffer1;
  CUDACHECK_TEST(cudaMalloc(&dataBuffer1, dataSize));
  CUDACHECK_TEST(cudaMemset(dataBuffer1, 0, dataSize)); // Initialize dest to 0

  // Allocate error counter on GPU 1
  uint32_t* errorCount_d;
  CUDACHECK_TEST(cudaMalloc(&errorCount_d, sizeof(uint32_t)));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  // Transport on GPU 0: local=barrier0, remote=barrier1
  // GPU 0 will use put() to copy from dataBuffer0 to dataBuffer1 via P2P
  LocalState localState0{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
  };
  RemoteState remoteState0{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
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

  // Transport on GPU 1: local=barrier1, remote=barrier0
  LocalState localState1{
      .dataBuffer = nullptr,
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer1, numBarriers),
  };
  RemoteState remoteState1{
      .dataBuffer = nullptr,
      .peerChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .myChunkStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(barrierBuffer0, numBarriers),
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

  const int numBlocks = 4; // 4 blocks = 4 warp groups (barrier ids 0-3)
  const int blockSize = 32; // 32 threads per block = 1 warp group per block

  // GPU 0: Use put() to copy data from local buffer to GPU 1's buffer,
  // then barrier sync
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  test::testBarrierWriteData(
      transport0_d,
      reinterpret_cast<char*>(dataBuffer1), // Remote buffer on GPU 1
      reinterpret_cast<const char*>(dataBuffer0), // Local source on GPU 0
      dataSize,
      numBlocks,
      blockSize,
      test::GroupType::WARP);

  // GPU 1: Barrier sync, then verify the data
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  test::testBarrierVerifyData(
      transport1_d,
      dataBuffer1, // Local buffer on GPU 1
      dataSize,
      fillValue,
      errorCount_d,
      numBlocks,
      blockSize,
      test::GroupType::WARP);

  // Wait for both GPUs to complete
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy error count back and verify
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(errorCount_h, 0)
      << "Data verification failed: " << errorCount_h << " bytes mismatched "
      << "out of " << dataSize << " bytes";

  // Cleanup
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(barrierBuffer0));
  CUDACHECK_TEST(cudaFree(dataBuffer0));
  CUDACHECK_TEST(cudaFree(transport0_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(barrierBuffer1));
  CUDACHECK_TEST(cudaFree(dataBuffer1));
  CUDACHECK_TEST(cudaFree(errorCount_d));
  CUDACHECK_TEST(cudaFree(transport1_d));
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
