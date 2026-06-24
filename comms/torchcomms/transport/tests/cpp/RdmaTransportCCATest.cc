// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include "comms/ctran/regcache/RegCache.h"
#include "comms/torchcomms/transport/RdmaTransport.h"
#include "comms/torchcomms/transport/RdmaTransportCCA.hpp"
#include "comms/utils/cvars/nccl_cvars.h"

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace torch::comms;

class RdmaTransportCCATest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclCvarInit();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!RdmaTransport::supported()) {
      GTEST_SKIP() << "RDMA/IB not supported on this platform";
    }
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    c10::cuda::CUDACachingAllocator::init(deviceCount);
  }

  // The CUDA caching-allocator trace tracker installed by attachRdmaMemoryHook
  // cannot be removed (no detach API) and outlives any single test. Clearing
  // the hook's callbacks between tests stops a prior test's hook from firing
  // extra (de)registrations during this test's allocations.
  void TearDown() override {
    detachRdmaMemoryHook();
  }
};

// Allocating a buffer through the CUDA caching allocator and then installing
// the hook registers the pre-existing segment with RegCache via
// registerMemPreHook(). searchIbRegHandle() returns null before the hook runs
// (segment not cached) and a valid handle afterwards (the IB registration is
// materialized lazily on lookup).
TEST_F(RdmaTransportCCATest, HookRegistersExistingCachingAllocatorSegments) {
  const int cudaDev = 0;
  const size_t bufferSize = 4 * 1024 * 1024; // 4MB, forces a large-pool segment

  void* buffer = c10::cuda::CUDACachingAllocator::raw_alloc(bufferSize);
  ASSERT_NE(buffer, nullptr);

  const auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  // Before the hook runs the segment is not registered in RegCache.
  EXPECT_EQ(regCache->searchIbRegHandle(buffer, bufferSize, cudaDev), nullptr);

  // Installing the hook snapshots and registers all pre-existing segments.
  attachRdmaMemoryHook(RdmaRegTensor, RdmaDeregTensor);

  // The segment is now registered; the IB handle is materialized lazily.
  EXPECT_NE(regCache->searchIbRegHandle(buffer, bufferSize, cudaDev), nullptr);

  c10::cuda::CUDACachingAllocator::raw_delete(buffer);
}

// Reproduces the torchcomm (NCCLX) hook and the RDMA hook both being installed:
// both ultimately call RegCache::globalRegister on SEGMENT_ALLOC and
// RegCache::globalDeregister on SEGMENT_FREE for the same caching-allocator
// segment. Two registrants must share a single registration, and the paired
// deregisters must not double-free: the first releases it, the second is a safe
// no-op. This verifies RegCache refcounts the shared segment correctly.
TEST_F(RdmaTransportCCATest, RegCacheRefCountWithTwoRegistrants) {
  const int cudaDev = 0;
  const size_t bufferSize = 4 * 1024 * 1024; // 4MB, forces a large-pool segment

  void* buffer = c10::cuda::CUDACachingAllocator::raw_alloc(bufferSize);
  ASSERT_NE(buffer, nullptr);

  const auto regCache = ctran::RegCache::getInstance();
  ASSERT_NE(regCache, nullptr);

  // Both registrants (NCCLX hook + RDMA hook) register the same buffer.
  EXPECT_EQ(
      regCache->globalRegister(
          buffer,
          bufferSize,
          /*forceReg=*/false,
          /*ncclManaged=*/false,
          cudaDev),
      commSuccess);
  EXPECT_EQ(
      regCache->globalRegister(
          buffer,
          bufferSize,
          /*forceReg=*/false,
          /*ncclManaged=*/false,
          cudaDev),
      commSuccess);

  // The segment is shared: a single IB handle is materialized for both.
  EXPECT_NE(regCache->searchIbRegHandle(buffer, bufferSize, cudaDev), nullptr);

  // Both registrants deregister (as both hooks fire on the same SEGMENT_FREE).
  // Neither call may abort or double-free; the second is a safe no-op.
  EXPECT_EQ(regCache->globalDeregister(buffer, bufferSize), commSuccess);
  EXPECT_EQ(regCache->globalDeregister(buffer, bufferSize), commSuccess);

  // End state is clean: the buffer is fully deregistered.
  EXPECT_EQ(regCache->searchIbRegHandle(buffer, bufferSize, cudaDev), nullptr);

  c10::cuda::CUDACachingAllocator::raw_delete(buffer);
}
