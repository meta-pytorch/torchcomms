// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsKeyRegistry.h"
#include "comms/utils/collstats/CollStatsReadoutDriver.h"
#include "comms/utils/collstats/CollStatsSpan.cuh"
#include "comms/utils/collstats/CollStatsTypes.h"

namespace meta::comms::collstats {
namespace {

__global__ void
spanKernel(CollStatsDeviceBlock* block, uint32_t keyId, uint64_t logicalBytes) {
  collStatsSpanEntry(block, /*slot=*/0);
  __syncthreads();
  if (threadIdx.x == 0) {
    collStatsSpanFinalizeElectedById(block, /*slot=*/0, keyId, logicalBytes);
  }
}

uint64_t totalCount(const std::vector<CollStatValue>& values) {
  uint64_t total = 0;
  for (const auto& v : values) {
    total += v.count;
  }
  return total;
}

// One instrumented collective: stream-ordered pre-reset, span kernel, then the
// driver tick — exactly the sequence the launch path will run.
void runCollective(
    CollStatsReadoutDriver& driver,
    const CollStatsDeviceBlockHandle& h,
    cudaStream_t stream,
    uint32_t keyId) {
  collStatsEnqueuePreReset(h, /*slot=*/0, stream);
  spanKernel<<<8, 64, 0, stream>>>(h.dev, keyId, 4096);
  driver.onCollective(stream);
}

TEST(CollStatsReadoutDriverGpuTest, PipelinesWindowsWithoutBlocking) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CollStatsKeyRegistry keys(capacity);
  const uint32_t key = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  std::vector<uint64_t> exportedCounts;
  const uint32_t cadence = 4;
  // Scoped so the driver's own teardown runs before the block is freed.
  {
    CollStatsReadoutDriver driver(
        h,
        cadence,
        [&](const CollStatSnapshot& snap) {
          exportedCounts.push_back(totalCount(snap.values));
        },
        keys);
    ASSERT_FALSE(driver.disabled());

    // Batch A: cadence collectives. The cadence-th tick issues window 0's copy
    // (pending), but harvests nothing yet.
    for (uint32_t i = 0; i < cadence; ++i) {
      runCollective(driver, h, stream, key);
    }
    EXPECT_TRUE(exportedCounts.empty());

    // Ensure window 0's copy has completed so the next harvest query succeeds.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Batch B: cadence more collectives. The cadence-th tick harvests window 0
    // (the pipeline path, during onCollective) and issues window 1.
    for (uint32_t i = 0; i < cadence; ++i) {
      runCollective(driver, h, stream, key);
    }
    ASSERT_EQ(exportedCounts.size(), 1u);
    EXPECT_EQ(exportedCounts[0], cadence); // window 0 = cadence observations

    // Drain and harvest the last window via flush (teardown path).
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    driver.flush();
    ASSERT_EQ(exportedCounts.size(), 2u);
    EXPECT_EQ(exportedCounts[1], cadence); // window 1 = cadence observations

    EXPECT_EQ(driver.windowsExported(), 2u);
    EXPECT_EQ(driver.windowsDropped(), 0u);
  }

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

TEST(CollStatsReadoutDriverGpuTest, NoReadoutBeforeCadenceReached) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CollStatsKeyRegistry keys(capacity);
  const uint32_t key = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  int exports = 0;
  // Scoped so the driver is destroyed while the block is still alive, the
  // order CtranAlgo guarantees by declaring the driver last.
  {
    CollStatsReadoutDriver driver(
        h, /*cadence=*/8, [&](const CollStatSnapshot&) { ++exports; }, keys);

    for (int i = 0; i < 7; ++i) { // one short of cadence
      runCollective(driver, h, stream, key);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    driver.flush(); // no window was ever issued, so nothing to harvest
    EXPECT_EQ(exports, 0);
    EXPECT_EQ(driver.windowsExported(), 0u);
  }
  // The destructor's flushFinal does pick them up; flush alone never would.
  EXPECT_EQ(exports, 1);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

// The collectives between the last cadence boundary and teardown are the tail
// of every run, and at the default cadence of 128 they can be most of a short
// one. flushFinal issues one extra window for them.
TEST(CollStatsReadoutDriverGpuTest, FinalFlushExportsTheTrailingWindow) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CollStatsKeyRegistry keys(capacity);
  const uint32_t key = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  std::vector<uint64_t> exportedCounts;
  const uint32_t cadence = 8;
  // Scoped so the driver's own teardown runs before the block is freed.
  {
    CollStatsReadoutDriver driver(
        h,
        cadence,
        [&](const CollStatSnapshot& snap) {
          exportedCounts.push_back(totalCount(snap.values));
        },
        keys);
    ASSERT_FALSE(driver.disabled());

    // One full window, then a partial one that never reaches a boundary.
    const int trailing = 3;
    for (uint32_t i = 0; i < cadence + trailing; ++i) {
      runCollective(driver, h, stream, key);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(exportedCounts.empty()); // window 0 issued, not yet harvested

    driver.flushFinal();
    ASSERT_EQ(exportedCounts.size(), 2u);
    EXPECT_EQ(exportedCounts[0], cadence); // the window that hit the boundary
    EXPECT_EQ(
        exportedCounts[1], trailing); // the tail, which flush() would drop
    EXPECT_EQ(driver.windowsExported(), 2u);
    EXPECT_EQ(driver.windowsDropped(), 0u);

    // Idempotent: the tail was consumed, so a second call has nothing to issue,
    // and neither does the destructor below.
    driver.flushFinal();
    EXPECT_EQ(exportedCounts.size(), 2u);
  }
  EXPECT_EQ(exportedCounts.size(), 2u);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

// The sink is caller-supplied and is invoked from the destructor's final flush,
// where an escaping exception would cross a noexcept boundary and call
// std::terminate -- the job dying over telemetry. The catch is the only thing
// preventing that, so it needs a test that actually throws.
TEST(CollStatsReadoutDriverGpuTest, ThrowingSinkIsCountedNotPropagated) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CollStatsKeyRegistry keys(capacity);
  const uint32_t key = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  uint32_t sinkCalls = 0;
  uint32_t callsBeforeScopeExit = 0;
  const uint32_t cadence = 4;
  {
    CollStatsReadoutDriver driver(
        h,
        cadence,
        [&](const CollStatSnapshot&) {
          ++sinkCalls;
          throw std::runtime_error("sink failed");
        },
        keys);
    ASSERT_FALSE(driver.disabled());

    for (uint32_t i = 0; i < cadence * 2; ++i) {
      runCollective(driver, h, stream, key);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // The throw is swallowed and counted; the window is still consumed, so the
    // driver keeps running rather than wedging on a bad sink.
    EXPECT_NO_THROW(driver.flushFinal());
    EXPECT_GT(sinkCalls, 0u);
    EXPECT_EQ(driver.sinkExceptions(), sinkCalls);
    EXPECT_FALSE(driver.disabled()) << "a throwing sink is not a CUDA fault";

    // Leave a tail unharvested so the destructor's own flushFinal also runs
    // through the throwing sink. That call is the noexcept boundary the catch
    // exists for, and the explicit flushFinal above would otherwise have
    // consumed everything before it.
    for (uint32_t i = 0; i < cadence - 1; ++i) {
      runCollective(driver, h, stream, key);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    callsBeforeScopeExit = sinkCalls;
  }
  // Reaching here at all is the assertion: the destructor exported the tail
  // through a sink that throws, and did not call std::terminate.
  EXPECT_GT(sinkCalls, callsBeforeScopeExit);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

// TODO(T282705070): "never blocks the enqueue thread" has no test, because it
// does not hold everywhere. Parking the instrumented stream behind a 2s spin
// kernel and timing one tick measures ~0ms on a devgpu H100 and ~2005ms under
// remote execution, with the driver enabled and the pinned staging allocated in
// both. Something in the issue path still serializes with the stream on some
// hardware; identify it before asserting the invariant here.

} // namespace

} // namespace meta::comms::collstats
