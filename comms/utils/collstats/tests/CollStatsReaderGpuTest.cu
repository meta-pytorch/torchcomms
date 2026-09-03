// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsKeyRegistry.h"
#include "comms/utils/collstats/CollStatsReader.h"
#include "comms/utils/collstats/CollStatsSpan.cuh"
#include "comms/utils/collstats/CollStatsTypes.h"

namespace meta::comms::collstats {
namespace {

// Records one observation per launch through the span, exactly as the
// instrumented collective kernel does: the value slot arrives as a kernel
// argument, already resolved on the host.
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

TEST(CollStatsReaderGpuTest, WindowReadoutIsPerWindowAndZeroesRetiredBank) {
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
  const uint32_t id = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  // Window 0: three collectives, each preceded by its stream-ordered pre-reset
  // (the finalizer emits only, so the pre-reset owns slot cleanup).
  for (int i = 0; i < 3; ++i) {
    collStatsEnqueuePreReset(h, /*slot=*/0, stream);
    spanKernel<<<8, 64, 0, stream>>>(h.dev, id, 4096);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  const CollStatSnapshot w0 = collStatsReadWindow(h, stream, keys);
  // Only the occupied prefix is transferred, not the bank's capacity.
  EXPECT_EQ(w0.numKeys, 1u);
  EXPECT_EQ(w0.values.size(), 2u); // one key plus the catch-all
  EXPECT_EQ(totalCount(w0.values), 3u);
  EXPECT_EQ(w0.catchAllCount, 0u);

  // Window 1: two collectives land in the flipped-to bank. The retired bank was
  // zeroed, so this window must report only the new two, not five.
  for (int i = 0; i < 2; ++i) {
    collStatsEnqueuePreReset(h, /*slot=*/0, stream);
    spanKernel<<<8, 64, 0, stream>>>(h.dev, id, 4096);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  const CollStatSnapshot w1 = collStatsReadWindow(h, stream, keys);
  EXPECT_EQ(totalCount(w1.values), 2u);
  EXPECT_EQ(w1.windowEpoch, w0.windowEpoch + 1); // epoch advanced once per read

  // An immediate re-read with no intervening collectives sees an empty window.
  const CollStatSnapshot w2 = collStatsReadWindow(h, stream, keys);
  EXPECT_EQ(totalCount(w2.values), 0u);

  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

TEST(CollStatsReaderGpuTest, SnapshotPairsValuesWithRecordedKey) {
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
  const uint32_t idA = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});
  const uint32_t idB = keys.resolve(
      CollStatKey{
          CollStatOp::AllGather,
          CollStatAlgo::Ring,
          CollStatProto::Unknown,
          2u,
          9u});

  collStatsEnqueuePreReset(h, /*slot=*/0, stream);
  spanKernel<<<8, 64, 0, stream>>>(h.dev, idA, 4096);
  collStatsEnqueuePreReset(h, /*slot=*/0, stream);
  spanKernel<<<8, 64, 0, stream>>>(h.dev, idB, 8192);
  collStatsEnqueuePreReset(h, /*slot=*/0, stream);
  spanKernel<<<8, 64, 0, stream>>>(h.dev, idB, 8192);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  const CollStatSnapshot snap = collStatsReadWindow(h, stream, keys);

  // values[i] and keys[i] describe the same key, and the counts follow the id
  // each observation was launched with.
  ASSERT_EQ(snap.numKeys, 2u);
  ASSERT_EQ(snap.keys.size(), 2u);
  EXPECT_EQ(snap.keys[idA].op, CollStatOp::AllReduce);
  EXPECT_EQ(snap.keys[idA].dtype, 7u);
  EXPECT_EQ(snap.keys[idA].sizeClass, 3u);
  EXPECT_EQ(snap.values[idA].count, 1u);

  EXPECT_EQ(snap.keys[idB].op, CollStatOp::AllGather);
  EXPECT_EQ(snap.keys[idB].algorithm, CollStatAlgo::Ring);
  EXPECT_EQ(snap.values[idB].count, 2u);

  // Nothing saturated, so the trailing catch-all slot stayed empty.
  EXPECT_EQ(snap.values[snap.numKeys].count, 0u);

  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

// Keys past capacity share the trailing slot, and their observations are still
// recorded rather than dropped.
TEST(CollStatsReaderGpuTest, SaturatedKeysLandInTheCatchAll) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 2;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  CollStatsKeyRegistry keys(capacity);
  keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          1u,
          1u});
  keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          2u,
          2u});
  const uint32_t overflow = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          3u,
          3u});
  ASSERT_EQ(overflow, keys.catchAllId());

  // No pre-reset, unlike the other cases here: this is the block's first
  // collective, so the slot still holds the sentinel the allocation wrote. The
  // pre-reset exists to clean a slot a predecessor left dirty, and there is no
  // predecessor.
  spanKernel<<<8, 64, 0, stream>>>(h.dev, overflow, 4096);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  const CollStatSnapshot snap = collStatsReadWindow(h, stream, keys);
  EXPECT_EQ(snap.numKeys, capacity);
  EXPECT_EQ(snap.values[snap.numKeys].count, 1u);
  EXPECT_EQ(snap.catchAllCount, 1u);

  cudaStreamDestroy(stream);
  collStatsFreeDeviceBlock(h);
}

// The event-gated readout must order correctly across the instrumented stream
// and the reader stream with no manual sync between phases: the reader waits
// the instrumented stream's finalizers before copying, and post-flip
// collectives are gated into the new bank. Running to completion also proves
// the cross-stream waits do not deadlock.
TEST(CollStatsReaderGpuTest, GatedReadoutOrdersAcrossStreamsWithoutManualSync) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t instrumented;
  cudaStream_t reader;
  ASSERT_EQ(cudaStreamCreate(&instrumented), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&reader), cudaSuccess);
  cudaEvent_t streamEvent;
  cudaEvent_t flipEvent;
  ASSERT_EQ(cudaEventCreate(&streamEvent), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&flipEvent), cudaSuccess);

  const cudaEvent_t streamEvents[1] = {streamEvent};
  CollStatsReadGating gating{};
  gating.instrumentedStreams = &instrumented;
  gating.streamEvents = streamEvents;
  gating.numStreams = 1;
  gating.flipEvent = flipEvent;

  CollStatsKeyRegistry keys(capacity);
  const uint32_t id = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  // Window 0: three collectives on the instrumented stream, left un-synced. The
  // gated readout must itself order the copy after them.
  for (int i = 0; i < 3; ++i) {
    collStatsEnqueuePreReset(h, /*slot=*/0, instrumented);
    spanKernel<<<8, 64, 0, instrumented>>>(h.dev, id, 4096);
  }
  const CollStatSnapshot w0 = collStatsReadWindow(h, reader, keys, &gating);
  EXPECT_EQ(totalCount(w0.values), 3u);

  // Window 1: two more collectives, gated into the new bank by the flip event.
  for (int i = 0; i < 2; ++i) {
    collStatsEnqueuePreReset(h, /*slot=*/0, instrumented);
    spanKernel<<<8, 64, 0, instrumented>>>(h.dev, id, 4096);
  }
  const CollStatSnapshot w1 = collStatsReadWindow(h, reader, keys, &gating);
  EXPECT_EQ(totalCount(w1.values), 2u);
  EXPECT_EQ(w1.windowEpoch, w0.windowEpoch + 1);

  ASSERT_EQ(cudaStreamSynchronize(instrumented), cudaSuccess);
  cudaEventDestroy(streamEvent);
  cudaEventDestroy(flipEvent);
  cudaStreamDestroy(instrumented);
  cudaStreamDestroy(reader);
  collStatsFreeDeviceBlock(h);
}

// The async issue path must not synchronize: it records a copy-done event, and
// the caller harvests the staging snapshot only after that event completes. The
// caller-tracked epoch must select the correct retired bank each window.
TEST(CollStatsReaderGpuTest, AsyncIssueHarvestsAfterCopyDoneEvent) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    GTEST_SKIP() << "no CUDA device";
  }

  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  cudaStream_t instrumented;
  cudaStream_t reader;
  ASSERT_EQ(cudaStreamCreate(&instrumented), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&reader), cudaSuccess);
  cudaEvent_t streamEvent;
  cudaEvent_t flipEvent;
  cudaEvent_t copyDone;
  ASSERT_EQ(cudaEventCreate(&streamEvent), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&flipEvent), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&copyDone), cudaSuccess);

  const cudaEvent_t streamEvents[1] = {streamEvent};
  CollStatsReadGating gating{};
  gating.instrumentedStreams = &instrumented;
  gating.streamEvents = streamEvents;
  gating.numStreams = 1;
  gating.flipEvent = flipEvent;

  CollStatsKeyRegistry keys(capacity);
  const uint32_t id = keys.resolve(
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});

  CollStatsPinnedStaging staging; // reused across windows, like the driver's
  ASSERT_TRUE(staging.allocate(capacity));
  CollStatSnapshot snap;
  uint64_t epoch = 0; // caller-tracked, in lockstep with the device flips

  // Window 0: three collectives, then an async issue. Harvest after copyDone.
  for (int i = 0; i < 3; ++i) {
    collStatsEnqueuePreReset(h, /*slot=*/0, instrumented);
    spanKernel<<<8, 64, 0, instrumented>>>(h.dev, id, 4096);
  }
  ASSERT_EQ(
      collStatsIssueReadWindow(
          h, reader, &gating, epoch, copyDone, staging, keys),
      cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(copyDone), cudaSuccess);
  staging.publish(epoch, keys, h.cfg, snap);
  ++epoch;
  EXPECT_EQ(snap.numKeys, 1u);
  EXPECT_EQ(snap.windowEpoch, 0u);
  EXPECT_EQ(totalCount(snap.values), 3u);

  // Window 1: two collectives; the tracked epoch selects the other retired
  // bank.
  for (int i = 0; i < 2; ++i) {
    collStatsEnqueuePreReset(h, /*slot=*/0, instrumented);
    spanKernel<<<8, 64, 0, instrumented>>>(h.dev, id, 4096);
  }
  ASSERT_EQ(
      collStatsIssueReadWindow(
          h, reader, &gating, epoch, copyDone, staging, keys),
      cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(copyDone), cudaSuccess);
  staging.publish(epoch, keys, h.cfg, snap);
  ++epoch;
  EXPECT_EQ(snap.windowEpoch, 1u);
  EXPECT_EQ(totalCount(snap.values), 2u);

  ASSERT_EQ(cudaStreamSynchronize(instrumented), cudaSuccess);
  cudaEventDestroy(streamEvent);
  cudaEventDestroy(flipEvent);
  cudaEventDestroy(copyDone);
  cudaStreamDestroy(instrumented);
  cudaStreamDestroy(reader);
  collStatsFreeDeviceBlock(h);
}

} // namespace
} // namespace meta::comms::collstats
