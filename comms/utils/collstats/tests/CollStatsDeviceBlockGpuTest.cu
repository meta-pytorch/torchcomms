// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsSpan.cuh"
#include "comms/utils/collstats/CollStatsTypes.h"

namespace meta::comms::collstats {
namespace {

// One elected thread per block records a single observation, mimicking the
// collective finalizer: block b maps to key (b % numDistinctKeys) and a
// duration that scales with the key, so the whole device claim + accumulate
// path runs under real cross-block atomic contention.
__global__ void recordKernel(
    CollStatsDeviceBlock* block,
    uint32_t numDistinctKeys,
    uint64_t baseNs) {
  if (threadIdx.x != 0) {
    return;
  }
  // One value slot per distinct key, as the host registry would have handed
  // them out.
  const uint32_t keyId = blockIdx.x % numDistinctKeys;
  const uint64_t dur = baseNs * (keyId + 1);
  collStatsRecordById(block, keyId, dur, 100);
}

// The skip is the same in every test here, so it lives in SetUp rather than
// being restated nine times.
class CollStatsDeviceBlockGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "no CUDA device";
    }
  }

  // The retired bank is always read back whole, so the copy and its sizing live
  // in one place: values past the claimed keys are zero and cost nothing to
  // compare.
  static std::vector<CollStatValue> readValues(
      const CollStatsDeviceBlockHandle& h,
      uint32_t capacity) {
    std::vector<CollStatValue> values(capacity + 1);
    EXPECT_EQ(
        cudaMemcpy(
            values.data(),
            h.values[0], // epoch started at 0
            values.size() * sizeof(CollStatValue),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return values;
  }
};

TEST_F(CollStatsDeviceBlockGpuTest, ConcurrentRecordAccumulatesOnDevice) {
  const uint32_t capacity = 64;
  const uint32_t numSlots = 4;
  const uint32_t numDistinctKeys = 5;
  const uint32_t numBlocks = 200;
  const uint64_t baseNs = 1'000'000; // 1 ms

  CollStatsDeviceBlockHandle h = collStatsAllocDeviceBlock(capacity, numSlots);
  ASSERT_NE(h.dev, nullptr);

  recordKernel<<<numBlocks, 32>>>(h.dev, numDistinctKeys, baseNs);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  CollStatsDeviceBlock hostBlock{};
  ASSERT_EQ(
      cudaMemcpy(
          &hostBlock,
          h.dev,
          sizeof(CollStatsDeviceBlock),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  // The readback is the only coverage the publish step has: the block the
  // device actually sees must carry the capacity, slot count and device
  // pointers the allocator was asked for, not just be copyable.
  EXPECT_EQ(hostBlock.bank.numKeys, capacity);
  EXPECT_EQ(hostBlock.numSlots, numSlots);
  EXPECT_EQ(hostBlock.bank.values[0], h.values[0]);
  EXPECT_EQ(hostBlock.bank.values[1], h.values[1]);
  EXPECT_EQ(hostBlock.span, h.span);
  EXPECT_EQ(
      hostBlock.hist.numBuckets, collStatDefaultHistGeometry().numBuckets);
  EXPECT_EQ(hostBlock.numThresholds, kMaxThresholds);

  const std::vector<CollStatValue> values = readValues(h, capacity);

  uint64_t totalCount = 0;
  uint64_t totalHistogram = 0;
  for (const auto& v : values) {
    totalCount += v.count;
    for (uint32_t b = 0; b < kHistMaxBuckets; ++b) {
      totalHistogram += v.histogram[b];
    }
  }
  EXPECT_EQ(totalCount, numBlocks);
  EXPECT_EQ(totalHistogram, numBlocks);

  collStatsFreeDeviceBlock(h);
}

// Mirrors the collective kernel's use of the span: every block records its
// start on entry, then after a block-wide barrier the elected thread finalizes;
// the last block to arrive records one observation.
// `threadIdx.x == 0` is deliberately the caller's whole election: it is the
// idiom every instrumented collective uses, so the span helpers must be correct
// under it for any block shape.
__global__ void spanKernel(
    CollStatsDeviceBlock* block,
    uint32_t slot,
    uint32_t keyId,
    uint64_t logicalBytes) {
  collStatsSpanEntry(block, slot);
  __syncthreads();
  if (threadIdx.x == 0) {
    collStatsSpanFinalizeElectedById(block, slot, keyId, logicalBytes);
  }
}

// Byte offsets rather than &h.span[slot].arrived: the latter is host code
// forming an address inside a device allocation, which sanitizers flag even
// though it never dereferences. collStatsEnqueuePreReset computes its addresses
// the same way for the same reason.
char* arrivedAddr(const CollStatsDeviceBlockHandle& h, uint32_t slot) {
  return reinterpret_cast<char*>(h.span) +
      static_cast<std::size_t>(slot) * sizeof(CollStatSpanScratch) +
      offsetof(CollStatSpanScratch, arrived);
}

uint32_t readArrived(const CollStatsDeviceBlockHandle& h, uint32_t slot) {
  uint32_t arrived = 0;
  EXPECT_EQ(
      cudaMemcpy(
          &arrived,
          arrivedAddr(h, slot),
          sizeof(arrived),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return arrived;
}

void writeArrived(
    const CollStatsDeviceBlockHandle& h,
    uint32_t slot,
    uint32_t value) {
  EXPECT_EQ(
      cudaMemcpy(
          arrivedAddr(h, slot), &value, sizeof(value), cudaMemcpyHostToDevice),
      cudaSuccess);
}

uint64_t totalCount(const std::vector<CollStatValue>& values) {
  uint64_t total = 0;
  for (const auto& v : values) {
    total += v.count;
  }
  return total;
}

TEST_F(CollStatsDeviceBlockGpuTest, SpanRecordsOneObservationPerLaunch) {
  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  // Value slot 0, as the host registry would have resolved it.
  constexpr uint32_t kKeyId = 0;

  // Mirror the launch path: a stream-ordered pre-reset before each collective.
  // The finalizer emits only, so the pre-reset owns slot cleanup.
  collStatsEnqueuePreReset(h, /*slot=*/0, /*stream=*/0);
  spanKernel<<<16, 64>>>(h.dev, /*slot=*/0, kKeyId, 4096);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(totalCount(readValues(h, capacity)), 1u);

  // A second collective, again pre-reset, records again.
  collStatsEnqueuePreReset(h, /*slot=*/0, /*stream=*/0);
  spanKernel<<<16, 64>>>(h.dev, /*slot=*/0, kKeyId, 4096);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const std::vector<CollStatValue> values = readValues(h, capacity);
  EXPECT_EQ(totalCount(values), 2u);

  uint64_t maxDur = 0;
  for (const auto& v : values) {
    maxDur = v.durMaxNs > maxDur ? v.durMaxNs : maxDur;
  }
  EXPECT_GT(maxDur, 0u);
  EXPECT_LT(maxDur, 1'000'000'000ull); // well under a second

  collStatsFreeDeviceBlock(h);
}

// An aborted sequence — entry without a finalize (e.g. fewer blocks than
// gridDim, or a kernel that returns early) — leaves the slot dirty. The
// stream-ordered pre-reset must clean it so the next collective still records a
// correct span. This is the property the old finalizer self-reset could not
// give.
__global__ void spanEntryOnlyKernel(CollStatsDeviceBlock* block) {
  collStatsSpanEntry(block, /*slot=*/0);
}

TEST_F(CollStatsDeviceBlockGpuTest, PreResetRecoversFromAbortedSequence) {
  const uint32_t capacity = 64;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  constexpr uint32_t kKeyId = 0;

  // Aborted sequence: blocks record entry but never finalize, so arrived is
  // left nonzero and start holds a stale minimum.
  spanEntryOnlyKernel<<<16, 64>>>(h.dev);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // The successor's pre-reset cleans the slot; the full sequence then records
  // exactly one observation with a plausible positive duration.
  collStatsEnqueuePreReset(h, /*slot=*/0, /*stream=*/0);
  spanKernel<<<16, 64>>>(h.dev, /*slot=*/0, kKeyId, 4096);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::vector<CollStatValue> values = readValues(h, capacity);
  EXPECT_EQ(totalCount(values), 1u);
  uint64_t maxDur = 0;
  for (const auto& v : values) {
    maxDur = v.durMaxNs > maxDur ? v.durMaxNs : maxDur;
  }
  EXPECT_GT(maxDur, 0u);
  EXPECT_LT(maxDur, 1'000'000'000ull);

  collStatsFreeDeviceBlock(h);
}

// The comm allocates a single span slot, because OrderedWorkStreamGuard leaves
// only one instrumented collective live at a time. Anything asking for a slot
// past that must be a no-op rather than scribbling past the allocation.
TEST_F(CollStatsDeviceBlockGpuTest, PreResetIgnoresOutOfRangeSlot) {
  const uint32_t capacity = 8;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/1);
  ASSERT_NE(h.dev, nullptr);

  // Dirty the one real slot so an out-of-range reset landing there would show.
  collStatsEnqueuePreReset(h, /*slot=*/0, /*stream=*/0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const uint32_t arrived = 7;
  writeArrived(h, /*slot=*/0, arrived);

  collStatsEnqueuePreReset(h, /*slot=*/1, /*stream=*/0);
  collStatsEnqueuePreReset(h, /*slot=*/~0u, /*stream=*/0);
  // A null handle is a no-op too (instrumentation off).
  CollStatsDeviceBlockHandle none{};
  collStatsEnqueuePreReset(none, /*slot=*/0, /*stream=*/0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  EXPECT_EQ(readArrived(h, /*slot=*/0), arrived)
      << "slot 0 untouched by the out-of-range and null-handle resets";

  collStatsFreeDeviceBlock(h);
}

// The finalizer elects internally, so a caller using the 1-D idiom in a 2-D
// block still contributes exactly one arrival per block. Asserted on `arrived`
// rather than on the duration because it is deterministic: without the internal
// election this reads gridDim * blockDim.y, the equality fires while most
// blocks are still running, and the recorded span is plausible but far too
// short -- a wrong number rather than a missing one.
TEST_F(CollStatsDeviceBlockGpuTest, SpanElectsInternallyInAMultiDimBlock) {
  const uint32_t capacity = 64;
  const uint32_t numBlocks = 16;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/4);
  ASSERT_NE(h.dev, nullptr);

  collStatsEnqueuePreReset(h, /*slot=*/0, /*stream=*/0);
  // 64x2: the y dimension is what a caller electing on threadIdx.x alone
  // misses.
  spanKernel<<<numBlocks, dim3(64, 2, 1)>>>(
      h.dev, /*slot=*/0, /*keyId=*/0, 4096);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readArrived(h, /*slot=*/0), numBlocks);

  const std::vector<CollStatValue> values = readValues(h, capacity);
  EXPECT_EQ(totalCount(values), 1u);

  collStatsFreeDeviceBlock(h);
}

// The device span helpers bound the slot themselves. The host pre-reset already
// does, but only the device side turns a bad slot into a write past the span
// allocation rather than a lost observation.
TEST_F(CollStatsDeviceBlockGpuTest, SpanIgnoresOutOfRangeSlotOnDevice) {
  const uint32_t capacity = 64;
  const uint32_t numSlots = 4;
  CollStatsDeviceBlockHandle h = collStatsAllocDeviceBlock(capacity, numSlots);
  ASSERT_NE(h.dev, nullptr);

  spanKernel<<<16, 64>>>(h.dev, /*slot=*/numSlots, /*keyId=*/0, 4096);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  const std::vector<CollStatValue> values = readValues(h, capacity);
  EXPECT_EQ(totalCount(values), 0u) << "an out-of-range slot records nothing";
  // The in-range slots are untouched, so the bad slot did not alias one.
  for (uint32_t s = 0; s < numSlots; ++s) {
    EXPECT_EQ(readArrived(h, s), 0u);
  }

  collStatsFreeDeviceBlock(h);
}

__global__ void recordAtKeyKernel(CollStatsDeviceBlock* block, uint32_t keyId) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    collStatsRecordById(block, keyId, /*durNs=*/5'000, /*logicalBytes=*/64);
  }
}

// A key id past the registry's capacity is clamped onto the trailing catch-all
// slot. This is the only branch in collStatsRecordById, and it is what keeps a
// saturated registry losing attribution rather than indexing past the bank.
TEST_F(CollStatsDeviceBlockGpuTest, RecordClampsIdsPastCapacityToCatchAll) {
  const uint32_t capacity = 8;
  CollStatsDeviceBlockHandle h =
      collStatsAllocDeviceBlock(capacity, /*numSlots=*/1);
  ASSERT_NE(h.dev, nullptr);

  recordAtKeyKernel<<<1, 32>>>(h.dev, /*keyId=*/capacity + 5);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::vector<CollStatValue> values = readValues(h, capacity);

  EXPECT_EQ(values[capacity].count, 1u) << "catch-all is the trailing slot";
  for (uint32_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(values[i].count, 0u) << "no real key was charged, i=" << i;
  }

  collStatsFreeDeviceBlock(h);
}

// The owner is the only thing that frees a block, so its move paths decide
// between a double free and a leak. A moved-from owner must be empty, and
// self-assignment must not free the block it is about to keep.
TEST_F(CollStatsDeviceBlockGpuTest, OwnerMovesWithoutDoubleFree) {
  CollStatsDeviceBlockOwner a{
      collStatsAllocDeviceBlock(/*keyCapacity=*/8, /*numSlots=*/1)};
  ASSERT_TRUE(a.valid());
  const CollStatsDeviceBlock* dev = a.handle().dev;

  CollStatsDeviceBlockOwner b{std::move(a)};
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.handle().dev, dev);
  EXPECT_FALSE(a.valid()) << "moved-from owner must not free the block again";

  // Through a reference, so this is a real self-assignment rather than one the
  // compiler diagnoses and folds away.
  CollStatsDeviceBlockOwner& alias = b;
  b = std::move(alias);
  EXPECT_TRUE(b.valid()) << "self-assignment kept the block";
  EXPECT_EQ(b.handle().dev, dev);

  {
    CollStatsDeviceBlockOwner empty;
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// Every rejection path guards a fixed-capacity device array against an
// out-of-range index, so each must report a null handle -- leaving
// instrumentation off -- rather than hand back a block that would be indexed
// past its own allocation.
TEST_F(CollStatsDeviceBlockGpuTest, AllocRejectsWhatItCannotBound) {
  // keyCapacity + 1 is the value-slot count, so UINT32_MAX wraps it to zero.
  EXPECT_EQ(collStatsAllocDeviceBlock(UINT32_MAX, /*numSlots=*/1).dev, nullptr);

  // A zero-slot block would leave every span entry addressing nothing.
  EXPECT_EQ(
      collStatsAllocDeviceBlock(/*keyCapacity=*/8, /*numSlots=*/0).dev,
      nullptr);

  // A count too small for its own bounds: logBucketNs derives the index from
  // the tMin/tMax/sub-bucket triple, not from numBuckets, so this passes a
  // range check and still indexes past histogram[].
  CollStatsBlockConfig understatedBuckets = collStatDefaultBlockConfig();
  understatedBuckets.hist.numBuckets -= 1;
  EXPECT_EQ(
      collStatsAllocDeviceBlock(
          /*keyCapacity=*/8, /*numSlots=*/1, understatedBuckets)
          .dev,
      nullptr);

  // tMinNs == 0 makes log2(dur / tMinNs) infinite and the cast to an index
  // undefined; the re-derivation reports zero buckets and the alloc refuses.
  CollStatsBlockConfig zeroTMin = collStatDefaultBlockConfig();
  zeroTMin.hist.tMinNs = 0;
  EXPECT_EQ(
      collStatsAllocDeviceBlock(/*keyCapacity=*/8, /*numSlots=*/1, zeroTMin)
          .dev,
      nullptr);

  // More thresholds than the device-resident array holds.
  CollStatsBlockConfig tooManyThresholds = collStatDefaultBlockConfig();
  tooManyThresholds.numThresholds = kMaxThresholds + 1;
  EXPECT_EQ(
      collStatsAllocDeviceBlock(
          /*keyCapacity=*/8, /*numSlots=*/1, tooManyThresholds)
          .dev,
      nullptr);
}

} // namespace
} // namespace meta::comms::collstats
