// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Unit-level properties of the low-precision relay path.
 *
 * WHY ppn=1 AND WHY THIS SUITE EXISTS SEPARATELY
 *
 * Nothing here needs a communicator. These are the properties the four
 * collectives will BUILD ON, and they are much easier to pin down here than
 * through an 8-rank collective: what a scale means, where a block ends, that
 * bytes() is additive, that a reduction accumulates in fp32. If one of these is
 * wrong, every collective is wrong in the same way, and a failure in this suite
 * says which property broke instead of "the allreduce answer is off by 3%".
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "meta/relay/sharded_relay_lp.h"
#include "nccl.h"

using namespace rcclx::relay;

namespace {

constexpr size_t kBlock = kLpBlockElems;

} // namespace

// ---------------------------------------------------------------------------
// Wire layout. Host only -- these are the arithmetic identities every offset
// expression in the four collectives is about to depend on.
// ---------------------------------------------------------------------------

TEST(ShardedRelayLpWire, BlockLayoutIsExactly33Over32) {
  EXPECT_EQ(kLpBlockElems, 128u);
  EXPECT_EQ(kLpBlockBytes, 132u);
  EXPECT_EQ(lpWireBytes(kBlock), kLpBlockBytes);
  for (size_t blocks = 1; blocks <= 4096; blocks *= 2) {
    const size_t elems = blocks * kBlock;
    EXPECT_EQ(lpWireBytes(elems) * 32, elems * 33) << "at " << elems;
  }
}

TEST(ShardedRelayLpWire, BytesIsAdditiveOnAlignedCounts) {
  // Exactly the shape every collective relies on: a region split into pieces
  // whose wire bytes must sum to the whole, and an offset that must land where
  // the previous piece ended.
  const std::vector<size_t> counts = {
      kBlock, 2 * kBlock, 8 * kBlock, 1000 * kBlock, 129024, 16777216};
  for (size_t a : counts) {
    for (size_t b : counts) {
      EXPECT_EQ(lpWireBytes(a + b), lpWireBytes(a) + lpWireBytes(b))
          << "a=" << a << " b=" << b;
    }
  }
  // And the three-way split the pipelined paths produce.
  const size_t total = 3 * 1024 * kBlock;
  const size_t p0 = 1024 * kBlock;
  const size_t p1 = 512 * kBlock;
  EXPECT_EQ(
      lpWireBytes(p0) + lpWireBytes(p1) + lpWireBytes(total - p0 - p1),
      lpWireBytes(total));
}

TEST(ShardedRelayLpWire, FullPrecisionWireReproducesTodaysArithmetic) {
  const RelayWire fp = lpFullPrecisionWire(ncclFloat32, sizeof(float));
  EXPECT_FALSE(fp.lp);
  EXPECT_EQ(fp.dtype, ncclFloat32);
  for (size_t n : {size_t{1}, size_t{7}, size_t{128}, size_t{1000000}}) {
    EXPECT_EQ(fp.bytes(n), n * sizeof(float));
    EXPECT_EQ(fp.count(n), n);
  }
}

TEST(ShardedRelayLpWire, LowPrecisionWireIsBytesOfUint8) {
  const RelayWire lp = lpWireFor(ncclBfloat16, sizeof(uint16_t), true);
  EXPECT_TRUE(lp.lp);
  EXPECT_EQ(lp.dtype, ncclUint8);
  // elemSize still describes the caller's buffers, not the wire.
  EXPECT_EQ(lp.elemSize, sizeof(uint16_t));
  for (size_t blocks : {size_t{1}, size_t{3}, size_t{1024}}) {
    const size_t n = blocks * kBlock;
    EXPECT_EQ(lp.bytes(n), lpWireBytes(n));
    // count == bytes is what makes one ncclSend move one self-describing blob.
    EXPECT_EQ(lp.count(n), lp.bytes(n));
  }
  // A bf16 message shrinks by 1.94x on the wire, an fp32 one by 3.88x.
  const size_t n = 1024 * kBlock;
  EXPECT_NEAR(
      static_cast<double>(n * sizeof(uint16_t)) / lp.bytes(n), 1.939, 0.01);
  EXPECT_NEAR(
      static_cast<double>(n * sizeof(float)) / lp.bytes(n), 3.879, 0.01);
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

TEST(ShardedRelayLpGate, SupportsOnlyBf16AndFp32) {
  EXPECT_TRUE(lpDtypeSupported(ncclBfloat16));
  EXPECT_TRUE(lpDtypeSupported(ncclFloat32));
  EXPECT_FALSE(lpDtypeSupported(ncclFloat16));
  EXPECT_FALSE(lpDtypeSupported(ncclInt32));
  EXPECT_FALSE(lpDtypeSupported(ncclFloat64));
}

TEST(ShardedRelayLpGate, RequiresEveryPerGroupCountToBeAWholeNumberOfBlocks) {
  const size_t aligned[] = {kBlock, 4 * kBlock, 16777216};
  const size_t oneOff[] = {kBlock, 4 * kBlock + 1, 16777216};
  const size_t withZero[] = {kBlock, 0};
  EXPECT_TRUE(lpCountsAligned(aligned, 3));
  EXPECT_FALSE(lpCountsAligned(oneOff, 3));
  EXPECT_FALSE(lpCountsAligned(withZero, 2));
  EXPECT_FALSE(lpCountsAligned(nullptr, 1));
  EXPECT_FALSE(lpCountsAligned(aligned, 0));
}

TEST(ShardedRelayLpGate, EachDeclineReasonIsCountedSeparately) {
  // Engagement is asserted through these counters everywhere low precision is
  // tested, because the gate declines SILENTLY -- an LP run that quietly fell
  // back looks exactly like a passing one. So the counters themselves need to
  // be trustworthy.
  const size_t good[] = {4 * kBlock};
  const size_t bad[] = {4 * kBlock + 3};
  const size_t big = static_cast<size_t>(64) << 20;

  LpGateInputs in;
  in.coll = LpCollective::AllReduce;
  in.datatype = ncclFloat32;
  in.counts = good;
  in.nGroups = 1;
  in.nActiveRanksPerGroup = 2;
  in.routeSizeBytes = big;
  in.relayRouteSelected = true;

  lpResetCounters();
  EXPECT_TRUE(lpEligible(in));
  EXPECT_EQ(lpDeclineCount(), 0u);

  LpGateInputs notRelay = in;
  notRelay.relayRouteSelected = false;
  EXPECT_FALSE(lpEligible(notRelay));
  EXPECT_EQ(lpDeclineCount(LpDecline::Route), 1u);

  LpGateInputs wrongDtype = in;
  wrongDtype.datatype = ncclInt32;
  EXPECT_FALSE(lpEligible(wrongDtype));
  EXPECT_EQ(lpDeclineCount(LpDecline::Dtype), 1u);

  LpGateInputs unaligned = in;
  unaligned.counts = bad;
  EXPECT_FALSE(lpEligible(unaligned));
  EXPECT_EQ(lpDeclineCount(LpDecline::Alignment), 1u);

  LpGateInputs tooSmall = in;
  tooSmall.routeSizeBytes = 4096;
  EXPECT_FALSE(lpEligible(tooSmall));
  EXPECT_EQ(lpDeclineCount(LpDecline::Size), 1u);

  EXPECT_EQ(lpDeclineCount(), 4u);
  lpRecordEngage();
  EXPECT_EQ(lpEngageCount(), 1u);
  lpResetCounters();
  EXPECT_EQ(lpEngageCount(), 0u);
  EXPECT_EQ(lpDeclineCount(), 0u);
}

TEST(ShardedRelayLpGate, SizeThresholdIsAboveTheLaunchBoundBand) {
  // Below ~576 KB the measured relay time is flat: that band is pure launch
  // cost, and low precision ADDS launches. A threshold inside it would make
  // things slower.
  for (int a : {2, 4}) {
    for (int g : {1, 4}) {
      EXPECT_GE(lpMinBytes(LpCollective::AllReduce, a, g), size_t{576} << 10);
      EXPECT_GE(lpMinBytes(LpCollective::AllToAll, a, g), size_t{576} << 10);
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
