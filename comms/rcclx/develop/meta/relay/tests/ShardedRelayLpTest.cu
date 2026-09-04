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
 *
 * HOW CORRECTNESS IS ASSERTED WITHOUT A HOST fp8 MODEL
 *
 * rccl_float8 is __hip_fp8_e4m3_fnuz on gfx942 and OCP __hip_fp8_e4m3
 * elsewhere, and the two encode the same value to different bytes. A host-side
 * reference implementation would therefore have to know which device it is
 * talking to, and would silently be testing itself on the arch it guessed
 * wrong. So nothing here models fp8:
 *
 *  - Round-trip cases assert PROPERTIES that hold for both flavours: a block
 *    whose elements are equal comes back bit-exact (the power-of-two
 *    normalization target is what buys that, see sharded_relay_lp.h), and
 *    anything else comes back inside e4m3's relative band.
 *  - Reduce cases build their reference from the DEQUANTIZED inputs, read back
 *    from the device. That takes input quantization out of the comparison
 *    entirely, so what is left under test is the reduction arithmetic and the
 * one rounding of its result -- which is the part these kernels actually own.
 */

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "meta/relay/sharded_relay_lp.h"
#include "meta/relay/sharded_relay_lp_arena.h"
#include "meta/relay/sharded_relay_lp_kernels.h"
#include "nccl.h"

using namespace rcclx::relay;

namespace {

constexpr size_t kBlock = kLpBlockElems;

// e4m3 has 3 explicit mantissa bits, so its half-ULP is 2^-4 = 6.25% relative.
constexpr float kE4m3RelBand = 0.0625f;

// RMS relative error of one e4m3 round-to-nearest, which is the floor the
// global L2 check can possibly sit at. Within a binade the relative ULP runs
// from 2^-3 at the bottom to 2^-4 at the top, and round-to-nearest error is
// uniform on
// +/- ULP/2, so the relative RMS is about 2^-4.5 / sqrt(3) = 0.026. Measured on
// normal data this suite reads 0.0247-0.0257, i.e. exactly that -- which is
// itself the strongest evidence the scale arithmetic is right, and the reason
// an earlier 0.02 limit here was unsatisfiable rather than strict.
//
// 0.035 leaves ~35% headroom over the floor while still being nowhere near what
// a real defect produces: a systematically wrong scale, a scale read from the
// neighbouring block, or a dropped scale all land at 0.5 or above.
constexpr float kL2LimitOneRounding = 0.035f;

// Absolute floor for elements far below their block's maximum. With the absmax
// normalized to 128, the smallest representable normalized magnitude is 2^-9
// (OCP) or 2^-10 (fnuz) so the true floor is blockAbsMax * 2^-16 at worst;
// /1024 leaves 64x of margin, which is the difference between a meaningful
// bound and a flaky one.
constexpr float kAbsBandFraction = 1.0f / 1024.0f;

class DeviceBuf {
 public:
  explicit DeviceBuf(size_t bytes) {
    if (bytes > 0 && hipMalloc(&p_, bytes) != hipSuccess) {
      p_ = nullptr;
    }
  }
  ~DeviceBuf() {
    if (p_ != nullptr) {
      hipFree(p_);
    }
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;

  void* get() const {
    return p_;
  }

 private:
  void* p_{nullptr};
};

void toDevice(void* dst, const std::vector<float>& src) {
  ASSERT_EQ(
      hipMemcpy(
          dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice),
      hipSuccess);
}

std::vector<float> fromDevice(const void* src, size_t count) {
  std::vector<float> out(count, 0.0f);
  EXPECT_EQ(
      hipMemcpy(out.data(), src, count * sizeof(float), hipMemcpyDeviceToHost),
      hipSuccess);
  return out;
}

// Quantize `in`, then dequantize it back. Returns what the wire round-trip
// produced -- which is also, for every reduce test below, the ground truth for
// what the wire actually holds.
std::vector<float> roundTrip(const std::vector<float>& in) {
  const size_t n = in.size();
  DeviceBuf dIn(n * sizeof(float));
  DeviceBuf dWire(lpWireBytes(n));
  DeviceBuf dOut(n * sizeof(float));
  toDevice(dIn.get(), in);
  launchLpQuantizeKernel<float>(dWire.get(), dIn.get(), n, nullptr);
  launchLpDequantizeKernel<float>(dOut.get(), dWire.get(), n, nullptr);
  EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
  return fromDevice(dOut.get(), n);
}

// Quantize `in` into an owned wire buffer, and report what that wire decodes
// to.
struct WireAndTruth {
  std::vector<float> truth;
};

WireAndTruth quantizeInto(void* wire, const std::vector<float>& in) {
  const size_t n = in.size();
  DeviceBuf dIn(n * sizeof(float));
  DeviceBuf dBack(n * sizeof(float));
  toDevice(dIn.get(), in);
  launchLpQuantizeKernel<float>(wire, dIn.get(), n, nullptr);
  launchLpDequantizeKernel<float>(dBack.get(), wire, n, nullptr);
  EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
  return WireAndTruth{fromDevice(dBack.get(), n)};
}

std::vector<float> blockAbsMaxPerElement(const std::vector<float>& v) {
  std::vector<float> out(v.size(), 0.0f);
  for (size_t b = 0; b * kBlock < v.size(); b++) {
    float m = 0.0f;
    for (size_t i = 0; i < kBlock; i++) {
      m = std::max(m, std::fabs(v[b * kBlock + i]));
    }
    for (size_t i = 0; i < kBlock; i++) {
      out[b * kBlock + i] = m;
    }
  }
  return out;
}

// |got - want| <= 6.25%*|want| + absMax/1024, plus a global relative L2 bound.
// The L2 check is not redundant: a 6.25% per-element band would otherwise pass
// a block whose scale is systematically wrong on smooth data.
void expectWithinE4m3Band(
    const std::vector<float>& got,
    const std::vector<float>& want,
    float l2Limit = kL2LimitOneRounding) {
  ASSERT_EQ(got.size(), want.size());
  const std::vector<float> absMax = blockAbsMaxPerElement(want);
  double num = 0.0;
  double den = 0.0;
  size_t reported = 0;
  for (size_t i = 0; i < want.size(); i++) {
    const float tol =
        kE4m3RelBand * std::fabs(want[i]) + kAbsBandFraction * absMax[i];
    const float err = std::fabs(got[i] - want[i]);
    if (err > tol && reported < 8) {
      reported++;
      ADD_FAILURE() << "element " << i << " (block " << i / kBlock << "): got "
                    << got[i] << ", want " << want[i] << ", tolerance " << tol;
    }
    num += static_cast<double>(err) * err;
    den += static_cast<double>(want[i]) * want[i];
  }
  if (den > 0.0) {
    EXPECT_LT(std::sqrt(num / den), l2Limit);
  }
}

void expectExact(
    const std::vector<float>& got,
    const std::vector<float>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < want.size(); i++) {
    ASSERT_FLOAT_EQ(got[i], want[i])
        << "at element " << i << " (block " << i / kBlock << ")";
  }
}

// A distinct constant per block, including a zero block and negatives. These
// are the values that must survive BIT-EXACTLY, which is what lets the
// collectives' existing constant-fill assertions stay exact under low
// precision.
std::vector<float> constantBlocks(size_t nBlocks) {
  static const float kConstants[] = {
      1.0f, -3.5f, 0.0f, 1024.0f, 7.25e-3f, -1.0f, 12345.0f, 0.125f};
  std::vector<float> v(nBlocks * kBlock, 0.0f);
  for (size_t b = 0; b < nBlocks; b++) {
    const float c = kConstants[b % (sizeof(kConstants) / sizeof(float))];
    std::fill(v.begin() + b * kBlock, v.begin() + (b + 1) * kBlock, c);
  }
  return v;
}

std::vector<float> randomValues(size_t n, uint32_t seed, float scale = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; i++) {
    v[i] = dist(rng);
  }
  return v;
}

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

TEST(ShardedRelayLpGate, CountAlignmentRequirementIsPerCall) {
  // 128 elements is the floor, but a schedule that subdivides a region needs
  // more. The flat A>2 allreduce splits its direct region into A per-owner
  // shards, and pD / A is only a whole number of wire blocks when the count is
  // a multiple of A * 128 -- so a count that satisfies the floor can still be
  // wrong for that schedule, and the gate has to be told which it is.
  const size_t blockAligned[] = {4 * kBlock}; // 512 elements
  EXPECT_TRUE(lpCountsAligned(blockAligned, 1));
  EXPECT_TRUE(lpCountsAligned(blockAligned, 1, 4 * kBlock));
  // A whole number of blocks, but not of 4 blocks: fine at A=2, not at A=4.
  const size_t threeBlocks[] = {3 * kBlock};
  EXPECT_TRUE(lpCountsAligned(threeBlocks, 1));
  EXPECT_FALSE(lpCountsAligned(threeBlocks, 1, 4 * kBlock));
  EXPECT_FALSE(lpCountsAligned(blockAligned, 1, 0));

  // And it reaches lpEligible through LpGateInputs.
  LpGateInputs in;
  in.datatype = ncclFloat32;
  in.counts = threeBlocks;
  in.nGroups = 1;
  in.nActiveRanksPerGroup = 4;
  in.routeSizeBytes = static_cast<size_t>(64) << 20;
  in.relayRouteSelected = true;
  lpResetCounters();
  EXPECT_TRUE(lpEligible(in));
  in.countAlignElems = 4 * kBlock;
  EXPECT_FALSE(lpEligible(in));
  EXPECT_EQ(lpDeclineCount(LpDecline::Alignment), 1u);
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

TEST(ShardedRelayLpArena, CapacityFollowsTheMessageProvisioning) {
  // A pure function of NCCL_SHARDED_RELAY_LP_MAX_MSG_MB, which is what makes it
  // safe for every rank to compare its footprint against independently.
  const size_t maxElems = lpMaxMsgBytes() / sizeof(uint16_t);
  EXPECT_EQ(
      lpArenaCapacityBytes(),
      kLpArenaShadowsPerMessage * lpWireBytesRoundUp(maxElems));
  EXPECT_GT(lpArenaCapacityBytes(), lpMaxMsgBytes());
}

TEST(ShardedRelayLpArena, CarverPartitionsDeterministicallyAndRefusesOverrun) {
  std::vector<char> backing(1 << 20);
  LpArenaLease lease{backing.data(), backing.size()};

  LpArenaCarver first(lease);
  char* a1 = first.take(1000);
  char* b1 = first.take(4096);
  ASSERT_NE(a1, nullptr);
  ASSERT_NE(b1, nullptr);

  // Same order, same addresses -- which is what lets a captured graph replay.
  LpArenaCarver second(lease);
  EXPECT_EQ(second.take(1000), a1);
  EXPECT_EQ(second.take(4096), b1);

  // 256-byte aligned, so the inline fp32 scales are always aligned too.
  EXPECT_EQ(reinterpret_cast<uintptr_t>(a1) % LpArenaCarver::kAlign, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b1) % LpArenaCarver::kAlign, 0u);
  EXPECT_GE(b1 - a1, 1000);

  LpArenaCarver small(LpArenaLease{backing.data(), 512});
  EXPECT_NE(small.take(256), nullptr);
  EXPECT_TRUE(small.ok());
  EXPECT_EQ(small.take(1024), nullptr);
  EXPECT_FALSE(small.ok());
}

// ---------------------------------------------------------------------------
// Quantize / dequantize
// ---------------------------------------------------------------------------

TEST(ShardedRelayLpKernels, ConstantBlocksRoundTripBitExactly) {
  // The property the whole test story rests on: with the absmax normalized to a
  // POWER OF TWO, a block of equal values comes back unchanged. That is why the
  // collectives' existing constant-fill assertions stay exact under low
  // precision and become detectors for a wrong scale, a wrong block boundary or
  // a dropped scale. If this fails, do not loosen it.
  const std::vector<float> in = constantBlocks(16);
  expectExact(roundTrip(in), in);
}

TEST(ShardedRelayLpKernels, RandomValuesRoundTripInsideTheE4m3Band) {
  const std::vector<float> in = randomValues(64 * kBlock, 12345);
  expectWithinE4m3Band(roundTrip(in), in);
}

TEST(ShardedRelayLpKernels, ScalesArePerBlockNotPerBuffer) {
  // Block 0 is enormous, block 1 is tiny. Under a single buffer-wide scale
  // every element of block 1 would quantize to zero; under per-block scales
  // block 1 is as accurate as if it were alone. This is the test that a scale
  // is being read from the right block.
  std::vector<float> in(2 * kBlock, 0.0f);
  for (size_t i = 0; i < kBlock; i++) {
    in[i] = 1.0e6f;
    in[kBlock + i] = 1.0e-6f * static_cast<float>(i + 1);
  }
  const std::vector<float> got = roundTrip(in);
  expectWithinE4m3Band(got, in);
  for (size_t i = 0; i < kBlock; i++) {
    EXPECT_GT(got[kBlock + i], 0.0f)
        << "small block element " << i << " was flattened to zero";
  }
}

TEST(ShardedRelayLpKernels, AllZeroBlockRoundTripsToZero) {
  // scale == 0 is the one case the encode has to special-case, or it divides by
  // zero and the block comes back NaN.
  std::vector<float> in(4 * kBlock, 0.0f);
  for (size_t i = 0; i < kBlock; i++) {
    in[kBlock + i] = 2.0f; // a non-zero neighbour, so a shared scale would show
  }
  const std::vector<float> got = roundTrip(in);
  expectExact(got, in);
}

TEST(ShardedRelayLpKernels, ANonFiniteElementDoesNotRoundTripToZero) {
  // The counterpart of the case above, and the reason the absmax fold is not
  // fmaxf: fmaxf returns the non-NaN operand, so a block holding a NaN would
  // take the absmax of its FINITE elements and an all-NaN block would take 0 --
  // which is the all-zero block's absmax, so the block would be written as 128
  // zero codes and come back as clean 0.0. A diverged gradient must not arrive
  // looking converged, or a post-collective isfinite() check stops firing on
  // exactly the runs that need it.
  std::vector<float> in = constantBlocks(4);
  in[0 * kBlock + 5] = std::numeric_limits<float>::quiet_NaN();
  in[2 * kBlock + 7] = std::numeric_limits<float>::infinity();
  const std::vector<float> got = roundTrip(in);

  // Asserted per BLOCK, not per element: a non-finite scale is a property of
  // the whole block, so its 127 finite neighbours are lost too. That is not a
  // regression -- an inf absmax already flattened them to zero -- and it is the
  // conservative direction, because non-finite never decodes back to finite.
  for (size_t i = 0; i < kBlock; i++) {
    EXPECT_FALSE(std::isfinite(got[0 * kBlock + i]))
        << "element " << i << " of the NaN-bearing block came back finite";
    EXPECT_FALSE(std::isfinite(got[2 * kBlock + i]))
        << "element " << i << " of the inf-bearing block came back finite";
  }
  // Blocks 1 and 3 are constant and untouched, so they still round-trip
  // bit-exactly: the fold is per block and a bad neighbour does not leak in.
  for (size_t i = 0; i < kBlock; i++) {
    EXPECT_FLOAT_EQ(got[1 * kBlock + i], in[1 * kBlock + i]) << "at " << i;
    EXPECT_FLOAT_EQ(got[3 * kBlock + i], in[3 * kBlock + i]) << "at " << i;
  }
}

TEST(
    ShardedRelayLpKernels,
    ReduceRequantizeKeepsANonFiniteContributionVisible) {
  // The path that matters most in practice: a region a HELPER reduces lands
  // straight in the caller's receive buffer, so if the requantize swallowed a
  // NaN there is no later kernel to notice. One contribution diverges; the sum
  // must not come back finite.
  constexpr int kContribs = 3;
  const size_t n = 4 * kBlock;
  DeviceBuf dContribs(kContribs * lpWireBytes(n));
  DeviceBuf dOut(lpWireBytes(n));
  DeviceBuf dBack(n * sizeof(float));

  for (int p = 0; p < kContribs; p++) {
    std::vector<float> in = constantBlocks(4);
    if (p == 1) {
      in[kBlock + 3] = std::numeric_limits<float>::quiet_NaN();
    }
    char* slot = static_cast<char*>(dContribs.get()) + p * lpWireBytes(n);
    (void)quantizeInto(slot, in);
  }

  launchLpReduceRequantizeKernel(
      dOut.get(), dContribs.get(), kContribs, n, /*divisor=*/1, nullptr);
  launchLpDequantizeKernel<float>(dBack.get(), dOut.get(), n, nullptr);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  const std::vector<float> got = fromDevice(dBack.get(), n);
  for (size_t i = 0; i < kBlock; i++) {
    EXPECT_FALSE(std::isfinite(got[kBlock + i]))
        << "element " << i << " of the diverged block survived the helper hop "
        << "as a finite value";
  }
}

TEST(ShardedRelayLpKernels, TailBlockIsQuantizedAndNeighboursAreUntouched) {
  // A count that is a whole number of blocks but not a power of two, so the
  // grid-stride loop's last iteration is partial.
  const size_t nBlocks = 8191;
  std::vector<float> in = randomValues(nBlocks * kBlock, 777);
  // Make the very last block constant, so the tail is checked exactly.
  std::fill(in.end() - kBlock, in.end(), -2.75f);
  const std::vector<float> got = roundTrip(in);
  for (size_t i = 0; i < kBlock; i++) {
    ASSERT_FLOAT_EQ(got[in.size() - kBlock + i], -2.75f)
        << "tail element " << i;
  }
  expectWithinE4m3Band(got, in);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

TEST(ShardedRelayLpKernels, ReduceRequantizeSumsWireContributionsInFp32) {
  constexpr int kContribs = 4;
  const size_t n = 32 * kBlock;
  DeviceBuf dContribs(kContribs * lpWireBytes(n));
  DeviceBuf dOut(lpWireBytes(n));
  DeviceBuf dBack(n * sizeof(float));

  std::vector<float> want(n, 0.0f);
  for (int p = 0; p < kContribs; p++) {
    const std::vector<float> in = randomValues(n, 100 + p);
    char* slot = static_cast<char*>(dContribs.get()) + p * lpWireBytes(n);
    // Reference is what the WIRE holds, not what we handed in, so input
    // quantization is out of the comparison.
    const WireAndTruth w = quantizeInto(slot, in);
    for (size_t i = 0; i < n; i++) {
      want[i] += w.truth[i];
    }
  }

  launchLpReduceRequantizeKernel(
      dOut.get(), dContribs.get(), kContribs, n, /*divisor=*/1, nullptr);
  launchLpDequantizeKernel<float>(dBack.get(), dOut.get(), n, nullptr);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  expectWithinE4m3Band(fromDevice(dBack.get(), n), want);
}

TEST(ShardedRelayLpKernels, ReduceRequantizeIsExactForEqualConstantBlocks) {
  // A sum of equal values has block absmax equal to that sum, so the
  // power-of-two normalization makes the requantized result exact too. This is
  // what keeps the collectives' constant-fill reduce assertions exact across
  // the helper hop.
  constexpr int kContribs = 3;
  const size_t n = 8 * kBlock;
  DeviceBuf dContribs(kContribs * lpWireBytes(n));
  DeviceBuf dOut(lpWireBytes(n));
  DeviceBuf dBack(n * sizeof(float));

  const std::vector<float> in = constantBlocks(8);
  for (int p = 0; p < kContribs; p++) {
    char* slot = static_cast<char*>(dContribs.get()) + p * lpWireBytes(n);
    const WireAndTruth w = quantizeInto(slot, in);
    expectExact(w.truth, in);
  }
  std::vector<float> want(n);
  for (size_t i = 0; i < n; i++) {
    want[i] = static_cast<float>(kContribs) * in[i];
  }

  launchLpReduceRequantizeKernel(
      dOut.get(), dContribs.get(), kContribs, n, /*divisor=*/1, nullptr);
  launchLpDequantizeKernel<float>(dBack.get(), dOut.get(), n, nullptr);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  expectExact(fromDevice(dBack.get(), n), want);
}

TEST(ShardedRelayLpKernels, ReduceRequantizeDivisorIsExactAndAppliedOnce) {
  // The divisor is always nActiveRanks, which the dispatchers require to be a
  // power of two. Scaling before the requantize therefore moves the block
  // absmax by the same exact power of two and leaves every fp8 code unchanged,
  // so dividing here is bit-identical to dividing after the dequantize. Checked
  // directly, because it is the reason the helper may own the divisor at all --
  // the regions a helper reduces have no active-side closing kernel to defer it
  // to.
  constexpr int kContribs = 4;
  constexpr int kDivisor = 4;
  const size_t n = 8 * kBlock;
  DeviceBuf dContribs(kContribs * lpWireBytes(n));
  DeviceBuf dPlain(lpWireBytes(n));
  DeviceBuf dDivided(lpWireBytes(n));
  DeviceBuf dPlainBack(n * sizeof(float));
  DeviceBuf dDividedBack(n * sizeof(float));

  for (int p = 0; p < kContribs; p++) {
    char* slot = static_cast<char*>(dContribs.get()) + p * lpWireBytes(n);
    (void)quantizeInto(slot, randomValues(n, 900 + p));
  }

  launchLpReduceRequantizeKernel(
      dPlain.get(), dContribs.get(), kContribs, n, /*divisor=*/1, nullptr);
  launchLpReduceRequantizeKernel(
      dDivided.get(), dContribs.get(), kContribs, n, kDivisor, nullptr);
  launchLpDequantizeKernel<float>(dPlainBack.get(), dPlain.get(), n, nullptr);
  launchLpDequantizeKernel<float>(
      dDividedBack.get(), dDivided.get(), n, nullptr);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  const std::vector<float> plain = fromDevice(dPlainBack.get(), n);
  const std::vector<float> divided = fromDevice(dDividedBack.get(), n);
  for (size_t i = 0; i < n; i++) {
    // Bit-exact, not approximate: no extra rounding may creep in.
    ASSERT_FLOAT_EQ(divided[i], plain[i] / static_cast<float>(kDivisor))
        << "at element " << i;
  }
}

TEST(ShardedRelayLpKernels, MultiReduceAccumulatesIntoDstAndDividesOnce) {
  constexpr int kContribs = 3;
  constexpr int kDivisor = 4; // == 1 seed + 3 contributions, i.e. ncclAvg
  const size_t n = 16 * kBlock;
  DeviceBuf dContribs(kContribs * lpWireBytes(n));
  DeviceBuf dDst(n * sizeof(float));

  const std::vector<float> dstIn = randomValues(n, 42);
  std::vector<float> want = dstIn;
  for (int p = 0; p < kContribs; p++) {
    const std::vector<float> in = randomValues(n, 200 + p);
    char* slot = static_cast<char*>(dContribs.get()) + p * lpWireBytes(n);
    const WireAndTruth w = quantizeInto(slot, in);
    for (size_t i = 0; i < n; i++) {
      want[i] += w.truth[i];
    }
  }
  for (size_t i = 0; i < n; i++) {
    want[i] /= static_cast<float>(kDivisor);
  }

  toDevice(dDst.get(), dstIn);
  launchLpMultiReduceKernel<float>(
      dDst.get(), dContribs.get(), kContribs, n, kDivisor, nullptr);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  // dst and the contributions are exact here (dst is never quantized, the
  // contributions were read back from the wire), so the only slack is fp32
  // summation order.
  const std::vector<float> got = fromDevice(dDst.get(), n);
  for (size_t i = 0; i < n; i++) {
    ASSERT_NEAR(got[i], want[i], 1e-4f * std::fabs(want[i]) + 1e-6f)
        << "at element " << i;
  }
}

TEST(ShardedRelayLpKernels, SeededMultiReduceReadsSeedNotDst) {
  constexpr int kContribs = 2;
  const size_t n = 8 * kBlock;
  DeviceBuf dContribs(kContribs * lpWireBytes(n));
  DeviceBuf dSeed(n * sizeof(float));
  DeviceBuf dDst(n * sizeof(float));

  const std::vector<float> seed = randomValues(n, 7);
  // Poison dst: if the kernel reads it instead of the seed, the answer is wrong
  // by a large, obvious amount rather than subtly.
  const std::vector<float> poison(n, 1.0e9f);

  std::vector<float> want = seed;
  for (int p = 0; p < kContribs; p++) {
    const std::vector<float> in = randomValues(n, 300 + p);
    char* slot = static_cast<char*>(dContribs.get()) + p * lpWireBytes(n);
    const WireAndTruth w = quantizeInto(slot, in);
    for (size_t i = 0; i < n; i++) {
      want[i] += w.truth[i];
    }
  }

  toDevice(dSeed.get(), seed);
  toDevice(dDst.get(), poison);
  launchLpSeededMultiReduceKernel<float>(
      dDst.get(), dSeed.get(), dContribs.get(), kContribs, n, 1, nullptr);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  const std::vector<float> got = fromDevice(dDst.get(), n);
  for (size_t i = 0; i < n; i++) {
    ASSERT_NEAR(got[i], want[i], 1e-4f * std::fabs(want[i]) + 1e-6f)
        << "at element " << i;
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
