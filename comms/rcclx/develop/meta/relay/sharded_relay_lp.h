/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "meta/relay/sharded_relay_route.h"
#include "nccl.h"

/**
 * Low-precision (fp8e4m3) wire format for the sharded-relay collectives.
 *
 * WHY THIS EXISTS
 *
 * Every win the relay collectives have came from moving fewer bytes per XGMI
 * link direction. This moves SMALLER bytes: data is quantized to fp8e4m3 before
 * it crosses a rank boundary and dequantized after. For bf16 that is 1.94x less
 * on the wire, for fp32 3.88x, on exactly the links the relay was built to keep
 * busy.
 *
 * THE ONE IDEA
 *
 * The wire format of a transfer becomes a per-call VALUE, not a second set of
 * collectives. Every boundary-crossing site in all four collectives has one
 * shape today:
 *
 *     ncclSend(base + offsetElems * elementSize, nElems, datatype, peer, ...)
 *
 * A RelayWire answers `bytes(elems)`, `count(elems)` and `dtype`, so each site
 * becomes
 *
 *     ncclSend(base + wire.bytes(offsetElems), wire.count(nElems), wire.dtype,
 * ...)
 *
 * With LP off every one of those expressions evaluates to exactly what it does
 * today, so the full-precision path is unchanged by construction rather than by
 * testing. Routing, group structure, pipelining and op counts are untouched,
 * and a helper that merely forwards bytes today keeps forwarding bytes and
 * never becomes LP-aware. If review ever finds a passthrough helper being made
 * LP-aware, this abstraction has leaked and the change is wrong.
 *
 * WHY 33/32
 *
 * The wire buffer is a sequence of 132-byte blocks: 128 fp8e4m3 payload bytes
 * followed by one fp32 scale derived from that block's absmax. 132/128 = 33/32
 * exactly, so bytes(n) = n * 33/32 is exact and -- critically -- ADDITIVE. The
 * scales ride inline, so one ncclSend still moves one self-describing blob.
 * Overhead is 3.1%: 1.031 effective bytes per element.
 *
 * ADDITIVITY NEEDS 128-ALIGNED COUNTS, SO THE GATE REQUIRES THEM
 *
 * bytes(a + b) == bytes(a) + bytes(b) only when both are multiples of 128, and
 * the violations are not tail-only. `dirBSizes[g] = counts[g] - dirBOffsets[g]`
 * inherits counts[g]'s alignment in all four collectives; all-gather and
 * all-to-all slot strides and reduce-scatter's sendBlockOffset are raw
 * per-group counts, which break in the MIDDLE of a buffer. Every one of them
 * becomes aligned by construction the moment the per-group count is, so
 * lpEligible() requires `counts[g] % kRelayChunkAlignElements == 0` for every
 * group. That is a size-only predicate and therefore agreed across ranks
 * without communication, the same discipline the one-shot gate follows. bytes()
 * asserts the alignment under LP, which turns every substituted site into a
 * self-test.
 *
 * NUMERICS: A POWER-OF-TWO NORMALIZATION TARGET
 *
 * A block's absmax m is normalized to kLpNormalizeMax = 128, NOT to the fp8
 * format's own maximum (240 for the fnuz flavour, 448 for OCP). 128 is a power
 * of two, which makes the whole scale chain exact:
 *
 *   scale  = m * 2^-7                   exact (power-of-two multiply)
 *   code   = fp8(v / scale)             for v == m the true quotient is exactly
 *                                       128, which fp8e4m3 represents exactly
 *   value' = code * scale               exact for code == 128
 *
 * so a block whose elements are all equal round-trips BIT-EXACTLY, and so does
 * a sum of such blocks. That is what lets the existing constant-fill tests keep
 * their exact comparators and become genuine LP bug detectors for a wrong
 * scale, a wrong block boundary or a dropped scale -- they must not be
 * loosened. Normalizing to 240 or 448 instead would make `m / FP8_MAX` a
 * rounded fp32 value and lose that property for no gain: relative precision
 * inside a block is set by e4m3's 3 explicit mantissa bits (half-ULP 2^-4
 * = 6.25%) either way, and 128 still leaves 17 binades of in-block dynamic
 * range before subnormals.
 *
 * The BYTE PATTERNS are still arch-local -- rccl_float8 is __hip_fp8_e4m3_fnuz
 * on gfx942 and OCP __hip_fp8_e4m3 elsewhere, and the two encode the same value
 * differently. Fine today: the relay is intra-node and homogeneous. Latent if a
 * relay ever spans a heterogeneous set of devices.
 *
 * THE SWITCH IS A PER-CALL ARGUMENT
 *
 * Low precision is requested per call by a trailing `low_precision` argument on
 * each collective, not by a process-wide env var, because a job legitimately
 * wants its gradient allreduce in fp8 while the optimizer-state collective
 * stays full precision -- same process, same communicator, same step.
 *
 * `low_precision` is a COLLECTIVE argument. It must be identical on every rank
 * of the call, exactly like datatype, op and the counts. Ranks that disagree
 * disagree on wire byte counts, so the call hangs or corrupts rather than
 * degrading. It is documented, not validated, because validating it would cost
 * an allreduce per call -- the same treatment datatype already gets.
 *
 * Requesting LP means "use low precision where it pays", not "quantize
 * unconditionally". On top of the caller's request the internal gate is
 * size-only, so ranks agree on it without communication: supported dtype, 128
 * aligned counts, a relay (not PureDirect, not one-shot) route, a size above
 * the measured crossover, and a call that fits the arena. Measured relay time
 * is flat across 4 KB..576 KB -- that regime is launch-bound with no bandwidth
 * term, and LP ADDS launches -- so small messages stay full precision even when
 * asked.
 *
 * The env vars here are tuning and provisioning only, never a feature switch.
 * NCCL_SHARDED_RELAY_LP_MIN_KB is the one worth knowing about: it moves the
 * size crossover, which is how the crossover gets measured in the first place
 * -- the built-in value declines small messages before anything is timed, so a
 * sweep that could not override it would have no way to see whether low
 * precision would have won there.
 */
namespace rcclx::relay {

// ---------------------------------------------------------------------------
// Wire layout
// ---------------------------------------------------------------------------

// Payload elements per wire block. A property of the FORMAT, so it is its own
// constant: the 33/32 wire ratio and the exactness of lpWireBytes() both depend
// on it being 128, and the static asserts below hold it there.
//
// It used to be defined as kRelayChunkAlignElements, on the reasoning that
// every region boundary must be a whole number of blocks and that constant is
// what the schedules align to. The REQUIREMENT is divisibility, not equality,
// and conflating them made the wire format hostage to a tuning constant --
// raising the chunk alignment to fix a misalignment stall would have silently
// redefined a block as 516 bytes and coarsened the scale granularity fourfold.
// The divisibility assertion below is the real invariant.
inline constexpr size_t kLpBlockElems = 128;

// One fp32 scale per block, trailing its payload.
inline constexpr size_t kLpScaleBytes = sizeof(float);

// 128 payload bytes + 4 scale bytes.
inline constexpr size_t kLpBlockBytes = kLpBlockElems + kLpScaleBytes;

static_assert(kLpBlockElems == 128, "the 33/32 wire ratio assumes 128");
static_assert(
    kRelayChunkAlignElements % kLpBlockElems == 0,
    "every relay chunk boundary must be a whole number of wire blocks, or a "
    "region offset in wire bytes would not be additive");

// A chunk boundary must also land on a 16-BYTE-ALIGNED wire offset, which is a
// strictly stronger requirement than being a whole number of blocks and is the
// reason kRelayChunkAlignElements is 512.
//
// A block is 132 bytes and 132 = 4 * 33, so a boundary of B blocks sits at
// 132*B bytes, which is 16-byte aligned only when B is a multiple of 4. Full
// precision never had this problem: its offsets are elements * elementSize, and
// a power-of-two element size turns element alignment into byte alignment for
// free. The 33/32 ratio is what breaks the implication, so this is a
// low-precision-specific constraint even though it is satisfied by a shared
// constant.
//
// MEASURED COST of getting this wrong, on the shapes where a boundary happened
// to land on an odd block count: fused all-to-all A=4 ran at 0.64x of
// full-precision relay at 32 MB and 0.65x at 40 MB while reading 1.16x-1.19x at
// every neighbouring size, and single-group all-gather sat at 0.78x-0.90x
// across 31.5-72 MB. With the boundary 4-block aligned those became 1.13x-1.19x
// and 1.19x-1.30x. The offending sizes were exactly those whose chunk count was
// not a multiple of 4; nothing about the data or the reduction changed.
static_assert(
    (kRelayChunkAlignElements / kLpBlockElems) % 4 == 0,
    "a relay chunk boundary must be a multiple of 4 wire blocks so its offset in "
    "wire bytes (132 per block) is 16-byte aligned");
static_assert(
    kLpBlockBytes * 32 == kLpBlockElems * 33,
    "wire bytes per element must be exactly 33/32");
static_assert(
    kLpBlockBytes % kLpScaleBytes == 0,
    "block stride must keep the inline scale 4-byte aligned, so every LP "
    "buffer offset -- always a whole number of blocks -- lands on an aligned "
    "float");

// Absmax normalization target. A power of two on purpose; see the file comment.
inline constexpr float kLpNormalizeMax = 128.0f;
inline constexpr float kLpInvNormalizeMax = 1.0f / kLpNormalizeMax;

static_assert(
    kLpNormalizeMax == 128.0f,
    "kLpInvNormalizeMax must stay an exact power of two, or the scale "
    "arithmetic stops being exact and constant blocks stop round-tripping");

// Wire bytes for `elems` elements. Free function so host sizing code and device
// kernels can share it without a RelayWire in hand.
inline constexpr size_t lpWireBytes(size_t elems) {
  return (elems / kLpBlockElems) * kLpBlockBytes;
}

// Wire bytes for `elems` elements, rounded up to a whole block. For SIZING
// only (arena provisioning, capacity checks) -- never for an offset, because
// rounding up is not additive and an offset computed this way would not agree
// with the region sizes around it.
inline constexpr size_t lpWireBytesRoundUp(size_t elems) {
  return ((elems + kLpBlockElems - 1) / kLpBlockElems) * kLpBlockBytes;
}

/**
 * The wire format of one transfer.
 *
 * Full precision is `{false, elementSize, datatype}` and reproduces today's
 * arithmetic exactly. Low precision is `{true, elementSize, ncclUint8}`, where
 * elementSize still describes the caller's FULL-PRECISION buffers -- the
 * quantize source and the dequantize destination, which exist on both paths --
 * and the wire itself is counted in bytes.
 */
struct RelayWire {
  bool lp{false};
  size_t elemSize{0};
  ncclDataType_t dtype{ncclFloat32};

  // Byte offset/length of `elems` elements on the wire. Additive in `elems` for
  // 128-multiples, which is what every offset expression in the four
  // collectives relies on.
  size_t bytes(size_t elems) const {
    if (!lp) {
      return elems * elemSize;
    }
    // Guaranteed by lpEligible()'s alignment gate. Asserted rather than
    // handled: there is no correct answer for an unaligned LP region, because
    // the peer computing the matching offset would have to make the same wrong
    // choice AND the surrounding regions would have to absorb the difference.
    assert(elems % kLpBlockElems == 0);
    return lpWireBytes(elems);
  }

  // What to pass as ncclSend/ncclRecv's `count`, given `elems` elements of the
  // caller's data.
  size_t count(size_t elems) const {
    return lp ? bytes(elems) : elems;
  }
};

// Full-precision wire for `datatype`. The identity case: this is what every
// site sees when low precision is off or declined.
inline RelayWire lpFullPrecisionWire(ncclDataType_t datatype, size_t elemSize) {
  return RelayWire{false, elemSize, datatype};
}

// Low-precision wire for `datatype`. `elemSize` is still the caller's element
// size, because the quantize source and dequantize destination are in the
// caller's dtype.
inline RelayWire lpWireFor(ncclDataType_t datatype, size_t elemSize, bool lp) {
  return lp ? RelayWire{true, elemSize, ncclUint8}
            : lpFullPrecisionWire(datatype, elemSize);
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

// Which relay collective is asking. Only used to pick a size threshold, so the
// four crossovers can be tuned independently from one sweep.
enum class LpCollective {
  AllReduce,
  ReduceScatter,
  AllGather,
  AllToAll,
};

/**
 * True for the dtypes the LP wire supports: bf16 and fp32.
 *
 * Everything else falls through to full precision untouched even with
 * `low_precision` set, which is why every existing ncclInt32 test keeps passing
 * verbatim with the flag on and doubles as the regression test for clean
 * fallback.
 */
bool lpDtypeSupported(ncclDataType_t datatype);

// Every per-group count is a whole number of `alignElems` elements. 128 (one
// wire block) is the floor, but a schedule whose geometry divides a region
// further needs more: the flat A>2 allreduce splits its direct region into A
// per-owner shards, and `pD / A` is only a whole number of blocks when the
// count is a multiple of A * 128. Passing that requirement in keeps the check
// size-only and therefore collective-consistent, which is what matters -- see
// the file comment.
bool lpCountsAligned(
    const size_t* counts,
    int nGroups,
    size_t alignElems = kLpBlockElems);

/**
 * Smallest message low precision is used for, in the same byte metric the
 * collective's route selector uses (its per-rank input label).
 *
 * PER DTYPE, because bf16 and fp32 are not one policy measured twice. fp32
 * quantized to e4m3 sends 4 bytes per element as 33/32, a 3.88x reduction;
 * bf16 sends 2 as 33/32, only 1.94x. Twice the saving buys a crossover roughly
 * a size band and a half earlier and a peak twice as high, so a single table
 * keyed only on (collective, width, grouping) has to pick one dtype to be
 * right about. It used to pick bf16, which left fp32 running full precision
 * across a range where it wins 1.1x-2.8x.
 *
 * MEASURED on MI350X, best-of-N over 10 reps x 20 iterations, low precision and
 * full precision timed back to back on the SAME communicator, 4 KB to 1 GB. The
 * ratio is LP-vs-FULL-PRECISION-RELAY: above 1.00x the wire format is faster
 * than the same relay carrying full-precision bytes. It is deliberately NOT a
 * ratio against NCCL, which would fold in the relay's own 2x-2.5x and make
 * every shape look like a win, including ones where enabling low precision
 * makes things slower.
 *
 * The bf16 history is worth keeping, because THREE times this table said "low
 * precision does not pay here" when it was measuring a stall: 2 of 16 shapes
 * enabled before the wavefront-absmax rewrite, 7 of 16 after it, 11 of 16 after
 * the chunk alignment was raised to 512, and 15 of 16 once allreduce actually
 * PICKED UP that alignment change -- it had its own hard-coded 128 and silently
 * did not get it. fp32 enables 15 of 16.
 *
 * The pattern is worth internalizing: every single time a shape looked like it
 * had a deep reason not to pay, it was an alignment or latency bug in this
 * code, not a property of the wire format. Treat a shape that does not improve
 * monotonically with size as a bug report against the schedule.
 *
 *                     bf16                      fp32
 *                     nGroups==1   fused        nGroups==1   fused
 *   allreduce      A=2    8 MiB     8 MiB         4.5 MiB     4.5 MiB
 *   allreduce      A=4   12 MiB    12 MiB         9 MiB       9 MiB
 *   reduce-scatter A=2   12 MiB    12 MiB         4.5 MiB     4.5 MiB
 *   reduce-scatter A=4   60 MiB    60 MiB         60 MiB      60 MiB
 *   all-gather     A=2    8 MiB     8 MiB         4.5 MiB     4.5 MiB
 *   all-gather     A=4    8 MiB    12 MiB         4.5 MiB    12 MiB
 *   all-to-all     A=2    --       12 MiB          --         4.5 MiB
 *   all-to-all     A=4   24 MiB    27 MiB        13.5 MiB    27 MiB
 *
 *   -- means off at every size.
 *
 * EVERY fp32 THRESHOLD IS EXACTLY THE SMALLEST SIZE MEASURED TO WIN, with two
 * stated exceptions. The sweep jumps 576 KB to 4.5 MB, so five of these shapes
 * win at the first size in the MB decade and their true crossovers are
 * somewhere in that gap. 4.5 MiB claims only what was measured; anyone wanting
 * it lower has to add sizes to the sweep, not round down.
 *
 * The exceptions are reduce-scatter A=4 (60 MiB, own first win 63 MB) and FUSED
 * all-gather A=4 (12 MiB, own first win 13.5 MB). Both take bf16's rounded-down
 * value, because in both cases the two dtypes first win at the SAME measured
 * point with nothing in the gap below it to separate them -- so fp32 sitting
 * ABOVE bf16 on identical evidence would break the ordering that makes the two
 * tables comparable, to move a threshold by a few percent.
 *
 * THE TABLES ARE ORDERED, and a test pins it: no fp32 threshold may be higher
 * than its bf16 counterpart. It follows from the
 * wire format rather than from any measurement -- the same shape has strictly
 * more to gain at 3.88x than at 1.94x -- so it holds across retunes and is what
 * catches a value copied into the wrong table. It has already earned that: the
 * fused all-gather exception above was a 13.5 MiB entry this test rejected.
 *
 * ONE MEASURED CAVEAT ON THE EXCLUSION. Single-group all-to-all A=2 is off in
 * both dtypes, and in fp32 that is a judgement call rather than a clear no: run
 * truly alone it peaks at 1.12x and fades to ~1.00x, but under FOUR CO-RESIDENT
 * jobs it holds 1.07x-1.20x across the whole range. Both cases are nGroups == 1
 * and the gate cannot tell them apart, so the conservative reading wins and
 * co-resident callers give up about 1.1x. Separating them needs a contention
 * signal the gate does not have.
 *
 * fp32 peaks, over each shape's enabled range: 2.04x-2.33x single-group
 * all-gather, 2.20x-2.26x single-group allreduce, up to 2.83x fused all-gather
 * A=4. The reductions are the ones that gain most from the dtype split, because
 * they were the ones bf16 gated latest.
 *
 * fp32 also settles two shapes bf16 could not:
 *   - reduce-scatter A=4 FUSED is ENABLED for fp32 and off for bf16. Both show
 *     the same flat ~1.00x through 40 MB and then a step, but fp32's step is to
 *     2.01x-2.63x across six consecutive sizes, which is a crossover rather
 *     than a plateau at the edge of the data.
 *   - single-group allreduce A=2 pays in BOTH dtypes now. It was fp32-only,
 * with bf16 off for an unexplained stall and fp32 showing that same stall as
 *     non-monotonic VARIANCE (1.19x to 1.83x) that the 3.88x saving stayed
 * ahead of. The allreduce chunk-alignment fix removed both: fp32 now rises
 * smoothly from 1.08x at 4.5 MB to 2.26x at 1 GB, and bf16 is enabled at 8 MiB.
 *
 * ONE shape stays off for fp32: single-group all-to-all A=2, which peaks at
 * 1.12x at 27 MB and falls back to 0.99x-1.02x from 63 MB up. Same shape as
 * bf16 (0.88x-0.97x), just shifted up by the extra saving -- still not a
 * threshold, because the trend runs the wrong way with size.
 *
 * CO-RESIDENT JOBS LAND IN THE nGroups == 1 COLUMN. A parallel relay job is a
 * single-group call, so that column is also what several independent jobs
 * sharing a node get, and it was measured with the node otherwise idle. The
 * parallel sweep says this is conservative rather than optimistic: every
 * enabled shape does slightly BETTER under four co-resident jobs.
 *
 * THE nGroups COLUMN DOES NOT TELL THE STORY. An earlier reading of it was that
 * contention is what low precision wins -- fused shapes paid, uncontended ones
 * did not -- which was a plausible bandwidth argument and wrong. Nearly every
 * single-group shape was sitting in an alignment stall; with that gone, six of
 * the eight pay in bf16 and seven in fp32. Contention still helps, since the
 * fused ratios are higher, but it is not the dividing line.
 *
 * BOTH DTYPES NOW ENABLE 15 OF 16 SHAPES, and the one exclusion is the same in
 * both. Two entries used to be listed here and neither survived contact with
 * more data: single-group allreduce A=2 was an unexplained "stall" that turned
 * out to be an alignment bug in this code, and fused reduce-scatter A=4 was a
 * plateau at the edge of a range that stopped at 72 MB.
 *
 * The one that stays off, in both dtypes:
 *
 *   - single-group all-to-all A=2: 0.95x-0.97x small and 0.88x at 135-144 MB in
 *     bf16; fp32 peaks at 1.12x and falls back to ~1.00x. STRUCTURAL rather
 * than a threshold or a bug. `foldDiagonalIntoGroup` is set exactly when
 *     nGroups == 1, and the folded self-peer diagonal stays FULL PRECISION
 *     because RCCL services it as a local copy that never crosses a boundary.
 * At A=2 that diagonal is one of two segments, so HALF the payload cannot
 *     shrink while the low-precision machinery is paid over all of it. At A=4
 *     the diagonal is a quarter and the shape pays 1.25x; fused does not fold
 * at all and pays 1.38x. Fixing it means not folding under low precision, or
 *     quantizing the diagonal -- a schedule change, not a threshold.
 *
 *     The co-resident measurement is the tell: the same shape reads 1.05x
 * (bf16) and 1.15x (fp32) under four parallel jobs. The gate declines a small
 * real win there because it cannot tell that case from the uncontended one --
 *     both are nGroups == 1.
 *
 * Fused all-to-all A=4 starts at 27 MiB in BOTH dtypes, which is also exactly
 * where its XOR-relay route starts when fused. That is not a coincidence to be
 * tidied away: below the route crossover the call is a direct exchange with the
 * helpers idle, so there is nothing staged for the wire format to shrink and
 * low precision would decline on route regardless. The two gates agree by
 * construction, and they agree for both dtypes because the route gate does not
 * depend on the wire format.
 *
 * NCCL_SHARDED_RELAY_LP_MIN_KB overrides all of it, which is what made the
 * table measurable: the built-in values decline before anything is timed. The
 * collective test suites set it too, so they cover the mechanism without being
 * coupled to this policy.
 */
size_t lpMinBytes(
    LpCollective coll,
    int nActiveRanksPerGroup,
    int nGroups,
    ncclDataType_t datatype);

/**
 * Everything the caller does not already know, in one place.
 *
 * `routeSizeBytes` is the metric the collective's own selector computed, passed
 * in rather than recomputed here so the LP threshold and the route threshold
 * are directly comparable and there is only one definition of each metric.
 *
 * `relayRouteSelected` is the caller's already-made decision: a relay route was
 * chosen and this is not the one-shot path. LP has nothing to offer a schedule
 * that does not cross a rank boundary through staging.
 */
struct LpGateInputs {
  LpCollective coll{LpCollective::AllReduce};
  ncclDataType_t datatype{ncclFloat32};
  const size_t* counts{nullptr};
  int nGroups{0};
  int nActiveRanksPerGroup{0};
  size_t routeSizeBytes{0};
  bool relayRouteSelected{false};
  // Elements the per-group counts must be a whole multiple of. Defaults to one
  // wire block; a schedule that subdivides a region further raises it. See
  // lpCountsAligned().
  size_t countAlignElems{kLpBlockElems};
};

// Why an LP request was declined. Recorded per reason because the gate declines
// SILENTLY on four independent grounds, so an "LP" run that quietly fell back
// looks exactly like a passing LP run -- every LP test, bench and demo asserts
// engagement through these counters rather than trusting the flag.
enum class LpDecline {
  Dtype,
  Alignment,
  Size,
  Route,
  Arena,
  GraphCapture,
  kNumReasons,
};

/**
 * The size-and-dtype half of the gate: true if low precision should be used for
 * this call.
 *
 * Derived only from values that are collective-consistent, so every rank of the
 * call reaches the same answer without communication. Callers AND this with the
 * arena's capacity answer, which is also size-only.
 *
 * Records a decline reason and logs at INFO on false.
 */
bool lpEligible(const LpGateInputs& in);

// Record a decline the caller detected itself -- arena capacity, or a first LP
// call inside a graph capture. Keeps the counters a single source of truth for
// "did LP engage".
void lpRecordDecline(LpDecline reason);

// Record that a call actually ran in low precision.
void lpRecordEngage();

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

// Calls that ran in low precision, process-wide.
uint64_t lpEngageCount();

// Calls that asked for low precision and did not get it, by reason.
uint64_t lpDeclineCount(LpDecline reason);

// All reasons summed.
uint64_t lpDeclineCount();

// For tests and benchmarks that measure engagement across a phase.
void lpResetCounters();

// ---------------------------------------------------------------------------
// Tuning and provisioning parameters
//
// None of these is a feature switch -- low precision is requested per call. A
// job that sets none of them gets the feature, lazily provisioned.
// ---------------------------------------------------------------------------

/**
 * NCCL_SHARDED_RELAY_LP_HOPS, default 2. How many hops of a two-hop relay are
 * quantized.
 *
 * 2 is what delivers the feature: in the A=2 relay the two hops are serialized
 * groups, so quantizing only the up-hop yields about 1.3x instead of 1.94x. But
 * it means the allreduce and reduce-scatter reduce path rounds to e4m3 twice,
 * the second time on an already-summed value, and e4m3's half-ULP is 6.25%
 * relative. 1 keeps the return hop in the caller's dtype, so an accuracy
 * complaint can be bisected without a rebuild, at roughly a third of the gain.
 *
 * An env var rather than a second call argument because it is a debugging knob,
 * not a per-call product decision. If a third knob ever has to be per-call,
 * replace the trailing flag with an options struct (the ncclConfig_t pattern)
 * rather than growing the parameter list again.
 */
int lpHops();

// NCCL_SHARDED_RELAY_LP_PREALLOC, default 0. Build the arena during
// ncclCommInitRank instead of on the first call that asks for low precision.
// Off by default because a per-call flag means init cannot know whether LP will
// ever be used, and provisioning the arena on every communicator whose caller
// never asks is pure waste. Jobs that will use LP set it, and get zero
// call-path allocation and graph-capture safety from their very first call.
bool lpPrealloc();

// NCCL_SHARDED_RELAY_LP_MAX_MSG_MB, default 1024. Largest full-precision
// per-rank message the arena is provisioned for. See sharded_relay_lp_arena.h
// for what that costs.
size_t lpMaxMsgBytes();

} // namespace rcclx::relay
