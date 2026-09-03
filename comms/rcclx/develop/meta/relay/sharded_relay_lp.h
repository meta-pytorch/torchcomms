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
 */
namespace rcclx::relay {

// ---------------------------------------------------------------------------
// Wire layout
// ---------------------------------------------------------------------------

// Payload elements per wire block. Tied to the relay's chunk alignment, not
// chosen independently: the whole format rests on every LP region boundary
// being a whole number of blocks, and kRelayChunkAlignElements is what the
// collectives already align every chunk, offset and tile to.
inline constexpr size_t kLpBlockElems = kRelayChunkAlignElements;

// One fp32 scale per block, trailing its payload.
inline constexpr size_t kLpScaleBytes = sizeof(float);

// 128 payload bytes + 4 scale bytes.
inline constexpr size_t kLpBlockBytes = kLpBlockElems + kLpScaleBytes;

static_assert(kLpBlockElems == 128, "the 33/32 wire ratio assumes 128");
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

// Every per-group count is a whole number of wire blocks. See the file comment
// for why this is the gate rather than a tail-padding scheme.
bool lpCountsAligned(const size_t* counts, int nGroups);

/**
 * Smallest message low precision is used for, in the same byte metric the
 * collective's route selector uses (its per-rank input label).
 *
 * PROVISIONAL. Low precision starts where the relay routes start, which is the
 * conservative choice: below ~576 KB the measured relay time is flat -- that
 * band is pure launch cost with no bandwidth term -- and LP adds launches, so a
 * gain there would mean the measurement is wrong. These get their measured
 * values from the LP sweep, with the same "measured crossover, here is the
 * provenance" convention sharded_relay_route.h uses for the route thresholds.
 */
size_t lpMinBytes(LpCollective coll, int nActiveRanksPerGroup, int nGroups);

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
