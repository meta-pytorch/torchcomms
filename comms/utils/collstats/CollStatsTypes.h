// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

#include "comms/utils/collstats/CollStatsKeys.h"

// Collective-stats data structures: the per-communicator, double-buffered,
// device-resident aggregate and the per-stream span scratch. This header is the
// shared vocabulary between the device finalizer (which accumulates) and the
// host reader (which snapshots); it deliberately carries no atomics, no CUDA
// types, and no key-provisioning logic, so it compiles on host for unit tests.

namespace meta::comms::collstats {

// Portability: device qualifiers vanish under a host-only compiler so the pure
// math in CollStatsHistogram.h is exercised directly by host gtest.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLLSTATS_HD __host__ __device__
#else
#define COLLSTATS_HD
#endif

/* ---------------------------------------------------------------------------
 * Histogram geometry.
 *
 * Durations are bucketed log-spaced: `subBucketsPerOctave` sub-buckets per
 * octave starting at `tMinNs`, plus one underflow bucket at index 0 and one
 * overflow bucket at the last index. The geometry is per communicator, built
 * at init from MCCL_GPU_COLLSTATS_HIST_TMIN_NS, _HIST_TMAX_NS and
 * _HIST_SUBBUCKETS and carried in the device block, so a job retunes it
 * without a rebuild. This header only defines the shape; the owner reads the
 * cvars and hands the result down.
 *
 * The array in CollStatValue is sized to kHistMaxBuckets at compile time. The
 * value struct is device-resident and copied out whole every window, so a
 * runtime-length array would make the bank stride and the reader's prefix copy
 * geometry-dependent. Init rejects a geometry needing more than the capacity.
 *
 * At the cvar defaults -- 8 sub-buckets per octave over [1us, 1024s], so 30
 * octaves and 242 buckets -- every interior bucket is 2^(1/8) wide, a constant
 * relative error of 9.05%:
 *
 *   bucket   0   underflow            dur <  1us
 *   bucket   1   [1000ns, 1091ns)     1us, the first interior bucket
 *   bucket   9   [2000ns, 2181ns)     2us, one octave in
 *   bucket  27                        10us
 *   bucket  80   [939us, 1024us)      1ms
 *   bucket 160   [962ms, 1049ms)      1s
 *   bucket 234                        600s, the watchdog backstop
 *   bucket 241   overflow             dur >= 1024s
 *
 * Absolute width scales with magnitude, which is the point: ~90ns at 1us,
 * ~85ms at 1s. Good enough to rank outliers, not a paging signal -- that is
 * what the exact threshold counters below are for.
 * ---------------------------------------------------------------------------
 */
inline constexpr uint32_t kHistMaxBuckets = 242;
inline constexpr uint32_t kHistUnderflowBucket = 0;

/* Per-communicator histogram geometry. `numBuckets` includes the underflow and
 * overflow buckets, so the overflow index is numBuckets - 1. */
struct CollStatHistGeometry {
  uint64_t tMinNs;
  uint64_t tMaxNs;
  uint32_t subBucketsPerOctave;
  uint32_t numBuckets;
};

/* Cvar defaults, and what kHistMaxBuckets was sized from. */
inline constexpr uint32_t kDefaultHistSubBucketsPerOctave = 8;
inline constexpr uint64_t kDefaultHistTMinNs = 1'000ull; // 1 us
inline constexpr uint64_t kDefaultHistTMaxNs =
    1'024ull * 1'000'000'000ull; // 1024 s
inline constexpr uint32_t kDefaultHistOctaves = 30; // [1us, 1024s]

/* Exact past-threshold counters. Durations >= each cut-point are counted with
 * an exact atomic add, so a threshold-based page carries no bucket-boundary
 * error. Cut-points are per communicator, read from
 * MCCL_GPU_COLLSTATS_THRESHOLDS_NS at init; the array is capacity-bounded for
 * the same reason the histogram is. */
inline constexpr uint32_t kMaxThresholds = 4;
inline constexpr uint64_t kDefaultThresholdsNs[kMaxThresholds] = {
    1ull * 1'000'000'000ull, // 1 s
    10ull * 1'000'000'000ull, // 10 s
    60ull * 1'000'000'000ull, // 60 s
    600ull * 1'000'000'000ull, // watchdog (CollTrace 10-min backstop)
};

/* Size-class capacity. sizeClass is a u8 key field, and a configured edge list
 * longer than this would produce classes it cannot represent; init rejects
 * one. 64 leaves room well past the 34 default power-of-two classes. */
inline constexpr uint32_t kMaxSizeClasses = 64;

/* ---------------------------------------------------------------------------
 * Aggregate key: (op, algorithm, protocol, dtype, sizeClass).
 *
 * Every field is a small code, so all five are u8 and the key is 5 bytes. op,
 * algorithm and protocol are the collstats vocabulary from CollStatsKeys.h,
 * which every producer translates into, so this header pulls in no NCCL or
 * MCCL headers and a zero-filled bank slot reads as Unknown rather than as a
 * real collective. dtype is the raw datatype value, which is small. sizeClass
 * is an index into the configured size-class edges, not a byte count -- see
 * CollStatsHistogram.h.
 *
 * Bucketing the size is what keeps key cardinality bounded. An exact size in
 * the key would make every distinct message size its own key, and a job with a
 * few hundred tensor shapes would exhaust the registry and spill the rest into
 * the catch-all.
 * ---------------------------------------------------------------------------
 */
struct CollStatKey {
  CollStatOp op;
  CollStatAlgo algorithm;
  CollStatProto protocol;
  uint8_t dtype;
  uint8_t sizeClass;
};

/* Per-key accumulated value. Per-window banking bounds every field far below
 * wrap within a window, so no saturating math is needed. count, logicalBytes,
 * durationSumNs, thresholdCounts and the histogram are add-only; durMaxNs and
 * durMinNsComplement are atomicMax. The device finalizer applies the atomics;
 * this struct is just the layout.
 *
 * The bucket and threshold counters are u32 for readout cost, not for range.
 * Each counts finalized collectives inside one window, so together they sum to
 * at most the collectives in that window -- far under u32. The width matters
 * because the histogram is 242 of the struct's 246 counters, and the reader
 * copies the full key capacity out of the retired bank every window whether or
 * not a slot is occupied: at u32 that is 1024 bytes per key and 0.5 MiB per
 * window at the default 512-key capacity, nearly twice that at u64. */
struct CollStatValue {
  uint64_t count;
  uint64_t logicalBytes; // logical message size (count x dtype), written once
  uint64_t durationSumNs;
  uint64_t durMaxNs;
  /* Minimum duration, stored bitwise-complemented so a zero-filled bank reads
   * as "no observation": ~0 is UINT64_MAX, the identity for a minimum, and
   * max(~a, ~b) == ~min(a, b), so one atomicMax maintains it. Storing the
   * minimum directly would force the between-window bank reset to write
   * UINT64_MAX into one field of every key instead of a single zero fill.
   * Readers should un-complement through a named accessor rather than by
   * hand, so the "count == 0 means unset" rule lives in one place. */
  uint64_t durMinNsComplement;
  uint32_t histogram[kHistMaxBuckets];
  uint32_t thresholdCounts[kMaxThresholds];
};

/* Exact per-window minimum duration (ns); 0 when there were no observations.
 * Lives here, beside the field and the rule above, rather than with the other
 * derived metrics: it is the only reader of the complemented encoding, and
 * keeping it here is what makes "never un-complement by hand" enforceable. */
inline uint64_t collStatDurMinNs(const CollStatValue& v) {
  return v.count == 0 ? 0ull : ~v.durMinNsComplement;
}

struct CollStatSpanScratch {
  uint64_t start; // min entry %globaltimer over blocks; init UINT64_MAX
  // Exit-barrier arrival count. The finalizer is the block that drives it to
  // collStatsGridBlocks(), read from gridDim at exit -- not expectedBlocks.
  uint32_t arrived;
  /* Reserved, and none of the three is read today. A multi-kernel collective
   * sharing one slot therefore records one observation per kernel rather than
   * one per collective; sequencing on kernelIndex/kernelCount, and taking the
   * block count from expectedBlocks instead of gridDim, are both future work.
   * Stated here because the barrier condition a reader assumes is authoritative
   * decides whether they trust a multi-kernel duration. */
  uint32_t expectedBlocks;
  uint32_t kernelIndex;
  uint32_t kernelCount;
};

inline constexpr uint64_t kSpanStartInit = ~0ull; // UINT64_MAX

} // namespace meta::comms::collstats
