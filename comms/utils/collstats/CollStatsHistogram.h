// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cmath>
#include <cstdint>

#include "comms/utils/collstats/CollStatsTypes.h"

/* Pure, deterministic bucketing math shared by the device finalizer and any
 * host-side reconstruction. Duration bucketing happens on the device at
 * finalize, so the duration must already be in nanoseconds: %globaltimer reads
 * in ns, so a single-GPU duration (end - start) needs no calibration constant
 * before bucketing. Size-class bucketing happens on the host at enqueue, where
 * the message size is already known. Kept header-only and __host__ __device__
 * so host gtest validates exactly what the kernel computes.
 *
 * Neither bucket set is compiled in. The duration geometry comes from
 * MCCL_GPU_COLLSTATS_HIST_TMIN_NS / _HIST_TMAX_NS / _HIST_SUBBUCKETS and the
 * size classes from MCCL_GPU_COLLSTATS_SIZE_CLASS_EDGES, both read by the
 * owner at init and passed in here. See CollStatHistGeometry and
 * CollStatSizeClasses. */

namespace meta::comms::collstats {

/* Log-bucket a duration (ns) into [0, geom.numBuckets). Index 0 is underflow
 * (dur < tMinNs), the last index is the overflow/tail bucket (dur >= tMaxNs),
 * and the interior is geom.subBucketsPerOctave sub-buckets per octave, giving a
 * constant relative error of 2^(1/S)-1 -- 9.05% at the default S of 8.
 *
 * Note: the interior split uses double log2. On one GPU this is one op per
 * finalize, not per element. Host and device IEEE-754 log2 can differ by at
 * most one bucket at an exact octave edge, which is smaller than the
 * histogram's own relative error and is therefore tolerated. */
COLLSTATS_HD inline uint32_t logBucketNs(
    const CollStatHistGeometry& geom,
    uint64_t durNs) {
  if (durNs < geom.tMinNs) {
    return kHistUnderflowBucket;
  }
  if (durNs >= geom.tMaxNs) {
    return geom.numBuckets - 1;
  }
  const double octaves =
      log2(static_cast<double>(durNs) / static_cast<double>(geom.tMinNs));
  const uint32_t sub =
      static_cast<uint32_t>(octaves * geom.subBucketsPerOctave);
  // The tMaxNs guard above bounds sub below the overflow index; shift past the
  // underflow bucket at index 0.
  return 1u + sub;
}

/* Build a geometry from configured bounds, deriving the bucket count from the
 * octave span. Returns numBuckets 0 -- which collStatsAllocDeviceBlock rejects
 * -- for bounds that are inverted, degenerate, or need more buckets than the
 * compile-time capacity, so a bad cvar fails at init rather than silently
 * reshaping the histogram. */
inline CollStatHistGeometry collStatMakeHistGeometry(
    uint64_t tMinNs,
    uint64_t tMaxNs,
    uint32_t subBucketsPerOctave) {
  CollStatHistGeometry geom{tMinNs, tMaxNs, subBucketsPerOctave, 0};
  if (tMinNs == 0 || tMaxNs <= tMinNs || subBucketsPerOctave == 0) {
    return geom;
  }
  const double octaves =
      std::log2(static_cast<double>(tMaxNs) / static_cast<double>(tMinNs));
  const uint64_t interior =
      static_cast<uint64_t>(std::ceil(octaves * subBucketsPerOctave));
  const uint64_t total = interior + 2;
  if (total > kHistMaxBuckets) {
    return geom;
  }
  geom.numBuckets = static_cast<uint32_t>(total);
  return geom;
}

/* The compiled-in defaults, derived through the same function rather than
 * asserted, so the default geometry cannot drift from the capacity it is
 * sized against. Host-only, like the derivation it calls: geometry is assembled
 * on the host and handed to the device, never computed there. */
inline CollStatHistGeometry collStatDefaultHistGeometry() {
  return collStatMakeHistGeometry(
      kDefaultHistTMinNs, kDefaultHistTMaxNs, kDefaultHistSubBucketsPerOctave);
}

/* Configured size-class edges, ascending. `edges[i]` is the inclusive lower
 * bound of class i + 1, so anything below edges[0] is class 0. Held by value
 * and capacity-bounded so it copies without allocating.
 *
 * Bounded by kMaxSizeClasses because sizeClass is a u8 field of CollStatKey,
 * and because bounded key cardinality is the whole reason the size is bucketed
 * rather than stored exactly. */
struct CollStatSizeClasses {
  uint64_t edges[kMaxSizeClasses];
  uint32_t n;
};

/* The default edge set: powers of two from 2 bytes to 8 GiB, which reproduces
 * exactly the floor(log2(bytes)) classes this started with, so data recorded
 * before the edges became configurable stays comparable.
 *
 *   class  0   [0 B,    2 B)     empty, or a single byte
 *   class  1   [2 B,    4 B)
 *   class 10   [1 KiB,  2 KiB)
 *   class 20   [1 MiB,  2 MiB)
 *   class 30   [1 GiB,  2 GiB)
 *   class 33   [8 GiB,  inf)     the tail bucket
 *
 * Powers of two spend most of their classes on sizes a training job never
 * sends, and two shapes 40% apart share a class. A job that cares about a
 * narrower range should configure edges around it. */
COLLSTATS_HD inline CollStatSizeClasses collStatDefaultSizeClasses() {
  CollStatSizeClasses sc{};
  sc.n = 0;
  for (uint32_t i = 1; i <= 33u && sc.n < kMaxSizeClasses; ++i) {
    sc.edges[sc.n++] = 1ull << i;
  }
  return sc;
}

/* Map a logical message size (bytes) to its class: the index of the last edge
 * at or below `msgBytes`, or 0 when it is below the first edge. Integer-only,
 * so host and device agree exactly. A linear scan because the edge list is
 * tens of entries and this runs once per launch on the enqueue thread. */
COLLSTATS_HD inline uint8_t sizeClassOf(
    const CollStatSizeClasses& sc,
    uint64_t msgBytes) {
  uint32_t cls = 0;
  for (uint32_t i = 0; i < sc.n; ++i) {
    if (msgBytes < sc.edges[i]) {
      break;
    }
    cls = i + 1;
  }
  return static_cast<uint8_t>(cls);
}

} // namespace meta::comms::collstats
