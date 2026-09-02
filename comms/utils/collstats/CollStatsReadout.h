// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cmath>
#include <cstdint>

#include "comms/utils/collstats/CollStatsTypes.h"

// Host-side metric derivation from a copied-out bank value.
// These run on the reader after the D2H copy, never on the device, so they are
// plain host functions. They turn the raw per-window counters into the reported
// numbers: the per-window average, the lower edge of a histogram bucket, and
// tail percentiles from the histogram (at its ~9% relative error). The exact
// minimum lives beside its encoding in CollStatsTypes.h (collStatDurMinNs) and
// the maximum is a raw field, so neither is derived here.
//
// Bus bandwidth is deliberately absent: it needs the per-op traffic factor, and
// the exported window carries the raw aggregates so the off-box consumer can
// apply it -- see CollStatsJson.h. Keeping the latency and throughput numbers
// distinct matters, because reporting one as the other is the main way this
// class of tool misroutes an incident.

namespace meta::comms::collstats {

/* Lower edge (ns) of a histogram bucket, i.e. the inverse of logBucketNs under
 * the same geometry. The underflow bucket starts at 0; the overflow bucket
 * starts at tMaxNs. */
inline double collStatBucketLowerNs(
    const CollStatHistGeometry& geom,
    uint32_t bucket) {
  if (bucket == kHistUnderflowBucket) {
    return 0.0;
  }
  if (bucket >= geom.numBuckets - 1) {
    return static_cast<double>(geom.tMaxNs);
  }
  const double octavesAbove =
      static_cast<double>(bucket - 1) / geom.subBucketsPerOctave;
  return static_cast<double>(geom.tMinNs) * std::exp2(octavesAbove);
}

// Exact per-window average duration (ns); 0 when no observations.
inline double collStatAvgDurationNs(const CollStatValue& v) {
  return v.count == 0
      ? 0.0
      : static_cast<double>(v.durationSumNs) / static_cast<double>(v.count);
}

// Nearest-rank quantile read from the histogram, returned as the lower edge of
// the containing bucket (ns). `p` is in (0, 1]. Adequate for ranking outliers;
// paging keys off the exact threshold counters, not this.
inline double collStatPercentileNs(
    const CollStatHistGeometry& geom,
    const CollStatValue& v,
    double p) {
  if (v.count == 0) {
    return 0.0;
  }
  const double target = p * static_cast<double>(v.count);
  uint64_t cum = 0;
  for (uint32_t b = 0; b < geom.numBuckets; ++b) {
    cum += v.histogram[b];
    if (static_cast<double>(cum) >= target) {
      return collStatBucketLowerNs(geom, b);
    }
  }
  return collStatBucketLowerNs(geom, geom.numBuckets - 1);
}

} // namespace meta::comms::collstats
