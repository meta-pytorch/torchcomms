// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

#include "comms/utils/collstats/CollStatsAtomics.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// The finalize update: one completed observation folded into a per-key value.
// The device finalizer (the collective's last block) calls this once per launch
// after computing dur = end - start in nanoseconds; many finalizers update one
// bank concurrently, so every field is touched with an atomic. Written
// host/device-shared so a host test drives the exact accumulation the kernel
// runs.

namespace meta::comms::collstats {

/* Fold one observation into `v`. `logicalBytes` is the collective's logical
 * message size (count x dtype), added once per finalize -- not accumulated at
 * send boundaries -- so Sum(logicalBytes) over the window is the bus-bandwidth
 * numerator. durMaxNs is an atomicMax and the minimum rides an atomicMax on the
 * complement, so both are exact per-window extremes. */
COLLSTATS_HD inline void collStatAccumulate(
    CollStatValue* v,
    uint64_t durNs,
    uint64_t logicalBytes,
    const CollStatHistGeometry& geom,
    const uint64_t* thresholdsNs,
    uint32_t numThresholds) {
  collStatAtomicAdd(&v->logicalBytes, logicalBytes);
  collStatAtomicAdd(&v->durationSumNs, durNs);
  collStatAtomicMax(&v->durMaxNs, durNs);
  // max of complements is the complement of the min; see CollStatValue.
  collStatAtomicMax(&v->durMinNsComplement, ~durNs);
  collStatAtomicInc(&v->histogram[logBucketNs(geom, durNs)]);
  for (uint32_t i = 0; i < numThresholds; ++i) {
    if (durNs >= thresholdsNs[i]) {
      collStatAtomicInc(&v->thresholdCounts[i]);
    }
  }
  // count last, because it is the published-ness flag: collStatDurMinNs gates
  // on it, so anyone who observes a non-zero count has already observed the
  // minimum it un-complements. The reader only touches a retired bank, so this
  // costs nothing today and keeps a live-bank peek (a debugger, a future
  // in-flight dump) from reading UINT64_MAX as the minimum.
  collStatAtomicInc(&v->count);
}

/* Host-only convenience wrapper using the default geometry and cut-points.
 * Device code passes the comm's device-resident configuration explicitly, since
 * a host constexpr array is not addressable from the device. */
inline void
collStatAccumulate(CollStatValue* v, uint64_t durNs, uint64_t logicalBytes) {
  collStatAccumulate(
      v,
      durNs,
      logicalBytes,
      collStatDefaultHistGeometry(),
      kDefaultThresholdsNs,
      kMaxThresholds);
}

} // namespace meta::comms::collstats
