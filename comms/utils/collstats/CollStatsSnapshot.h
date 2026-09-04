// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <vector>

#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// A readout window after it has left the device. Deliberately its own header,
// and free of the CUDA runtime: the reader produces one of these, but the
// consumers -- the exporter's queue, the serializers -- are plain host code,
// and pulling them through CollStatsReader.h would make every one of them a
// GPU target for a struct of integers and vectors.

namespace meta::comms::collstats {

// One window's worth of per-key aggregates, copied to the host.
//
// `values` and `keys` are both indexed by the dense id the host key registry
// assigned, so values[i] belongs to keys[i]. `values` carries one extra
// trailing entry, values[numKeys], holding everything that resolved to the
// catch-all; it has no entry in `keys` because it is not one key.
//
// Only the occupied prefix is transferred. Ids are handed out densely and never
// recycled, so numKeys is the registry's size at the moment the window was
// issued, not the bank's capacity.
struct CollStatSnapshot {
  uint32_t numKeys{0};
  uint64_t windowEpoch{0}; // pre-flip epoch value; a monotonic window sequence
  uint64_t catchAllCount{0}; // cumulative, from the host registry
  // Wall-clock bounds of the window, unix nanoseconds, or 0 when the producer
  // did not stamp them. Wall rather than monotonic because their purpose is
  // lining a window up against events outside this process; durations come
  // from the device clock and never from here, so a clock step can misplace a
  // boundary but cannot corrupt a measurement.
  //
  // The span also makes the window's duty cycle computable: the per-key
  // duration sums over this wall interval is the fraction of the period the
  // rank spent inside collectives.
  uint64_t windowStartUnixNs{0};
  uint64_t windowEndUnixNs{0};
  /* The bucketing this window was produced under. Exported alongside the
   * buckets so a consumer never has to assume the defaults, and so a window
   * recorded before a retune stays interpretable. Defaulted rather than
   * zero-initialized: a zero geometry has numBuckets 0, and the readout's
   * `numBuckets - 1` overflow index would wrap. */
  CollStatHistGeometry hist{collStatDefaultHistGeometry()};
  uint64_t thresholdsNs[kMaxThresholds]{
      kDefaultThresholdsNs[0],
      kDefaultThresholdsNs[1],
      kDefaultThresholdsNs[2],
      kDefaultThresholdsNs[3]};
  uint32_t numThresholds{kMaxThresholds};
  /* The size-class edges this window was produced under, carried for the same
   * reason as `hist`: a key's sizeClass is an index into these, so without them
   * an exported row cannot be turned back into byte bounds off-box. */
  CollStatSizeClasses sizeClasses{collStatDefaultSizeClasses()};
  std::vector<CollStatKey> keys; // [numKeys]
  std::vector<CollStatValue> values; // [numKeys + 1]
};

} // namespace meta::comms::collstats
