/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

// Maximum number of groups supported by the multi-group relay collectives.
// Mirrors SHARDED_RELAY_MAX_GROUPS in sharded_relay_allreduce.h; redefined here
// so this header is self-contained (the two values must stay in sync).
#ifndef SHARDED_RELAY_MAX_GROUPS
#define SHARDED_RELAY_MAX_GROUPS 8
#endif

/**
 * Internal route selection for the sharded-relay collectives.
 *
 * NOT part of the public API. Included only by the relay implementations and
 * their tests -- it is deliberately absent from the four public relay headers
 * so that no caller can see or influence the route.
 *
 * Whether a relay collective goes through the passthrough helpers or straight
 * across the intra links is owned entirely by the collective and derived only
 * from the message size. These functions ARE that decision: each
 * implementation calls its selector at the point it dispatches, so there is a
 * single definition of the size -> route mapping rather than one copy in the
 * implementation and another hard-coded in the tests. They take no communicator
 * and no stream, touch no global state, and cannot be used to override the
 * route -- they only answer "at this size, which route does the design
 * specify?".
 *
 * Tests assert against them so that a regression which collapses the routing
 * (for example always taking the direct path) fails loudly. That check used to
 * ride on a side effect: helpers staged relay traffic in the caller-supplied
 * buffer, so a test could infer the route from whether that buffer had been
 * written. Once helper staging moved into kernel-owned internal scratch the
 * caller's buffer is untouched on every route, and the route stopped being
 * observable from outside.
 *
 * The thresholds are measured crossovers (MI350X, bf16, 8 GPUs); see the
 * comments on each selector for the provenance of the numbers.
 */
namespace rcclx::relay {

// Route taken by a multi-group sharded-relay all-to-all call.
enum class AllToAllRoute {
  // Exact direct all-to-all across the intra links; helpers idle.
  PureDirect,
  // Original 2-active passthrough relay through the 6 dedicated helpers.
  A2Relay,
  // No-pack XOR/Latin relay for A==4 with 4 helpers.
  A4XorRelay,
};

// Route taken by a multi-group sharded-relay all-gather call.
enum class AllGatherRoute {
  // Direct shard exchange; helpers idle.
  PureDirect,
  // Original 2-active passthrough relay.
  A2Relay,
  // Flat scatter->forward relay with the 2-hop helper offload enabled.
  FlatOffload,
};

// Route taken by a multi-group sharded-relay reduce-scatter call.
enum class ReduceScatterRoute {
  // Single-group direct exchange plus a local reduce; helpers idle.
  PureDirect,
  // 2-active 2-hop helper relay.
  A2Relay,
  // Flat relay with reduce-at-helper offload enabled.
  FlatOffload,
};

// Route taken by a multi-group sharded-relay allreduce call.
enum class AllReduceRoute {
  // Full exchange between the active ranks plus a local reduce; helpers idle.
  // For A > 2 this is the direct reduce-scatter + all-gather with the offload
  // disabled.
  PureDirect,
  // 2-active helper relay.
  A2Relay,
  // Flat helper-reduce-and-broadcast with the 2-hop offload enabled.
  FlatOffload,
};

// GPU memory access alignment in elements for relay chunk sizing. The relay
// implementations define their CHUNK_ALIGN_ELEMENTS from this.
inline constexpr size_t kRelayChunkAlignElements = 128;

// Largest per-group count, the value every size threshold is measured against.
// All groups march through the same schedule to keep XGMI traffic phase-synced,
// so the route is tuned off the largest per-group message.
inline size_t relayMaxCount(const size_t* counts, int nGroups) {
  size_t maxCount = 0;
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] > maxCount) {
      maxCount = counts[g];
    }
  }
  return maxCount;
}

/**
 * All-to-all route for the given geometry.
 *
 * Size metric is the per-active-rank input size,
 * A * max(segmentCounts) * elementSize, which equals the bench's per-rank input
 * label.
 *
 * A==2 crossover: the fused relay overtakes the direct exchange at ~9 MB and an
 * independent call at ~27 MB (independent has no cross-group contention, so
 * direct holds on longer); cross over below each. A==4 uses the XOR relay only
 * inside [63 MiB, 256 MiB).
 */
inline AllToAllRoute selectAllToAllRoute(
    int nActiveRanksPerGroup,
    int numHelpers,
    int nGroups,
    const size_t* segmentCounts,
    size_t elementSize) {
  const size_t maxBytes = static_cast<size_t>(nActiveRanksPerGroup) *
      relayMaxCount(segmentCounts, nGroups) * elementSize;

  if (nActiveRanksPerGroup == 2) {
    const size_t pureDirectMaxBytes = (nGroups > 1)
        ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
        : (static_cast<size_t>(27) << 20); // independent: < 27 MB
    return (maxBytes < pureDirectMaxBytes) ? AllToAllRoute::PureDirect
                                           : AllToAllRoute::A2Relay;
  }

  bool allSegmentCountsPositive = true;
  for (int g = 0; g < nGroups; g++) {
    allSegmentCountsPositive &= segmentCounts[g] > 0;
  }
  constexpr size_t kXorRelayMinBytes = static_cast<size_t>(63) << 20;
  constexpr size_t kXorRelayMaxBytes = static_cast<size_t>(256) << 20;
  const bool useXorRelay = nActiveRanksPerGroup == 4 && numHelpers == 4 &&
      allSegmentCountsPositive && maxBytes >= kXorRelayMinBytes &&
      maxBytes < kXorRelayMaxBytes;
  return useXorRelay ? AllToAllRoute::A4XorRelay : AllToAllRoute::PureDirect;
}

// Per-source relay chunk of the A==4 XOR all-to-all. The direct-B region
// absorbs both the /3 remainder and the alignment loss, so
// 3 * relayCount <= segmentCount always holds.
inline size_t allToAllA4RelayCount(size_t segmentCount) {
  return (segmentCount / 3 / kRelayChunkAlignElements) *
      kRelayChunkAlignElements;
}

/**
 * All-gather route for the given geometry.
 *
 * Size metric is max(sendCounts) * elementSize -- the per-rank input shard
 * label, with no active-rank factor.
 *
 * A==2 crossover: the fused 2-active relay overtakes the direct exchange at
 * ~4.5 MB and an independent call at ~13.5 MB; cross over below each. For A>2
 * the flat path turns a profit on the 2-hop offload from ~12 MB when fused and
 * ~8 MB when independent (an independent call has the cross links to itself).
 * A==2 never takes the offload: it only reaches the flat path in the
 * small-message regime where the relay was already ruled out, so offloading
 * would re-add the hop it was routed there to avoid.
 */
inline AllGatherRoute selectAllGatherRoute(
    int nActiveRanksPerGroup,
    int numHelpers,
    int nGroups,
    const size_t* sendCounts,
    size_t elementSize) {
  const size_t maxBytes = relayMaxCount(sendCounts, nGroups) * elementSize;

  if (nActiveRanksPerGroup == 2) {
    const size_t pureDirectMaxBytes = (nGroups > 1)
        ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
        : (static_cast<size_t>(9) << 20); // independent: < 9 MB
    return (maxBytes < pureDirectMaxBytes) ? AllGatherRoute::PureDirect
                                           : AllGatherRoute::A2Relay;
  }

  const size_t offloadMinBytes = (nGroups > 1)
      ? (static_cast<size_t>(12) << 20) // fused: >= 12 MB
      : (static_cast<size_t>(8) << 20); // independent: >= 8 MB
  const bool useOffload = (numHelpers > 0) && (nActiveRanksPerGroup > 2) &&
      (maxBytes >= offloadMinBytes);
  return useOffload ? AllGatherRoute::FlatOffload : AllGatherRoute::PureDirect;
}

/**
 * Reduce-scatter route for the given geometry.
 *
 * Size metric is A * max(recvCounts) * elementSize, the bench per-rank input
 * label. The A==2 fast path measures 2 * recvCount * elementSize, which is the
 * same value because it is only reachable when A==2.
 *
 * A==2 crossover: the fused relay overtakes the direct exchange at ~9 MB and an
 * independent call at ~27 MB; cross over just below each. For A>2 the offload's
 * extra hop and second group boundary only pay for themselves past ~48 MB;
 * below that the single-group pure-direct reduce-scatter wins outright.
 */
inline ReduceScatterRoute selectReduceScatterRoute(
    int nActiveRanksPerGroup,
    int numHelpers,
    int nGroups,
    const size_t* recvCounts,
    size_t elementSize) {
  const size_t maxBytes = static_cast<size_t>(nActiveRanksPerGroup) *
      relayMaxCount(recvCounts, nGroups) * elementSize;

  if (nActiveRanksPerGroup == 2) {
    const size_t pureDirectMaxBytes = (nGroups > 1)
        ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
        : (static_cast<size_t>(27) << 20); // independent: < 27 MB
    return (maxBytes < pureDirectMaxBytes) ? ReduceScatterRoute::PureDirect
                                           : ReduceScatterRoute::A2Relay;
  }

  constexpr size_t kOffloadMinBytes = static_cast<size_t>(48) << 20;
  const bool useOffload = (numHelpers > 0) && (maxBytes >= kOffloadMinBytes);
  return useOffload ? ReduceScatterRoute::FlatOffload
                    : ReduceScatterRoute::PureDirect;
}

/**
 * Allreduce route for the given geometry.
 *
 * Size metric is max(counts) * elementSize -- the bench per-rank input label,
 * with no active-rank factor.
 *
 * A==2 crossover: the relay wins big at large sizes (one group spread across
 * all helpers) so the crossover is low, and an independent call has no
 * cross-group contention on the direct link so pure-direct holds on longer
 * (2 MB fused, 6 MB independent). For A>2 the fused sweep phase-syncs every
 * group so the offload cross links stay clean and it pays off almost
 * immediately, while an independent call contends with whatever else is in
 * flight (2 MB fused, 9 MB independent); below the crossover the offload only
 * adds helper-hop latency, so the flat path runs a pure-direct
 * reduce-scatter + all-gather among the active ranks with helpers idle.
 *
 * Unlike the other three selectors this one does not consult numHelpers: the
 * allreduce predicates it replaces never did.
 */
inline AllReduceRoute selectAllReduceRoute(
    int nActiveRanksPerGroup,
    int nGroups,
    const size_t* counts,
    size_t elementSize) {
  const size_t maxBytes = relayMaxCount(counts, nGroups) * elementSize;

  if (nActiveRanksPerGroup == 2) {
    const size_t pureDirectMaxBytes = (nGroups > 1)
        ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
        : (static_cast<size_t>(6) << 20); // independent: < 6 MB
    return (maxBytes < pureDirectMaxBytes) ? AllReduceRoute::PureDirect
                                           : AllReduceRoute::A2Relay;
  }

  const size_t pureDirectMaxBytes = (nGroups > 1)
      ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
      : (static_cast<size_t>(9) << 20); // independent: < 9 MB
  return (maxBytes < pureDirectMaxBytes) ? AllReduceRoute::PureDirect
                                         : AllReduceRoute::FlatOffload;
}

// Permille of the count that the flat A>2 allreduce sends over the 2-hop helper
// offload. Equal direct and offload regions minimize the two-group critical
// path; below the crossover the offload is disabled entirely.
inline size_t allReduceOffloadPermille(AllReduceRoute route) {
  return route == AllReduceRoute::FlatOffload ? 500 : 0;
}

} // namespace rcclx::relay
