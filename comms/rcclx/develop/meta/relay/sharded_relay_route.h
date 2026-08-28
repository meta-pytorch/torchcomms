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
 * direct holds on longer); cross over below each. A==4 relays from 27 MB fused
 * / 9 MB independent up, with no upper bound: the 256 MiB ceiling this used to
 * carry was covering an alignment bug in the relay geometry, not a real
 * crossover.
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
  constexpr size_t kXorRelayMinBytes = static_cast<size_t>(9) << 20;
  const size_t xorRelayMinBytes = (nGroups > 1)
      ? (static_cast<size_t>(27) << 20) // fused: >= 27 MB
      : kXorRelayMinBytes; // independent: >= 9 MB
  const bool useXorRelay = nActiveRanksPerGroup == 4 && numHelpers == 4 &&
      allSegmentCountsPositive && maxBytes >= xorRelayMinBytes;
  return useXorRelay ? AllToAllRoute::A4XorRelay : AllToAllRoute::PureDirect;
}

// Per-source relay chunk of the A==4 XOR all-to-all, and equally the size of
// its leading direct region: the schedule's two serialized phases each carry
// one such chunk per link, so the three regions are directA = relay = this
// count with directB absorbing the /3 remainder and the alignment loss. Every
// region boundary is therefore 128-element aligned; a plain segmentCount/3
// directA is misaligned for every power-of-two segment (2^n is never divisible
// by 3), which pushed the relay region onto an unaligned offset and cost more
// than the relay saved.
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

// Smallest message the single-group 2-active reduce-scatter overlaps its owner
// reduce for, measured against the two bounds that bracket the change (the
// unoverlapped schedule, and the same schedule with the reduce skipped
// entirely). Fraction of that interval closed, per size:
//
//    63-72 MB   the overlap LOSES 6-8%
//   135-144 MB  a wash (-0.5% to +1.1%)
//   256 MB      47%
//   512 MB      65%
//     1 GB      80%   (3.82x -> 4.68x vs NCCL; the ceiling is T/(T+1) = 89%)
//
// Two effects push the crossover up: the T+1 event record/wait pairs are a
// fixed cost, and the last region's reduce can never be hidden because no
// transfer follows it -- at small sizes the depth is lower, so that unhidden
// region is a larger share of the whole. 256 MiB is the smallest size measured
// to win clearly; there is no sample between 144 MiB and it.
inline constexpr size_t kRelayOverlapReduceMinBytes = static_cast<size_t>(256)
    << 20;

// Smallest message for which the pipelined A>2 all-gather issues its direct
// exchange as v-sized pieces instead of one (A-1)*v operation.
//
// Both directions of every link carry the same 3v per group, but the cross
// links carried it as three v-sized ops while the intra links carried it as a
// single 3v op, and RCCL budgets channels per operation -- so the single-op
// link got a third of the channels and gated the group. Making every op in a
// group the same size provisions all seven links alike.
//
// The mirrored reduce-scatter shape takes this unconditionally because it wins
// everywhere its offload route is active (1.30x -> 1.61x at 1 GB, 1.26x
// -> 1.46x at 135 MB). All-gather has far less to gain, being already the
// closest variant to the roofline, and the extra ops do not amortize at
// moderate sizes:
//
//   135 MB  1.77x -> 1.73x     256 MB  1.81x -> 1.83x
//   144 MB  1.76x -> 1.74x     512 MB  1.83x -> 1.88x
//                                1 GB  1.83x -> 1.91x
//
// so it is gated to where it measurably pays.
inline constexpr size_t kRelayUniformDirectOpMinBytes = static_cast<size_t>(256)
    << 20;

// Largest software-pipeline depth the single-group relays will use.
//
// 8 is where the measured optimum stops moving. Depth 16 is a wash at best
// (2-active allreduce at 1 GB: 4.252 ms at depth 8, 4.253 ms at 16) and a loss
// where the per-stage op count is already high (4-active all-to-all at 512 MB:
// 1.836 ms -> 1.993 ms), so there is nothing past 8 to take.
//
// This was briefly capped at 4, on the belief that depth 8 destabilized
// co-resident jobs. That was a measurement error worth recording, because the
// sweep it came from will mislead the same way again: the parallel sweep's
// relay times vary run to run by ~20-30%, in two loose clusters, and a single
// run against a single-run baseline reads that as a regression. It is not the
// pipeline: the spread is present at depth 1 with EQUAL OR LARGER magnitude
// (max/min over 4 runs at 144 MB: 1.78 at depth 1 versus 1.24 at depth 4), and
// it is identical across co-resident jobs to within 3%, so it is not cross-job
// skew either.
//
// Contributors identified, in descending order of how sure we are:
//  - Co-tenant GPU work on a shared node. Confirmed: while another workload had
//    the GPUs, a 4-active single-group all-gather at 256 MB read 8.3-10.6 ms
//    for the NCCL baseline against 5.08 ms on an idle node, and per-rep spreads
//    went from 1.00-1.12 to 1.25-1.45. Always check BENCH_SWEEP_PER_REP spreads
//    before trusting a number.
//  - p2p channel resourcing. Pinning NCCL_MIN/MAX_P2P_NCHANNELS makes the
//  timing
//    deterministic to +/-1% but costs 1.75x at 32 channels and 20x at 8, since
//    the default resolves to 64 channels with 8 per peer and forcing it lower
//    puts all 7 peers on shared channels. Suggestive, not conclusive.
// Ruled out: clocks (fclk stays pinned at 1500 MHz, sclk moves ~6%, and neither
// tracks the slow runs) and the host allocator (expandable_segments changes
// nothing).
//
// The single-group sweep does NOT show this: on an idle node it repeats to
// within 0.3-0.6%, which is why it is the sweep the size gating above is tuned
// on. Parallel comparisons need repeated invocations and must compare minima.
inline constexpr int kRelayMaxPipelineTiles = 8;

// What one extra ncclGroup boundary costs, expressed as the per-link bytes that
// take the same time: a grouped-P2P launch plus work-FIFO upload plus fence is
// ~25 us, which at the measured ~50 GB/s per XGMI link is ~768 KiB. Deepening
// the pipeline trades per-link bytes for boundaries, so this is what decides
// where that trade stops paying.
inline constexpr size_t kRelayPipelineBoundaryBytes = 768u << 10;

/**
 * Geometry of a single-group relay's software pipeline, in units of the
 * smallest chunk the schedule moves.
 *
 * A schedule is described by two affine functions of the depth T, each a
 * (perTile, fixed) pair:
 *
 * - totalUnits(T): how many units the relayed count divides into. One unit is
 *   count / totalUnits(T).
 * - linkUnits(T): how many units the busiest link direction carries, which is
 *   what the call actually costs.
 *
 * Depth 1 always reproduces the existing two-group schedule, which is the check
 * that a shape is written correctly.
 *
 * The shipped shapes, for A active ranks and H helpers:
 *
 * - 2-active, all four collectives. link {1, 1}, total {H+1, 1}. One unit per
 *   link per group; count/4 at T = 1 falling towards count/7 at H = 6.
 * - 4-active all-to-all. link {1, 1}, total {2, 1}. One unit up and one down
 * per cross link per group, matched by one direct unit on the intra links;
 *   2*count/3 at T = 1 falling towards count/2.
 * - A>2 all-gather fanout. link {A-1, 1}, total {H+A-1, 1}. Asymmetric: the
 *   helper fans each source's slice out to the A-1 other destinations, so one
 *   direction of the cross link carries A-1 units for every 1 the other
 *   carries and merging is bounded by the heavy direction. At A = H = 4 that
 *   is link {3, 1} / total {7, 1}: count/2 at T = 1 falling towards
 *   3*count/7, which is also the hard floor since an active rank must take in
 *   A-1 whole buffers across its 7 links.
 */
struct RelayPipelineShape {
  int linkPerTile;
  int linkFixed;
  int totalPerTile;
  int totalFixed;
};

inline constexpr RelayPipelineShape relayShapeA2(int numHelpers) {
  return {1, 1, numHelpers + 1, 1};
}
inline constexpr RelayPipelineShape kRelayShapeA4AllToAll = {1, 1, 2, 1};

// The all-gather fanout schedule's layout consumes H*T + 1 + (T-1)*(A-1) units
// before its final direct chunk, so totalUnits must be at least that for
// directSize(T) = count - directOffset(T) not to underflow. Deriving both
// members from A and H keeps the shape and the kernel's own layout in step for
// every geometry buildShardedRelayRankConfig() accepts, not just A = H = 4.
inline constexpr RelayPipelineShape relayShapeFanout(
    int nActiveRanks,
    int numHelpers) {
  return {nActiveRanks - 1, 1, numHelpers + nActiveRanks - 1, 1};
}

/**
 * Software-pipeline depth for a single-group relay.
 *
 * With nGroups == 1 the active ranks and the helpers are DISJOINT sets, so a
 * cross link carries the scatter (active -> helper) in one direction and the
 * forward (helper -> active) in the other. Running those as two serialized
 * ncclGroups therefore leaves every cross link HALF-DUPLEX for the whole call:
 * the forward direction is idle for all of group 1 and the scatter direction
 * for all of group 2.
 *
 * Tiling the relay into T tiles and issuing tile t's forward in the SAME group
 * as tile t+1's scatter fills both directions. What that is worth depends on
 * the schedule's shape (see RelayPipelineShape); at 2 active ranks it takes the
 * per-link cost from count/4 towards count/7, which is also the hard floor
 * since an active rank must move its buffer out across its 7 links exactly
 * once.
 *
 * A FUSED call gains nothing here, which is why this is gated on nGroups == 1:
 * there every rank is active for one group and a helper for the others, so its
 * scatter and its forward are egress on the SAME link direction. They add
 * rather than overlap, and the extra group boundaries are pure cost.
 *
 * The depth is whichever power of two minimizes
 * linkUnits(T) * unitBytes + (T + 1) * kRelayPipelineBoundaryBytes, which is
 * the two competing terms stated directly rather than a size threshold per
 * depth.
 */
inline int relayPipelineTiles(
    int nGroups,
    RelayPipelineShape shape,
    size_t maxCount,
    size_t elementSize) {
  if (nGroups != 1 || shape.totalPerTile < 1) {
    return 1;
  }
  const size_t bytes = maxCount * elementSize;
  int bestTiles = 1;
  size_t bestCost = 0;
  for (int tiles = 1; tiles <= kRelayMaxPipelineTiles; tiles *= 2) {
    const size_t totalUnits =
        static_cast<size_t>(shape.totalPerTile) * static_cast<size_t>(tiles) +
        static_cast<size_t>(shape.totalFixed);
    // A unit still has to be a whole aligned chunk.
    if (maxCount / totalUnits < kRelayChunkAlignElements) {
      break;
    }
    const size_t linkUnits =
        static_cast<size_t>(shape.linkPerTile) * static_cast<size_t>(tiles) +
        static_cast<size_t>(shape.linkFixed);
    const size_t cost = linkUnits * (bytes / totalUnits) +
        static_cast<size_t>(tiles + 1) * kRelayPipelineBoundaryBytes;
    if (bestCost == 0 || cost < bestCost) {
      bestCost = cost;
      bestTiles = tiles;
    }
  }
  return bestTiles;
}

/**
 * Software-pipeline depth for the single-group A>2 allreduce.
 *
 * This one needs its own selector because, unlike every other relay, its
 * depth-1 form is NOT the shipped two-group schedule, so it cannot be expressed
 * as a RelayPipelineShape whose depth-1 case reproduces today's behaviour.
 *
 * The two-group allreduce splits the count into equal direct and offload halves
 * and moves count/4 per link. The pipelined form runs T direct tiles (a
 * reduce-scatter tile per group, its all-gather one group later) woven with T
 * offload tiles, and moves 2*(T+1)*count/((A + 2H)*T) per link: at A = H = 4
 * that is count/3 at T = 1 and exactly count/4 at T = 2 -- worse, then
 * break-even with an extra boundary. It first pays at depth 4 (5*count/24) and
 * again at 8 (3*count/16), so only those are offered, and only when they also
 * cover their extra boundaries.
 *
 * The unit count is A + 2H rather than a fixed 12 because that is what the
 * layout actually consumes: the offload region takes 2*H*T units and the direct
 * region needs A*T for its per-owner shards to tile. At the shipped A = H = 4
 * geometry that is 12, but A = 4 with H = 5..8 (a 9-to-12-rank comm) or A = 8
 * with H = 8 (a 16-rank one) also reach this path, and a fixed 12 would push pO
 * past count and underflow pD. The depth THRESHOLDS below are still calibrated
 * against A = H = 4 measurements; other geometries get a correct layout from a
 * cost model that has not been re-measured for them.
 *
 * Its offload is what makes it worth doing at all: the helper takes one chunk
 * per active rank and returns one reduced chunk per active rank, so the cross
 * link carries the same in each direction and merging is not throttled by a
 * heavy side, unlike the all-gather and reduce-scatter shapes.
 *
 * A stage here is priced at TWICE kRelayPipelineBoundaryBytes because this is
 * the only schedule that issues two reduce kernels per stage -- the owner
 * folding its direct shard's tile and the helper summing its offload tile -- on
 * top of the group boundary itself. Measured, that is what it takes: the depth
 * curve wants the two-group schedule at or below 72 MB, depth 4 from 135 MB to
 * 256 MB, and depth 8 from 512 MB, which pins the per-stage price to
 * between 1.33 and 1.875 MiB. At the shared 768 KiB it crosses over near 54 MB
 * instead and loses 7-8% at 63-72 MB.
 */
// Units the pipelined allreduce layout divides the count into per tile:
// 2*H for the offload region (the helper takes one chunk per active rank and
// returns one, so a tile is 2y) plus A for the direct region's per-owner
// shards.
inline constexpr int relayAllReducePipelineUnitsPerTile(
    int nActiveRanks,
    int numHelpers) {
  return nActiveRanks + 2 * numHelpers;
}

inline int relayAllReducePipelineTiles(
    int nGroups,
    int nActiveRanks,
    int numHelpers,
    size_t maxCount,
    size_t elementSize) {
  if (nGroups != 1) {
    return 1;
  }
  const size_t bytes = maxCount * elementSize;
  // Depth 1 means "keep the two-group schedule", so that is what to beat.
  constexpr size_t kStageBytes = 2 * kRelayPipelineBoundaryBytes;
  size_t bestCost = bytes / 4 + 2 * kStageBytes;
  int bestTiles = 1;
  const size_t unitsPerTile = static_cast<size_t>(
      relayAllReducePipelineUnitsPerTile(nActiveRanks, numHelpers));
  for (int tiles = 4; tiles <= kRelayMaxPipelineTiles; tiles *= 2) {
    const size_t units = unitsPerTile * static_cast<size_t>(tiles);
    if (maxCount / units < kRelayChunkAlignElements) {
      break;
    }
    const size_t cost = 2 * static_cast<size_t>(tiles + 1) * (bytes / units) +
        static_cast<size_t>(tiles + 1) * kStageBytes;
    if (cost < bestCost) {
      bestCost = cost;
      bestTiles = tiles;
    }
  }
  return bestTiles;
}

} // namespace rcclx::relay
