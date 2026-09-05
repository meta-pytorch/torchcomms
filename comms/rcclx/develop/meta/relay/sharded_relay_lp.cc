/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/sharded_relay_lp.h"

#include <atomic>
#include <limits>

#include "debug.h"
#include "param.h"

namespace rcclx::relay {

namespace {

// Process-wide, because the question these answer is "did low precision engage
// at all during this phase", which is asked by tests, the perf bench and the
// examples -- none of which has a communicator handy at the point they ask.
// Relaxed ordering: they are diagnostics, never read to make a decision, and a
// count that lags a call by a few nanoseconds is not a wrong count by the time
// anybody looks.
std::atomic<uint64_t>& engageCounter() {
  static std::atomic<uint64_t> c{0};
  return c;
}

std::atomic<uint64_t>* declineCounters() {
  static std::atomic<uint64_t> c[static_cast<int>(LpDecline::kNumReasons)]{};
  return c;
}

const char* declineName(LpDecline reason) {
  switch (reason) {
    case LpDecline::Dtype:
      return "unsupported dtype";
    case LpDecline::Alignment:
      return "a per-group count is not a multiple of 128";
    case LpDecline::Size:
      return "below the measured crossover";
    case LpDecline::Route:
      return "not a relay route";
    case LpDecline::Arena:
      return "does not fit the arena";
    case LpDecline::GraphCapture:
      return "no arena yet and the stream is capturing";
    case LpDecline::kNumReasons:
      break;
  }
  return "unknown";
}

// Thresholds shared by both dtype tables. Named for the size they are, so a
// table entry reads as the measured number it came from.
constexpr size_t kNever = std::numeric_limits<size_t>::max();
constexpr size_t k4p5Mib = (static_cast<size_t>(9) << 20) / 2;
constexpr size_t k8Mib = static_cast<size_t>(8) << 20;
constexpr size_t k9Mib = static_cast<size_t>(9) << 20;
constexpr size_t k12Mib = static_cast<size_t>(12) << 20;
constexpr size_t k13p5Mib = (static_cast<size_t>(27) << 20) / 2;
constexpr size_t k24Mib = static_cast<size_t>(24) << 20;
constexpr size_t k27Mib = static_cast<size_t>(27) << 20;
// 48 MiB is not a tuned number. It is kOffloadMinBytes from
// selectReduceScatterRoute(), mirrored here: below it the 4-active
// reduce-scatter takes PureDirect, so lpEligible() declines on ROUTE and the
// wire format is not consulted at all. Same arrangement as the 4-active
// all-to-all's 27 MiB against kXorRelayMinBytes -- the two gates agree because
// one is the other, not because they were fitted to the same data.
constexpr size_t k48Mib = static_cast<size_t>(48) << 20;

// bf16 crossovers. 1.94x fewer wire bytes per element, so this is the later of
// the two tables; 11 of 16 shapes pay. Full provenance in the header.
size_t
lpMinBytesBf16(LpCollective coll, int nActiveRanksPerGroup, int nGroups) {
  if (nGroups <= 1) {
    // Uncontended, measured across the full range. CO-RESIDENT JOBS LAND HERE:
    // a parallel relay job is nGroups == 1, so these are also what several
    // independent jobs sharing a node get, and they were measured with the node
    // otherwise idle.
    switch (coll) {
      case LpCollective::AllReduce:
        // RE-MEASURED after the allreduce chunk-alignment fix, which changed
        // both entries and retired the "stall" this table used to describe. The
        // old reading was that A=2 wins 1.09x-1.13x from 13.5 to 27 MB and then
        // drops to 0.75x-0.92x out to 144 MB, and that it was a memory-system
        // effect needing hardware counters. It was not: allreduce was rounding
        // its chunks to 128 elements instead of the shared 512, so its wire
        // offsets were 4-byte aligned wherever the pipeline reached four tiles.
        //
        // A=2 now rises monotonically with size, which is what every healthy
        // shape here does: 1.08x at 9 MB, 1.16x at 13.5, 1.22x at 31.5 -- the
        // size that used to collapse -- and 1.30x-1.35x from 67.5 MB to 1 GB.
        //
        // A=4: 0.95x at 4.5 MB, 1.04x at 9 MB, 1.14x at 13.5 MB, so the first
        // clear win is at 13.5 and 12 MiB sits just under it. It also holds
        // 1.29x-1.40x to 144 MB and 1.37x/1.34x at 256/512 MB, where before the
        // fix it REGRESSED to 0.79x/0.74x.
        return nActiveRanksPerGroup == 4 ? k12Mib : k8Mib;
      case LpCollective::AllGather:
        // 8 MiB, not 12: BOTH widths win at 9 MB (1.10x at A=2, 1.11x at A=4)
        // and only A=4 loses at 4.5 MB. This is the earliest-paying collective
        // here, same as in the fused table.
        return k8Mib;
      case LpCollective::ReduceScatter:
        // A=2: 1.05x at 9 MB, 1.13x at 13.5 MB, up to 1.30x. 12 MiB.
        //
        // A=4 is 48 MiB, and the number is now MECHANISM rather than fit. The
        // old entry said "60 MiB rather than 48: there is no data between 40
        // and 63 MB", which was the right call on the evidence and the wrong
        // answer: densifying that gap puts the step between 44 MB (0.99x) and
        // 48 MB (1.27x), a 28-point jump across one 4 MB stride. A crossover
        // ramps as a fixed quantize cost amortizes; this steps, then sits flat
        // at 1.25x-1.35x for six consecutive sizes.
        //
        // 48 MiB is kOffloadMinBytes in selectReduceScatterRoute(). Below it a
        // 4-active reduce-scatter is PureDirect -- no helpers, no relayed hop
        // -- so lpEligible() declines on ROUTE and the wire format never runs.
        // The flat sub-48 MiB readings were never low precision losing; they
        // were low precision absent, which is why forcing this gate open does
        // not move them.
        //
        // It also explains the coincidence that made this shape suspicious:
        // both dtypes appeared to cross at the same size despite fp32
        // saving 3.88x per element against bf16's 1.94x. They share a
        // BYTE-keyed route gate. Past it they diverge exactly as the wire
        // format predicts -- bf16 1.25x-1.35x, fp32 1.76x-2.09x.
        return nActiveRanksPerGroup == 2 ? k12Mib : k48Mib;
      case LpCollective::AllToAll:
        // A=4: flat 0.96x-0.97x through 13.5 MB, then 1.07x-1.17x from 27 MB.
        //
        // A=2 is ENABLED NOW. It used to be the one shape that never won at any
        // size and got WORSE with size (0.95x-0.97x small, 0.88x at
        // 135-144 MB), which was recorded here as "whatever it is, it is not a
        // threshold". That was right: it was the DIAGONAL FOLD. The 2-active
        // schedules kept the diagonal as a full-precision self P2P pair inside
        // a wire-format comm group, where it is ~8x (non-pipelined) to ~11x
        // (pipelined, T=4) the size of every op sharing its group once low
        // precision halves those ops, so it became the group's critical path.
        // The all-gather, which moves the same bytes over the same links with
        // the same 50/50 local-to-wire split, has always issued that copy
        // outside its groups and always paid 1.30x.
        //
        // Unfolded under low precision only, A=2 now rises monotonically:
        // 1.10x at 13.5 MB, 1.32x at 27, 1.12x-1.16x across 31.5-40, then
        // 1.25x-1.37x to 144 MB and 1.50x/1.55x/1.64x at 256/512 MB/1 GB --
        // the best all-to-all shape in this table. 9 MB reads 0.97x and 4.5 MB
        // 0.93x, so 12 MiB sits just below the first win.
        //
        // A=4 was measured both ways and KEPT ITS FOLD: its ratio is only ~3.2x
        // at T=2, mild enough that the in-kernel copy still beats a serialized
        // memcpy. Folded 1.24x-1.31x against unfolded 1.24x-1.27x at
        // 63-144 MB.
        return nActiveRanksPerGroup == 4 ? k24Mib : k12Mib;
    }
    return kNever;
  }

  switch (coll) {
    case LpCollective::AllReduce:
      // Also re-measured after the allreduce chunk-alignment fix, which lifted
      // the fused ratios by roughly 0.15-0.20 and moved A=2's crossover down a
      // band. A=2: 1.10x at 9 MB, 1.21x at 13.5, then 1.35x-1.49x. A=4: 1.07x
      // at 9 MB is break-even, 1.18x at 13.5 is the first clear win, rising to
      // 1.49x.
      return nActiveRanksPerGroup == 2 ? k8Mib : k12Mib;
    case LpCollective::AllGather:
      // A=2 pays earliest of anything measured, 1.14x already at 9 MB, so it
      // gets the lower threshold. A=4: 1.30x at 13.5 MB up to 1.45x, still the
      // largest win in the table.
      return nActiveRanksPerGroup == 2 ? k8Mib : k12Mib;
    case LpCollective::ReduceScatter:
      // A=2: 1.17x at 13.5 MB rising to 1.42x, the best of the fused
      // reductions.
      //
      // A=4 is 48 MiB == kOffloadMinBytes, for the reason given in the
      // single-group entry: below it the route is PureDirect and low precision
      // declines on ROUTE, so nothing under 48 MiB was ever a measurement of
      // the wire format. The old entry read the flat sub-40 MB region as
      // evidence about a crossover; it was evidence that the wire format never
      // ran.
      //
      // Measured fused with the gate forced open: 1.00x at 44 MB and 1.31x at
      // 48 MB in bf16, 1.03x and 1.88x in fp32. Same step at the same size in
      // both groupings and both dtypes -- as it must be, since the route gate
      // is keyed on bytes and does not depend on nGroups.
      return nActiveRanksPerGroup == 2 ? k12Mib : k48Mib;
    case LpCollective::AllToAll:
      // A=2: 1.14x at 13.5 MB, 1.23x-1.33x above. A=4 reads 1.00x at 13.5 MB
      // and 1.16x at 27 MB, and 27 MiB is also exactly where its XOR-relay
      // route starts when fused (kXorRelayMinBytes) -- below that the call is a
      // direct exchange and low precision would decline on route anyway, so the
      // two gates agree by construction rather than by coincidence.
      return nActiveRanksPerGroup == 2 ? k12Mib : k27Mib;
  }
  return kNever;
}

// fp32 crossovers. 3.88x fewer wire bytes per element -- twice bf16's saving --
// so every threshold is at or below its bf16 counterpart and 15 of 16 shapes
// pay. EVERY VALUE IS EXACTLY THE SMALLEST SIZE MEASURED TO WIN: the sweep
// jumps 576 KB to 4.5 MB, so a 4.5 MiB entry means "wins at the first size in
// the MB decade and the decade below is unmeasured", not "wins from 4.5 MiB".
size_t
lpMinBytesFp32(LpCollective coll, int nActiveRanksPerGroup, int nGroups) {
  if (nGroups <= 1) {
    switch (coll) {
      case LpCollective::AllReduce:
        // A=2 is ENABLED here and off for bf16. 1.09x at 4.5 MB and never below
        // 1.19x after, out to 2.26x at 1 GB. The bf16 stall is still visible as
        // VARIANCE -- 1.63x at 31.5 MB against 1.25x at 32 MB, non-monotonic in
        // size -- but the 3.88x saving stays ahead of it at every size. This is
        // not a claim that the stall was fixed.
        //
        // A=4: 1.02x at 4.5 MB is a tie, 1.21x at 9 MB is the first clear win.
        return nActiveRanksPerGroup == 4 ? k9Mib : k4p5Mib;
      case LpCollective::AllGather:
        // Both widths win 1.14x at 4.5 MB, the earliest size measured, and rise
        // to 2.04x (A=2) and 2.33x (A=4).
        return k4p5Mib;
      case LpCollective::ReduceScatter:
        // A=2: 1.08x at 4.5 MB up to 2.31x.
        //
        // A=4 is 48 MiB == kOffloadMinBytes. The old entry agonized over using
        // 60 rather than the first measured win of 63 MB, on the grounds that
        // "both dtypes' first win is that same point and there is no data in
        // the 40-63 MB gap to separate them" -- and read that shared crossover
        // as a reason to keep the two tables ordered. The shared crossover was
        // real but it was not about the wire format: below 48 MiB the route is
        // PureDirect and low precision declines on ROUTE in both dtypes alike.
        //
        // Densified, gate forced open: 0.99x at 44 MB then 1.76x at 48 MB, and
        // 1.76x-2.09x across every size from there to 144 MB. bf16 steps at the
        // same 48 MB to only 1.27x, so once the route admits low precision the
        // dtypes separate by roughly the ratio of what they save -- which is
        // what makes 48 MiB the same number in both tables for a defensible
        // reason rather than a coincidence to be preserved.
        return nActiveRanksPerGroup == 2 ? k4p5Mib : k48Mib;
      case LpCollective::AllToAll:
        // A=4: 1.02x at 9 MB, 1.18x at 13.5 MB, up to 1.90x.
        //
        // A=2 is ENABLED NOW, and it is the largest win in either table. It was
        // the ONE fp32 exclusion, described here as peaking at 1.12x at 27 MB
        // and falling back to 0.99x-1.02x from 63 MB up -- "a band that closes
        // again", which a min-bytes gate cannot express. Same cause as bf16:
        // the full-precision diagonal folded into a wire-format comm group.
        // Note the old peak was at 27 MB, the only size on the non-pipelined
        // path, and the fall-back began exactly where the pipeline deepens.
        //
        // fp32 is hurt worse by the fold than bf16 -- the self op carries
        // 4 B/elem against ~1.03 B/elem of wire, so it is roughly twice the
        // outlier -- and correspondingly gains more from unfolding: 1.06x at
        // 9 MB, 1.20x at 13.5, 1.62x at 27, then 1.35x-1.93x to 144 MB and
        // 2.22x/2.35x/2.53x at 256/512 MB/1 GB. 4.5 MB is a 1.00x tie, so
        // 8 MiB sits just below the first win.
        return nActiveRanksPerGroup == 4 ? k13p5Mib : k8Mib;
    }
    return kNever;
  }

  switch (coll) {
    case LpCollective::AllReduce:
      // A=2: 1.08x at 4.5 MB to 2.65x at 1 GB. A=4: 1.03x at 4.5 MB is
      // break-even, 1.26x at 9 MB is the first clear win.
      return nActiveRanksPerGroup == 4 ? k9Mib : k4p5Mib;
    case LpCollective::AllGather:
      // A=2: 1.14x at 4.5 MB up to 2.44x.
      //
      // A=4 reads 1.01x at 4.5 MB and 1.00x at 9 MB, then jumps to 1.92x at
      // 13.5 MB -- a step, not a ramp. The SECOND of the two entries that take
      // bf16's rounded-down value (12 MiB) instead of their own first measured
      // win (13.5 MB): bf16 also reads 1.00x at 9 MB and also first wins at
      // 13.5 MB, so the evidence is identical and fp32 sitting above bf16 on it
      // would break the ordering the two tables are compared by.
      return nActiveRanksPerGroup == 2 ? k4p5Mib : k12Mib;
    case LpCollective::ReduceScatter:
      // A=2: 1.09x at 4.5 MB up to 2.68x.
      //
      // A=4 is 48 MiB == kOffloadMinBytes, for the reason given in the
      // single-group entry. The old entry read this shape as "the clearest case
      // for splitting the table by dtype", on identical flat ~1.00x through
      // 40 MB in both. That flatness was a shared ROUTE decline, not a dtype
      // difference, so it argued for nothing. Measured fused with the gate
      // forced open: 1.03x at 44 MB, 1.88x at 48 MB, 1.91x-2.08x above.
      return nActiveRanksPerGroup == 2 ? k4p5Mib : k48Mib;
    case LpCollective::AllToAll:
      // A=2: 1.08x at 4.5 MB up to 2.41x. A=4: 1.02x at 13.5 MB, 1.64x at
      // 27 MB -- the same 27 MiB as bf16, and again exactly where the fused
      // XOR-relay route starts, because the route gate does not depend on the
      // wire format.
      return nActiveRanksPerGroup == 2 ? k4p5Mib : k27Mib;
  }
  return kNever;
}

} // namespace

// Tuning and provisioning only. There is deliberately no
// NCCL_SHARDED_RELAY_LP_MODE_ENABLE: low precision is requested per call, so a
// process-wide switch would be a second, contradictory answer to the same
// question.
NCCL_PARAM(ShardedRelayLpHops, "SHARDED_RELAY_LP_HOPS", 2);
NCCL_PARAM(ShardedRelayLpPrealloc, "SHARDED_RELAY_LP_PREALLOC", 0);
NCCL_PARAM(ShardedRelayLpMaxMsgMb, "SHARDED_RELAY_LP_MAX_MSG_MB", 1024);
NCCL_PARAM(ShardedRelayLpMinKb, "SHARDED_RELAY_LP_MIN_KB", 0);

int lpHops() {
  const int64_t hops = ncclParamShardedRelayLpHops();
  // Anything but 1 means "quantize both hops". Clamped rather than validated so
  // a typo degrades to the default behaviour instead of disabling the feature.
  return hops == 1 ? 1 : 2;
}

bool lpPrealloc() {
  return ncclParamShardedRelayLpPrealloc() == 1;
}

size_t lpMaxMsgBytes() {
  int64_t mb = ncclParamShardedRelayLpMaxMsgMb();
  if (mb < 1) {
    mb = 1;
  }
  return static_cast<size_t>(mb) << 20;
}

bool lpDtypeSupported(ncclDataType_t datatype) {
  // bf16 and fp32 only. Both quantize to e4m3 through an fp32 intermediate, and
  // both are what the relay actually carries in the workloads this targets.
  // fp16 is deliberately absent: its 5-bit exponent already overlaps e4m3's
  // range closely enough that the 1.94x is not obviously worth a second
  // rounding, and adding it is a measurement away rather than a guess.
  return datatype == ncclBfloat16 || datatype == ncclFloat32;
}

bool lpCountsAligned(const size_t* counts, int nGroups, size_t alignElems) {
  if (counts == nullptr || nGroups <= 0 || alignElems == 0) {
    return false;
  }
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0 || counts[g] % alignElems != 0) {
      return false;
    }
  }
  return true;
}

size_t lpMinBytes(
    LpCollective coll,
    int nActiveRanksPerGroup,
    int nGroups,
    ncclDataType_t datatype) {
  // NCCL_SHARDED_RELAY_LP_MIN_KB overrides the crossover. This exists so the
  // crossover can be MEASURED: the built-in values below refuse most shapes
  // outright, and the gate declines before anything is timed, so without an
  // override a sweep could never see whether low precision would have won. It
  // is also what the collective test suites set, because they cover the
  // MECHANISM and must not be coupled to this tuning policy.
  //
  // Tuning only, and it can only make low precision apply MORE widely or less
  // -- never change what the wire format does. Read through NCCL_PARAM, which
  // caches on first read, so it has to be set before the first relay call; that
  // is the same contract every other knob here has.
  //
  // Deliberately dtype-INDEPENDENT: an override is a single number a human
  // typed to move one gate out of the way, so making it mean two different
  // things depending on the tensor would defeat the point.
  const int64_t minKb = ncclParamShardedRelayLpMinKb();
  if (minKb > 0) {
    return static_cast<size_t>(minKb) << 10;
  }

  // Two independently measured tables. fp32 is the earlier one everywhere,
  // because quantizing 4 bytes to 33/32 saves twice what quantizing 2 does.
  // Anything else falls back to bf16, the more conservative table --
  // unreachable in practice, since lpEligible() rejects unsupported dtypes
  // before the size gate, but a wrong answer here would silently widen the
  // policy.
  return datatype == ncclFloat32
      ? lpMinBytesFp32(coll, nActiveRanksPerGroup, nGroups)
      : lpMinBytesBf16(coll, nActiveRanksPerGroup, nGroups);
}

bool lpEligible(const LpGateInputs& in) {
  if (!in.relayRouteSelected) {
    lpRecordDecline(LpDecline::Route);
    return false;
  }
  if (!lpDtypeSupported(in.datatype)) {
    lpRecordDecline(LpDecline::Dtype);
    return false;
  }
  if (!lpCountsAligned(in.counts, in.nGroups, in.countAlignElems)) {
    lpRecordDecline(LpDecline::Alignment);
    return false;
  }
  if (in.routeSizeBytes <
      lpMinBytes(in.coll, in.nActiveRanksPerGroup, in.nGroups, in.datatype)) {
    lpRecordDecline(LpDecline::Size);
    return false;
  }
  return true;
}

void lpRecordDecline(LpDecline reason) {
  const int idx = static_cast<int>(reason);
  if (idx < 0 || idx >= static_cast<int>(LpDecline::kNumReasons)) {
    return;
  }
  const uint64_t n =
      declineCounters()[idx].fetch_add(1, std::memory_order_relaxed) + 1;
  // Logged only the first few times per reason. The point is to learn whether
  // real traffic misses the gate, which one line answers; a line per call would
  // bury it.
  if (n <= 4) {
    INFO(
        NCCL_COLL,
        "Sharded relay: low precision declined (%s); running in full precision",
        declineName(reason));
  }
}

void lpRecordEngage() {
  engageCounter().fetch_add(1, std::memory_order_relaxed);
}

uint64_t lpEngageCount() {
  return engageCounter().load(std::memory_order_relaxed);
}

uint64_t lpDeclineCount(LpDecline reason) {
  const int idx = static_cast<int>(reason);
  if (idx < 0 || idx >= static_cast<int>(LpDecline::kNumReasons)) {
    return 0;
  }
  return declineCounters()[idx].load(std::memory_order_relaxed);
}

uint64_t lpDeclineCount() {
  uint64_t total = 0;
  for (int i = 0; i < static_cast<int>(LpDecline::kNumReasons); i++) {
    total += declineCounters()[i].load(std::memory_order_relaxed);
  }
  return total;
}

void lpResetCounters() {
  engageCounter().store(0, std::memory_order_relaxed);
  for (int i = 0; i < static_cast<int>(LpDecline::kNumReasons); i++) {
    declineCounters()[i].store(0, std::memory_order_relaxed);
  }
}

} // namespace rcclx::relay
