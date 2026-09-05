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
constexpr size_t k60Mib = static_cast<size_t>(60) << 20;

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
        // A=4 sits at 0.90x-1.07x through 40 MB and then holds 1.18x-1.27x
        // across FIVE consecutive sizes from 63 MB to 144 MB. 60 MiB rather
        // than 48: there is no data between 40 and 63 MB, and 31.5 MB reads
        // 0.90x, so the threshold goes just below the first measured win
        // instead of into the unmeasured gap.
        return nActiveRanksPerGroup == 2 ? k12Mib : k60Mib;
      case LpCollective::AllToAll:
        // A=4: flat 0.96x-0.97x through 13.5 MB, then 1.07x-1.17x from 27 MB.
        //
        // A=2 is the only shape that never wins at any size in either grouping,
        // and it gets WORSE with size (0.95x-0.97x small, 0.88x at 135-144 MB),
        // which is the opposite of every other shape here. Whatever it is, it
        // is not a threshold. fp32 declines it too, for the same trend.
        return nActiveRanksPerGroup == 4 ? k24Mib : kNever;
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
      // A=4 reaches 1.32x only at 63 MB, the top of the range it was tuned on,
      // and a plateau at the edge of the data is not a crossover. fp32 at this
      // same shape DOES cross cleanly, so this is a candidate for re-measuring
      // rather than a settled no.
      return nActiveRanksPerGroup == 2 ? k12Mib : kNever;
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
        // A=4 has the same profile as its bf16 twin -- 0.94x-1.03x through
        // 40 MB, nothing in the 40-63 MB gap, then a step -- but fp32's step is
        // to 1.89x-2.24x across six consecutive sizes.
        //
        // 60 MiB, the SAME value as bf16, and the one place this table does not
        // use the first measured win (63 MB). Both dtypes' first win is that
        // same point and there is no data in the 40-63 MB gap to separate them,
        // so 63 MiB here would put fp32 above bf16 on identical evidence --
        // breaking the ordering that makes the two tables comparable, to move a
        // threshold by 5%.
        return nActiveRanksPerGroup == 2 ? k4p5Mib : k60Mib;
      case LpCollective::AllToAll:
        // A=4: 1.02x at 9 MB, 1.18x at 13.5 MB, up to 1.90x.
        //
        // A=2 is the ONE fp32 exclusion. It peaks at 1.12x at 27 MB and falls
        // back to 0.99x-1.02x from 63 MB up -- the same wrong-way-with-size
        // trend as bf16 (0.88x-0.97x), shifted up by the extra saving. A
        // min-bytes gate cannot express a band that closes again.
        return nActiveRanksPerGroup == 4 ? k13p5Mib : kNever;
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
      // A=4 is ENABLED here and off for bf16, which is the clearest case for
      // splitting the table by dtype: identical flat ~1.00x through 40 MB in
      // both, then bf16 manages 1.32x at one size at the edge of its range
      // while fp32 holds 2.01x-2.63x across six. 60 MiB to match the
      // single-group entry, for the reason given there.
      return nActiveRanksPerGroup == 2 ? k4p5Mib : k60Mib;
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
