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

size_t lpMinBytes(LpCollective coll, int nActiveRanksPerGroup, int nGroups) {
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
  const int64_t minKb = ncclParamShardedRelayLpMinKb();
  if (minKb > 0) {
    return static_cast<size_t>(minKb) << 10;
  }

  // MEASURED after the chunk-alignment fix, which changed the answer again: low
  // precision now pays for ELEVEN of the sixteen (collective, width, grouping)
  // shapes. It paid for two before the wavefront-absmax rewrite and seven
  // before the alignment fix -- both of those were measuring stalls, not the
  // wire format. The full table with provenance is in the header.
  //
  // Three shapes stay off, and none of them because low precision loses
  // everywhere:
  //
  //  - reduce-scatter A=4, either grouping, sits at 0.98x-1.03x through 40 MB
  //  and
  //    only reaches 1.24x-1.32x at 63 MB, the TOP of the measured range. A
  //    threshold at the edge of the data is a guess; it needs measuring past 72
  //    MB.
  //  - single-group allreduce A=2 wins 1.09x-1.14x from 13.5 MB to 27 MB, then
  //    drops to 0.78x-0.92x from 31.5 MB up. A min-bytes gate cannot say "this
  //    band only", and the alignment fix did NOT move it, so it is a separate
  //    mechanism still being profiled.
  //  - single-group all-to-all A=2 is the only shape that genuinely never wins:
  //    0.91x-0.96x at every size, before and after every fix.
  constexpr size_t kNever = std::numeric_limits<size_t>::max();
  constexpr size_t k8Mib = static_cast<size_t>(8) << 20;
  constexpr size_t k12Mib = static_cast<size_t>(12) << 20;
  constexpr size_t k24Mib = static_cast<size_t>(24) << 20;
  constexpr size_t k27Mib = static_cast<size_t>(27) << 20;

  if (nGroups <= 1) {
    // Uncontended. Still wins for most shapes, which is itself a change: before
    // the alignment fix nearly every single-group shape was troughed, and that
    // looked like "there is no bandwidth term to win when the links are slack".
    // It was a stall.
    switch (coll) {
      case LpCollective::AllReduce:
        // A=4: 1.09x at 13.5 MB rising to 1.29x. A=2 is the profiling target
        // above -- it wins 13.5-27 MB and troughs from 31.5 MB.
        return nActiveRanksPerGroup == 4 ? k12Mib : kNever;
      case LpCollective::AllGather:
        // A=2: 1.17x at 13.5 MB, 1.19x-1.23x above. A=4: 1.20x
        // then 1.24x-1.30x.
        return k12Mib;
      case LpCollective::ReduceScatter:
        // A=2: 1.11x at 13.5 MB rising to 1.28x.
        return nActiveRanksPerGroup == 2 ? k12Mib : kNever;
      case LpCollective::AllToAll:
        // A=4: 1.08x at 27 MB rising to 1.16x, and flat 0.97x-1.00x below, so
        // the crossover sits between 13.5 and 27 MB.
        return nActiveRanksPerGroup == 4 ? k24Mib : kNever;
    }
    return kNever;
  }

  switch (coll) {
    case LpCollective::AllReduce:
      // A=2: 1.20x at 13.5 MB, 1.26x-1.30x above. A=4: 1.15x then 1.23x-1.28x.
      return k12Mib;
    case LpCollective::AllGather:
      // A=2 pays earliest of anything measured, 1.14x already at 9 MB, so it
      // gets the lower threshold. A=4: 1.30x at 13.5 MB up to 1.45x, still the
      // largest win in the table.
      return nActiveRanksPerGroup == 2 ? k8Mib : k12Mib;
    case LpCollective::ReduceScatter:
      // A=2: 1.17x at 13.5 MB rising to 1.42x, the best of the fused
      // reductions.
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
      lpMinBytes(in.coll, in.nActiveRanksPerGroup, in.nGroups)) {
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
