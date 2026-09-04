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

  // MEASURED after the wavefront-absmax rewrite, which changed the answer
  // substantially: low precision now wins for SIX of the eight shapes when the
  // groups contend, where before it won for two. The full table with provenance
  // is in the header.
  //
  // Two things shape this policy:
  //
  //  - CONTENTION IS WHAT LOW PRECISION WINS. nGroups > 1 puts several groups
  //  on
  //    the XGMI links at once, so there is a bandwidth term for halved wire
  //    bytes to shrink. Every fused shape below is enabled; almost no
  //    single-group one is.
  //  - A MIN-BYTES THRESHOLD CANNOT EXCLUDE A MIDDLE BAND. Two shapes win
  //  either
  //    side of a reproducible trough (fused all-to-all A=4 craters to 0.64x at
  //    exactly 32 MB and 0.65x at 40 MB; most nGroups == 1 shapes dip through
  //    31.5-63 MB). Enabling those would buy a 1.2x plateau at the price of a
  //    0.65x cliff inside it, which is not a trade this knob can express. They
  //    stay off until the trough is explained.
  constexpr size_t kNever = std::numeric_limits<size_t>::max();
  constexpr size_t k8Mib = static_cast<size_t>(8) << 20;
  constexpr size_t k12Mib = static_cast<size_t>(12) << 20;

  if (nGroups <= 1) {
    // The one uncontended shape that is monotone rather than troughed: 1.08x at
    // 13.5 MB rising steadily to 1.30x at 72 MB, with no dip anywhere. Every
    // other single-group shape oscillates through the 31.5-63 MB band.
    if (coll == LpCollective::AllReduce && nActiveRanksPerGroup == 4) {
      return k12Mib;
    }
    return kNever;
  }

  switch (coll) {
    case LpCollective::AllReduce:
      // A=2: 1.17x at 13.5 MB, 1.26x-1.31x above. A=4: 1.14x then 1.24x-1.28x.
      return k12Mib;
    case LpCollective::AllGather:
      // A=2 pays earliest of anything measured -- 1.14x already at 9
      // MB, 1.24x-1.33x above -- so it gets the lower threshold. A=4: 1.30x
      // at 13.5 MB up to 1.45x, the largest win in the table.
      return nActiveRanksPerGroup == 2 ? k8Mib : k12Mib;
    case LpCollective::ReduceScatter:
      // A=2: 1.16x at 13.5 MB rising to 1.40x. A=4 is deliberately off: it sits
      // at 0.95x-1.04x through 40 MB and only reaches 1.29x at 63 MB, which is
      // the top of the measured range -- a threshold set at the edge of the
      // data is a guess, and it needs measuring past 72 MB first.
      return nActiveRanksPerGroup == 2 ? k12Mib : kNever;
    case LpCollective::AllToAll:
      // A=2: 1.14x at 13.5 MB, 1.21x-1.32x above. A=4 wins 1.16x-1.19x from 27
      // MB but craters to 0.64x/0.65x at 32 MB and 40 MB, INSIDE that range, so
      // a minimum cannot capture the win without the cliff.
      return nActiveRanksPerGroup == 2 ? k12Mib : kNever;
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
