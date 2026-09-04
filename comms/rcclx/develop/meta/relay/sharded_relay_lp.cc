/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meta/relay/sharded_relay_lp.h"

#include <atomic>

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

bool lpCountsAligned(const size_t* counts, int nGroups) {
  if (counts == nullptr || nGroups <= 0) {
    return false;
  }
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0 || counts[g] % kLpBlockElems != 0) {
      return false;
    }
  }
  return true;
}

size_t lpMinBytes(LpCollective coll, int nActiveRanksPerGroup, int nGroups) {
  (void)coll;
  (void)nActiveRanksPerGroup;
  (void)nGroups;
  // PROVISIONAL, pending the LP sweep. 4 MiB is above the entire flat
  // launch-bound band (measured flat across 4 KB..576 KB) and at or above every
  // relay-route crossover in sharded_relay_route.h, so low precision is only
  // ever considered where a relay route is already active and there is a real
  // bandwidth term for it to shrink. The signature carries the collective and
  // the geometry so the measured per-shape crossovers can be filled in here
  // without touching a single call site.
  return static_cast<size_t>(4) << 20;
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
  if (!lpCountsAligned(in.counts, in.nGroups)) {
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
