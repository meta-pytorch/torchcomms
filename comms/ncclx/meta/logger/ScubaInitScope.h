// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <optional>
#include <string>
#include <utility>

#include <fmt/core.h>

#include "comms/utils/logger/EventsScubaUtil.h"
#include "comms/utils/logger/NcclScubaSample.h"

namespace meta::comms::ncclx {

// RAII scope for a comm-init entrypoint. Bundles the standard Scuba sticky
// context (num_ranks, rank, and cuda_dev when known) with the enclosing "INIT"
// sample, matching the hand-rolled scaffolding it replaces: the sticky guards
// are pushed first and the sample is logged on scope exit while they are still
// live. `sample()` is exposed so the caller attaches communicator metadata and
// sets the exec result exactly as before. Hoisted out of the forked init
// entrypoints; the sample keeps the call site's identity via the
// NCCLX_SCUBA_INIT_SCOPE macro.
class ScubaInitScope {
 public:
  ScubaInitScope(
      const char* fileName,
      int line,
      const char* functionName,
      int nRanks,
      int rank)
      : ctxNRanks_(ScubaContextKeys::num_ranks, fmt::format("{}", nRanks)),
        ctxRank_(ScubaContextKeys::rank, fmt::format("{}", rank)),
        guard_("INIT", fileName, line, functionName) {}

  ScubaInitScope(
      const char* fileName,
      int line,
      const char* functionName,
      int nRanks,
      int rank,
      int cudaDev)
      : ctxNRanks_(ScubaContextKeys::num_ranks, fmt::format("{}", nRanks)),
        ctxRank_(ScubaContextKeys::rank, fmt::format("{}", rank)),
        ctxCudaDev_(
            std::in_place,
            ScubaContextKeys::cuda_dev,
            fmt::format("{}", cudaDev)),
        guard_("INIT", fileName, line, functionName) {}

  ScubaInitScope(const ScubaInitScope&) = delete;
  ScubaInitScope& operator=(const ScubaInitScope&) = delete;
  ScubaInitScope(ScubaInitScope&&) = delete;
  ScubaInitScope& operator=(ScubaInitScope&&) = delete;

  NcclScubaSample& sample() {
    return guard_.sample();
  }

 private:
  // Declaration order is load-bearing: the sticky guards are constructed before
  // `guard_` (so the sample sees them) and, in reverse, `guard_` logs before
  // they are popped.
  ::EventsScubaUtil::StickyContextGuard ctxNRanks_;
  ::EventsScubaUtil::StickyContextGuard ctxRank_;
  std::optional<::EventsScubaUtil::StickyContextGuard> ctxCudaDev_;
  ::EventsScubaUtil::SampleGuard guard_;
};

} // namespace meta::comms::ncclx

// Declares a ScubaInitScope named `var` for the enclosing init entrypoint,
// capturing the call site's __FILE__ / __LINE__ / __FUNCTION__. Pass either
// (nRanks, rank) or (nRanks, rank, cudaDev).
#define NCCLX_SCUBA_INIT_SCOPE(var, ...)    \
  ::meta::comms::ncclx::ScubaInitScope var( \
      __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
