// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cstdint>

namespace comms::prims::test {

enum class NvlSignalTrapCase : uint32_t {
  AggregateDepthTooLarge,
  RankCountTooLarge,
  ArrivalCountMismatch,
  PerPeerGroupTooSmall,
  WaitTimeout,
  ZeroRound,
  SignalsPerChannelMismatch,
};

void launchNvlSignalTrap(NvlSignalTrapCase testCase);
void launchNvlSignalRankBoundary(int nvlRanks, uint64_t* output);

} // namespace comms::prims::test
