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
  SerialMinWaitTimeout,
  TreeMinWaitTimeout,
  ButterflyMinWaitTimeout,
  SignalsPerChannelMismatch,
  UpperWordWaitAllTimeout,
  UpperWordSerialMinTimeout,
  UpperWordTreeMinTimeout,
  UpperWordButterflyMinTimeout,
};

enum class NvlSignalRankBoundaryWaitPolicy : uint32_t {
  WaitAll,
  SerialMin,
  TreeMin,
  ButterflyMin,
};

void launchNvlSignalTrap(NvlSignalTrapCase testCase);
void launchNvlSignalRankBoundary(
    int nvlRanks,
    NvlSignalRankBoundaryWaitPolicy waitPolicy,
    uint64_t roundValue,
    uint64_t* output);

} // namespace comms::prims::test
