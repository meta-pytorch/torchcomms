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
};

void launchNvlSignalTrap(NvlSignalTrapCase testCase);

} // namespace comms::prims::test
