// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::prims::test {

enum class StageLayoutTrapCase : uint32_t {
  ZeroGeometry,
  TooManyGroups,
  BadSignalsPerLane,
  InsufficientLocalSignals,
  InsufficientMultimemSignals,
};

void launchStageLayoutTrap(StageLayoutTrapCase testCase);

} // namespace comms::prims::test
