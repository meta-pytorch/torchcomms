// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef COMMS_PRIMS_TESTS_MULTIMEM_NVL_STAGE_LAYOUT_TRAP_TEST_CUH_
#define COMMS_PRIMS_TESTS_MULTIMEM_NVL_STAGE_LAYOUT_TRAP_TEST_CUH_

#include <cstdint>

namespace comms::prims::test {

enum class StageLayoutTrapCase : uint32_t {
  ZeroGeometry,
  TooManyGroups,
  BadSignalsPerChannel,
  InsufficientLocalSignals,
  InsufficientMultimemSignals,
};

void launchStageLayoutTrap(StageLayoutTrapCase testCase);

} // namespace comms::prims::test

#endif
