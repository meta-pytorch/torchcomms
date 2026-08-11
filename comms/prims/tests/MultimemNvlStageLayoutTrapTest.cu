// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlStageLayoutTrapTest.cuh"

#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStageLayout.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

namespace comms::prims::test {
namespace {

__global__ void stageLayoutTrapKernel(StageLayoutTrapCase testCase) {
  constexpr uint32_t kNvlRanks = 4;
  constexpr uint32_t kExpectedSignalsPerLane =
      multimem_staging_signals_per_lane(kNvlRanks);
  SignalState signal;
  const uint32_t localSignalCount =
      testCase == StageLayoutTrapCase::InsufficientLocalSignals
      ? kExpectedSignalsPerLane - 1
      : kExpectedSignalsPerLane;
  const uint32_t multimemSignalCount =
      testCase == StageLayoutTrapCase::InsufficientMultimemSignals
      ? kExpectedSignalsPerLane - 1
      : kExpectedSignalsPerLane;
  MultimemNvlTransportDevice transport{
      .localData = reinterpret_cast<char*>(1),
      .multimemData = reinterpret_cast<char*>(1),
      .internalLocalSignals =
          DeviceSpan<SignalState>(&signal, localSignalCount),
      .internalMultimemSignals =
          DeviceSpan<SignalState>(&signal, multimemSignalCount),
      .dataBufferSize = 64,
      .nvlRank = 0,
      .nvlRanks = kNvlRanks,
      .pipelineDepth = 1,
      .maxChannels = 1,
      .signalsPerLane = kExpectedSignalsPerLane,
  };
  switch (testCase) {
    case StageLayoutTrapCase::ZeroGeometry:
      transport.pipelineDepth = 0;
      break;
    case StageLayoutTrapCase::TooManyGroups:
      break;
    case StageLayoutTrapCase::BadSignalsPerLane:
      transport.signalsPerLane = kExpectedSignalsPerLane - 1;
      break;
    case StageLayoutTrapCase::InsufficientLocalSignals:
    case StageLayoutTrapCase::InsufficientMultimemSignals:
      break;
  }
  auto group = make_block_group();
  static_cast<void>(multimem::make_stage_layout<uint32_t>(transport, group));
}

} // namespace

void launchStageLayoutTrap(StageLayoutTrapCase testCase) {
  const uint32_t numGroups =
      testCase == StageLayoutTrapCase::TooManyGroups ? 2 : 1;
  stageLayoutTrapKernel<<<numGroups, 32>>>(
      testCase); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

} // namespace comms::prims::test
