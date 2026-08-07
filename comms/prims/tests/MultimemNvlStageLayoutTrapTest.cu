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
  SignalState signal;
  const uint32_t localSignalCount =
      testCase == StageLayoutTrapCase::InsufficientLocalSignals ? 11 : 12;
  const uint32_t multimemSignalCount =
      testCase == StageLayoutTrapCase::InsufficientMultimemSignals ? 11 : 12;
  MultimemNvlTransportDevice transport{
      .localData = reinterpret_cast<char*>(1),
      .multimemData = reinterpret_cast<char*>(1),
      .internalLocalSignals =
          DeviceSpan<SignalState>(&signal, localSignalCount),
      .internalMultimemSignals =
          DeviceSpan<SignalState>(&signal, multimemSignalCount),
      .dataBufferSize = 64,
      .pipelineDepth = 1,
      .maxGroups = 1,
      .signalsPerLane = 12,
      .nvlRank = 0,
      .nvlRanks = 4,
  };
  switch (testCase) {
    case StageLayoutTrapCase::ZeroGeometry:
      transport.pipelineDepth = 0;
      break;
    case StageLayoutTrapCase::TooManyGroups:
      break;
    case StageLayoutTrapCase::BadSignalsPerLane:
      transport.signalsPerLane = 11;
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
