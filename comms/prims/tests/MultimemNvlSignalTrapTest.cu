// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlSignalTrapTest.cuh"

#include "comms/common/fault_tolerance/TestAbort.h"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/nvl/MultimemNvlSignal.cuh"

namespace comms::prims::test {
namespace {

__device__ alignas(SignalState) uint64_t
    trapSignalStorage[16 * sizeof(SignalState) / sizeof(uint64_t)];

__global__ void nvlSignalTrapKernel(
    NvlSignalTrapCase testCase,
    Timeout waitAbort) {
  auto* trapSignals = reinterpret_cast<SignalState*>(trapSignalStorage);
  SignalState* peerSignals[4] = {
      trapSignals, trapSignals, trapSignals, trapSignals};
  const bool tooManyRanks = testCase == NvlSignalTrapCase::RankCountTooLarge;
  const uint32_t pipelineDepth =
      testCase == NvlSignalTrapCase::AggregateDepthTooLarge ? 33 : 1;
  const int nvlRanks = tooManyRanks ? 65 : 4;
  const uint32_t signalsPerChannel =
      static_cast<uint32_t>(3 * nvlRanks + 4 * pipelineDepth);
  MultimemNvlTransportDevice transport{
      .internalLocalSignals =
          DeviceSpan<SignalState>(trapSignals, signalsPerChannel),
      .internalMultimemSignals =
          DeviceSpan<SignalState>(trapSignals, signalsPerChannel),
      .nvlRank = 0,
      .nvlRanks = nvlRanks,
      .pipelineDepth = pipelineDepth,
      .maxChannels = 1,
      .signalsPerChannel = signalsPerChannel,
      .internalUnicastSignalsByRank = DeviceSpan<SignalState*>(peerSignals, 4),
  };
  const NvlSignalParticipants participants{
      .publisherMask = 1,
      .waiterMask = 1,
      .expectedArrivals =
          testCase == NvlSignalTrapCase::ArrivalCountMismatch ? 2u : 1u,
  };
  const StageRound round{
      .channel = 0,
      .value = testCase == NvlSignalTrapCase::ZeroRound ? 0u : 1u,
  };
  if (testCase == NvlSignalTrapCase::WaitTimeout ||
      testCase == NvlSignalTrapCase::SerialMinWaitTimeout ||
      testCase == NvlSignalTrapCase::TreeMinWaitTimeout ||
      testCase == NvlSignalTrapCase::ButterflyMinWaitTimeout) {
    auto group = make_block_group();
    waitAbort.start();
    const NvlSignalParticipants waitParticipants{
        .publisherMask = uint64_t{1} << 1,
        .waiterMask = 1,
        .expectedArrivals = 1,
    };
    if (testCase == NvlSignalTrapCase::WaitTimeout) {
      signal_wait<
          NvlSignalAccess::Unicast,
          NvlSignalTopology::PerPeer,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::WaitAll>(
          transport, round, waitParticipants, group, waitAbort);
    } else if (testCase == NvlSignalTrapCase::SerialMinWaitTimeout) {
      signal_wait<
          NvlSignalAccess::Unicast,
          NvlSignalTopology::PerPeer,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::SerialMin>(
          transport, round, waitParticipants, group, waitAbort);
    } else if (testCase == NvlSignalTrapCase::TreeMinWaitTimeout) {
      signal_wait<
          NvlSignalAccess::Unicast,
          NvlSignalTopology::PerPeer,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::TreeMin>(
          transport, round, waitParticipants, group, waitAbort);
    } else {
      signal_wait<
          NvlSignalAccess::Unicast,
          NvlSignalTopology::PerPeer,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::ButterflyMin>(
          transport, round, waitParticipants, group, waitAbort);
    }
    return;
  }
  if (testCase == NvlSignalTrapCase::AggregateDepthTooLarge ||
      testCase == NvlSignalTrapCase::ArrivalCountMismatch) {
    auto group = make_warp_group();
    signal_publish_and_wait<
        NvlSignalAccess::Multimem,
        NvlSignalTopology::Aggregate,
        NvlSignalPhase::Ready>(
        transport, round, participants, group, Timeout{});
    return;
  }
  auto group = make_block_group();
  signal_publish<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::PerPeer,
      NvlSignalPhase::Ready>(transport, round, participants, group);
}

} // namespace

void launchNvlSignalTrap(NvlSignalTrapCase testCase) {
  const uint32_t threads =
      testCase == NvlSignalTrapCase::PerPeerGroupTooSmall ? 32 : 64;
  auto waitAbort = comms::fault_tolerance::testing::testAbortDevice();
  waitAbort.setOpTimeoutMs(1);
  nvlSignalTrapKernel<<<1, threads>>>(
      testCase,
      waitAbort); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

} // namespace comms::prims::test
