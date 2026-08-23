// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlSignalTrapTest.cuh"

#include "comms/common/fault_tolerance/TestAbort.h"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/nvl/MultimemNvlSignal.cuh"

namespace comms::prims::test {
namespace {

constexpr auto kZeroRankMask = NvlSignalRankMask::first(0);
constexpr auto kTooManyRankMask =
    NvlSignalRankMask::first(kMaxNvlSignalRanks + 1);
static_assert(kZeroRankMask.low == 0 && kZeroRankMask.high == 0);
static_assert(kTooManyRankMask.low == 0 && kTooManyRankMask.high == 0);
static_assert(
    nvl_signal_per_peer_group_size(0) == kNvlSignalSmallPerPeerThreads);
static_assert(
    nvl_signal_per_peer_group_size(kMaxNvlSignalRanks + 1) ==
    kNvlSignalLargePerPeerThreads);

__device__ alignas(SignalState) uint64_t
    trapSignalStorage[16 * sizeof(SignalState) / sizeof(uint64_t)];

constexpr int kBoundaryMaxRanks = static_cast<int>(kMaxNvlSignalRanks);
constexpr uint32_t kBoundaryPipelineDepth = 1;
constexpr uint32_t kBoundarySignalCount =
    static_cast<uint32_t>(multimem_staging_signals_per_channel(
        kBoundaryMaxRanks,
        kBoundaryPipelineDepth));
__device__ alignas(SignalState) uint64_t boundarySignalStorage
    [kBoundarySignalCount * sizeof(SignalState) / sizeof(uint64_t)];
__device__ SignalState* boundaryPeerSignals[kBoundaryMaxRanks];

__global__ void nvlSignalTrapKernel(
    NvlSignalTrapCase testCase,
    AbortDevice waitAbort) {
  auto* trapSignals = reinterpret_cast<SignalState*>(trapSignalStorage);
  SignalState* peerSignals[4] = {
      trapSignals, trapSignals, trapSignals, trapSignals};
  const bool tooManyRanks = testCase == NvlSignalTrapCase::RankCountTooLarge;
  const uint32_t pipelineDepth =
      testCase == NvlSignalTrapCase::AggregateDepthTooLarge ? 33 : 1;
  const int nvlRanks =
      tooManyRanks ? static_cast<int>(kMaxNvlSignalRanks + 1) : 4;
  uint32_t signalsPerChannel =
      static_cast<uint32_t>(multimem_staging_signals_per_channel(
          static_cast<uint64_t>(nvlRanks), pipelineDepth));
  if (testCase == NvlSignalTrapCase::SignalsPerChannelMismatch) {
    --signalsPerChannel;
  }
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
  if (testCase == NvlSignalTrapCase::DuplicateChannelOwner ||
      testCase == NvlSignalTrapCase::AggregatePartialWarp ||
      testCase == NvlSignalTrapCase::AggregateNon1DGrid) {
    auto group = make_warp_group();
    signal_publish_and_wait<
        NvlSignalAccess::Unicast,
        NvlSignalTopology::Aggregate,
        NvlSignalPhase::Ready>(
        transport, round, participants, group, AbortDevice{});
    return;
  }
  if (testCase == NvlSignalTrapCase::AggregateDepthTooLarge ||
      testCase == NvlSignalTrapCase::ArrivalCountMismatch ||
      testCase == NvlSignalTrapCase::SignalsPerChannelMismatch) {
    auto group = make_warp_group();
    signal_publish_and_wait<
        NvlSignalAccess::Multimem,
        NvlSignalTopology::Aggregate,
        NvlSignalPhase::Ready>(
        transport, round, participants, group, AbortDevice{});
    return;
  }
  auto group = make_block_group();
  if (testCase == NvlSignalTrapCase::BlockBarrierNon1DBlock ||
      testCase == NvlSignalTrapCase::BlockBarrierDuplicateChannelOwner ||
      testCase == NvlSignalTrapCase::BlockBarrierNon1DGrid) {
    nvl_signal_detail::validate_block_barrier(transport, /*channel=*/0, group);
    return;
  }
  if (testCase == NvlSignalTrapCase::PerPeerDuplicateChannelOwner ||
      testCase == NvlSignalTrapCase::PerPeerNon1DBlock) {
    signal_publish<
        NvlSignalAccess::Unicast,
        NvlSignalTopology::PerPeer,
        NvlSignalPhase::Ready>(transport, round, participants, group);
    return;
  }
  signal_publish<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::PerPeer,
      NvlSignalPhase::Ready>(transport, round, participants, group);
}

template <NvlPerPeerWaitPolicy waitPolicy>
__global__ void nvlSignalRankBoundaryKernel(
    int nvlRanks,
    uint64_t roundValue,
    uint64_t* output) {
  auto group = make_block_group();
  auto* signals = reinterpret_cast<SignalState*>(boundarySignalStorage);
  for (int rank = static_cast<int>(threadIdx.x); rank < nvlRanks;
       rank += static_cast<int>(blockDim.x)) {
    boundaryPeerSignals[rank] = signals;
  }
  group.sync();

  const int selectedRank = nvlRanks - 1;
  auto publishers =
      NvlSignalRankMask::single(static_cast<uint32_t>(selectedRank));
  publishers.low |= uint64_t{1};
  if (group.is_leader()) {
    signals[0].store(roundValue);
  }
  group.sync();
  const auto selected =
      NvlSignalRankMask::single(static_cast<uint32_t>(selectedRank));
  const uint32_t signalsPerChannel = static_cast<uint32_t>(
      multimem_staging_signals_per_channel(nvlRanks, kBoundaryPipelineDepth));
  MultimemNvlTransportDevice transport{
      .internalLocalSignals =
          DeviceSpan<SignalState>(signals, signalsPerChannel),
      .internalMultimemSignals =
          DeviceSpan<SignalState>(signals, signalsPerChannel),
      .nvlRank = selectedRank,
      .nvlRanks = nvlRanks,
      .pipelineDepth = kBoundaryPipelineDepth,
      .maxChannels = 1,
      .signalsPerChannel = signalsPerChannel,
      .internalUnicastSignalsByRank =
          DeviceSpan<SignalState*>(boundaryPeerSignals, nvlRanks),
  };
  signal_publish_and_wait<
      NvlSignalAccess::Unicast,
      NvlSignalTopology::PerPeer,
      NvlSignalPhase::Ready,
      waitPolicy>(
      transport,
      StageRound{.channel = 0, .value = roundValue},
      NvlSignalParticipants{
          .publisherMask = publishers,
          .waiterMask = selected,
          .expectedArrivals = nvl_signal_detail::mask_rank_count(publishers),
      },
      group,
      AbortDevice{});
  if (group.is_leader()) {
    *output = signals[selectedRank].load();
  }
}

template <NvlPerPeerWaitPolicy waitPolicy>
__global__ void nvlSignalUpperWordWaitTimeoutKernel(AbortDevice waitAbort) {
  auto group = make_block_group();
  auto* signals = reinterpret_cast<SignalState*>(boundarySignalStorage);
  for (int rank = static_cast<int>(threadIdx.x); rank < kBoundaryMaxRanks;
       rank += static_cast<int>(blockDim.x)) {
    boundaryPeerSignals[rank] = signals;
  }
  group.sync();

  constexpr int kNvlRanks = 65;
  constexpr int kUpperWordRank = 64;
  constexpr uint64_t kRoundValue = 1;
  constexpr auto kPublisherMask = NvlSignalRankMask::single(kUpperWordRank);
  constexpr auto kWaiterMask = NvlSignalRankMask::single(0);
  if (threadIdx.x < kUpperWordRank) {
    signals[threadIdx.x].store(kRoundValue);
  }
  group.sync();
  const uint32_t signalsPerChannel = static_cast<uint32_t>(
      multimem_staging_signals_per_channel(kNvlRanks, kBoundaryPipelineDepth));
  MultimemNvlTransportDevice transport{
      .internalLocalSignals =
          DeviceSpan<SignalState>(signals, signalsPerChannel),
      .internalMultimemSignals =
          DeviceSpan<SignalState>(signals, signalsPerChannel),
      .nvlRank = 0,
      .nvlRanks = kNvlRanks,
      .pipelineDepth = kBoundaryPipelineDepth,
      .maxChannels = 1,
      .signalsPerChannel = signalsPerChannel,
      .internalUnicastSignalsByRank =
          DeviceSpan<SignalState*>(boundaryPeerSignals, kNvlRanks),
  };
  waitAbort.start();
  signal_wait<
      NvlSignalAccess::Unicast,
      NvlSignalTopology::PerPeer,
      NvlSignalPhase::Ready,
      waitPolicy>(
      transport,
      StageRound{.channel = 0, .value = kRoundValue},
      NvlSignalParticipants{
          .publisherMask = kPublisherMask,
          .waiterMask = kWaiterMask,
          .expectedArrivals = 1,
      },
      group,
      waitAbort);
}

dim3 signal_trap_threads(NvlSignalTrapCase testCase) {
  switch (testCase) {
    case NvlSignalTrapCase::AggregatePartialWarp:
      return dim3(kWarpSize / 2);
    case NvlSignalTrapCase::DuplicateChannelOwner:
      return dim3(2 * kWarpSize);
    case NvlSignalTrapCase::PerPeerNon1DBlock:
    case NvlSignalTrapCase::BlockBarrierNon1DBlock:
      return dim3(kNvlSignalSmallPerPeerThreads, 2);
    case NvlSignalTrapCase::AggregateDepthTooLarge:
    case NvlSignalTrapCase::ArrivalCountMismatch:
    case NvlSignalTrapCase::SignalsPerChannelMismatch:
    case NvlSignalTrapCase::AggregateNon1DGrid:
    case NvlSignalTrapCase::PerPeerGroupTooSmall:
      return dim3(kWarpSize);
    default:
      return dim3(kNvlSignalSmallPerPeerThreads);
  }
}

dim3 signal_trap_blocks(NvlSignalTrapCase testCase) {
  switch (testCase) {
    case NvlSignalTrapCase::PerPeerDuplicateChannelOwner:
    case NvlSignalTrapCase::BlockBarrierDuplicateChannelOwner:
      return dim3(2);
    case NvlSignalTrapCase::AggregateNon1DGrid:
    case NvlSignalTrapCase::BlockBarrierNon1DGrid:
      return dim3(1, 2);
    default:
      return dim3(1);
  }
}

} // namespace

void launchNvlSignalTrap(NvlSignalTrapCase testCase) {
  auto waitAbort = comms::fault_tolerance::testing::testAbortDevice();
  waitAbort.setOpTimeoutMs(1);
  if (testCase == NvlSignalTrapCase::UpperWordWaitAllTimeout) {
    nvlSignalUpperWordWaitTimeoutKernel<NvlPerPeerWaitPolicy::WaitAll>
        <<<1, kNvlSignalLargePerPeerThreads>>>(
            waitAbort); // NOLINT(facebook-cuda-safe-kernel-call-check)
    return;
  }
  if (testCase == NvlSignalTrapCase::UpperWordSerialMinTimeout) {
    nvlSignalUpperWordWaitTimeoutKernel<NvlPerPeerWaitPolicy::SerialMin>
        <<<1, kNvlSignalLargePerPeerThreads>>>(
            waitAbort); // NOLINT(facebook-cuda-safe-kernel-call-check)
    return;
  }
  if (testCase == NvlSignalTrapCase::UpperWordTreeMinTimeout) {
    nvlSignalUpperWordWaitTimeoutKernel<NvlPerPeerWaitPolicy::TreeMin>
        <<<1, kNvlSignalLargePerPeerThreads>>>(
            waitAbort); // NOLINT(facebook-cuda-safe-kernel-call-check)
    return;
  }
  if (testCase == NvlSignalTrapCase::UpperWordButterflyMinTimeout) {
    nvlSignalUpperWordWaitTimeoutKernel<NvlPerPeerWaitPolicy::ButterflyMin>
        <<<1, kNvlSignalLargePerPeerThreads>>>(
            waitAbort); // NOLINT(facebook-cuda-safe-kernel-call-check)
    return;
  }
  nvlSignalTrapKernel<<<
      signal_trap_blocks(testCase),
      signal_trap_threads(testCase)>>>(
      testCase,
      waitAbort); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

void launchNvlSignalRankBoundary(
    int nvlRanks,
    NvlSignalRankBoundaryWaitPolicy waitPolicy,
    uint64_t roundValue,
    uint64_t* output) {
  const uint32_t threads =
      nvl_signal_per_peer_group_size(static_cast<uint32_t>(nvlRanks));
  switch (waitPolicy) {
    case NvlSignalRankBoundaryWaitPolicy::WaitAll:
      nvlSignalRankBoundaryKernel<NvlPerPeerWaitPolicy::WaitAll>
          <<<1, threads>>>(
              nvlRanks,
              roundValue,
              output); // NOLINT(facebook-cuda-safe-kernel-call-check)
      break;
    case NvlSignalRankBoundaryWaitPolicy::SerialMin:
      nvlSignalRankBoundaryKernel<NvlPerPeerWaitPolicy::SerialMin>
          <<<1, threads>>>(
              nvlRanks,
              roundValue,
              output); // NOLINT(facebook-cuda-safe-kernel-call-check)
      break;
    case NvlSignalRankBoundaryWaitPolicy::TreeMin:
      nvlSignalRankBoundaryKernel<NvlPerPeerWaitPolicy::TreeMin>
          <<<1, threads>>>(
              nvlRanks,
              roundValue,
              output); // NOLINT(facebook-cuda-safe-kernel-call-check)
      break;
    case NvlSignalRankBoundaryWaitPolicy::ButterflyMin:
      nvlSignalRankBoundaryKernel<NvlPerPeerWaitPolicy::ButterflyMin>
          <<<1, threads>>>(
              nvlRanks,
              roundValue,
              output); // NOLINT(facebook-cuda-safe-kernel-call-check)
      break;
  }
}

} // namespace comms::prims::test
