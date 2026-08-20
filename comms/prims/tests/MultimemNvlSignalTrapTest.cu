// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlSignalTrapTest.cuh"

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

__global__ void nvlSignalTrapKernel(NvlSignalTrapCase testCase) {
  auto* trapSignals = reinterpret_cast<SignalState*>(trapSignalStorage);
  SignalState* peerSignals[4] = {
      trapSignals, trapSignals, trapSignals, trapSignals};
  const bool tooManyRanks = testCase == NvlSignalTrapCase::RankCountTooLarge;
  const uint32_t pipelineDepth =
      testCase == NvlSignalTrapCase::AggregateDepthTooLarge ? 33 : 1;
  const int nvlRanks =
      tooManyRanks ? static_cast<int>(kMaxNvlSignalRanks + 1) : 4;
  uint32_t signalsPerChannel =
      static_cast<uint32_t>(3 * nvlRanks + 4 * pipelineDepth);
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
  if (testCase == NvlSignalTrapCase::WaitTimeout) {
    auto group = make_block_group();
    Timeout timeout{1};
    timeout.start();
    signal_wait<
        NvlSignalAccess::Unicast,
        NvlSignalTopology::PerPeer,
        NvlSignalPhase::Ready>(
        transport,
        round,
        NvlSignalParticipants{
            .publisherMask = uint64_t{1} << 1,
            .waiterMask = 1,
            .expectedArrivals = 1,
        },
        group,
        timeout);
    return;
  }
  if (testCase == NvlSignalTrapCase::AggregateDepthTooLarge ||
      testCase == NvlSignalTrapCase::ArrivalCountMismatch ||
      testCase == NvlSignalTrapCase::SignalsPerChannelMismatch ||
      testCase == NvlSignalTrapCase::DuplicateChannelOwner ||
      testCase == NvlSignalTrapCase::AggregatePartialWarp ||
      testCase == NvlSignalTrapCase::AggregateNon1DGrid) {
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

__global__ void nvlSignalRankBoundaryKernel(int nvlRanks, uint64_t* output) {
  auto group = make_block_group();
  auto* signals = reinterpret_cast<SignalState*>(boundarySignalStorage);
  for (int rank = static_cast<int>(threadIdx.x); rank < nvlRanks;
       rank += static_cast<int>(blockDim.x)) {
    boundaryPeerSignals[rank] = signals;
  }
  group.sync();

  const int selectedRank = nvlRanks - 1;
  const uint64_t roundValue = static_cast<uint64_t>(nvlRanks);
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
      NvlSignalPhase::Ready>(
      transport,
      StageRound{.channel = 0, .value = roundValue},
      NvlSignalParticipants{
          .publisherMask = publishers,
          .waiterMask = selected,
          .expectedArrivals = nvl_signal_detail::mask_rank_count(publishers),
      },
      group,
      Timeout{});
  if (group.is_leader()) {
    *output = signals[selectedRank].load();
  }
}

dim3 signal_trap_threads(NvlSignalTrapCase testCase) {
  switch (testCase) {
    case NvlSignalTrapCase::AggregatePartialWarp:
      return dim3(kWarpSize / 2);
    case NvlSignalTrapCase::DuplicateChannelOwner:
      return dim3(2 * kWarpSize);
    case NvlSignalTrapCase::PerPeerNon1DBlock:
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
      return dim3(2);
    case NvlSignalTrapCase::AggregateNon1DGrid:
      return dim3(1, 2);
    default:
      return dim3(1);
  }
}

} // namespace

void launchNvlSignalTrap(NvlSignalTrapCase testCase) {
  nvlSignalTrapKernel<<<
      signal_trap_blocks(testCase),
      signal_trap_threads(testCase)>>>(
      testCase); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

void launchNvlSignalRankBoundary(int nvlRanks, uint64_t* output) {
  const uint32_t threads =
      nvl_signal_per_peer_group_size(static_cast<uint32_t>(nvlRanks));
  nvlSignalRankBoundaryKernel<<<1, threads>>>(
      nvlRanks, output); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

} // namespace comms::prims::test
