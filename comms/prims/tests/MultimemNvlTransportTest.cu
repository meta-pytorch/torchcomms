// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/MultimemNvlTransportTest.cuh"

#include <stdexcept>
#include <type_traits>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/transport/nvl/MultimemNvlReduce.cuh"
#include "comms/prims/transport/nvl/MultimemNvlSignal.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStageLayout.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStore.cuh"

namespace comms::prims::test {

namespace {

__global__ void setUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal(group, signalId, SignalOp::SIGNAL_SET, value);
}

__global__ void setInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal_internal(group, signalId, SignalOp::SIGNAL_SET, value);
}

__global__ void addUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal(group, signalId, SignalOp::SIGNAL_ADD, value);
}

__global__ void addInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value) {
  auto group = make_warp_group();
  transport.signal_internal(group, signalId, SignalOp::SIGNAL_ADD, value);
}

__global__ void waitAndReadUserSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out) {
  auto group = make_warp_group();
  transport.wait_signal_until(group, signalId, op, expected);
  if (group.is_leader()) {
    *out = transport.read_signal(signalId);
  }
}

__global__ void waitAndReadInternalSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out) {
  auto group = make_warp_group();
  transport.wait_internal_signal_until(group, signalId, op, expected);
  if (group.is_leader()) {
    *out = transport.read_internal_signal(signalId);
  }
}

__global__ void readUserAndInternalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = transport.read_signal(userId);
    out[1] = transport.read_internal_signal(internalId);
  }
}

__global__ void setAllPeerInternalSignalsKernel(
    MultimemNvlTransportDevice transport,
    uint64_t value) {
  for (auto destination = blockIdx.x * blockDim.x + threadIdx.x;
       destination < transport.nvlRanks;
       destination += blockDim.x * gridDim.x) {
    auto* destinationSignals =
        transport.internalUnicastSignalsByRank[destination];
    destinationSignals[transport.nvlRank].signal(SignalOp::SIGNAL_SET, value);
  }
}

__global__ void readPeerInternalSignalsKernel(
    MultimemNvlTransportDevice transport,
    uint64_t* out) {
  for (auto source = blockIdx.x * blockDim.x + threadIdx.x;
       source < transport.nvlRanks;
       source += blockDim.x * gridDim.x) {
    out[source] = transport.internalLocalSignals[source].load();
  }
}

__device__ NvlSignalParticipants
makeTestParticipants(const MultimemNvlTransportDevice& transport, bool fanIn) {
  const auto allRanks =
      NvlSignalRankMask::first(static_cast<uint32_t>(transport.nvlRanks));
  return NvlSignalParticipants{
      .publisherMask = fanIn ? allRanks.without(0) : allRanks,
      .waiterMask = fanIn ? uint64_t{1} : allRanks,
      .expectedArrivals = static_cast<uint32_t>(
          fanIn ? transport.nvlRanks - 1 : transport.nvlRanks),
  };
}

__device__ NvlSignalParticipants
makeAckParticipants(const MultimemNvlTransportDevice& transport) {
  const auto allRanks =
      NvlSignalRankMask::first(static_cast<uint32_t>(transport.nvlRanks));
  return NvlSignalParticipants{
      .publisherMask = 1,
      .waiterMask = allRanks,
      .expectedArrivals = 1,
  };
}

template <NvlSignalAccess access, NvlSignalPhase phase>
__global__ void aggregateSignalProtocolKernel(
    MultimemNvlTransportDevice transport,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out) {
  auto group = make_warp_group();
  const StageRound round{.channel = 0, .value = roundValue};
  signal_publish_and_wait<access, NvlSignalTopology::Aggregate, phase>(
      transport,
      round,
      makeTestParticipants(transport, fanIn),
      group,
      AbortDevice{});

  const uint32_t lane = group.thread_id_in_group;
  if (lane < transport.pipelineDepth) {
    const uint64_t phaseOffset =
        phase == NvlSignalPhase::Ready ? uint64_t{0} : uint64_t{2};
    const uint64_t laneBase =
        static_cast<uint64_t>(3 * transport.nvlRanks) + 4 * lane;
    out[lane] = transport.internalLocalSignals[laneBase + phaseOffset].load();
    out[transport.pipelineDepth + lane] =
        transport.internalLocalSignals[laneBase + phaseOffset + 1].load();
  }
}

template <
    NvlSignalAccess access,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy>
__global__ void perPeerSignalProtocolKernel(
    MultimemNvlTransportDevice transport,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out) {
  auto group = make_block_group();
  const StageRound round{.channel = 0, .value = roundValue};
  signal_publish_and_wait<
      access,
      NvlSignalTopology::PerPeer,
      phase,
      waitPolicy>(
      transport,
      round,
      makeTestParticipants(transport, fanIn),
      group,
      AbortDevice{});

  const int source = static_cast<int>(group.thread_id_in_group);
  if (source < transport.nvlRanks) {
    uint64_t stripe = 0;
    if constexpr (phase == NvlSignalPhase::Ack) {
      stripe = static_cast<uint64_t>(transport.nvlRanks);
    } else if constexpr (phase == NvlSignalPhase::Consumed) {
      stripe = static_cast<uint64_t>(2 * transport.nvlRanks);
    }
    out[source] = transport.internalLocalSignals[stripe + source].load();
  }
}

__global__ void aggregateAckSignalProtocolKernel(
    MultimemNvlTransportDevice transport,
    uint64_t roundValue,
    uint64_t* out) {
  auto group = make_warp_group();
  signal_publish_and_wait<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::Aggregate,
      NvlSignalPhase::Ack>(
      transport,
      StageRound{.channel = 0, .value = roundValue},
      makeAckParticipants(transport),
      group,
      AbortDevice{});

  const uint32_t lane = group.thread_id_in_group;
  if (lane < transport.pipelineDepth) {
    const uint64_t signalId =
        static_cast<uint64_t>(3 * transport.nvlRanks) + 4 * lane + 2;
    out[lane] = transport.internalLocalSignals[signalId].load();
    out[transport.pipelineDepth + lane] =
        transport.internalLocalSignals[signalId + 1].load();
  }
}

__global__ void multiChannelAggregateSignalKernel(
    MultimemNvlTransportDevice transport,
    uint64_t* out) {
  auto group = make_warp_group();
  const auto layout = multimem::make_stage_layout<uint64_t>(transport, group);
  const StageRound round{
      .channel = group.group_id,
      .value = 1,
  };
  signal_publish_and_wait<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::Aggregate,
      NvlSignalPhase::Ready>(
      transport,
      round,
      makeTestParticipants(transport, /*fanIn=*/false),
      group,
      AbortDevice{});
  const uint32_t lane = group.thread_id_in_group;
  if (lane < transport.pipelineDepth) {
    const uint64_t signalId = layout.signalBase +
        static_cast<uint64_t>(3 * transport.nvlRanks) + 4 * lane;
    const uint64_t outputBase =
        static_cast<uint64_t>(group.group_id) * 2 * transport.pipelineDepth;
    out[outputBase + lane] = transport.internalLocalSignals[signalId].load();
    out[outputBase + transport.pipelineDepth + lane] =
        transport.internalLocalSignals[signalId + 1].load();
  }
}

__global__ void aggregateMultimemWaiterTransitionKernel(
    MultimemNvlTransportDevice transport,
    uint64_t* out) {
  auto group = make_warp_group();
  const StageRound round{.channel = 0, .value = 1};
  signal_publish_and_wait<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::Aggregate,
      NvlSignalPhase::Ready>(
      transport,
      round,
      makeTestParticipants(transport, /*fanIn=*/true),
      group,
      AbortDevice{});

  if (transport.nvlRank == transport.nvlRanks - 1) {
    __nanosleep(100'000'000);
  }
  group.sync();
  signal_publish_and_wait<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::Aggregate,
      NvlSignalPhase::Ready>(
      transport,
      round,
      makeTestParticipants(transport, /*fanIn=*/false),
      group,
      AbortDevice{});

  const uint32_t lane = group.thread_id_in_group;
  if (lane < transport.pipelineDepth) {
    const uint64_t signalId =
        static_cast<uint64_t>(3 * transport.nvlRanks) + 4 * lane;
    out[lane] = transport.internalLocalSignals[signalId].load();
    out[transport.pipelineDepth + lane] =
        transport.internalLocalSignals[signalId + 1].load();
  }
}

__global__ void aggregateMultimemRelaxedPayloadKernel(
    MultimemNvlTransportDevice transport,
    uint64_t* observedPayload) {
  auto group = make_warp_group();
  const NvlSignalParticipants participants{
      .publisherMask = NvlSignalRankMask::single(1),
      .waiterMask = NvlSignalRankMask::single(0),
      .expectedArrivals = 1,
  };
  const uint64_t payloadId = static_cast<uint64_t>(2) * transport.nvlRanks;
  constexpr uint32_t kPayloadLane = kWarpSize - 1;
  auto* remotePayload =
      &transport.internalUnicastSignalsByRank[0][payloadId].signal_;
  const auto* localPayload = &transport.internalLocalSignals[payloadId].signal_;

  if (transport.nvlRank == 1 && group.thread_id_in_group == kPayloadLane) {
    comms::device::st_relaxed_sys_global(remotePayload, 11);
  }
  signal_publish_and_wait<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::Aggregate,
      NvlSignalPhase::Ready>(
      transport,
      StageRound{.channel = 0, .value = 1},
      participants,
      group,
      AbortDevice{});
  if (transport.nvlRank == 0 && group.thread_id_in_group == kPayloadLane) {
    *observedPayload = comms::device::ld_relaxed_sys_global(localPayload);
  }
}

__global__ void perPeerMultimemRelaxedPayloadKernel(
    MultimemNvlTransportDevice transport,
    uint64_t* observedPayload) {
  auto group = make_block_group();
  const NvlSignalParticipants participants{
      .publisherMask = NvlSignalRankMask::single(1),
      .waiterMask = NvlSignalRankMask::single(0),
      .expectedArrivals = 1,
  };
  const uint64_t payloadId = static_cast<uint64_t>(2) * transport.nvlRanks;
  const uint64_t waiterReadyId = payloadId + 1;
  constexpr uint32_t kPayloadThread = kWarpSize;
  auto* remotePayload =
      &transport.internalUnicastSignalsByRank[0][payloadId].signal_;
  const auto* localPayload = &transport.internalLocalSignals[payloadId].signal_;

  if (transport.nvlRank == 0 && group.is_leader()) {
    transport.internalUnicastSignalsByRank[1][waiterReadyId].store(1);
  }
  if (transport.nvlRank == 1) {
    if (group.is_leader()) {
      while (transport.internalLocalSignals[waiterReadyId].load() != 1) {
      }
    }
    group.sync();
  }
  if (transport.nvlRank == 1 && group.thread_id_in_group == kPayloadThread) {
    __nanosleep(100'000'000);
    comms::device::st_relaxed_sys_global(remotePayload, 22);
  }
  signal_publish_and_wait<
      NvlSignalAccess::Multimem,
      NvlSignalTopology::PerPeer,
      NvlSignalPhase::Ready>(
      transport,
      StageRound{.channel = 0, .value = 1},
      participants,
      group,
      AbortDevice{});
  if (transport.nvlRank == 0 && group.is_leader()) {
    *observedPayload = comms::device::ld_relaxed_sys_global(localPayload);
  }
}

__global__ void separatePublishAndWaitKernel(
    MultimemNvlTransportDevice transport,
    uint64_t roundValue,
    uint64_t* out) {
  auto group = make_block_group();
  const auto allRanks =
      NvlSignalRankMask::first(static_cast<uint32_t>(transport.nvlRanks));
  const NvlSignalParticipants participants{
      .publisherMask = allRanks.without(0),
      .waiterMask = 1,
      .expectedArrivals = static_cast<uint32_t>(transport.nvlRanks - 1),
  };
  const StageRound round{.channel = 0, .value = roundValue};
  if (transport.nvlRank == 0) {
    signal_wait<
        NvlSignalAccess::Unicast,
        NvlSignalTopology::PerPeer,
        NvlSignalPhase::Ready>(
        transport, round, participants, group, AbortDevice{});
  } else {
    signal_publish<
        NvlSignalAccess::Unicast,
        NvlSignalTopology::PerPeer,
        NvlSignalPhase::Ready>(transport, round, participants, group);
  }
  const int source = static_cast<int>(group.thread_id_in_group);
  if (transport.nvlRank == 0 && source < transport.nvlRanks) {
    out[source] = transport.internalLocalSignals[source].load();
  }
}

__global__ void initializeAggregateSignalsKernel(
    MultimemNvlTransportDevice transport,
    uint64_t counterValue,
    uint64_t epochValue) {
  const uint32_t lane = threadIdx.x;
  if (lane < transport.pipelineDepth) {
    const uint64_t laneBase = static_cast<uint64_t>(3) * transport.nvlRanks +
        static_cast<uint64_t>(lane) * 4;
    transport.internalLocalSignals[laneBase].store(counterValue);
    transport.internalLocalSignals[laneBase + 1].store(epochValue);
  }
}

__global__ void blockAggregateBarrierKernel(
    MultimemNvlTransportDevice transport,
    uint32_t epochs,
    int32_t* reducedValues,
    uint64_t* signalValues) {
  auto block = make_block_group();
  auto* local = reinterpret_cast<int32_t*>(transport.localData);
  const auto* multicast =
      reinterpret_cast<const int32_t*>(transport.multimemData);
  for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
    const std::size_t offset =
        (static_cast<std::size_t>(epoch) * gridDim.x + blockIdx.x) * blockDim.x;
    local[offset + threadIdx.x] = transport.nvlRank + 1 +
        static_cast<int32_t>(10 * epoch + 100 * blockIdx.x);
    nvl_block_barrier(
        transport, static_cast<uint32_t>(blockIdx.x), block, Timeout{});
    multimem::load_reduce_at<int32_t>(
        block, reducedValues + offset, multicast + offset, blockDim.x);
  }

  block.sync();
  if (threadIdx.x < transport.pipelineDepth) {
    const uint64_t signalId =
        static_cast<uint64_t>(blockIdx.x) * transport.signalsPerChannel +
        static_cast<uint64_t>(3 * transport.nvlRanks) +
        static_cast<uint64_t>(4 * threadIdx.x);
    const std::size_t outputBase =
        static_cast<std::size_t>(blockIdx.x) * 2 * transport.pipelineDepth;
    signalValues[outputBase + threadIdx.x] =
        transport.internalLocalSignals[signalId].load();
    signalValues[outputBase + transport.pipelineDepth + threadIdx.x] =
        transport.internalLocalSignals[signalId + 1].load();
  }
}

__global__ void perPeerWaitOnlyKernel(
    MultimemNvlTransportDevice transport,
    uint64_t roundValue,
    uint64_t* out) {
  auto group = make_block_group();
  const auto participants = makeTestParticipants(transport, /*fanIn=*/false);
  signal_wait<
      NvlSignalAccess::Unicast,
      NvlSignalTopology::PerPeer,
      NvlSignalPhase::Ready>(
      transport,
      StageRound{.channel = 0, .value = roundValue},
      participants,
      group,
      AbortDevice{});
  const int source = static_cast<int>(group.thread_id_in_group);
  if (source < transport.nvlRanks) {
    out[source] = transport.internalLocalSignals[source].load();
  }
}

template <NvlSignalAccess access, NvlSignalPhase phase>
void launchAggregateSignalProtocolTyped(
    MultimemNvlTransportDevice transport,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  aggregateSignalProtocolKernel<access, phase>
      <<<1, 32, 0, stream>>>(transport, fanIn, roundValue, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

template <
    NvlSignalAccess access,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy>
void launchPerPeerSignalProtocolTyped(
    MultimemNvlTransportDevice transport,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  perPeerSignalProtocolKernel<access, phase, waitPolicy>
      <<<1,
         nvl_signal_per_peer_group_size(
             static_cast<uint32_t>(transport.nvlRanks)),
         0,
         stream>>>(transport, fanIn, roundValue, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchMultimemReadyPerPeerSignalProtocolTyped(
    MultimemNvlTransportDevice transport,
    NvlPerPeerWaitPolicy waitPolicy,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  switch (waitPolicy) {
    case NvlPerPeerWaitPolicy::WaitAll:
      return launchPerPeerSignalProtocolTyped<
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::WaitAll>(
          transport, fanIn, roundValue, out, stream);
    case NvlPerPeerWaitPolicy::SerialMin:
      return launchPerPeerSignalProtocolTyped<
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::SerialMin>(
          transport, fanIn, roundValue, out, stream);
    case NvlPerPeerWaitPolicy::TreeMin:
      return launchPerPeerSignalProtocolTyped<
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::TreeMin>(
          transport, fanIn, roundValue, out, stream);
    case NvlPerPeerWaitPolicy::ButterflyMin:
      return launchPerPeerSignalProtocolTyped<
          NvlSignalAccess::Multimem,
          NvlSignalPhase::Ready,
          NvlPerPeerWaitPolicy::ButterflyMin>(
          transport, fanIn, roundValue, out, stream);
  }
}

template <typename T>
__device__ T reductionValue(float value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    return static_cast<T>(value);
  }
}

template <typename T>
__global__ void fillReductionInputKernel(
    MultimemNvlTransportDevice transport,
    float value,
    std::size_t elems,
    std::size_t sourceOffsetElems) {
  auto* source = reinterpret_cast<T*>(transport.localData) + sourceOffsetElems;
  for (std::size_t i = threadIdx.x; i < elems; i += blockDim.x) {
    source[i] = reductionValue<T>(value);
  }
}

template <typename T, bool kAccF32>
__global__ void loadReduceKernel(
    MultimemNvlTransportDevice transport,
    T* output,
    std::size_t elems,
    std::size_t sourceOffsetElems) {
  auto group = make_warp_group();
  const auto* source =
      reinterpret_cast<const T*>(transport.multimemData) + sourceOffsetElems;
  multimem::load_reduce_at<T, multimem::MultimemRedOp::Add, kAccF32>(
      group, output, source, elems);
}

template <int kUnroll>
__global__ void multimemStoreKernel(
    MultimemNvlTransportDevice transport,
    std::size_t destinationOffset,
    const void* source,
    std::size_t bytes) {
  auto group = make_warp_group();
  multimem::store<kUnroll>(
      group, transport.multimem_data_ptr(destinationOffset), source, bytes);
}

template <typename T, bool kAccF32>
__global__ void phasedReduceBlockKernel(
    MultimemNvlTransportDevice transport,
    T* output) {
  auto block = make_block_group();
  constexpr std::size_t kElements = sizeof(uint4) / sizeof(T);
  auto* local = reinterpret_cast<T*>(transport.localData);
  if (threadIdx.x < kElements) {
    local[threadIdx.x] = reductionValue<T>(transport.nvlRank + 1);
  }
  nvl_block_barrier(transport, /*channel=*/0, block);

  uint4 reduced{};
  if (block.is_leader()) {
    reduced = multimem::load_reduce_block16<T, kAccF32>(
        reinterpret_cast<const T*>(transport.multimemData));
  }
  nvl_block_barrier(transport, /*channel=*/0, block);

  if (block.is_leader()) {
    const std::size_t firstLane = kElements *
        static_cast<std::size_t>(transport.nvlRank) /
        static_cast<std::size_t>(transport.nvlRanks);
    const std::size_t endLane = kElements *
        static_cast<std::size_t>(transport.nvlRank + 1) /
        static_cast<std::size_t>(transport.nvlRanks);
    multimem::store_reduced_block16_range(
        output + firstLane, reduced, firstLane, endLane - firstLane);
  }
  nvl_block_barrier(transport, /*channel=*/0, block);
}

__global__ void stageLayoutKernel(
    MultimemNvlTransportDevice transport,
    StageLayoutResult* results) {
  auto group = make_block_group();
  const auto layout = multimem::make_stage_layout<uint32_t>(transport, group);
  if (group.is_leader()) {
    const int lastRank = layout.nvlRanks - 1;
    results[group.group_id] = StageLayoutResult{
        .channelBeginBytes = layout.channelBeginBytes,
        .stagingBytes = layout.stagingBytes,
        .signalBase = layout.signalBase,
        .signalsPerChannel = layout.signalsPerChannel,
        .readyFirst = multimem::ready_signal_id(layout, 0),
        .readyLast = multimem::ready_signal_id(layout, lastRank),
        .ackFirst = multimem::ack_signal_id(layout, 0),
        .ackLast = multimem::ack_signal_id(layout, lastRank),
        .consumedFirst = multimem::consumed_signal_id(layout, 0),
        .consumedLast = multimem::consumed_signal_id(layout, lastRank),
        .lane0ReadyCounter = multimem::ready_counter_signal_id(layout, 0),
        .lane0ReadyEpoch = multimem::ready_epoch_signal_id(layout, 0),
        .lane0AckCounter = multimem::ack_counter_signal_id(layout, 0),
        .lane0AckEpoch = multimem::ack_epoch_signal_id(layout, 0),
        .lane1ReadyCounter = multimem::ready_counter_signal_id(layout, 1),
        .lane1ReadyEpoch = multimem::ready_epoch_signal_id(layout, 1),
        .lane1AckCounter = multimem::ack_counter_signal_id(layout, 1),
        .lane1AckEpoch = multimem::ack_epoch_signal_id(layout, 1),
        .pipelineDepth = layout.pipelineDepth,
    };
  }
}

template <typename T>
void launchFillReductionInputTyped(
    MultimemNvlTransportDevice transport,
    float value,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  fillReductionInputKernel<T>
      <<<1, 32, 0, stream>>>(transport, value, elems, sourceOffsetElems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launchLoadReduceTyped(
    MultimemNvlTransportDevice transport,
    bool accF32,
    void* output,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  if (accF32) {
    loadReduceKernel<T, true><<<1, 32, 0, stream>>>(
        transport, static_cast<T*>(output), elems, sourceOffsetElems);
  } else {
    loadReduceKernel<T, false><<<1, 32, 0, stream>>>(
        transport, static_cast<T*>(output), elems, sourceOffsetElems);
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launchPhasedReduceBlockTyped(
    MultimemNvlTransportDevice transport,
    bool accF32,
    void* output,
    cudaStream_t stream) {
  if (accF32) {
    phasedReduceBlockKernel<T, true>
        <<<1, 128, 0, stream>>>(transport, static_cast<T*>(output));
  } else {
    phasedReduceBlockKernel<T, false>
        <<<1, 128, 0, stream>>>(transport, static_cast<T*>(output));
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace

void launchSetUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  setUserSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchSetInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  setInternalSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAddUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  addUserSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAddInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    uint64_t value,
    cudaStream_t stream) {
  addInternalSignalKernel<<<1, 32, 0, stream>>>(transport, signalId, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchWaitAndReadUserSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream) {
  waitAndReadUserSignalKernel<<<1, 32, 0, stream>>>(
      transport, signalId, op, expected, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchWaitAndReadInternalSignal(
    MultimemNvlTransportDevice transport,
    uint64_t signalId,
    CmpOp op,
    uint64_t expected,
    uint64_t* out,
    cudaStream_t stream) {
  waitAndReadInternalSignalKernel<<<1, 32, 0, stream>>>(
      transport, signalId, op, expected, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchReadUserAndInternal(
    MultimemNvlTransportDevice transport,
    uint64_t userId,
    uint64_t internalId,
    uint64_t* out,
    cudaStream_t stream) {
  readUserAndInternalKernel<<<1, 32, 0, stream>>>(
      transport, userId, internalId, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchSetAllPeerInternalSignals(
    MultimemNvlTransportDevice transport,
    uint64_t value,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (transport.nvlRanks + kThreads - 1) / kThreads;
  setAllPeerInternalSignalsKernel<<<blocks, kThreads, 0, stream>>>(
      transport, value);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchReadPeerInternalSignals(
    MultimemNvlTransportDevice transport,
    uint64_t* out,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (transport.nvlRanks + kThreads - 1) / kThreads;
  readPeerInternalSignalsKernel<<<blocks, kThreads, 0, stream>>>(
      transport, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAggregateSignalProtocol(
    MultimemNvlTransportDevice transport,
    NvlSignalAccess access,
    NvlSignalPhase phase,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  if (phase == NvlSignalPhase::Consumed) {
    throw std::runtime_error("aggregate consumed signal is unsupported");
  }
  if (access == NvlSignalAccess::Unicast) {
    if (phase == NvlSignalPhase::Ready) {
      return launchAggregateSignalProtocolTyped<
          NvlSignalAccess::Unicast,
          NvlSignalPhase::Ready>(transport, fanIn, roundValue, out, stream);
    }
    return launchAggregateSignalProtocolTyped<
        NvlSignalAccess::Unicast,
        NvlSignalPhase::Ack>(transport, fanIn, roundValue, out, stream);
  }
  if (phase == NvlSignalPhase::Ready) {
    return launchAggregateSignalProtocolTyped<
        NvlSignalAccess::Multimem,
        NvlSignalPhase::Ready>(transport, fanIn, roundValue, out, stream);
  }
  return launchAggregateSignalProtocolTyped<
      NvlSignalAccess::Multimem,
      NvlSignalPhase::Ack>(transport, fanIn, roundValue, out, stream);
}

void launchAggregateAckSignalProtocol(
    MultimemNvlTransportDevice transport,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  aggregateAckSignalProtocolKernel<<<1, 32, 0, stream>>>(
      transport, roundValue, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchPerPeerWaitAllSignalProtocol(
    MultimemNvlTransportDevice transport,
    NvlSignalAccess access,
    NvlSignalPhase phase,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
#define LAUNCH_PER_PEER_WAIT_ALL(ACCESS, PHASE) \
  return launchPerPeerSignalProtocolTyped<      \
      ACCESS,                                   \
      PHASE,                                    \
      NvlPerPeerWaitPolicy::WaitAll>(           \
      transport, fanIn, roundValue, out, stream)
  if (access == NvlSignalAccess::Unicast) {
    if (phase == NvlSignalPhase::Ready) {
      LAUNCH_PER_PEER_WAIT_ALL(NvlSignalAccess::Unicast, NvlSignalPhase::Ready);
    }
    if (phase == NvlSignalPhase::Ack) {
      LAUNCH_PER_PEER_WAIT_ALL(NvlSignalAccess::Unicast, NvlSignalPhase::Ack);
    }
    LAUNCH_PER_PEER_WAIT_ALL(
        NvlSignalAccess::Unicast, NvlSignalPhase::Consumed);
  }
  if (phase == NvlSignalPhase::Ready) {
    LAUNCH_PER_PEER_WAIT_ALL(NvlSignalAccess::Multimem, NvlSignalPhase::Ready);
  }
  if (phase == NvlSignalPhase::Ack) {
    LAUNCH_PER_PEER_WAIT_ALL(NvlSignalAccess::Multimem, NvlSignalPhase::Ack);
  }
  LAUNCH_PER_PEER_WAIT_ALL(NvlSignalAccess::Multimem, NvlSignalPhase::Consumed);
#undef LAUNCH_PER_PEER_WAIT_ALL
}

void launchMultimemReadyPerPeerSignalProtocol(
    MultimemNvlTransportDevice transport,
    NvlPerPeerWaitPolicy waitPolicy,
    bool fanIn,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  launchMultimemReadyPerPeerSignalProtocolTyped(
      transport, waitPolicy, fanIn, roundValue, out, stream);
}

void launchMultiChannelAggregateSignal(
    MultimemNvlTransportDevice transport,
    uint32_t channels,
    uint64_t* out,
    cudaStream_t stream) {
  multiChannelAggregateSignalKernel<<<1, channels * kWarpSize, 0, stream>>>(
      transport, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAggregateMultimemWaiterTransition(
    MultimemNvlTransportDevice transport,
    uint64_t* out,
    cudaStream_t stream) {
  aggregateMultimemWaiterTransitionKernel<<<1, 32, 0, stream>>>(transport, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchAggregateMultimemRelaxedPayload(
    MultimemNvlTransportDevice transport,
    uint64_t* observedPayload,
    cudaStream_t stream) {
  aggregateMultimemRelaxedPayloadKernel<<<1, 32, 0, stream>>>(
      transport, observedPayload);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchPerPeerMultimemRelaxedPayload(
    MultimemNvlTransportDevice transport,
    uint64_t* observedPayload,
    cudaStream_t stream) {
  perPeerMultimemRelaxedPayloadKernel<<<
      1,
      nvl_signal_per_peer_group_size(static_cast<uint32_t>(transport.nvlRanks)),
      0,
      stream>>>(transport, observedPayload);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchSeparatePublishAndWait(
    MultimemNvlTransportDevice transport,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  separatePublishAndWaitKernel<<<
      1,
      nvl_signal_per_peer_group_size(static_cast<uint32_t>(transport.nvlRanks)),
      0,
      stream>>>(transport, roundValue, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchInitializeAggregateSignals(
    MultimemNvlTransportDevice transport,
    uint64_t counterValue,
    uint64_t epochValue,
    cudaStream_t stream) {
  initializeAggregateSignalsKernel<<<1, 32, 0, stream>>>(
      transport, counterValue, epochValue);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchBlockAggregateBarrier(
    MultimemNvlTransportDevice transport,
    uint32_t channels,
    uint32_t epochs,
    int32_t* reducedValues,
    uint64_t* signalValues,
    cudaStream_t stream) {
  blockAggregateBarrierKernel<<<channels, 128, 0, stream>>>(
      transport, epochs, reducedValues, signalValues);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchPerPeerWaitOnly(
    MultimemNvlTransportDevice transport,
    uint64_t roundValue,
    uint64_t* out,
    cudaStream_t stream) {
  perPeerWaitOnlyKernel<<<
      1,
      nvl_signal_per_peer_group_size(static_cast<uint32_t>(transport.nvlRanks)),
      0,
      stream>>>(transport, roundValue, out);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchFillReductionInput(
    MultimemNvlTransportDevice transport,
    MultimemReductionTestType type,
    float value,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  switch (type) {
    case MultimemReductionTestType::Float:
      return launchFillReductionInputTyped<float>(
          transport, value, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Int32:
      return launchFillReductionInputTyped<int32_t>(
          transport, value, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Float16:
      return launchFillReductionInputTyped<__half>(
          transport, value, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Bfloat16:
      return launchFillReductionInputTyped<__nv_bfloat16>(
          transport, value, elems, sourceOffsetElems, stream);
  }
}

void launchLoadReduce(
    MultimemNvlTransportDevice transport,
    MultimemReductionTestType type,
    bool accF32,
    void* output,
    std::size_t elems,
    std::size_t sourceOffsetElems,
    cudaStream_t stream) {
  switch (type) {
    case MultimemReductionTestType::Float:
      return launchLoadReduceTyped<float>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Int32:
      return launchLoadReduceTyped<int32_t>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Float16:
      return launchLoadReduceTyped<__half>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
    case MultimemReductionTestType::Bfloat16:
      return launchLoadReduceTyped<__nv_bfloat16>(
          transport, accF32, output, elems, sourceOffsetElems, stream);
  }
}

void launchMultimemStore(
    MultimemNvlTransportDevice transport,
    std::size_t destinationOffset,
    const void* source,
    std::size_t bytes,
    int unroll,
    cudaStream_t stream) {
  if (unroll == 1) {
    multimemStoreKernel<1>
        <<<1, 32, 0, stream>>>(transport, destinationOffset, source, bytes);
  } else if (unroll == 4) {
    multimemStoreKernel<4>
        <<<1, 32, 0, stream>>>(transport, destinationOffset, source, bytes);
  } else {
    throw std::invalid_argument("test supports unroll 1 or 4");
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

void launchPhasedReduceBlock(
    MultimemNvlTransportDevice transport,
    MultimemReductionTestType type,
    bool accF32,
    void* output,
    cudaStream_t stream) {
  switch (type) {
    case MultimemReductionTestType::Float16:
      return launchPhasedReduceBlockTyped<__half>(
          transport, accF32, output, stream);
    case MultimemReductionTestType::Bfloat16:
      return launchPhasedReduceBlockTyped<__nv_bfloat16>(
          transport, accF32, output, stream);
    case MultimemReductionTestType::Float:
    case MultimemReductionTestType::Int32:
      throw std::runtime_error("phased reduce block requires a 2-byte type");
  }
}

void launchStageLayout(
    MultimemNvlTransportDevice transport,
    StageLayoutResult* results,
    uint32_t numGroups,
    cudaStream_t stream) {
  stageLayoutKernel<<<numGroups, 32, 0, stream>>>(transport, results);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
