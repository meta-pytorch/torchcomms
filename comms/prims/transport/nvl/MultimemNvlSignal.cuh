// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cstdint>
#include <limits>

#include "comms/common/fault_tolerance/AbortMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

namespace comms::prims {

/** Selects the virtual-address path used to publish a signal. */
enum class NvlSignalAccess { Unicast, Multimem };

/** Selects peer-owned slots or lane-owned aggregate counters. */
enum class NvlSignalTopology { PerPeer, Aggregate };

/** Selects one stripe from the transport's internal signal layout. */
enum class NvlSignalPhase { Ready, Ack, Consumed };

/** Selects how a per-peer waiter combines peer completion state. */
enum class NvlPerPeerWaitPolicy { WaitAll, SerialMin, TreeMin, ButterflyMin };

inline constexpr uint32_t kMaxNvlSignalRanks = 72;
inline constexpr uint32_t kNvlSignalRanksPerMaskWord =
    std::numeric_limits<uint64_t>::digits;
inline constexpr uint32_t kNvlSignalSmallPerPeerThreads =
    kNvlSignalRanksPerMaskWord;
inline constexpr uint32_t kNvlSignalLargePerPeerThreads =
    2 * kNvlSignalSmallPerPeerThreads;

__host__ __device__ constexpr bool nvl_signal_rank_count_supported(
    uint32_t ranks) {
  return ranks > 0 && ranks <= kMaxNvlSignalRanks;
}

/**
 * Returns a launchable one-dimensional per-peer CUDA block size.
 *
 * Rank validity is checked separately so an invalid count still reaches
 * device-side protocol validation instead of producing a zero-thread launch.
 */
__host__ __device__ constexpr uint32_t nvl_signal_per_peer_group_size(
    uint32_t ranks) {
  return ranks <= kNvlSignalSmallPerPeerThreads ? kNvlSignalSmallPerPeerThreads
                                                : kNvlSignalLargePerPeerThreads;
}

inline constexpr uint32_t kMaxNvlSignalPerPeerThreads =
    nvl_signal_per_peer_group_size(kMaxNvlSignalRanks);
inline constexpr uint32_t kMaxNvlSignalPerPeerWarps =
    kMaxNvlSignalPerPeerThreads / kWarpSize;

static_assert(kMaxNvlSignalPerPeerThreads % kWarpSize == 0);

/** Identifies one channel and one monotonic per-peer round value. */
struct StageRound {
  uint32_t channel;
  uint64_t value;
};

/** Selects up to 72 NVL-local ranks; ranks 64-71 use the high word. */
struct NvlSignalRankMask {
  uint64_t low{0};
  uint64_t high{0};

  __host__ __device__ constexpr NvlSignalRankMask() = default;
  __host__ __device__ constexpr NvlSignalRankMask(uint64_t lowBits)
      : low(lowBits) {}
  __host__ __device__ constexpr NvlSignalRankMask(
      uint64_t lowBits,
      uint64_t highBits)
      : low(lowBits), high(highBits) {}

  __host__ __device__ static constexpr NvlSignalRankMask first(uint32_t ranks) {
    if (!nvl_signal_rank_count_supported(ranks)) {
      return {};
    }
    if (ranks < kNvlSignalRanksPerMaskWord) {
      return {(uint64_t{1} << ranks) - 1};
    }
    if (ranks == kNvlSignalRanksPerMaskWord) {
      return {~uint64_t{0}};
    }
    return {
        ~uint64_t{0},
        (uint64_t{1} << (ranks - kNvlSignalRanksPerMaskWord)) - 1};
  }

  __host__ __device__ static constexpr NvlSignalRankMask single(uint32_t rank) {
    if (rank < kNvlSignalRanksPerMaskWord) {
      return {uint64_t{1} << rank};
    }
    if (rank < kMaxNvlSignalRanks) {
      return {0, uint64_t{1} << (rank - kNvlSignalRanksPerMaskWord)};
    }
    return {};
  }

  __host__ __device__ constexpr NvlSignalRankMask without(uint32_t rank) const {
    auto result = *this;
    if (rank < kNvlSignalRanksPerMaskWord) {
      result.low &= ~(uint64_t{1} << rank);
    } else if (rank < kMaxNvlSignalRanks) {
      result.high &= ~(uint64_t{1} << (rank - kNvlSignalRanksPerMaskWord));
    }
    return result;
  }
};

/** Describes the ranks participating in one signal operation. */
struct NvlSignalParticipants {
  NvlSignalRankMask publisherMask{};
  NvlSignalRankMask waiterMask{};
  uint32_t expectedArrivals{0};
};

namespace nvl_signal_detail {

__device__ __forceinline__ bool rank_selected(
    const NvlSignalRankMask& mask,
    int rank) {
  if (rank < 0 || rank >= static_cast<int>(kMaxNvlSignalRanks)) {
    return false;
  }
  const uint64_t word = rank < static_cast<int>(kNvlSignalRanksPerMaskWord)
      ? mask.low
      : mask.high;
  return ((word >> (rank % kNvlSignalRanksPerMaskWord)) & uint64_t{1}) != 0;
}

__device__ __forceinline__ bool mask_is_empty(const NvlSignalRankMask& mask) {
  return mask.low == 0 && mask.high == 0;
}

__device__ __forceinline__ bool mask_fits(
    const NvlSignalRankMask& mask,
    const NvlSignalRankMask& valid) {
  return (mask.low & ~valid.low) == 0 && (mask.high & ~valid.high) == 0;
}

__device__ __forceinline__ uint32_t
mask_rank_count(const NvlSignalRankMask& mask) {
#if defined(__CUDA_ARCH__)
  return static_cast<uint32_t>(__popcll(mask.low) + __popcll(mask.high));
#else
  (void)mask;
  return 0;
#endif
}

__device__ __forceinline__ bool sequence_reached(
    uint64_t observed,
    uint64_t expected) {
  // Sequence values must remain within half of the uint64_t range so modular
  // ordering distinguishes an advanced value across wraparound.
  return observed - expected < (uint64_t{1} << 63);
}

template <
    NvlSignalAccess access,
    NvlSignalTopology topology,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy>
__device__ __forceinline__ void validate_protocol() {
  static_assert(
      access == NvlSignalAccess::Unicast ||
      access == NvlSignalAccess::Multimem);
  static_assert(
      topology == NvlSignalTopology::PerPeer ||
      topology == NvlSignalTopology::Aggregate);
  static_assert(
      phase == NvlSignalPhase::Ready || phase == NvlSignalPhase::Ack ||
      phase == NvlSignalPhase::Consumed);
  static_assert(
      waitPolicy == NvlPerPeerWaitPolicy::WaitAll ||
      waitPolicy == NvlPerPeerWaitPolicy::SerialMin ||
      waitPolicy == NvlPerPeerWaitPolicy::TreeMin ||
      waitPolicy == NvlPerPeerWaitPolicy::ButterflyMin);
  static_assert(
      topology == NvlSignalTopology::PerPeer ||
      waitPolicy == NvlPerPeerWaitPolicy::WaitAll);
  static_assert(
      topology == NvlSignalTopology::PerPeer ||
      phase != NvlSignalPhase::Consumed);
}

__device__ __forceinline__ void wait_until_reached(
    const SignalState& signal,
    uint64_t expected,
    const AbortDevice& abortDevice) {
  while (!sequence_reached(signal.load(), expected)) {
    if (FT_ABORT_CHECK(
            abortDevice,
            "NVL signal wait for sequence=%llu",
            static_cast<unsigned long long>(expected))) {
      FT_DEVICE_TRAP();
    }
  }
}

template <NvlSignalPhase phase>
__device__ __forceinline__ uint64_t peer_signal_id(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    int sourceRank) {
  const uint64_t channelBase =
      static_cast<uint64_t>(round.channel) * transport.signalsPerChannel;
  uint64_t stripe = 0;
  if constexpr (phase == NvlSignalPhase::Ack) {
    stripe = static_cast<uint64_t>(transport.nvlRanks);
  } else if constexpr (phase == NvlSignalPhase::Consumed) {
    stripe = static_cast<uint64_t>(2) * transport.nvlRanks;
  }
  return channelBase + stripe + static_cast<uint64_t>(sourceRank);
}

template <NvlSignalPhase phase>
__device__ __forceinline__ uint64_t aggregate_counter_id(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    uint32_t lane) {
  static_assert(
      phase != NvlSignalPhase::Consumed,
      "aggregate consumed signaling has no allocated counter");
  const uint64_t channelBase =
      static_cast<uint64_t>(round.channel) * transport.signalsPerChannel;
  const uint64_t laneBase = channelBase +
      static_cast<uint64_t>(3) * static_cast<uint64_t>(transport.nvlRanks) +
      static_cast<uint64_t>(lane) * detail::kMultimemSignalsPerLane;
  if constexpr (phase == NvlSignalPhase::Ready) {
    return laneBase;
  } else {
    return laneBase + 2;
  }
}

__device__ __forceinline__ void validate_common(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants) {
#if defined(__CUDA_ARCH__)
  const bool validRankCount = transport.nvlRanks > 0 &&
      nvl_signal_rank_count_supported(
                                  static_cast<uint32_t>(transport.nvlRanks));
  const auto validRankMask = NvlSignalRankMask::first(
      validRankCount ? static_cast<uint32_t>(transport.nvlRanks) : 0);
  const uint64_t expectedSignalsPerChannel = validRankCount
      ? multimem_staging_signals_per_channel(
            static_cast<uint64_t>(transport.nvlRanks), transport.pipelineDepth)
      : 0;
  if (!validRankCount || transport.nvlRank < 0 ||
      transport.nvlRank >= transport.nvlRanks ||
      mask_is_empty(participants.publisherMask) ||
      mask_is_empty(participants.waiterMask) ||
      !mask_fits(participants.publisherMask, validRankMask) ||
      !mask_fits(participants.waiterMask, validRankMask) ||
      participants.expectedArrivals == 0 ||
      participants.expectedArrivals !=
          mask_rank_count(participants.publisherMask) ||
      round.channel >= transport.maxChannels || round.value == 0 ||
      transport.signalsPerChannel != expectedSignalsPerChannel) {
    printf(
        "NVL signal invalid geometry: rank=%d ranks=%d channel=%u "
        "round=%llu maxChannels=%u publishers=(%llu,%llu) "
        "waiters=(%llu,%llu) arrivals=%u\n",
        transport.nvlRank,
        transport.nvlRanks,
        static_cast<unsigned>(round.channel),
        static_cast<unsigned long long>(round.value),
        static_cast<unsigned>(transport.maxChannels),
        static_cast<unsigned long long>(participants.publisherMask.low),
        static_cast<unsigned long long>(participants.publisherMask.high),
        static_cast<unsigned long long>(participants.waiterMask.low),
        static_cast<unsigned long long>(participants.waiterMask.high),
        static_cast<unsigned>(participants.expectedArrivals));
    __trap();
  }
#else
  (void)transport;
  (void)round;
  (void)participants;
#endif
}

template <NvlSignalAccess access>
__device__ __forceinline__ void validate_per_peer(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__)
  const uint64_t requiredSignals =
      static_cast<uint64_t>(transport.maxChannels) *
      transport.signalsPerChannel;
  const uint32_t requiredGroupSize =
      nvl_signal_per_peer_group_size(static_cast<uint32_t>(transport.nvlRanks));
  if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
      blockDim.x != requiredGroupSize || group.scope != SyncScope::BLOCK ||
      group.group_size != requiredGroupSize ||
      group.group_id != round.channel ||
      transport.nvlRanks > static_cast<int>(kMaxNvlSignalRanks) ||
      requiredSignals > transport.internalLocalSignals.size() ||
      requiredSignals > transport.internalMultimemSignals.size() ||
      (access == NvlSignalAccess::Unicast &&
       transport.internalUnicastSignalsByRank.size() !=
           static_cast<uint32_t>(transport.nvlRanks))) {
    printf(
        "NVL per-peer signal invalid execution geometry: block=(%u,%u,%u) "
        "grid=(%u,%u,%u) groupId=%u blockId=%u channel=%u groupSize=%u "
        "ranks=%d "
        "localSignals=%u multimemSignals=%u peers=%u\n",
        static_cast<unsigned>(blockDim.x),
        static_cast<unsigned>(blockDim.y),
        static_cast<unsigned>(blockDim.z),
        static_cast<unsigned>(gridDim.x),
        static_cast<unsigned>(gridDim.y),
        static_cast<unsigned>(gridDim.z),
        static_cast<unsigned>(group.group_id),
        static_cast<unsigned>(group.block_id),
        static_cast<unsigned>(round.channel),
        static_cast<unsigned>(group.group_size),
        transport.nvlRanks,
        static_cast<unsigned>(transport.internalLocalSignals.size()),
        static_cast<unsigned>(transport.internalMultimemSignals.size()),
        static_cast<unsigned>(transport.internalUnicastSignalsByRank.size()));
    __trap();
  }
#else
  (void)transport;
  (void)round;
  (void)group;
#endif
}

template <NvlSignalPhase phase>
__device__ __forceinline__ void wait_per_peer_all(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  const int source = static_cast<int>(group.thread_id_in_group);
  if (source < transport.nvlRanks &&
      rank_selected(participants.publisherMask, source)) {
    wait_until_reached(
        transport.internalLocalSignals[peer_signal_id<phase>(
            transport, round, source)],
        round.value,
        abortDevice);
  }
  group.sync();
}

template <NvlSignalPhase phase>
__device__ __forceinline__ void wait_per_peer_serial(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  if (group.is_leader()) {
    bool complete = false;
    while (!complete) {
      complete = true;
      for (int source = 0; source < transport.nvlRanks; ++source) {
        if (rank_selected(participants.publisherMask, source) &&
            !sequence_reached(
                transport
                    .internalLocalSignals[peer_signal_id<phase>(
                        transport, round, source)]
                    .load(),
                round.value)) {
          complete = false;
        }
      }
      if (!complete) {
        if (FT_ABORT_CHECK(
                abortDevice,
                "NVL serial per-peer wait for round=%llu",
                static_cast<unsigned long long>(round.value))) {
          FT_DEVICE_TRAP();
        }
      }
    }
  }
  group.sync();
}

template <NvlSignalPhase phase>
__device__ __forceinline__ void wait_per_peer_tree(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  __shared__ uint32_t completeByThread[kMaxNvlSignalPerPeerThreads];
  const uint32_t thread = group.thread_id_in_group;
  bool complete = false;
  while (!complete) {
    const int source = static_cast<int>(thread);
    completeByThread[thread] = source >= transport.nvlRanks ||
            !rank_selected(participants.publisherMask, source) ||
            sequence_reached(transport
                                 .internalLocalSignals[peer_signal_id<phase>(
                                     transport, round, source)]
                                 .load(),
                             round.value)
        ? 1
        : 0;
    group.sync();
    for (uint32_t stride = group.group_size / 2; stride != 0; stride /= 2) {
      if (thread < stride) {
        completeByThread[thread] &= completeByThread[thread + stride];
      }
      group.sync();
    }
    complete = completeByThread[0] != 0;
    if (!complete && group.is_leader()) {
      if (FT_ABORT_CHECK(
              abortDevice,
              "NVL tree per-peer wait for round=%llu",
              static_cast<unsigned long long>(round.value))) {
        FT_DEVICE_TRAP();
      }
    }
    group.sync();
  }
}

template <NvlSignalPhase phase>
__device__ __forceinline__ void wait_per_peer_butterfly(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  __shared__ uint32_t completeByWarp[kMaxNvlSignalPerPeerWarps];
  const uint32_t thread = group.thread_id_in_group;
  const uint32_t lane = thread % kWarpSize;
  const uint32_t warp = thread / kWarpSize;
  bool complete = false;
  while (!complete) {
    const int source = static_cast<int>(thread);
    uint32_t laneComplete = source >= transport.nvlRanks ||
            !rank_selected(participants.publisherMask, source) ||
            sequence_reached(transport
                                 .internalLocalSignals[peer_signal_id<phase>(
                                     transport, round, source)]
                                 .load(),
                             round.value)
        ? 1
        : 0;
#if defined(__CUDA_ARCH__)
    for (uint32_t delta = 1; delta < kWarpSize; delta *= 2) {
      laneComplete &= __shfl_xor_sync(~0u, laneComplete, delta);
    }
#endif
    if (lane == 0) {
      completeByWarp[warp] = laneComplete;
    }
    group.sync();
    complete = true;
    for (uint32_t index = 0; index < group.group_size / kWarpSize; ++index) {
      complete &= completeByWarp[index] != 0;
    }
    if (!complete && group.is_leader()) {
      if (FT_ABORT_CHECK(
              abortDevice,
              "NVL butterfly per-peer wait for round=%llu",
              static_cast<unsigned long long>(round.value))) {
        FT_DEVICE_TRAP();
      }
    }
    group.sync();
  }
}

template <NvlSignalAccess access>
__device__ __forceinline__ void validate_aggregate(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__)
  const uint64_t requiredSignals =
      static_cast<uint64_t>(transport.maxChannels) *
      transport.signalsPerChannel;
  const uint32_t threadInBlock = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
  const uint32_t warpInBlock = threadInBlock / kWarpSize;
  if (blockDim.x < kWarpSize || blockDim.x % kWarpSize != 0 ||
      blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
      group.scope != SyncScope::WARP || group.group_size != kWarpSize ||
      group.group_id != round.channel || transport.pipelineDepth == 0 ||
      transport.pipelineDepth > kWarpSize ||
      requiredSignals > transport.internalLocalSignals.size() ||
      requiredSignals > transport.internalMultimemSignals.size() ||
      (access == NvlSignalAccess::Unicast &&
       transport.internalUnicastSignalsByRank.size() !=
           static_cast<uint32_t>(transport.nvlRanks))) {
    printf(
        "NVL aggregate signal invalid execution geometry: block=(%u,%u,%u) "
        "grid=(%u,%u,%u) groupId=%u blockId=%u channel=%u warp=%u "
        "groupSize=%u "
        "pipelineDepth=%u localSignals=%u multimemSignals=%u peers=%u\n",
        static_cast<unsigned>(blockDim.x),
        static_cast<unsigned>(blockDim.y),
        static_cast<unsigned>(blockDim.z),
        static_cast<unsigned>(gridDim.x),
        static_cast<unsigned>(gridDim.y),
        static_cast<unsigned>(gridDim.z),
        static_cast<unsigned>(group.group_id),
        static_cast<unsigned>(group.block_id),
        static_cast<unsigned>(round.channel),
        static_cast<unsigned>(warpInBlock),
        static_cast<unsigned>(group.group_size),
        static_cast<unsigned>(transport.pipelineDepth),
        static_cast<unsigned>(transport.internalLocalSignals.size()),
        static_cast<unsigned>(transport.internalMultimemSignals.size()),
        static_cast<unsigned>(transport.internalUnicastSignalsByRank.size()));
    __trap();
  }
#else
  (void)transport;
  (void)round;
  (void)group;
#endif
}

__device__ __forceinline__ void validate_block_barrier(
    const MultimemNvlTransportDevice& transport,
    uint32_t channel,
    const ThreadGroup& block) {
#if defined(__CUDA_ARCH__)
  const bool validRankCount = transport.nvlRanks > 0 &&
      nvl_signal_rank_count_supported(
                                  static_cast<uint32_t>(transport.nvlRanks));
  const uint64_t expectedSignalsPerChannel = validRankCount
      ? multimem_staging_signals_per_channel(
            static_cast<uint64_t>(transport.nvlRanks), transport.pipelineDepth)
      : 0;
  const uint64_t requiredSignals =
      static_cast<uint64_t>(transport.maxChannels) *
      transport.signalsPerChannel;
  if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
      block.scope != SyncScope::BLOCK || block.group_size != blockDim.x ||
      block.group_size < kWarpSize || block.group_size % kWarpSize != 0 ||
      block.group_id != channel || !validRankCount ||
      transport.pipelineDepth == 0 || channel >= transport.maxChannels ||
      transport.signalsPerChannel != expectedSignalsPerChannel ||
      requiredSignals > transport.internalLocalSignals.size() ||
      requiredSignals > transport.internalMultimemSignals.size()) {
    printf(
        "NVL block barrier invalid geometry: block=(%u,%u,%u) "
        "grid=(%u,%u,%u) groupId=%u groupSize=%u ranks=%d "
        "channel=%u maxChannels=%u pipelineDepth=%u "
        "signalsPerChannel=%u expectedSignalsPerChannel=%llu\n",
        static_cast<unsigned>(blockDim.x),
        static_cast<unsigned>(blockDim.y),
        static_cast<unsigned>(blockDim.z),
        static_cast<unsigned>(gridDim.x),
        static_cast<unsigned>(gridDim.y),
        static_cast<unsigned>(gridDim.z),
        static_cast<unsigned>(block.group_id),
        static_cast<unsigned>(block.group_size),
        transport.nvlRanks,
        static_cast<unsigned>(channel),
        static_cast<unsigned>(transport.maxChannels),
        static_cast<unsigned>(transport.pipelineDepth),
        static_cast<unsigned>(transport.signalsPerChannel),
        static_cast<unsigned long long>(expectedSignalsPerChannel));
    __trap();
  }
#else
  (void)transport;
  (void)channel;
  (void)block;
#endif
}

} // namespace nvl_signal_detail

/**
 * Publishes one signal operation from every selected publisher rank.
 *
 * Aggregate topology requires one warp and maps active threads to pipeline
 * lanes.
 * Per-peer topology requires one block sized by
 * `nvl_signal_per_peer_group_size()` and maps threads to peers.
 *
 * @param transport Exchanged multimem transport device view.
 * @param round Logical channel and per-peer round value.
 * @param participants Rank masks and expected arrival count.
 * @param group Topology-specific cooperative thread group.
 */
namespace nvl_signal_detail {

template <
    NvlSignalAccess access,
    NvlSignalTopology topology,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy>
__device__ __forceinline__ void signal_publish_impl(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group) {
  nvl_signal_detail::validate_protocol<access, topology, phase, waitPolicy>();
  nvl_signal_detail::validate_common(transport, round, participants);
  if constexpr (topology == NvlSignalTopology::Aggregate) {
    nvl_signal_detail::validate_aggregate<access>(transport, round, group);
  } else {
    nvl_signal_detail::validate_per_peer<access>(transport, round, group);
  }

  comms::device::fence_acq_rel_sys();
  group.sync();
  if (nvl_signal_detail::rank_selected(
          participants.publisherMask, transport.nvlRank)) {
    if constexpr (topology == NvlSignalTopology::Aggregate) {
      const uint32_t lane = group.thread_id_in_group;
      if (lane < transport.pipelineDepth) {
        const uint64_t signalId =
            nvl_signal_detail::aggregate_counter_id<phase>(
                transport, round, lane);
        if constexpr (access == NvlSignalAccess::Multimem) {
          transport
              .template signal_internal_scalar_prefenced<SignalOp::SIGNAL_ADD>(
                  signalId, 1);
        } else {
          for (int destination = 0; destination < transport.nvlRanks;
               ++destination) {
            if (nvl_signal_detail::rank_selected(
                    participants.waiterMask, destination)) {
              transport.internalUnicastSignalsByRank[destination][signalId]
                  .atomic_fetch_add(1);
            }
          }
        }
      }
    } else {
      const uint64_t signalId = nvl_signal_detail::peer_signal_id<phase>(
          transport, round, transport.nvlRank);
      if constexpr (access == NvlSignalAccess::Multimem) {
        if (group.is_leader()) {
          transport
              .template signal_internal_scalar_prefenced<SignalOp::SIGNAL_SET>(
                  signalId, round.value);
        }
      } else {
        const int destination = static_cast<int>(group.thread_id_in_group);
        if (destination < transport.nvlRanks &&
            nvl_signal_detail::rank_selected(
                participants.waiterMask, destination)) {
          transport.internalUnicastSignalsByRank[destination][signalId].store(
              round.value);
        }
      }
    }
  }
  group.sync();
}

} // namespace nvl_signal_detail

/** Publishes one split signal operation for a supported protocol. */
template <
    NvlSignalAccess access,
    NvlSignalTopology topology,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy = NvlPerPeerWaitPolicy::WaitAll>
__device__ __forceinline__ void signal_publish(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group) {
  static_assert(
      access != NvlSignalAccess::Multimem ||
          topology != NvlSignalTopology::Aggregate,
      "aggregate multimem requires signal_publish_and_wait on every rank");
  nvl_signal_detail::signal_publish_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group);
}

/**
 * Waits for the selected publishers on every selected waiter rank.
 *
 * A caller that enables abortDevice checking must call `AbortDevice::start()`
 * before this function.
 *
 * @param transport Exchanged multimem transport device view.
 * @param round Logical channel and per-peer round value.
 * @param participants Rank masks and expected arrival count.
 * @param group Topology-specific cooperative thread group.
 * @param abortDevice Started abortDevice or the disabled default abortDevice.
 */
namespace nvl_signal_detail {

template <
    NvlSignalAccess access,
    NvlSignalTopology topology,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy>
__device__ __forceinline__ void signal_wait_impl(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  nvl_signal_detail::validate_protocol<access, topology, phase, waitPolicy>();
  nvl_signal_detail::validate_common(transport, round, participants);
  if constexpr (topology == NvlSignalTopology::Aggregate) {
    nvl_signal_detail::validate_aggregate<access>(transport, round, group);
  } else {
    nvl_signal_detail::validate_per_peer<access>(transport, round, group);
  }

  const bool isWaiter = nvl_signal_detail::rank_selected(
      participants.waiterMask, transport.nvlRank);
  if constexpr (topology == NvlSignalTopology::Aggregate) {
    const uint32_t lane = group.thread_id_in_group;
    if (lane < transport.pipelineDepth) {
      const uint64_t signalId = nvl_signal_detail::aggregate_counter_id<phase>(
          transport, round, lane);
      auto* counter = transport.internalLocalSignals.data() + signalId;
      auto* epoch = counter + 1;
      const uint64_t expected =
          epoch->load() + static_cast<uint64_t>(participants.expectedArrivals);
      if (isWaiter) {
        nvl_signal_detail::wait_until_reached(*counter, expected, abortDevice);
      }
      if constexpr (access == NvlSignalAccess::Multimem) {
        epoch->store(expected);
      } else if (isWaiter) {
        epoch->store(expected);
      }
    }
    group.sync();
  } else if (!isWaiter) {
    return;
  } else if constexpr (waitPolicy == NvlPerPeerWaitPolicy::WaitAll) {
    nvl_signal_detail::wait_per_peer_all<phase>(
        transport, round, participants, group, abortDevice);
  } else if constexpr (waitPolicy == NvlPerPeerWaitPolicy::SerialMin) {
    nvl_signal_detail::wait_per_peer_serial<phase>(
        transport, round, participants, group, abortDevice);
  } else if constexpr (waitPolicy == NvlPerPeerWaitPolicy::TreeMin) {
    nvl_signal_detail::wait_per_peer_tree<phase>(
        transport, round, participants, group, abortDevice);
  } else {
    nvl_signal_detail::wait_per_peer_butterfly<phase>(
        transport, round, participants, group, abortDevice);
  }
  (void)access;
}

} // namespace nvl_signal_detail

/** Waits for one split signal operation for a supported protocol. */
template <
    NvlSignalAccess access,
    NvlSignalTopology topology,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy = NvlPerPeerWaitPolicy::WaitAll>
__device__ __forceinline__ void signal_wait(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  static_assert(
      access != NvlSignalAccess::Multimem ||
          topology != NvlSignalTopology::Aggregate,
      "aggregate multimem requires signal_publish_and_wait on every rank");
  nvl_signal_detail::signal_wait_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group, abortDevice);
}

/**
 * Publishes and then waits using the same compile-time protocol.
 *
 * @param transport Exchanged multimem transport device view.
 * @param round Logical channel and per-peer round value.
 * @param participants Rank masks and expected arrival count.
 * @param group Topology-specific cooperative thread group.
 * @param abortDevice Started abortDevice or the disabled default abortDevice.
 */
template <
    NvlSignalAccess access,
    NvlSignalTopology topology,
    NvlSignalPhase phase,
    NvlPerPeerWaitPolicy waitPolicy = NvlPerPeerWaitPolicy::WaitAll>
__device__ __forceinline__ void signal_publish_and_wait(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const AbortDevice& abortDevice) {
  nvl_signal_detail::signal_publish_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group);
  nvl_signal_detail::signal_wait_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group, abortDevice);
}

/**
 * Synchronizes one CUDA block with the same channel block on every NVL rank.
 *
 * Every NVL rank must enter this barrier for the same channel in the same
 * sequence. The barrier consumes the channel's aggregate Ready lane zero and
 * must not be called from rank-local or otherwise divergent control flow.
 *
 * The first block synchronization makes every thread's prior writes visible
 * to the leader. The leader publishes one release-add through lane zero of the
 * channel's aggregate counter, waits for every NVL rank, and advances the
 * local epoch. The final block synchronization makes the acquire wait visible
 * to the remaining threads.
 */
__device__ __forceinline__ void nvl_block_barrier(
    const MultimemNvlTransportDevice& transport,
    uint32_t channel,
    ThreadGroup& block,
    const AbortDevice& abortDevice = AbortDevice{}) {
  nvl_signal_detail::validate_block_barrier(transport, channel, block);

  comms::device::fence_acq_rel_sys();
  block.sync();
  if (block.is_leader()) {
    const StageRound round{.channel = channel, .value = 1};
    const uint64_t signalId =
        nvl_signal_detail::aggregate_counter_id<NvlSignalPhase::Ready>(
            transport, round, /*lane=*/0);
    auto* counter = transport.internalLocalSignals.data() + signalId;
    auto* epoch = counter + 1;
    const uint64_t expected =
        epoch->load() + static_cast<uint64_t>(transport.nvlRanks);
    transport.template signal_internal_scalar_prefenced<SignalOp::SIGNAL_ADD>(
        signalId, 1);
    nvl_signal_detail::wait_until_reached(*counter, expected, abortDevice);
    epoch->store(expected);
  }
  block.sync();
}

} // namespace comms::prims
