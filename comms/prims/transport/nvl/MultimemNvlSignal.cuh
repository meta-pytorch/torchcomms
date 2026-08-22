// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cstdint>

#include "comms/common/fault_tolerance/AbortMacros.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
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

/** Identifies one channel and one monotonic per-peer round value. */
struct StageRound {
  uint32_t channel;
  uint64_t value;
};

/** Describes the ranks participating in one signal operation. */
struct NvlSignalParticipants {
  uint64_t publisherMask{0};
  uint64_t waiterMask{0};
  uint32_t expectedArrivals{0};
};

namespace nvl_signal_detail {

__device__ __forceinline__ bool rank_selected(uint64_t mask, int rank) {
  return rank >= 0 && rank < 64 && ((mask >> rank) & uint64_t{1}) != 0;
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
    const Timeout& timeout) {
  while (!sequence_reached(signal.load(), expected)) {
    if (FT_ABORT_CHECK(
            timeout,
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
  const bool validRankCount =
      transport.nvlRanks > 0 && transport.nvlRanks <= 64;
  uint64_t validRankMask = ~uint64_t{0};
  if (validRankCount && transport.nvlRanks < 64) {
    validRankMask = (uint64_t{1} << transport.nvlRanks) - 1;
  }
  if (!validRankCount || transport.nvlRank < 0 ||
      transport.nvlRank >= transport.nvlRanks ||
      participants.publisherMask == 0 || participants.waiterMask == 0 ||
      (participants.publisherMask & ~validRankMask) != 0 ||
      (participants.waiterMask & ~validRankMask) != 0 ||
      participants.expectedArrivals == 0 ||
      participants.expectedArrivals !=
          static_cast<uint32_t>(__popcll(participants.publisherMask)) ||
      round.channel >= transport.maxChannels || round.value == 0 ||
      transport.signalsPerChannel == 0) {
    printf(
        "NVL signal invalid geometry: rank=%d ranks=%d channel=%u "
        "round=%llu maxChannels=%u publishers=%llu waiters=%llu arrivals=%u\n",
        transport.nvlRank,
        transport.nvlRanks,
        static_cast<unsigned>(round.channel),
        static_cast<unsigned long long>(round.value),
        static_cast<unsigned>(transport.maxChannels),
        static_cast<unsigned long long>(participants.publisherMask),
        static_cast<unsigned long long>(participants.waiterMask),
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
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__)
  const uint64_t requiredSignals =
      static_cast<uint64_t>(transport.maxChannels) *
      transport.signalsPerChannel;
  if (group.scope != SyncScope::BLOCK || group.group_size != 64 ||
      transport.nvlRanks > 64 ||
      requiredSignals > transport.internalLocalSignals.size() ||
      requiredSignals > transport.internalMultimemSignals.size() ||
      (access == NvlSignalAccess::Unicast &&
       transport.internalUnicastSignalsByRank.size() !=
           static_cast<uint32_t>(transport.nvlRanks))) {
    printf(
        "NVL per-peer signal invalid execution geometry: groupSize=%u "
        "ranks=%d localSignals=%u multimemSignals=%u peers=%u\n",
        static_cast<unsigned>(group.group_size),
        transport.nvlRanks,
        static_cast<unsigned>(transport.internalLocalSignals.size()),
        static_cast<unsigned>(transport.internalMultimemSignals.size()),
        static_cast<unsigned>(transport.internalUnicastSignalsByRank.size()));
    __trap();
  }
#else
  (void)transport;
  (void)group;
#endif
}

template <NvlSignalPhase phase>
__device__ __forceinline__ void wait_per_peer_all(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const Timeout& timeout) {
  const int source = static_cast<int>(group.thread_id_in_group);
  if (source < transport.nvlRanks &&
      rank_selected(participants.publisherMask, source)) {
    wait_until_reached(
        transport.internalLocalSignals[peer_signal_id<phase>(
            transport, round, source)],
        round.value,
        timeout);
  }
  group.sync();
}

template <NvlSignalPhase phase>
__device__ __forceinline__ void wait_per_peer_serial(
    const MultimemNvlTransportDevice& transport,
    const StageRound& round,
    const NvlSignalParticipants& participants,
    ThreadGroup& group,
    const Timeout& timeout) {
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
                timeout,
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
    const Timeout& timeout) {
  __shared__ uint32_t completeByPeer[64];
  const uint32_t thread = group.thread_id_in_group;
  bool complete = false;
  while (!complete) {
    const int source = static_cast<int>(thread);
    completeByPeer[thread] = source >= transport.nvlRanks ||
            !rank_selected(participants.publisherMask, source) ||
            sequence_reached(transport
                                 .internalLocalSignals[peer_signal_id<phase>(
                                     transport, round, source)]
                                 .load(),
                             round.value)
        ? 1
        : 0;
    group.sync();
    for (uint32_t stride = 32; stride != 0; stride /= 2) {
      if (thread < stride) {
        completeByPeer[thread] &= completeByPeer[thread + stride];
      }
      group.sync();
    }
    complete = completeByPeer[0] != 0;
    if (!complete && group.is_leader()) {
      if (FT_ABORT_CHECK(
              timeout,
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
    const Timeout& timeout) {
  __shared__ uint32_t completeByWarp[2];
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
    complete = completeByWarp[0] != 0 && completeByWarp[1] != 0;
    if (!complete && group.is_leader()) {
      if (FT_ABORT_CHECK(
              timeout,
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
    const ThreadGroup& group) {
#if defined(__CUDA_ARCH__)
  const uint64_t requiredSignals =
      static_cast<uint64_t>(transport.maxChannels) *
      transport.signalsPerChannel;
  if (group.scope != SyncScope::WARP || group.group_size != kWarpSize ||
      transport.pipelineDepth == 0 || transport.pipelineDepth > kWarpSize ||
      requiredSignals > transport.internalLocalSignals.size() ||
      requiredSignals > transport.internalMultimemSignals.size() ||
      (access == NvlSignalAccess::Unicast &&
       transport.internalUnicastSignalsByRank.size() !=
           static_cast<uint32_t>(transport.nvlRanks))) {
    printf(
        "NVL aggregate signal invalid execution geometry: groupSize=%u "
        "pipelineDepth=%u localSignals=%u multimemSignals=%u peers=%u\n",
        static_cast<unsigned>(group.group_size),
        static_cast<unsigned>(transport.pipelineDepth),
        static_cast<unsigned>(transport.internalLocalSignals.size()),
        static_cast<unsigned>(transport.internalMultimemSignals.size()),
        static_cast<unsigned>(transport.internalUnicastSignalsByRank.size()));
    __trap();
  }
#else
  (void)transport;
  (void)group;
#endif
}

} // namespace nvl_signal_detail

/**
 * Publishes one signal operation from every selected publisher rank.
 *
 * Aggregate topology requires one warp and maps active threads to pipeline
 * lanes.
 * Per-peer topology requires one 64-thread block and maps threads to peers.
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
    nvl_signal_detail::validate_aggregate<access>(transport, group);
  } else {
    nvl_signal_detail::validate_per_peer<access>(transport, group);
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
          transport.template signal_internal_scalar<SignalOp::SIGNAL_ADD>(
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
          transport.template signal_internal_scalar<SignalOp::SIGNAL_SET>(
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
 * A caller that enables timeout checking must call `Timeout::start()` before
 * this function.
 *
 * @param transport Exchanged multimem transport device view.
 * @param round Logical channel and per-peer round value.
 * @param participants Rank masks and expected arrival count.
 * @param group Topology-specific cooperative thread group.
 * @param timeout Started timeout or the disabled default timeout.
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
    const Timeout& timeout) {
  nvl_signal_detail::validate_protocol<access, topology, phase, waitPolicy>();
  nvl_signal_detail::validate_common(transport, round, participants);
  if constexpr (topology == NvlSignalTopology::Aggregate) {
    nvl_signal_detail::validate_aggregate<access>(transport, group);
  } else {
    nvl_signal_detail::validate_per_peer<access>(transport, group);
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
        nvl_signal_detail::wait_until_reached(*counter, expected, timeout);
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
        transport, round, participants, group, timeout);
  } else if constexpr (waitPolicy == NvlPerPeerWaitPolicy::SerialMin) {
    nvl_signal_detail::wait_per_peer_serial<phase>(
        transport, round, participants, group, timeout);
  } else if constexpr (waitPolicy == NvlPerPeerWaitPolicy::TreeMin) {
    nvl_signal_detail::wait_per_peer_tree<phase>(
        transport, round, participants, group, timeout);
  } else {
    nvl_signal_detail::wait_per_peer_butterfly<phase>(
        transport, round, participants, group, timeout);
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
    const Timeout& timeout) {
  static_assert(
      access != NvlSignalAccess::Multimem ||
          topology != NvlSignalTopology::Aggregate,
      "aggregate multimem requires signal_publish_and_wait on every rank");
  nvl_signal_detail::signal_wait_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group, timeout);
}

/**
 * Publishes and then waits using the same compile-time protocol.
 *
 * @param transport Exchanged multimem transport device view.
 * @param round Logical channel and per-peer round value.
 * @param participants Rank masks and expected arrival count.
 * @param group Topology-specific cooperative thread group.
 * @param timeout Started timeout or the disabled default timeout.
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
    const Timeout& timeout) {
  nvl_signal_detail::signal_publish_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group);
  nvl_signal_detail::signal_wait_impl<access, topology, phase, waitPolicy>(
      transport, round, participants, group, timeout);
}

} // namespace comms::prims
