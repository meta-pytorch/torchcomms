// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Staging-window layout + signal-slot addressing for single-NVL-domain
// collectives.
//
// Shared infrastructure used by all three staging paths (ReduceScatter /
// AllGather / AllReduce): the per-CTA staging-window geometry (`StageLayout` /
// `make_stage_layout`) and the internal-signal slot-id helpers (ready/ack
// per-peer SET slots and the staging ADD barrier slots). It depends only on the
// transport struct in MultimemNvlTransportDevice.cuh, not on the reduce/store
// PTX.

// clang-tidy analyzes this .cuh as a standalone main file and misflags the
// pragma; it is a genuine include-once header. False positive, so suppress it.
// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

namespace comms::prims::multimem {

namespace detail {

__device__ __forceinline__ uint64_t
checked_signal_product(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > ~uint64_t{0} / lhs) {
#if defined(__CUDA_ARCH__)
    printf(
        "NvlMultimem signal layout overflow: %llu * %llu\n",
        static_cast<unsigned long long>(lhs),
        static_cast<unsigned long long>(rhs));
    __trap();
#endif
    return 0;
  }
  return lhs * rhs;
}

} // namespace detail

/**
 * Layout of one CTA-group's slice of the shared staging window.
 *
 * The flat per-rank data buffer is split across CTAs (groups), and each group's
 * slice is further split into `pipelineDepth` lanes. A round picks its lane
 * round-robin so consecutive rounds use disjoint physical windows (pipelining).
 */
struct StageLayout {
  std::size_t groupBeginBytes{0}; // this group's slice origin into the buffer
  std::size_t stagingElems{0}; // one lane window, in elements
  std::size_t stagingBytes{0}; // one lane window, in bytes
  uint64_t signalBase{0}; // this group's base into the internal signals
  uint64_t signalsPerLane{0}; // 2 * nvlRanks + 4
  uint32_t pipelineDepth{1};
};

template <typename T>
__device__ __forceinline__ StageLayout make_stage_layout(
    const comms::prims::MultimemNvlTransportDevice& transport,
    uint32_t pipelineDepth,
    comms::prims::ThreadGroup& group) {
  static_assert(sizeof(T) <= 16, "staging payloads must fit in 16 bytes");
  static_assert(16 % sizeof(T) == 0, "staging payloads must divide 16 bytes");

#if defined(__CUDA_ARCH__)
  if (pipelineDepth == 0) {
    printf("NvlMultimem staging layout: pipelineDepth must be non-zero\n");
    __trap();
  }
#endif

  // 16-byte (uint4) alignment for every staging-window slice, so the per-rank
  // lane offsets are 128-bit aligned and the v4 multimem.ld_reduce / store fast
  // paths engage for all element types (not just when offsets happen to land on
  // 16B). The backing buffer base is already page-aligned.
  const std::size_t alignBytes = sizeof(T) < 16 ? 16 : sizeof(T);
  const std::size_t elemsPerAlign = alignBytes / sizeof(T);
  const std::size_t totalUnits = transport.dataBufferSize / alignBytes;

#if defined(__CUDA_ARCH__)
  if (transport.localData == nullptr || transport.multimemData == nullptr) {
    printf("NvlMultimem transport has null data pointers\n");
    __trap();
  }
  // Trap before total_groups is used as a divisor below (division by zero is UB
  // and would happen strictly before the stagingUnits guard could catch it).
  if (group.total_groups == 0 || group.group_id >= group.total_groups) {
    printf(
        "NvlMultimem staging layout: invalid group_id=%u total_groups=%u\n",
        static_cast<unsigned>(group.group_id),
        static_cast<unsigned>(group.total_groups));
    __trap();
  }
#endif

  const std::size_t totalGroups = group.total_groups;
  const std::size_t groupId = group.group_id;
  const std::size_t unitsPerGroup = totalUnits / totalGroups;
  const std::size_t extraUnits = totalUnits % totalGroups;
  const std::size_t groupBeginUnit =
      unitsPerGroup * groupId + (groupId < extraUnits ? groupId : extraUnits);
  const std::size_t groupUnits =
      unitsPerGroup + static_cast<std::size_t>(groupId < extraUnits);
  const std::size_t stagingUnits =
      groupUnits / static_cast<std::size_t>(pipelineDepth);

#if defined(__CUDA_ARCH__)
  if (stagingUnits == 0) {
    printf(
        "NvlMultimem staging window too small: dataBufferSize=%llu "
        "groups=%u pipelineDepth=%u alignBytes=%llu\n",
        static_cast<unsigned long long>(transport.dataBufferSize),
        static_cast<unsigned>(group.total_groups),
        static_cast<unsigned>(pipelineDepth),
        static_cast<unsigned long long>(alignBytes));
    __trap();
  }
#endif

  // Layout per (group, lane):
  //   [0, nvlRanks)             ready[rank]           (SET, per-peer)
  //   [nvlRanks, 2*nvlRanks)    ack[rank]             (SET, per-peer)
  //   2*nvlRanks + 0            staging_ready_counter (ADD, multicast)
  //   2*nvlRanks + 1            staging_ready_epoch   (ADD, this rank baseline)
  //   2*nvlRanks + 2            staging_ack_counter   (ADD, multicast)
  //   2*nvlRanks + 3            staging_ack_epoch     (ADD, this rank baseline)
  // The four ADD-mode barrier slots MUST live outside the SET-mode ready/ack
  // region: without this separation, flipping stagingArrivalBarrier between ops
  // in a single process leaves ADD counter residue in slots that a later
  // SET-mode op reads with CMP_GE, and any past-op residue trivially satisfies
  // the wait.
#if defined(__CUDA_ARCH__)
  if (transport.nvlRanks <= 0 ||
      static_cast<uint64_t>(transport.nvlRanks) >
          (static_cast<uint64_t>(~uint32_t{0}) - 4) / 2) {
    printf(
        "NvlMultimem staging layout: invalid nvlRanks=%d\n",
        transport.nvlRanks);
    __trap();
  }
#endif
  const uint64_t signalsPerLane = multimem_staging_signals_per_lane(
      static_cast<uint32_t>(transport.nvlRanks));
  const uint64_t signalsPerGroup = detail::checked_signal_product(
      static_cast<uint64_t>(pipelineDepth), signalsPerLane);
  const uint64_t requiredSignals = detail::checked_signal_product(
      static_cast<uint64_t>(group.total_groups), signalsPerGroup);
#if defined(__CUDA_ARCH__)
  if (requiredSignals > transport.internalLocalSignals.size() ||
      requiredSignals > transport.internalMultimemSignals.size()) {
    printf(
        "NvlMultimem requires %llu internal signals "
        "(local=%u, multimem=%u, groups=%u, pipelineDepth=%u, nvlRanks=%d)\n",
        static_cast<unsigned long long>(requiredSignals),
        static_cast<unsigned>(transport.internalLocalSignals.size()),
        static_cast<unsigned>(transport.internalMultimemSignals.size()),
        static_cast<unsigned>(group.total_groups),
        static_cast<unsigned>(pipelineDepth),
        transport.nvlRanks);
    __trap();
  }
#endif

  return StageLayout{
      .groupBeginBytes = groupBeginUnit * alignBytes,
      .stagingElems = stagingUnits * elemsPerAlign,
      .stagingBytes = stagingUnits * alignBytes,
      .signalBase = detail::checked_signal_product(
          static_cast<uint64_t>(group.group_id), signalsPerGroup),
      .signalsPerLane = signalsPerLane,
      .pipelineDepth = pipelineDepth,
  };
}

__device__ __forceinline__ uint64_t
ready_signal_id(const StageLayout& layout, uint32_t lane, int rank) {
  return layout.signalBase +
      static_cast<uint64_t>(lane) * layout.signalsPerLane +
      static_cast<uint64_t>(rank);
}

__device__ __forceinline__ uint64_t ack_signal_id(
    const StageLayout& layout,
    uint32_t lane,
    int nvlRanks,
    int rank) {
  return layout.signalBase +
      static_cast<uint64_t>(lane) * layout.signalsPerLane +
      static_cast<uint64_t>(nvlRanks + rank);
}

__device__ __forceinline__ uint64_t
round_id(uint64_t roundBase, uint64_t primitiveRound) {
  return roundBase + primitiveRound + 1;
}

__device__ __forceinline__ std::size_t lane_begin(
    const StageLayout& layout,
    uint64_t primitiveRound) {
  const uint32_t lane =
      static_cast<uint32_t>(primitiveRound % layout.pipelineDepth);
  return layout.groupBeginBytes +
      static_cast<std::size_t>(lane) * layout.stagingBytes;
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
