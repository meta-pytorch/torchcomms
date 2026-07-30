// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Generic NVL multimem staging protocol for single-NVL-domain collectives.
//
// The reduction path follows the NVLS staging model: all local ranks write
// their contribution to the same offset in their private unicast backing, then
// the reducer reads that offset through the multicast VA with the
// `load_reduce_at` free function (multimem.ld_reduce). Broadcast uses the
// `store` free function (multimem.st). This header carries no collective- or
// algorithm-specific state (no KernArgs, no IB, no segment math); it reads its
// rank topology and buffers straight from MultimemNvlTransportDevice and is
// driven by thin per-collective loops (ReduceScatter / AllGather / AllReduce).
//
// The building blocks are split across sibling headers, all pulled in here:
//   - MultimemNvlStore.cuh       multimem.st store PTX + `store<>`
//   - MultimemNvlReduce.cuh      multimem.ld_reduce PTX + `load_reduce_at<>`
//   - MultimemNvlStageLayout.cuh staging-window layout + signal-slot addressing
// This umbrella header adds the round-level staging orchestration that composes
// them (`reduce_from_multimem`, `stage_and_wait_all_inputs`,
// `reduce_round_to_all_ranks`, `reduce_scatter_allgather_round`).
//
// Namespace: comms::prims::multimem.

#pragma once

#if defined(ENABLE_PRIMS)

#include <cstddef>
#include <cstdint>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/nvl/MultimemNvlReduce.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStageLayout.cuh"
#include "comms/prims/transport/nvl/MultimemNvlStore.cuh"

namespace comms::prims::multimem {

template <typename T, bool kAccF32 = true>
__device__ __forceinline__ void reduce_from_multimem(
    const comms::prims::MultimemNvlTransportDevice& transport,
    const StageLayout& layout,
    uint64_t primitiveRound,
    T* dst,
    std::size_t elems,
    comms::prims::ThreadGroup& group) {
  const T* mc = reinterpret_cast<const T*>(
      transport.multimem_data_ptr(lane_begin(layout, primitiveRound)));
  load_reduce_at<
      T,
      comms::prims::MultimemRedOp::Add,
      kReduceUnroll,
      /*kAliased=*/false,
      kAccF32>(group, dst, mc, elems);
}

template <typename T>
__device__ __forceinline__ void stage_and_wait_all_inputs(
    const comms::prims::MultimemNvlTransportDevice& transport,
    const StageLayout& layout,
    uint64_t roundId,
    uint64_t primitiveRound,
    const T* src,
    std::size_t elems,
    comms::prims::ThreadGroup& group) {
  const uint32_t lane =
      static_cast<uint32_t>(primitiveRound % layout.pipelineDepth);
  const std::size_t bytes = elems * sizeof(T);
  // Stage this rank's input into the shared window with 8-deep vectorized copy
  // (matches the broadcast path); 16B uint4 stores, 8 per thread per pass.
  comms::prims::memcpy_vectorized<8>(
      transport.local_data_ptr(lane_begin(layout, primitiveRound)),
      reinterpret_cast<const char*>(src),
      bytes,
      group);
  if (transport.stagingArrivalBarrier) {
    // O(1) arrival-counter barrier: the +nvlRanks accumulation across rounds
    // (lane reused every pipelineDepth rounds) and across ops is handled by the
    // per-slot epoch, exactly like the no-copy path. `roundId` is unused here.
    (void)roundId;
    transport.arrival_barrier(
        group,
        staging_ready_counter_id(layout, lane, transport.nvlRanks),
        staging_ready_epoch_id(layout, lane, transport.nvlRanks));
    return;
  }
  transport.signal_internal(
      group,
      ready_signal_id(layout, lane, transport.nvlRank),
      comms::prims::SignalOp::SIGNAL_SET,
      roundId);

  for (int rank = 0; rank < transport.nvlRanks; ++rank) {
    transport.wait_internal_signal_until(
        group,
        ready_signal_id(layout, lane, rank),
        comms::prims::CmpOp::CMP_GE,
        roundId);
  }
}

template <typename T, bool kAccF32 = true>
__device__ __forceinline__ void reduce_round_to_all_ranks(
    const comms::prims::MultimemNvlTransportDevice& transport,
    const StageLayout& layout,
    uint64_t roundId,
    uint64_t primitiveRound,
    T* dst,
    std::size_t elems,
    comms::prims::ThreadGroup& group) {
  const uint32_t lane =
      static_cast<uint32_t>(primitiveRound % layout.pipelineDepth);
  reduce_from_multimem<T, kAccF32>(
      transport, layout, primitiveRound, dst, elems, group);
  if (transport.stagingArrivalBarrier) {
    // O(1) arrival barrier replacing the per-peer ack full barrier. Only
    // `direct` calls this; `direct_rsag` keeps its per-owner ack interleaving
    // in reduce_scatter_allgather_round, so the ack[] SET slots stay live there
    // and the arrival counter/epoch slots are chosen disjoint from them.
    (void)roundId;
    transport.arrival_barrier(
        group,
        staging_ack_counter_id(layout, lane, transport.nvlRanks),
        staging_ack_epoch_id(layout, lane, transport.nvlRanks));
    return;
  }
  transport.signal_internal(
      group,
      ack_signal_id(layout, lane, transport.nvlRanks, transport.nvlRank),
      comms::prims::SignalOp::SIGNAL_SET,
      roundId);
  for (int rank = 0; rank < transport.nvlRanks; ++rank) {
    transport.wait_internal_signal_until(
        group,
        ack_signal_id(layout, lane, transport.nvlRanks, rank),
        comms::prims::CmpOp::CMP_GE,
        roundId);
  }
}

/**
 * Fused reduce-scatter + all-gather for one staging round (the cnvlmm AllReduce
 * `direct_rsag` large-message path). Requires `stage_and_wait_all_inputs` to
 * have already staged every rank's input slice into the window for this round.
 *
 * Unlike `reduce_round_to_all_ranks` (where EVERY rank multimem.ld_reduces the
 * FULL slice -> ~nvlRanks x switch egress, which plateaus at large sizes), each
 * rank here reduces ONLY its own 1/nvlRanks shard (a disjoint sub-region) into
 * its output, then broadcasts that reduced shard to all peers via
 * multimem::store (multimem.st). Every rank ends with the full reduced slice.
 * Switch traffic is ~1x reduce-read + ~1x broadcast (the NVLS
 * reduce-scatter+all-gather shape).
 *
 * Shards use the ceil convention; idle/partial tail shards stay in lockstep
 * (still signal/wait). The per-shard ack wait is a full barrier (each rank
 * waits all other owners), bounding cross-rank drift to < 1 round so lane reuse
 * under pipelining is safe.
 */
template <typename T, bool kAccF32 = true>
__device__ __forceinline__ void reduce_scatter_allgather_round(
    const comms::prims::MultimemNvlTransportDevice& transport,
    const StageLayout& layout,
    uint64_t roundId,
    uint64_t primitiveRound,
    T* dst,
    std::size_t elems,
    comms::prims::ThreadGroup& group) {
  const uint32_t lane =
      static_cast<uint32_t>(primitiveRound % layout.pipelineDepth);
  const std::size_t laneBeginBytes = lane_begin(layout, primitiveRound);
  const int nvlRanks = transport.nvlRanks;
  const int self = transport.nvlRank;
  // Round the per-rank shard up to the 16-byte vector width so every rank's
  // shard base (self * shardElems) is 16-byte aligned in the multicast window.
  // A non-16B-aligned multimem.ld_reduce base is run-to-run NON-deterministic
  // for fp16/bf16 (the switch's cross-rank accumulation of a misaligned read
  // varies by ~1 ULP); a 16B-aligned base uses the deterministic v4 read, so
  // the 2-shot reduce-scatter matches the 1-shot (aligned base 0) path bit for
  // bit. Coverage stays complete (shards over-cover; tail shards clamp/idle).
  const std::size_t kVecElems = 16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1;
  const std::size_t rawShard =
      (elems + static_cast<std::size_t>(nvlRanks) - 1) /
      static_cast<std::size_t>(nvlRanks);
  const std::size_t shardElems =
      ((rawShard + kVecElems - 1) / kVecElems) * kVecElems;

  // Reduce-scatter: reduce this rank's own shard from the multicast window into
  // its output slot.
  const std::size_t myBegin = static_cast<std::size_t>(self) * shardElems;
  const std::size_t myElems = myBegin < elems
      ? (elems - myBegin < shardElems ? elems - myBegin : shardElems)
      : 0;
  if (myElems > 0) {
    const T* myMc = reinterpret_cast<const T*>(
        transport.multimem_data_ptr(laneBeginBytes + myBegin * sizeof(T)));
    load_reduce_at<
        T,
        comms::prims::MultimemRedOp::Add,
        kReduceUnroll,
        /*kAliased=*/false,
        kAccF32>(group, dst + myBegin, myMc, myElems);
  }
  // load_reduce wrote dst[myBegin..]; the broadcast store reads it back, and
  // the two map threads to elements differently, so order the write before the
  // read.
  group.sync();
  // Broadcast the reduced shard to all backings' window slot (multimem.st).
  // Each rank stores its OWN shard byte-count here (asymmetric across ranks -
  // tail/idle shards differ), which is fine despite store()'s "same bytes"
  // note: store() does no cross-rank coordination, and the all-gather readers
  // below recompute each owner's shard size deterministically from the shared
  // shardElems/elems, so the per-owner byte counts always match.
  if (myElems > 0) {
    store<8>(
        group,
        transport.multimem_data_ptr(laneBeginBytes + myBegin * sizeof(T)),
        dst + myBegin,
        myElems * sizeof(T));
  }
  transport.signal_internal(
      group,
      ack_signal_id(layout, lane, nvlRanks, self),
      comms::prims::SignalOp::SIGNAL_SET,
      roundId);

  // All-gather: once each owner's broadcast is visible (the ack release/acquire
  // pairs the multimem.st with these reads), copy its reduced shard into dst.
  for (int s = 0; s < nvlRanks; ++s) {
    if (s == self) {
      continue;
    }
    transport.wait_internal_signal_until(
        group,
        ack_signal_id(layout, lane, nvlRanks, s),
        comms::prims::CmpOp::CMP_GE,
        roundId);
    const std::size_t sBegin = static_cast<std::size_t>(s) * shardElems;
    const std::size_t sElems = sBegin < elems
        ? (elems - sBegin < shardElems ? elems - sBegin : shardElems)
        : 0;
    if (sElems > 0) {
      comms::prims::memcpy_vectorized<8>(
          reinterpret_cast<char*>(dst + sBegin),
          transport.local_data_ptr(laneBeginBytes + sBegin * sizeof(T)),
          sElems * sizeof(T),
          group);
    }
  }
}

} // namespace comms::prims::multimem

#endif // ENABLE_PRIMS
