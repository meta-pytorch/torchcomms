// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/DeviceCheck.cuh"

namespace comms::pipes::collectives {

// Forward declaration - BroadcastContext is defined in BroadcastContext.cuh
struct BroadcastContext;

namespace detail {

/**
 * Pipelined ring broadcast with fused dual-destination copies.
 *
 * At intermediate ranks, each warp performs a fused dual-destination copy:
 * reading from the predecessor's staging buffer once and writing to both
 * the local output buffer and the successor's staging buffer simultaneously.
 * This eliminates the extra HBM read that would occur with two sequential
 * copies (staging->output, output->next staging).
 *
 * Uses RecvStream/SendStream primitives for clean, reusable iteration.
 *
 * Prerequisites:
 * - Standard transport setup
 * - Supports arbitrary numBlocks
 */
__device__ __forceinline__ void broadcast_pipelined_ring(
    BroadcastContext& ctx) {
#ifdef __CUDA_ARCH__
  int next_virtual = (ctx.virtual_rank + 1) % ctx.nranks;
  int prev_virtual = (ctx.virtual_rank - 1 + ctx.nranks) % ctx.nranks;
  int next_rank = ctx.actual_rank(next_virtual);
  int prev_rank = ctx.actual_rank(prev_virtual);

  bool is_root = (ctx.virtual_rank == 0);
  bool is_last = (ctx.virtual_rank == ctx.nranks - 1);

  if (is_root) {
    PIPES_DEVICE_CHECK(
        ctx.transports[next_rank].type == TransportType::P2P_NVL);
    auto& next_transport = ctx.transports[next_rank].p2p_nvl;
    auto send = next_transport.send_stream(ctx.nbytes);
    send.for_each_slot(ctx.group, [&](auto slot) {
      memcpy_vectorized(
          slot.data, // successor's staging (via NVLink)
          ctx.buff + slot.offset, // local source
          slot.size,
          ctx.group);
    });
  } else if (is_last) {
    PIPES_DEVICE_CHECK(
        ctx.transports[prev_rank].type == TransportType::P2P_NVL);
    auto& prev_transport = ctx.transports[prev_rank].p2p_nvl;
    auto recv = prev_transport.recv_stream(ctx.nbytes);
    recv.for_each_ready_chunk(ctx.group, [&](auto chunk) {
      memcpy_vectorized(
          ctx.buff + chunk.offset, // local output
          chunk.data, // predecessor's staging (local read)
          chunk.size,
          ctx.group);
    });
  } else {
    // Intermediate rank: fused recv + forward using RecvStream + SendStream
    PIPES_DEVICE_CHECK(
        ctx.transports[prev_rank].type == TransportType::P2P_NVL);
    PIPES_DEVICE_CHECK(
        ctx.transports[next_rank].type == TransportType::P2P_NVL);

    auto& prev_transport = ctx.transports[prev_rank].p2p_nvl;
    auto& next_transport = ctx.transports[next_rank].p2p_nvl;

    // Config validation: recv and send transports must have matching configs
    // for slot_for() mapping to work correctly
#ifndef NDEBUG
    PIPES_DEVICE_CHECK(
        prev_transport.chunk_size() == next_transport.chunk_size());
    PIPES_DEVICE_CHECK(
        prev_transport.data_buffer_size() == next_transport.data_buffer_size());
    PIPES_DEVICE_CHECK(
        prev_transport.pipeline_depth() == next_transport.pipeline_depth());
#endif

    auto recv = prev_transport.recv_stream(ctx.nbytes);
    auto send = next_transport.send_stream(ctx.nbytes);

    recv.for_each_ready_chunk(ctx.group, [&](auto chunk) {
      // slot_for() reverse-maps chunk.offset -> send staging pointer
      auto slot = send.slot_for(ctx.group, chunk);

      // Fused dual-destination copy: staging -> {output, next staging}
      std::array<char*, 2> dsts = {{ctx.buff + chunk.offset, slot.data}};
      memcpy_vectorized_multi_dest<2>(
          dsts,
          chunk.data, // predecessor's staging (local)
          chunk.size,
          ctx.group);

      // Signal successor: data ready (MANUAL commit required with slot_for)
      send.commit_slot(ctx.group, slot);

      // Note: RecvStream auto-releases predecessor's slot after this callback
    });
  }
#endif
}

} // namespace detail

} // namespace comms::pipes::collectives
