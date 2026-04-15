// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// User-facing TileChannel API for IB (RDMA) pipelined send/recv.
//
// TileChannel encapsulates the signal/counter management and staging buffer
// layout for GPU-initiated RDMA. Users call send_tile()/recv_tile() as
// single-step primitives — internal chunking handles tiles larger than
// the staging slot.
//
// Host-safe header for the launch function. Device-side APIs require
// including SendRecv.cu (compiled by nvcc).

#pragma once

#include <cuda_runtime.h>
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLXTypes.hpp"

namespace comms::pipes::ib {

using torchcomms::device::DeviceWindowNCCL;
using torchcomms::device::RegisteredBufferNCCL;

/**
 * TileChannel — lightweight device-side handle for pipelined RDMA send/recv.
 *
 * Encapsulates the window, staging buffer, and signal/counter IDs for one
 * block's tile communication with a peer. Created once per block, used
 * across multiple send_tile()/recv_tile() calls.
 *
 * The user provides their own staging and window buffers (BYOB). The channel
 * derives per-block signal/counter IDs from the block's tile index (pid).
 *
 * Internal chunking: if the user's tile is larger than what fits in one
 * staging slot (staging_slot_bytes / num_blocks), send_tile/recv_tile
 * internally pipeline the tile through the staging ring buffer in multiple
 * chunks. This decouples staging memory from data size.
 *
 * Signal layout: 2 * num_blocks signals.
 *   [0, num_blocks)             — DATA_READY (piggybacked on put)
 *   [num_blocks, 2*num_blocks)  — SLOT_FREE  (receiver → sender)
 * Counter layout: num_blocks counters.
 *   [0, num_blocks)             — NIC_DONE   (staging reuse)
 */
struct TileChannel {
  DeviceWindowNCCL* win;
  RegisteredBufferNCCL
      staging_buf; ///< Registered buffer (sender uses for put).
  float* ring_buf_ptr; ///< Ring buffer base. Sender: staging. Receiver: window.
  int peer_rank;
  int pid; ///< This block's tile index.
  int num_blocks;
  int pipeline_depth;
  size_t
      staging_slot_bytes; ///< Size of one pipeline slot (full, not per-block).

  // Per-block signal/counter IDs (derived from pid).
  int data_signal;
  int slot_free_signal;
  int nic_counter;

  /// Step counter — monotonically increasing across calls. Tracks signal
  /// values for backpressure and CUDA graph replay safety.
  int step;
};

/**
 * Initialize a TileChannel for a sender or receiver block.
 *
 * @param win              Device window handle.
 * @param staging_buf      Registered staging buffer (sender uses for put).
 * @param ring_buf_ptr     Ring buffer base pointer. For senders this is the
 *                         staging buffer; for receivers this is the window
 *                         buffer. Both share the same ring layout.
 * @param peer_rank        Rank to communicate with.
 * @param pid              This block's tile index (0..num_blocks-1).
 * @param num_blocks       Total sender (= receiver) blocks.
 * @param pipeline_depth   Number of ring-buffer slots.
 * @param staging_slot_bytes Size of one pipeline slot in bytes
 *                           (staging ring = pipeline_depth *
 * staging_slot_bytes).
 * @param signal_base      Initial step counter value (for CUDA graph replay).
 */
__device__ TileChannel make_tile_channel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL staging_buf,
    float* ring_buf_ptr,
    int peer_rank,
    int pid,
    int num_blocks,
    int pipeline_depth,
    size_t staging_slot_bytes,
    int signal_base);

/**
 * Send a tile of data to the peer via pipelined RDMA.
 *
 * Copies src_ptr → staging, then RDMA puts staging → peer's window.
 * If nbytes > per-block staging capacity, internally pipelines through
 * the staging ring buffer in multiple chunks.
 *
 * @param ch      TileChannel (step counter is advanced).
 * @param src_ptr Source data for this block's tile.
 * @param nbytes  Bytes to send.
 */
__device__ void send_tile(TileChannel& ch, const float* src_ptr, size_t nbytes);

/**
 * Receive a tile of data from the peer.
 *
 * Waits for RDMA data to arrive in window, then copies window → dst_ptr.
 * If nbytes > per-block staging capacity, internally receives in multiple
 * chunks.
 *
 * @param ch      TileChannel (step counter is advanced).
 * @param dst_ptr Destination for this block's tile.
 * @param nbytes  Bytes to receive.
 */
__device__ void recv_tile(TileChannel& ch, float* dst_ptr, size_t nbytes);

/**
 * Drain outstanding SLOT_FREE signals.
 *
 * Call after all send_tile() calls in an iteration to ensure the peer
 * has consumed all data before starting the next iteration.
 *
 * @param ch TileChannel.
 */
__device__ void drain(TileChannel& ch);

/**
 * Launch a pipelined send/recv kernel over IB (RDMA).
 *
 * Convenience launcher that creates TileChannels and calls send_tile/recv_tile
 * in a loop. For custom kernels, create TileChannels directly and call
 * send_tile/recv_tile as needed.
 *
 * Grid: 2 * num_blocks (first half sends, second half receives).
 * Block: 256 threads.
 */
void launch_send_recv_kernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL staging_buf,
    float* src_ptr,
    float* staging_ptr,
    float* win_ptr,
    float* dst_ptr,
    size_t total_bytes,
    size_t section_bytes,
    int pipeline_depth,
    int dst_rank,
    int src_rank,
    int num_blocks,
    int signal_base,
    int iterations,
    cudaStream_t stream);

} // namespace comms::pipes::ib
