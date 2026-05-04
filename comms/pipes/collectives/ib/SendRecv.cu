// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// TileChannel-based pipelined send/recv for IB (RDMA).
//
// send_tile/recv_tile are user-facing primitives. Each call handles one
// tile of data, internally chunking through the staging ring buffer if
// the tile is larger than the per-block staging capacity.
//
// The step counter tracks chunks (not tiles) — each chunk consumes one
// pipeline slot. The caller must ensure that the tile fits within
// pipeline_depth slots: nbytes <= per_block_chunk_bytes * pipeline_depth.

#include "comms/pipes/collectives/ib/SendRecv.cuh"

#include <stdexcept>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

namespace comms::pipes::ib {

using torchcomms::device::CmpOp;
using torchcomms::device::CoopScope;
using torchcomms::device::DeviceWindowNCCL;
using torchcomms::device::RegisteredBufferNCCL;
using torchcomms::device::SignalOp;

// ---- helpers ---------------------------------------------------------------

/**
 * Cooperative element-wise copy across all threads in a block.
 *
 * @param dst   Destination pointer (device global memory).
 * @param src   Source pointer (device global memory).
 * @param count Number of float elements to copy.
 */
__device__ __forceinline__ void
local_copy(float* dst, const float* src, size_t count) {
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ---- TileChannel -----------------------------------------------------------

__device__ TileChannel make_tile_channel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL staging_buf,
    float* ring_buf_ptr,
    int peer_rank,
    int pid,
    int num_blocks,
    int pipeline_depth,
    size_t staging_slot_bytes,
    int64_t* step_state_ptr) {
  int64_t initial_step = step_state_ptr ? *step_state_ptr : 0;
  return TileChannel{
      .win = win,
      .staging_buf = staging_buf,
      .ring_buf_ptr = ring_buf_ptr,
      .peer_rank = peer_rank,
      .pid = pid,
      .num_blocks = num_blocks,
      .pipeline_depth = pipeline_depth,
      .staging_slot_bytes = staging_slot_bytes,
      .data_signal = pid,
      .slot_free_signal = num_blocks + pid,
      .nic_counter = pid,
      .step = initial_step,
      .step_state_ptr = step_state_ptr,
  };
}

// ---- send_tile -------------------------------------------------------------

__device__ void
send_tile(TileChannel& ch, const float* src_ptr, size_t nbytes) {
  size_t per_block_chunk_bytes =
      (ch.staging_slot_bytes / ch.num_blocks) & ~15ULL;
  size_t total_chunks =
      (nbytes + per_block_chunk_bytes - 1) / per_block_chunk_bytes;

  // Pre-compute loop-invariant values.
  size_t staging_slot_floats = ch.staging_slot_bytes / sizeof(float);
  size_t per_block_slot_floats = per_block_chunk_bytes / sizeof(float);

  for (size_t c = 0; c < total_chunks; c++) {
    int slot = ch.step % ch.pipeline_depth;

    size_t data_off_floats = c * per_block_slot_floats;
    size_t chunk_bytes = (c + 1) * per_block_chunk_bytes <= nbytes
        ? per_block_chunk_bytes
        : (nbytes - c * per_block_chunk_bytes);

    // Staging destination: slot within ring + block within slot.
    float* staging_dst = ch.ring_buf_ptr + slot * staging_slot_floats +
        ch.pid * per_block_slot_floats;

    // 1. Wait NIC_DONE — staging slot safe to overwrite
    if (ch.step >= ch.pipeline_depth) {
      uint64_t nic_expected =
          static_cast<uint64_t>(ch.step - ch.pipeline_depth + 1);
      ch.win->wait_counter(
          ch.nic_counter, CmpOp::GE, nic_expected, CoopScope::BLOCK);
    }

    // 2. Local copy: src chunk → staging (vectorized uint4 loads)
    {
      auto group = comms::pipes::make_block_group();
      comms::pipes::memcpy_vectorized(
          reinterpret_cast<char*>(staging_dst),
          reinterpret_cast<const char*>(src_ptr + data_off_floats),
          chunk_bytes,
          group);
    }
    __syncthreads();
    __threadfence_system();

    // 3. Wait SLOT_FREE — remote window slot safe for new put
    if (ch.step >= ch.pipeline_depth) {
      uint64_t slot_expected =
          static_cast<uint64_t>(ch.step - ch.pipeline_depth + 1);
      ch.win->wait_signal_from(
          ch.peer_rank,
          ch.slot_free_signal,
          CmpOp::GE,
          slot_expected,
          CoopScope::BLOCK);
    }

    // 4. Put: staging → remote window (offset is symmetric: local staging
    //    layout matches remote window layout).
    size_t byte_off =
        static_cast<size_t>(staging_dst - ch.ring_buf_ptr) * sizeof(float);
    ch.win->put(
        byte_off,
        ch.staging_buf,
        byte_off,
        ch.peer_rank,
        chunk_bytes,
        ch.data_signal,
        ch.nic_counter,
        CoopScope::BLOCK);

    ch.step++;
  }
}

// ---- recv_tile -------------------------------------------------------------

__device__ void recv_tile(TileChannel& ch, float* dst_ptr, size_t nbytes) {
  size_t per_block_chunk_bytes =
      (ch.staging_slot_bytes / ch.num_blocks) & ~15ULL;
  size_t total_chunks =
      (nbytes + per_block_chunk_bytes - 1) / per_block_chunk_bytes;

  // Pre-compute loop-invariant values.
  size_t staging_slot_floats = ch.staging_slot_bytes / sizeof(float);
  size_t per_block_slot_floats = per_block_chunk_bytes / sizeof(float);

  for (size_t c = 0; c < total_chunks; c++) {
    int slot = ch.step % ch.pipeline_depth;

    size_t data_off_floats = c * per_block_slot_floats;
    size_t chunk_bytes = (c + 1) * per_block_chunk_bytes <= nbytes
        ? per_block_chunk_bytes
        : (nbytes - c * per_block_chunk_bytes);

    // Window source: slot within ring + block within slot.
    float* window_src = ch.ring_buf_ptr + slot * staging_slot_floats +
        ch.pid * per_block_slot_floats;

    // 1. Wait DATA_READY
    uint64_t expected = static_cast<uint64_t>(ch.step + 1);
    ch.win->wait_signal_from(
        ch.peer_rank, ch.data_signal, CmpOp::GE, expected, CoopScope::BLOCK);

    // 2. Local copy: window → dst (vectorized uint4 loads)
    {
      auto group = comms::pipes::make_block_group();
      comms::pipes::memcpy_vectorized(
          reinterpret_cast<char*>(dst_ptr + data_off_floats),
          reinterpret_cast<const char*>(window_src),
          chunk_bytes,
          group);
    }

    // 3. Signal SLOT_FREE
    ch.win->signal(
        ch.peer_rank, ch.slot_free_signal, SignalOp::ADD, 1, CoopScope::BLOCK);

    ch.step++;
  }
}

// ---- drain -----------------------------------------------------------------

__device__ void drain(TileChannel& ch) {
  // Wait for all outstanding SLOT_FREE signals.
  // NIC_DONE drain is unnecessary — send_tile waits lazily before reusing
  // each staging slot.
  uint64_t drain_val = static_cast<uint64_t>(ch.step);
  ch.win->wait_signal_from(
      ch.peer_rank,
      ch.slot_free_signal,
      CmpOp::GE,
      drain_val,
      CoopScope::BLOCK);
}

// ---- kernel ----------------------------------------------------------------

__global__ void send_recv_kernel(
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
    int64_t* step_state) {
  auto bid = blockIdx.x;
  bool is_sender = bid < num_blocks;
  int pid = is_sender ? bid : bid - num_blocks;
  int peer = is_sender ? dst_rank : src_rank;

  // step_state layout: [0, num_blocks) for senders,
  //                     [num_blocks, 2*num_blocks) for receivers.
  int64_t* my_step_state = step_state + (is_sender ? pid : num_blocks + pid);

  // Sender's ring buffer is the staging buffer; receiver's is the window.
  TileChannel ch = make_tile_channel(
      win,
      staging_buf,
      is_sender ? staging_ptr : win_ptr,
      peer,
      pid,
      num_blocks,
      pipeline_depth,
      section_bytes,
      my_step_state);

  size_t section_floats = section_bytes / sizeof(float);
  int total_steps = total_bytes / section_bytes;

  for (int step = 0; step < total_steps; step++) {
    TiledBuffer<float> tiles(
        (is_sender ? src_ptr : dst_ptr) + step * section_floats,
        section_floats,
        num_blocks);

    if (is_sender) {
      send_tile(ch, tiles.tile_data(pid), tiles.tile_bytes(pid));
    } else {
      recv_tile(ch, tiles.tile_data(pid), tiles.tile_bytes(pid));
    }
  }

  if (is_sender) {
    drain(ch);
  }

  // Persist step counter for next kernel launch.
  if (threadIdx.x == 0) {
    *my_step_state = ch.step;
  }
}

// ---- host launch -----------------------------------------------------------

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
    int64_t* step_state,
    cudaStream_t stream) {
  if (section_bytes == 0 || total_bytes % section_bytes != 0) {
    throw std::runtime_error(
        "launch_send_recv_kernel: total_bytes must be a positive multiple of section_bytes");
  }
  if (pipeline_depth > static_cast<int>(total_bytes / section_bytes)) {
    throw std::runtime_error(
        "launch_send_recv_kernel: pipeline_depth must not exceed total_steps");
  }
  send_recv_kernel<<<2 * num_blocks, 256, 0, stream>>>(
      win,
      staging_buf,
      src_ptr,
      staging_ptr,
      win_ptr,
      dst_ptr,
      total_bytes,
      section_bytes,
      pipeline_depth,
      dst_rank,
      src_rank,
      num_blocks,
      step_state);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("send_recv_kernel launch failed: ") +
        cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::ib
