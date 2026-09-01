// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/collectives/ReduceScatterDirectIbCore.cuh"
#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define PRIMS_DIRECT_IB_PHASE_HOST_DEVICE_INLINE \
  __host__ __device__ __forceinline__
#else
#define PRIMS_DIRECT_IB_PHASE_HOST_DEVICE_INLINE inline
#endif

namespace comms::prims {

/** The local operand source for a blocking Direct IB ReduceScatter phase. */
enum class DirectIbOutputSeed : std::uint8_t {
  PACKED_INPUT,
  OUTPUT_ALREADY_SEEDED,
};

/** Equal logical chunks packed with an optional padded physical stride. */
template <typename T>
struct DirectIbPackedInputView {
  const T* data{nullptr};
  std::size_t chunk_elements{0};
  std::size_t chunk_stride_elements{0};

  PRIMS_DIRECT_IB_PHASE_HOST_DEVICE_INLINE constexpr const T* chunk_data(
      int phase_rank) const {
    return data + static_cast<std::size_t>(phase_rank) * chunk_stride_elements;
  }
};

/** Final local output and whether a prior phase already seeded it. */
template <typename T>
struct DirectIbFinalOutputView {
  T* data{nullptr};
  DirectIbOutputSeed seed{DirectIbOutputSeed::PACKED_INPUT};
};

/**
 * Run one blocking, non-quantized Direct IB ReduceScatter phase.
 *
 * The input contains one equal logical chunk per phase rank. The output is
 * either disjoint from the packed input, in which case the first receive
 * combines the local packed chunk with its peer, or it is already seeded by a
 * prior phase. Callers validate buffer overlap, packed stride, and peer
 * readiness before launch. Alignment is part of the selected ReduceOp's
 * contract: callers using a 16-byte-only operation must pad the stride or
 * select an unaligned-safe operation. Callers also own whole-block
 * synchronization at phase boundaries; this helper synchronizes only its
 * role-local groups. The peer table is indexed in the same phase-local rank
 * space as `phase_rank` and `phase_size`; the self entry is unused.
 */
template <
    typename T,
    typename ReduceOp,
    int kSendThreads,
    int kRecvThreads,
    int kBlockThreads,
    bool kStaggerChannels = true,
    std::uint32_t kRecvBarrierId = ThreadGroup::kAutoBarrierId,
    std::uint32_t kSendBarrierId = ThreadGroup::kAutoBarrierId>
__device__ __forceinline__ void direct_ib_reduce_scatter_phase(
    int phase_rank,
    int phase_size,
    DirectIbPackedInputView<T> input,
    DirectIbFinalOutputView<T> output,
    std::size_t signaling_data_size,
    const P2pIbTransportDevice* peers,
    const ThreadGroup& block,
    const AbortDevice& abort_device) {
  auto role_group = make_direct_ib_reduce_scatter_role_group<
      kSendThreads,
      kRecvThreads,
      kBlockThreads,
      kRecvBarrierId,
      kSendBarrierId>(block);
  ThreadGroup& group = role_group.group;
  const bool is_receive = role_group.role == DirectIbReduceScatterRole::RECEIVE;

  const int channel = static_cast<int>(group.group_id);
  TiledBuffer<T> output_tile(output.data, input.chunk_elements, group);
  const std::size_t tile_bytes = output_tile.bytes();
  if (tile_bytes == 0) {
    return;
  }

  T* output_data = output_tile.data();
  char* output_bytes = reinterpret_cast<char*>(output_data);

  if (is_receive) {
    const T* own_source = input.chunk_data(phase_rank) +
        static_cast<std::size_t>(channel) * output_tile.tile_elements;

    if (phase_size <= 1) {
      if (output.seed == DirectIbOutputSeed::PACKED_INPUT) {
        memcpy_vectorized(
            output_bytes,
            reinterpret_cast<const char*>(own_source),
            tile_bytes,
            group);
      }
      return;
    }

    for (int step = 0; step < phase_size - 1; ++step) {
      const int peer = direct_ib_reduce_scatter_peer_for_step(
          phase_rank,
          phase_size,
          channel,
          step,
          DirectIbReduceScatterRole::RECEIVE,
          kStaggerChannels);
      const char* local_input =
          output.seed == DirectIbOutputSeed::PACKED_INPUT && step == 0
          ? reinterpret_cast<const char*>(own_source)
          : output_bytes;
      auto transport = peers[peer];
      transport.template recv<ReduceOp>(
          group,
          output_bytes,
          tile_bytes,
          signaling_data_size,
          abort_device,
          local_input);
    }
    return;
  }

  if (phase_size <= 1) {
    return;
  }

  for (int step = 0; step < phase_size - 1; ++step) {
    const int peer = direct_ib_reduce_scatter_peer_for_step(
        phase_rank,
        phase_size,
        channel,
        step,
        DirectIbReduceScatterRole::SEND,
        kStaggerChannels);
    TiledBuffer<const T> send_tile(
        input.chunk_data(peer), input.chunk_elements, group);
    auto transport = peers[peer];
    transport.send(
        group,
        reinterpret_cast<const char*>(send_tile.data()),
        send_tile.bytes(),
        signaling_data_size,
        abort_device);
  }
}

} // namespace comms::prims

#undef PRIMS_DIRECT_IB_PHASE_HOST_DEVICE_INLINE
