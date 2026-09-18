// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/ReduceScatterDirectIbV2.cuh"

#include "comms/prims/collectives/ReduceScatterDirectIbCore.cuh"

#include "comms/prims/core/Checks.h"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/P2pIbTransportProgressImpl.cuh"

namespace comms::prims {

namespace {

// Peers whose streams are in flight together. At W <= 9 this is one wave and
// the accumulator is hoisted across every peer; larger worlds run in waves and
// fold each wave into the output, costing an extra read+write per wave.
constexpr int kMaxPeersInFlight = 8;

// 8 floats as two float4s. Deliberately not a float[8]: passing a local array
// by reference defeats register allocation and the accumulator spills to local
// memory, which is the HBM traffic the hoist exists to avoid.
struct Acc8 {
  float4 lo;
  float4 hi;
};

__device__ __forceinline__ Acc8 unpack_bf16x8(const uint4& raw) {
  const auto* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  const float2 a = __bfloat1622float2(h[0]);
  const float2 b = __bfloat1622float2(h[1]);
  const float2 c = __bfloat1622float2(h[2]);
  const float2 d = __bfloat1622float2(h[3]);
  Acc8 r;
  r.lo = make_float4(a.x, a.y, b.x, b.y);
  r.hi = make_float4(c.x, c.y, d.x, d.y);
  return r;
}

__device__ __forceinline__ void acc_add(Acc8& a, const Acc8& b) {
  a.lo.x += b.lo.x;
  a.lo.y += b.lo.y;
  a.lo.z += b.lo.z;
  a.lo.w += b.lo.w;
  a.hi.x += b.hi.x;
  a.hi.y += b.hi.y;
  a.hi.z += b.hi.z;
  a.hi.w += b.hi.w;
}

__device__ __forceinline__ bool all_aligned(
    const detail::RecvChunkAcquisition* views,
    int count) {
  for (int i = 0; i < count; ++i) {
    if ((reinterpret_cast<uintptr_t>(views[i].staging) & 15u) != 0) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ std::size_t send_offset_bytes(
    const DirectReduceScatterIbV2Args& args,
    int peer,
    std::size_t tile_offset_elements) {
  return (static_cast<std::size_t>(peer) * args.chunk_elements +
          tile_offset_elements) *
      sizeof(__nv_bfloat16);
}

// Sum this rank's own shard with every peer's landed tile and write FP32 once.
//
// The peer loop is unrolled over a compile-time bound and the staging pointers
// are hoisted into registers first. With a runtime peer count the compiler
// cannot unroll, and every global load then waits on a shared-memory load to
// resolve its own address, serialising the eight loads.
template <int kUnroll>
__device__ __forceinline__ void reduce_chunk(
    float* out_base,
    const __nv_bfloat16* own_base,
    const detail::RecvChunkAcquisition* views,
    int count,
    int wave_base,
    std::size_t tid,
    std::size_t stride) {
  // validBytes, not protocolBytes: the RDMA write covers only validBytes,
  // while the rounding and tail padding are credit-only and hold stale bytes
  // from whichever chunk last used this slot. Deriving elems from anything
  // else here reads plausible-looking garbage.
  static_assert(
      kUnroll == 1,
      "the vector loop drops a trailing partial group when kUnroll > 1; "
      "write the residual loop before raising it");
  const std::size_t elems = views[0].validBytes / sizeof(__nv_bfloat16);
  const std::size_t elem_off = views[0].dataOff / sizeof(__nv_bfloat16);
  float* const out = out_base + elem_off;
  const __nv_bfloat16* const own = own_base + elem_off;

  constexpr std::size_t kVec = 8;
  const bool vec_ok = elems >= kVec &&
      ((reinterpret_cast<uintptr_t>(out) & 15u) == 0) &&
      ((reinterpret_cast<uintptr_t>(own) & 15u) == 0) &&
      all_aligned(views, count);
  const std::size_t nvec = vec_ok ? elems / kVec : 0;

  const uint4* sp[kMaxPeersInFlight];
#pragma unroll
  for (int i = 0; i < kMaxPeersInFlight; ++i) {
    sp[i] = reinterpret_cast<const uint4*>(views[i < count ? i : 0].staging);
  }
  const uint4* const ownv = reinterpret_cast<const uint4*>(own);
  float4* const outv = reinterpret_cast<float4*>(out);

  std::size_t v = tid;
  for (; v + (kUnroll - 1) * stride < nvec; v += kUnroll * stride) {
    Acc8 acc[kUnroll];
    // Wave 0 seeds from this rank's own shard; later waves fold into what the
    // previous wave already wrote.
    if (wave_base == 0) {
      uint4 o[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        o[u] = ownv[v + u * stride];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        acc[u] = unpack_bf16x8(o[u]);
      }
    } else {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        acc[u].lo = outv[2 * (v + u * stride)];
        acc[u].hi = outv[2 * (v + u * stride) + 1];
      }
    }
#pragma unroll
    for (int i = 0; i < kMaxPeersInFlight; ++i) {
      if (i >= count) {
        continue;
      }
      uint4 raw[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        raw[u] = sp[i][v + u * stride];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        acc_add(acc[u], unpack_bf16x8(raw[u]));
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      outv[2 * (v + u * stride)] = acc[u].lo;
      outv[2 * (v + u * stride) + 1] = acc[u].hi;
    }
  }
  // Ragged tail, and the whole chunk when alignment rules out the vector path.
  for (std::size_t idx = nvec * kVec + tid; idx < elems; idx += stride) {
    float a = (wave_base == 0) ? __bfloat162float(own[idx]) : out[idx];
    for (int i = 0; i < count; ++i) {
      a += __bfloat162float(
          reinterpret_cast<const __nv_bfloat16*>(views[i].staging)[idx]);
    }
    out[idx] = a;
  }
}

} // namespace

// BF16 in, BF16 on the wire, FP32 out. The receive side is warp-specialised:
// one warp owns the network while the rest only reduce, so a chunk's
// DATA_READY latency and the previous chunk's SLOT_FREE round trip overlap the
// current chunk's arithmetic instead of serialising with it.
template <int kRecvThreads, int kBlockSize, int kUnroll>
__global__
__launch_bounds__(kBlockSize, 1) void direct_reduce_scatter_ib_v2_kernel(
    const __grid_constant__ DirectReduceScatterIbV2Args args,
    AbortDevice abortDevice) {
#ifdef __CUDA_ARCH__
  abortDevice.start();

  // The send only needs its leader to post a WQE; one warp keeps its sync at
  // __syncwarp() and hands every other thread to the reduce.
  constexpr int kSendThreads = comms::device::kWarpSize;
  static_assert(kRecvThreads % comms::device::kWarpSize == 0);
  static_assert(kSendThreads + kRecvThreads == kBlockSize);

  const ThreadGroup block = make_block_group();
  const bool is_recv = block.thread_id_in_group < kRecvThreads;
  ThreadGroup group = is_recv
      ? ThreadGroup{
            .thread_id_in_group = block.thread_id_in_group,
            .group_size = kRecvThreads,
            .group_id = block.group_id,
            .block_id = block.block_id,
            .total_groups = block.total_groups,
            .scope = SyncScope::MULTIWARP,
            .barrier_id = 1}
      : ThreadGroup{
            .thread_id_in_group = block.thread_id_in_group - kRecvThreads,
            .group_size = kSendThreads,
            .group_id = block.group_id,
            .block_id = block.block_id,
            .total_groups = block.total_groups,
            .scope = SyncScope::WARP,
            .barrier_id = ThreadGroup::kAutoBarrierId};

  const int channel = static_cast<int>(group.group_id);
  const int my_rank = args.my_rank;
  const int W = args.num_ranks;
  const std::size_t max_sig = args.signaling_data_size;
  if (W <= 1) {
    return;
  }

  // Both roles tile in wire (bf16) elements off the same chunk_elements so the
  // two sides agree on each channel's byte range. Tiling the fp32 output
  // instead would round to a different multiple and desync them.
  TiledBuffer<const __nv_bfloat16> wire_tile(
      args.input, args.chunk_elements, group);
  const std::size_t wire_bytes = wire_tile.bytes();
  if (wire_bytes == 0) {
    return;
  }
  const std::size_t tile_offset_elements =
      static_cast<std::size_t>(channel) * wire_tile.tile_elements;
  const int num_peers = W - 1;

  // Solo THREAD-scope group: the progress calls are leader-only and their
  // internal group.sync() degenerates to a no-op, so one thread can drive every
  // peer without dragging the other warps into a barrier per peer.
  ThreadGroup solo{
      0,
      1,
      group.group_id,
      group.block_id,
      group.total_groups,
      SyncScope::THREAD,
      ThreadGroup::kAutoBarrierId};

  if (is_recv) {
    // Index output and own shard by the SAME element offsets as the wire tile.
    float* const out_base = args.output + tile_offset_elements;
    const __nv_bfloat16* const own_base = args.input +
        static_cast<std::size_t>(my_rank) * args.chunk_elements +
        tile_offset_elements;

    // Two slots so the bookkeeper can acquire chunk c+1 while the reducers are
    // still on chunk c. With one slot the acquire cannot start until the reduce
    // finishes, and the network stops overlapping the arithmetic entirely.
    constexpr int kSlots = 2;
    // Chunk c-1 is released only after chunk c is acquired, so the pipeline
    // must hold two chunks at once. A shallower pipeline never completes the
    // acquire and the bookkeeper stalls with no diagnostic.
    if (group.is_leader()) {
      auto probe = args.peers[(my_rank + 1) % W];
      if (probe.pipeline_depth() < kSlots) {
        printf(
            "[PIPES] FATAL: DirectIB v2 needs pipeline_depth >= %d, got %d\n",
            kSlots,
            probe.pipeline_depth());
        PIPES_DEVICE_TRAP();
      }
    }
    __shared__ int rpeer_of[kMaxPeersInFlight];
    __shared__ detail::RecvChunkAcquisition rviews[kSlots][kMaxPeersInFlight];
    __shared__ volatile int s_published;
    __shared__ volatile int s_consumed;
    __shared__ volatile int s_nchunks;

    constexpr int kBookThreads = comms::device::kWarpSize;
    const bool is_book = group.thread_id_in_group < kBookThreads;

    for (int base = 0; base < num_peers; base += kMaxPeersInFlight) {
      const int count = min(kMaxPeersInFlight, num_peers - base);

      if (group.is_leader()) {
        for (int i = 0; i < count; ++i) {
          const int step = base + i;
          rpeer_of[i] = direct_ib_reduce_scatter_peer_for_step(
              my_rank, W, channel, step, DirectIbReduceScatterRole::RECEIVE);
          for (int sl = 0; sl < kSlots; ++sl) {
            rviews[sl][i] = detail::RecvChunkAcquisition{};
          }
          auto transport = args.peers[rpeer_of[i]];
          // The acquire path hands back staging pointers and reduces out of
          // them directly, so this operation has no destination buffer.
          transport.init_recv_progress(
              solo, /*dst=*/nullptr, wire_bytes, max_sig);
        }
        s_published = -1;
        s_consumed = -1;
        s_nchunks = -1;
      }
      group.sync();

      if (is_book) {
        if (group.thread_id_in_group == 0) {
          int c = 0;
          int released = -1;
          while (true) {
            const int s = c % kSlots;
            // Acquire chunk c from every peer. Each call is non-blocking, so a
            // slow peer never stops us polling the others.
            int ready = 0;
            bool finished = false;
            bool aborted = false;
            while (ready < count && !finished && !aborted) {
              ready = 0;
              for (int i = 0; i < count; ++i) {
                if (rviews[s][i].staging != nullptr) {
                  ++ready;
                  continue;
                }
                detail::RecvChunkAcquisition view{};
                auto transport = args.peers[rpeer_of[i]];
                const auto st = transport.progress_recv_acquire_once(
                    solo, abortDevice, view);
                // `Aborted` is terminal and must be handled explicitly. It used
                // to fall through this chain: the acquire had already driven
                // the slot to `Done` via abandon_progress_state(), so the NEXT
                // acquire returned `Done`, the loop set `finished` and the
                // collective reported ordinary completion over a partially
                // reduced accumulator. Stopping here keeps "aborted" and
                // "stream complete" distinguishable.
                if (st == IbgdaSendRecvProgressStatus::Aborted) {
                  aborted = true;
                  break;
                }
                if (st == IbgdaSendRecvProgressStatus::Done) {
                  finished = true;
                  break;
                }
                if (st == IbgdaSendRecvProgressStatus::Progressed) {
                  rviews[s][i] = view;
                  ++ready;
                }
              }
            }
            if (finished || aborted) {
              s_nchunks = c;
              __threadfence_block();
              s_published = c;
              break;
            }

            __threadfence_block();
            s_published = c;

            // Release chunk c-1 only now: the acquire above already overlapped
            // the reducers' work on it, which is the point of the second slot.
            if (c >= 1) {
              while (s_consumed < c - 1) {
              }
              const int ps = (c - 1) % kSlots;
              for (int i = 0; i < count; ++i) {
                auto transport = args.peers[rpeer_of[i]];
                transport.progress_recv_release_once(
                    solo, abortDevice, rviews[ps][i]);
                rviews[ps][i] = detail::RecvChunkAcquisition{};
              }
              released = c - 1;
            }
            ++c;
          }
          // Drain the slots the reducers still hold.
          while (released < c - 1) {
            const int d = released + 1;
            while (s_consumed < d) {
            }
            const int ds = d % kSlots;
            for (int i = 0; i < count; ++i) {
              auto transport = args.peers[rpeer_of[i]];
              transport.progress_recv_release_once(
                  solo, abortDevice, rviews[ds][i]);
              rviews[ds][i] = detail::RecvChunkAcquisition{};
            }
            released = d;
          }
        }
      } else {
        // Reducers get their own barrier id so the bookkeeper warp is not
        // counted in it.
        ThreadGroup rgroup{
            .thread_id_in_group = group.thread_id_in_group - kBookThreads,
            .group_size = kRecvThreads - kBookThreads,
            .group_id = group.group_id,
            .block_id = group.block_id,
            .total_groups = group.total_groups,
            .scope = SyncScope::MULTIWARP,
            .barrier_id = 3};
        int c = 0;
        while (true) {
          while (s_published < c) {
            if (s_nchunks >= 0 && c >= s_nchunks) {
              break;
            }
          }
          if (s_nchunks >= 0 && c >= s_nchunks) {
            break;
          }
          __threadfence_block();
          reduce_chunk<kUnroll>(
              out_base,
              own_base,
              rviews[c % kSlots],
              count,
              base,
              rgroup.thread_id_in_group,
              rgroup.group_size);
          rgroup.sync();
          if (rgroup.thread_id_in_group == 0) {
            __threadfence_block();
            s_consumed = c;
          }
          ++c;
        }
      }
      group.sync();
    }
  } else if (group.is_leader()) {
    // Chunk-outer / peer-inner: chunk c goes to every peer before chunk c+1,
    // so a receiver gets the same byte range from all peers at once (what the
    // hoist needs) and all W-1 QPs stay busy instead of one.
    //
    // progress_registered_send_once() is non-blocking for its protocol
    // dependencies but not for queue space: the WQE reservation spins when a
    // peer's send queue is full. It is a stall, not a deadlock -- each QP has
    // one driver thread -- but it idles this round-robin. Outstanding WQEs are
    // (window / chunkWire) * (ceil(chunkWire / max_write) + 1), which must stay
    // under sq_wqe_num; a small max_signal_bytes is what breaches it.
    for (int base = 0; base < num_peers; base += kMaxPeersInFlight) {
      const int count = min(kMaxPeersInFlight, num_peers - base);
      int peer_of[kMaxPeersInFlight];
      bool posted[kMaxPeersInFlight];
      for (int i = 0; i < count; ++i) {
        const int step = base + i;
        peer_of[i] = direct_ib_reduce_scatter_peer_for_step(
            my_rank, W, channel, step, DirectIbReduceScatterRole::SEND);
        posted[i] = false;
        auto transport = args.peers[peer_of[i]];
        transport.init_registered_send_progress(
            solo,
            args.input_reg.subBuffer(
                send_offset_bytes(args, peer_of[i], tile_offset_elements)),
            wire_bytes,
            max_sig);
      }

      int remaining = count;
      while (remaining > 0) {
        for (int i = 0; i < count; ++i) {
          if (posted[i]) {
            continue;
          }
          auto transport = args.peers[peer_of[i]];
          const auto st =
              transport.progress_registered_send_once(solo, abortDevice);
          if (st == IbgdaRegisteredSendProgressStatus::Posted) {
            posted[i] = true;
            --remaining;
          }
        }
      }

      // The NIC reads the caller's input buffer directly, so the transfer is
      // not finished when the last WQE is posted. Drain before returning or the
      // caller may reuse that buffer while it is still being read.
      for (int i = 0; i < count; ++i) {
        auto transport = args.peers[peer_of[i]];
        while (
            transport.progress_registered_send_drain_once(solo, abortDevice) !=
            IbgdaRegisteredSendProgressStatus::Drained) {
        }
      }
    }
  }
#endif
}

template __global__ void direct_reduce_scatter_ib_v2_kernel<224, 256, 1>(
    const __grid_constant__ DirectReduceScatterIbV2Args,
    AbortDevice);
template __global__ void direct_reduce_scatter_ib_v2_kernel<480, 512, 1>(
    const __grid_constant__ DirectReduceScatterIbV2Args,
    AbortDevice);
template __global__ void direct_reduce_scatter_ib_v2_kernel<736, 768, 1>(
    const __grid_constant__ DirectReduceScatterIbV2Args,
    AbortDevice);
template __global__ void direct_reduce_scatter_ib_v2_kernel<992, 1024, 1>(
    const __grid_constant__ DirectReduceScatterIbV2Args,
    AbortDevice);

void launch_direct_reduce_scatter_ib_v2(
    const DirectReduceScatterIbV2Args& args,
    int num_blocks,
    int block_threads,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  switch (block_threads) {
    case 256:
      direct_reduce_scatter_ib_v2_kernel<224, 256, 1>
          <<<num_blocks, 256, 0, stream>>>(args, abortDevice);
      break;
    case 512:
      direct_reduce_scatter_ib_v2_kernel<480, 512, 1>
          <<<num_blocks, 512, 0, stream>>>(args, abortDevice);
      break;
    case 768:
      direct_reduce_scatter_ib_v2_kernel<736, 768, 1>
          <<<num_blocks, 768, 0, stream>>>(args, abortDevice);
      break;
    default:
      direct_reduce_scatter_ib_v2_kernel<992, 1024, 1>
          <<<num_blocks, 1024, 0, stream>>>(args, abortDevice);
      break;
  }
  PIPES_CUDA_CHECK(cudaGetLastError());
}

} // namespace comms::prims
