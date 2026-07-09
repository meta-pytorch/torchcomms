// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Public header for the ANS-compressed flavour of the sendrecv-tile
// kernel. Provided by the `:sendrecv_tile_compressed` BUCK target (which
// device-links the nvcompdx fatbin and pulls in the `:copy_op_compress`
// headers). Consumers that want compressed IBGDA chunks should depend on
// that target and `#include` this header for:
//
//   - `sendrecv_tile_compressed_kernel<Compressor, NumWarps>` kernel symbol
//   - `SendRecvAnsCompressStats` +
//     `fetch_and_reset_sendrecv_ans_compress_stats(stream)`
//
// The compressed kernel reuses `SendRecvTileArgs` from the sibling
// `SendRecvTile.cuh`. Compress requires `blockDim.x` to be one of {32,
// 64, 128, 256, 512, 1024}. NO dynamic shared memory needs to be
// reserved by the launcher: `AnsCompress::send` / `recv` declare their
// per-direction `__shared__` scratch internally, sized at compile time
// off nvcompdx's `constexpr shmem_size_group()`. Pass `0` as
// `cudaLaunchKernel`'s dynamic-shmem argument. NVL transport peers
// always use plain `Memcpy` regardless of which kernel is launched —
// compression only applies to IBGDA peers.
//
// SendRecvTile is self-contained: the device-only dispatchers below are
// a private copy (not shared with AllToAllvTile), so this collective
// depends only on the core `:copy_op_compress` layer and can sit
// anywhere in a stack that has the ANS CopyOp available. The dispatchers
// are gated by `#ifdef PIPES_ENABLE_ANS_COMPRESSION` so host-only
// consumers (benchmarks, tests) see only the public kernel decls + stats
// accessor.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "comms/prims/collectives/SendRecvTile.cuh"

namespace comms::prims {

// Host-visible tag selecting a compressor family for
// `sendrecv_tile_compressed_kernel`. The kernel is generic over the
// compressor: the device translation unit maps each tag + NumWarps to a
// concrete CopyOp (see `compressor_copyop` below). This tag is nameable from
// host TUs (benchmarks, tests) that cannot include the device-only
// `CopyOp.cuh`, so they can select the compressor at the launch site and read
// the launch-time constants below. Plug in a different compression algorithm
// by adding a new tag here, a `compressor_copyop` specialization, and its
// explicit kernel instantiations in `SendRecvTileCompressed.cu`.
//
// `kMaxUncompBytes` is the max uncompressed sub-chunk the compressor stages; a
// host launcher sizes the per-block realign aux buffer as
// `gridDim.x * kMaxUncompBytes`. It has to be a host-visible literal because
// the concrete CopyOp's copy of it lives inside the device-only
// `PIPES_ENABLE_ANS_COMPRESSION` region of `CopyOp.cuh`; a device-side
// `static_assert` (below) pins it to `AnsCompress::kMaxUncompBytes` so the two
// can never drift.
struct AnsCompressor {
  static constexpr std::size_t kMaxUncompBytes = 256ULL * 1024;
};

#ifdef __CUDACC__
/**
 * Compressed point-to-point tile send/recv. Same shape as
 * `sendrecv_tile_kernel` but compresses every IBGDA sub-chunk.
 *
 * Templated on a `Compressor` tag (e.g. `AnsCompressor`) and
 * `NumWarps = blockDim.x / 32`. The device TU maps `(Compressor, NumWarps)`
 * to a concrete CopyOp via `compressor_copyop` (for `AnsCompressor` this is
 * `AnsCompress<NumWarps, kAnsMaxUncompBytes>`), so the per-block static
 * `__shared__` footprint is just that one CopyOp's scratch. Callers pick the
 * matching `Compressor` + `NumWarps` at the launch site. Explicit
 * instantiations are emitted in `SendRecvTileCompressed.cu` for
 * `AnsCompressor` and `NumWarps in {1,2,4,8,16,32,64}`; callers that launch
 * with another tag / NumWarps must add the explicit instantiation there.
 */
template <typename Compressor, int NumWarps, int MinBlocksPerSM = 2>
__global__
    __launch_bounds__(NumWarps * 32, MinBlocksPerSM) void sendrecv_tile_compressed_kernel(
        const __grid_constant__ SendRecvTileArgs args,
        Timeout timeout = Timeout());
#else
template <typename Compressor, int NumWarps, int MinBlocksPerSM = 2>
void sendrecv_tile_compressed_kernel(SendRecvTileArgs args, Timeout timeout);
#endif

/**
 * Snapshot + reset of the running ANS compression byte counters fed by
 * `AnsCompress::send` inside the IBGDA dispatcher. Returns
 * `(uncompressed_bytes, compressed_bytes)` accumulated since the last
 * call, then atomically resets both device counters to 0.
 *
 * Reads the same `g_pipes_ans_total_*` device globals as AllToAllvTile's
 * `fetch_and_reset_ans_compress_stats` (they are `inline __device__`, so
 * merged to a single definition across TUs); the distinct name here only
 * avoids a host-symbol clash if a binary links both collectives'
 * compressed objects.
 */
struct SendRecvAnsCompressStats {
  uint64_t uncompressed_bytes;
  uint64_t compressed_bytes;
};

SendRecvAnsCompressStats fetch_and_reset_sendrecv_ans_compress_stats(
    cudaStream_t stream = nullptr);

#if defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)
} // namespace comms::prims

// CopyOp.cuh is only safe to include when both PIPES_ENABLE_ANS_COMPRESSION
// AND __CUDACC__ are defined. Pulled in here (out of the namespace) so
// the device helpers below can refer to `AnsCompress<>` and the global
// byte counters. Host consumers skip both the include and the
// dispatchers and only see the public decls above.
#include "comms/prims/core/CopyOp.cuh"

namespace comms::prims {

namespace {

// Maximum uncompressed sub-chunk size — sourced from the
// `PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES` macro defined in CopyOp.cuh, so
// the 256 KiB literal lives in a single place.
constexpr std::size_t kAnsMaxUncompBytes = PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES;

// CopyOp tag templated on `NumWarps`. Each `AnsCopyOp<N>` is a distinct
// type — instantiate exactly one per kernel so the static `__shared__`
// arrays declared inside `AnsCompress<N, ...>::send` / `::recv` only
// allocate space for the single NumWarps the kernel actually uses.
template <int NumWarps>
using AnsCopyOp = AnsCompress<NumWarps, kAnsMaxUncompBytes>;

// Maps a host-visible compressor tag + NumWarps to the concrete CopyOp that
// `sendrecv_tile_compressed_kernel` drives via `sendrecv_tile_impl<CopyOp>`.
// The kernel is generic over the tag; add a specialization here to support a
// new compressor family (and add its explicit instantiations in the .cu).
template <typename Compressor, int NumWarps>
struct compressor_copyop;

template <int NumWarps>
struct compressor_copyop<AnsCompressor, NumWarps> {
  using type = AnsCopyOp<NumWarps>;
};

// The host-visible tag constant MUST equal the concrete CopyOp's max
// uncompressed chunk (both ultimately `PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES`),
// or a host launcher would under-size the per-block realign aux buffer.
static_assert(
    AnsCompressor::kMaxUncompBytes == AnsCopyOp<1>::kMaxUncompBytes,
    "AnsCompressor::kMaxUncompBytes must match the device "
    "AnsCompress::kMaxUncompBytes");

// Sentinel-only default for the compressed IB path. Maps the
// `0 = one signal per perBlockSlot` sentinel to the LARGEST multiple of
// `kAnsMaxUncompBytes` whose worst-case on-wire chunked-ANS layout fits
// within `perBlockSlot`. Non-zero values are passed through untouched.
template <typename CopyOp>
__device__ __forceinline__ std::size_t default_max_signal_bytes_for_compress(
    std::size_t max_signal_bytes,
    P2pIbgdaTransportDevice& tr,
    int active_blocks) {
  if (max_signal_bytes != 0) {
    return max_signal_bytes;
  }
  const auto& state = tr.send_recv_state();
  const int effActive = active_blocks > 0 ? active_blocks : state.maxGroups;
  const std::size_t perBlockSlot = (state.dataBufferSize / effActive) & ~15ULL;
  return CopyOp::max_safe_chunk_size_for_slot(perBlockSlot);
}

// Compile-time-typed send dispatcher. Templated on the concrete
// `CopyOp = AnsCompress<NumWarps, MaxUncomp>` so the caller's kernel
// instantiation picks exactly one `AnsCompress<...>::send` instantiation.
template <typename CopyOp>
__device__ __forceinline__ void ibgda_send_compressed(
    P2pIbgdaTransportDevice& tr,
    ThreadGroup& group,
    void* src,
    std::size_t nbytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const Timeout& timeout,
    char* alignedAuxBuf) {
  const std::size_t effective_max_signal_bytes =
      default_max_signal_bytes_for_compress<CopyOp>(
          max_signal_bytes, tr, active_blocks);
  tr.send<CopyOp>(
      group,
      src,
      nbytes,
      active_blocks,
      effective_max_signal_bytes,
      timeout,
      alignedAuxBuf);
}

// Compile-time-typed recv dispatcher. Mirror of `ibgda_send_compressed`.
template <typename CopyOp>
__device__ __forceinline__ void ibgda_recv_compressed(
    P2pIbgdaTransportDevice& tr,
    ThreadGroup& group,
    void* dst,
    std::size_t nbytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const Timeout& timeout) {
  const std::size_t effective_max_signal_bytes =
      default_max_signal_bytes_for_compress<CopyOp>(
          max_signal_bytes, tr, active_blocks);
  tr.recv<CopyOp>(
      group, dst, nbytes, active_blocks, effective_max_signal_bytes, timeout);
}

} // namespace

#endif // defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)

} // namespace comms::prims
