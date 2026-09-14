// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// ANS-compressed flavour of the sendrecv-tile kernel. Compiled with
// PIPES_ENABLE_ANS_COMPRESSION (set by `:copy_op_compress`) and
// `--device-c` so nvcompdx's relocatable device symbols can be
// device-linked against `:nvcompdx_fatbin` via
// `gen_alltoallv_tile_dlink_cmd`.
//
// The plain (no-compression) kernel symbol lives in the sibling
// SendRecvTile.cu translation unit (`:sendrecv_tile`).
// `SendRecvTileCommon.cuh` is self-contained -- it does NOT go through
// `AllToAllvTileCommon.cuh`. Under PIPES_ENABLE_ANS_COMPRESSION it pulls in
// `CompressorTypes.cuh`, which owns the compressed dispatchers, the
// `AnsCopyOp<NumWarps>` alias and `kAnsMaxUncompBytes`, so this TU gets full
// ANS instantiation while the plain TU never sees nvcompdx symbols.
//
// Each `NumWarps` value that a caller actually launches must be
// explicitly instantiated here (kernels are not implicitly instantiated
// across TUs under `--device-c`).

#include "comms/prims/collectives/SendRecvTileCommon.cuh"
#include "comms/prims/collectives/SendRecvTileCompressed.cuh"
#include "comms/prims/collectives/SendRecvTileCompressedKernel.cuh"
#include "comms/prims/core/Checks.h"

namespace comms::prims {

template <typename Compressor, int NumWarps, int MinBlocksPerSM>
__global__
__launch_bounds__(NumWarps * 32, MinBlocksPerSM) void sendrecv_tile_compressed_kernel(
    const __grid_constant__ SendRecvTileArgs args,
    AbortDevice abortDevice) {
  // `__launch_bounds__` only caps the maximum, it does not pin the shape: a
  // `<AnsCompressor, 16>` kernel launches happily with 256 threads even though
  // nvcompdx was instantiated as `BlockWarp<16>`. The codec is cooperative and
  // sizes its `__shared__` scratch off the compile-time warp count, so a
  // mismatch is not a slow path -- the block reads and writes scratch for warps
  // that are not there. Reject it up front with a message instead.
  //
  // Also require a 1-D block and grid: `sendrecv_tile_impl` derives its tiling
  // from `blockIdx.x` / `gridDim.x` alone, so a y/z extent would silently give
  // several blocks the same tile.
  if (blockDim.x != static_cast<unsigned>(NumWarps) * 32u || blockDim.y != 1u ||
      blockDim.z != 1u || gridDim.y != 1u || gridDim.z != 1u) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf(
          "[PIPES] FATAL: sendrecv_tile_compressed_kernel<_, %d> requires "
          "blockDim=(%u,1,1) and a 1-D grid, got blockDim=(%u,%u,%u) "
          "gridDim=(%u,%u,%u)\n",
          NumWarps,
          static_cast<unsigned>(NumWarps) * 32u,
          blockDim.x,
          blockDim.y,
          blockDim.z,
          gridDim.x,
          gridDim.y,
          gridDim.z);
    }
    PIPES_DEVICE_TRAP();
  }
  abortDevice.start();
  // `SendRecvStatsTag` names this device-link unit as the owner of the ANS byte
  // counters, so they get a symbol distinct from any other compressed
  // collective linked into the same binary. It is inert unless this TU is built
  // with PIPES_ANS_COLLECT_STATS.
  sendrecv_tile_impl<
      typename compressor_copyop<Compressor, NumWarps, SendRecvStatsTag>::type>(
      args, abortDevice);
}

// Explicit instantiations. Each `(Compressor, NumWarps)` value that a caller
// actually launches must be explicitly instantiated here (kernels are not
// implicitly instantiated across TUs under `--device-c`). Today only the
// `AnsCompressor` tag is wired up (NumWarps ∈ {1, 2, 4, 8, 16, 32} ⇔
// blockDim.x ∈ {32, 64, 128, 256, 512, 1024}). 64 warps is NOT instantiated:
// it would expand to __launch_bounds__(2048, ...), and CUDA caps a block at
// 1024 threads, so no launch could ever match it. Each instantiation is its
// own `__global__` symbol with its own static `__shared__` reserved for ONLY
// that CopyOp's scratch. MinBlocksPerSM is the 2nd `__launch_bounds__` arg
// (resident blocks/SM ptxas must allow; caps registers). Default callers use
// 2; the SM-limited 256-thread bench config uses (8, 8) for 8 blocks/SM =
// 100% occupancy.
#define PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(                   \
    COMPRESSOR, NUM_WARPS, MIN_BLOCKS)                                   \
  template __global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS) void \
  sendrecv_tile_compressed_kernel<COMPRESSOR, NUM_WARPS, MIN_BLOCKS>(    \
      const __grid_constant__ SendRecvTileArgs args, AbortDevice abortDevice);

PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 1, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 2, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 4, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 8, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 16, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 32, 2)

// 100%-occupancy variant for 256 threads/block (NumWarps=8): m=8 caps
// registers at 65536/(256*8)=32/thread so 8 blocks/SM = 64 warps reside.
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 8, 8)

#undef PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS

// Host-side map from a runtime launch shape to the matching instantiation
// above. This is the authoritative statement of what is supported: the seven
// cases here must mirror the seven instantiations exactly. A pair that exists
// in one list and not the other is the bug this indirection exists to catch --
// a missing instantiation becomes an undefined symbol at device link, and a
// missing case here becomes a `nullptr` the caller must handle.
#define PIPES_SENDRECV_COMPRESSED_KERNEL(NUM_WARPS, MIN_BLOCKS) \
  reinterpret_cast<void*>(                                      \
      &sendrecv_tile_compressed_kernel<AnsCompressor, NUM_WARPS, MIN_BLOCKS>)

void* pick_sendrecv_tile_compressed_kernel(
    int threads_per_block,
    int min_blocks_per_sm) {
  switch (threads_per_block) {
    case 32:
      return min_blocks_per_sm == 2 ? PIPES_SENDRECV_COMPRESSED_KERNEL(1, 2)
                                    : nullptr;
    case 64:
      return min_blocks_per_sm == 2 ? PIPES_SENDRECV_COMPRESSED_KERNEL(2, 2)
                                    : nullptr;
    case 128:
      return min_blocks_per_sm == 2 ? PIPES_SENDRECV_COMPRESSED_KERNEL(4, 2)
                                    : nullptr;
    case 256:
      // The only shape with a second MinBlocksPerSM: m=8 caps registers at
      // 65536/(256*8)=32/thread, so 8 blocks/SM = 64 warps reside.
      if (min_blocks_per_sm == 8) {
        return PIPES_SENDRECV_COMPRESSED_KERNEL(8, 8);
      }
      return min_blocks_per_sm == 2 ? PIPES_SENDRECV_COMPRESSED_KERNEL(8, 2)
                                    : nullptr;
    case 512:
      return min_blocks_per_sm == 2 ? PIPES_SENDRECV_COMPRESSED_KERNEL(16, 2)
                                    : nullptr;
    case 1024:
      return min_blocks_per_sm == 2 ? PIPES_SENDRECV_COMPRESSED_KERNEL(32, 2)
                                    : nullptr;
    default:
      // Includes 2048 (64 warps), which cannot exist -- CUDA caps a block at
      // 1024 threads, so `__launch_bounds__(2048, ...)` has no valid launch.
      return nullptr;
  }
}

#undef PIPES_SENDRECV_COMPRESSED_KERNEL

namespace {

// Evaluates the sizing helper under the REAL `__CUDA_ARCH__`. One thread: this
// is a pure compile-time-ish query, not a parallel computation.
template <int NumWarps>
__global__ void ans_worst_case_stride_kernel(
    std::size_t chunk_bytes,
    unsigned long long* out) {
  *out = static_cast<unsigned long long>(
      compressor_copyop<AnsCompressor, NumWarps, SendRecvStatsTag>::type::
          worst_case_chunk_stride(chunk_bytes));
}

template <int NumWarps>
std::size_t query_stride(std::size_t chunk_bytes) {
  unsigned long long* d_out = nullptr;
  PIPES_CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
  ans_worst_case_stride_kernel<NumWarps><<<1, 1>>>(chunk_bytes, d_out);
  PIPES_CUDA_CHECK(cudaGetLastError());
  unsigned long long host = 0;
  PIPES_CUDA_CHECK(
      cudaMemcpy(&host, d_out, sizeof(host), cudaMemcpyDeviceToHost));
  PIPES_CUDA_CHECK(cudaFree(d_out));
  return static_cast<std::size_t>(host);
}

} // namespace

std::size_t sendrecv_ans_worst_case_chunk_stride(
    int num_warps,
    std::size_t chunk_bytes) {
  switch (num_warps) {
    case 1:
      return query_stride<1>(chunk_bytes);
    case 2:
      return query_stride<2>(chunk_bytes);
    case 4:
      return query_stride<4>(chunk_bytes);
    case 8:
      return query_stride<8>(chunk_bytes);
    case 16:
      return query_stride<16>(chunk_bytes);
    case 32:
      return query_stride<32>(chunk_bytes);
    default:
      return 0;
  }
}

#ifdef PIPES_ANS_COLLECT_STATS
// Reads and then zeroes the global ANS byte counters on `stream`.
//
// Compiled ONLY into the stats-enabled flavour of this TU
// (`:sendrecv_tile_compressed_stats`). The production
// `:sendrecv_tile_compressed` is built without `-DPIPES_ANS_COLLECT_STATS`, so
// `ans_counters::*` do not exist there and neither does this function --
// deliberately, so that calling it without the stats dep is an undefined
// symbol at link time rather than a silent stream of zeros.
//
// PRECONDITION: the caller must have quiesced *all* streams that run compressed
// send/recv kernels before calling. The read and the subsequent reset are
// separated by a `cudaStreamSynchronize(stream)`; any kernel on another stream
// that increments this unit's `ans_counters::*<SendRecvStatsTag>` in that
// window would have its contribution silently dropped by the reset. Using a
// single stream for both the compressed kernels and this call satisfies the
// precondition.
namespace {

// The four cudaMemcpy*Symbol* calls differ only in symbol and direction;
// factor them so the snapshot-and-reset reads as what it is rather than as
// memcpy plumbing.
template <typename Symbol>
unsigned long long read_counter(const Symbol& symbol, cudaStream_t stream) {
  unsigned long long value = 0;
  PIPES_CUDA_CHECK(cudaMemcpyFromSymbolAsync(
      &value,
      symbol,
      sizeof(unsigned long long),
      0,
      cudaMemcpyDeviceToHost,
      stream));
  return value;
}

template <typename Symbol>
void reset_counter(const Symbol& symbol, cudaStream_t stream) {
  static const unsigned long long kZero = 0ULL;
  PIPES_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      symbol,
      &kZero,
      sizeof(unsigned long long),
      0,
      cudaMemcpyHostToDevice,
      stream));
}

} // namespace

SendRecvAnsCompressStats fetch_and_reset_sendrecv_ans_compress_stats(
    cudaStream_t stream) {
  const unsigned long long uncomp =
      read_counter(ans_counters::uncomp_bytes<SendRecvStatsTag>, stream);
  const unsigned long long comp =
      read_counter(ans_counters::comp_bytes<SendRecvStatsTag>, stream);
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream));
  reset_counter(ans_counters::uncomp_bytes<SendRecvStatsTag>, stream);
  reset_counter(ans_counters::comp_bytes<SendRecvStatsTag>, stream);
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream));
  return SendRecvAnsCompressStats{
      static_cast<uint64_t>(uncomp), static_cast<uint64_t>(comp)};
}
#endif // PIPES_ANS_COLLECT_STATS

} // namespace comms::prims
