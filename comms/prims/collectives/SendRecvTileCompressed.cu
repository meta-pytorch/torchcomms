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
// `SendRecvTileCommon.cuh` reuses `AllToAllvTileCommon.cuh`, which
// conditionally pulls in `AllToAllvTileCompressed.cuh` (compressed-only
// dispatchers + `AnsCopyOp<NumWarps>` typedef + `kAnsMaxUncompBytes`)
// under PIPES_ENABLE_ANS_COMPRESSION, so this TU gets full ANS
// instantiation while the plain TU never sees nvcompdx symbols.
//
// Each `NumWarps` value that a caller actually launches must be
// explicitly instantiated here (kernels are not implicitly instantiated
// across TUs under `--device-c`).

#include "comms/prims/collectives/SendRecvTileCommon.cuh"
#include "comms/prims/collectives/SendRecvTileCompressed.cuh"
#include "comms/prims/core/Checks.h"

namespace comms::prims {

template <typename Compressor, int NumWarps, int MinBlocksPerSM>
__global__
__launch_bounds__(NumWarps * 32, MinBlocksPerSM) void sendrecv_tile_compressed_kernel(
    const __grid_constant__ SendRecvTileArgs args,
    Timeout timeout) {
  timeout.start();
  sendrecv_tile_impl<typename compressor_copyop<Compressor, NumWarps>::type>(
      args, timeout);
}

// Explicit instantiations. Each `(Compressor, NumWarps)` value that a caller
// actually launches must be explicitly instantiated here (kernels are not
// implicitly instantiated across TUs under `--device-c`). Today only the
// `AnsCompressor` tag is wired up (NumWarps ∈ {1, 2, 4, 8, 16, 32, 64} ⇔
// blockDim.x ∈ {32, 64, 128, 256, 512, 1024, 2048}). Each instantiation is its
// own `__global__` symbol with its own static `__shared__` reserved for ONLY
// that CopyOp's scratch. MinBlocksPerSM is the 2nd `__launch_bounds__` arg
// (resident blocks/SM ptxas must allow; caps registers). Default callers use
// 2; the SM-limited 256-thread bench config uses (8, 8) for 8 blocks/SM =
// 100% occupancy.
#define PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(                   \
    COMPRESSOR, NUM_WARPS, MIN_BLOCKS)                                   \
  template __global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS) void \
  sendrecv_tile_compressed_kernel<COMPRESSOR, NUM_WARPS, MIN_BLOCKS>(    \
      const __grid_constant__ SendRecvTileArgs args, Timeout timeout);

PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 1, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 2, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 4, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 8, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 16, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 32, 2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 64, 2)

// 100%-occupancy variant for 256 threads/block (NumWarps=8): m=8 caps
// registers at 65536/(256*8)=32/thread so 8 blocks/SM = 64 warps reside.
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(AnsCompressor, 8, 8)

#undef PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS

// Reads and then zeroes the global ANS byte counters on `stream`.
//
// PRECONDITION: the caller must have quiesced *all* streams that run compressed
// send/recv kernels before calling. The read and the subsequent reset are
// separated by a `cudaStreamSynchronize(stream)`; any kernel on another stream
// that increments `g_pipes_ans_total_*` in that window would have its
// contribution silently dropped by the reset. Using a single stream for both
// the compressed kernels and this call satisfies the precondition.
SendRecvAnsCompressStats fetch_and_reset_sendrecv_ans_compress_stats(
    cudaStream_t stream) {
  unsigned long long uncomp = 0, comp = 0;
  PIPES_CUDA_CHECK(cudaMemcpyFromSymbolAsync(
      &uncomp,
      g_pipes_ans_total_uncomp_bytes,
      sizeof(unsigned long long),
      0,
      cudaMemcpyDeviceToHost,
      stream));
  PIPES_CUDA_CHECK(cudaMemcpyFromSymbolAsync(
      &comp,
      g_pipes_ans_total_comp_bytes,
      sizeof(unsigned long long),
      0,
      cudaMemcpyDeviceToHost,
      stream));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream));
  const unsigned long long zero = 0ULL;
  PIPES_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      g_pipes_ans_total_uncomp_bytes,
      &zero,
      sizeof(unsigned long long),
      0,
      cudaMemcpyHostToDevice,
      stream));
  PIPES_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      g_pipes_ans_total_comp_bytes,
      &zero,
      sizeof(unsigned long long),
      0,
      cudaMemcpyHostToDevice,
      stream));
  PIPES_CUDA_CHECK(cudaStreamSynchronize(stream));
  return SendRecvAnsCompressStats{
      static_cast<uint64_t>(uncomp), static_cast<uint64_t>(comp)};
}

} // namespace comms::prims
