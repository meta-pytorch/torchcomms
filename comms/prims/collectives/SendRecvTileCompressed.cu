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

namespace comms::prims {

template <int NumWarps>
__global__
__launch_bounds__(NumWarps * 32, 2) void sendrecv_tile_compressed_kernel(
    const __grid_constant__ SendRecvTileArgs args,
    Timeout timeout) {
  timeout.start();
  sendrecv_tile_impl<AnsCopyOp<NumWarps>>(args, timeout);
}

// Explicit instantiations for NumWarps ∈ {1, 2, 4, 8, 16, 32, 64}
// (⇔ blockDim.x ∈ {32, 64, 128, 256, 512, 1024, 2048}). Each
// instantiation is its own `__global__` symbol with its own static
// `__shared__` reserved for ONLY that NumWarps' `AnsCompress` scratch.
#define PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(NUM_WARPS) \
  template __global__ __launch_bounds__(NUM_WARPS * 32, 2) void  \
  sendrecv_tile_compressed_kernel<NUM_WARPS>(                    \
      const __grid_constant__ SendRecvTileArgs args, Timeout timeout);

PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(1)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(2)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(4)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(8)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(16)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(32)
PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS(64)

#undef PIPES_INSTANTIATE_SENDRECV_COMPRESSED_KERNELS

SendRecvAnsCompressStats fetch_and_reset_sendrecv_ans_compress_stats(
    cudaStream_t stream) {
  unsigned long long uncomp = 0, comp = 0;
  cudaMemcpyFromSymbolAsync(
      &uncomp,
      g_pipes_ans_total_uncomp_bytes,
      sizeof(unsigned long long),
      0,
      cudaMemcpyDeviceToHost,
      stream);
  cudaMemcpyFromSymbolAsync(
      &comp,
      g_pipes_ans_total_comp_bytes,
      sizeof(unsigned long long),
      0,
      cudaMemcpyDeviceToHost,
      stream);
  cudaStreamSynchronize(stream);
  const unsigned long long zero = 0ULL;
  cudaMemcpyToSymbolAsync(
      g_pipes_ans_total_uncomp_bytes,
      &zero,
      sizeof(unsigned long long),
      0,
      cudaMemcpyHostToDevice,
      stream);
  cudaMemcpyToSymbolAsync(
      g_pipes_ans_total_comp_bytes,
      &zero,
      sizeof(unsigned long long),
      0,
      cudaMemcpyHostToDevice,
      stream);
  cudaStreamSynchronize(stream);
  return SendRecvAnsCompressStats{
      static_cast<uint64_t>(uncomp), static_cast<uint64_t>(comp)};
}

} // namespace comms::prims
