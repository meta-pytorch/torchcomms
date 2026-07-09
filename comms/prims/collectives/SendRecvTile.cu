// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Plain (no-compression) flavour of the sendrecv-tile kernel. Built
// without PIPES_ENABLE_ANS_COMPRESSION; never sees nvcompdx symbols. The
// compressed kernel symbols (`sendrecv_tile_compressed_kernel<Compressor,
// NumWarps>`) and `fetch_and_reset_sendrecv_ans_compress_stats` live in the
// sibling SendRecvTileCompressed.cu translation unit.

#include "comms/prims/collectives/SendRecvTileCommon.cuh"

namespace comms::prims {

__global__ __launch_bounds__(512, 2) void sendrecv_tile_kernel(
    const __grid_constant__ SendRecvTileArgs args,
    Timeout timeout) {
  timeout.start();
  sendrecv_tile_impl<Memcpy>(args, timeout);
}

} // namespace comms::prims
