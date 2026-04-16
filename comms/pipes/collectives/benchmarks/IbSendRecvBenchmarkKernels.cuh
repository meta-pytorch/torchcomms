// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Benchmark-only CUDA kernel declarations for IB put BW measurement.
// Host-safe header — can be included from .cc files compiled by clang.

#pragma once

#include <cuda_runtime.h>
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLXTypes.hpp"

namespace comms::pipes::ib::benchmark {

using torchcomms::device::DeviceWindowNCCL;
using torchcomms::device::RegisteredBufferNCCL;

/**
 * Launch a zero-copy put BW kernel (benchmark only).
 *
 * Each block puts total_bytes/num_blocks per iteration.
 * Signal layout: 2*num_blocks signals.
 *   [0, num_blocks)            — DATA_READY (piggybacked on put)
 *   [num_blocks, 2*num_blocks) — ACK (receiver signals sender)
 *
 * @param win          Device window handle.
 * @param src_buf      Registered source buffer.
 * @param total_bytes  Total transfer size per iteration.
 * @param dst_rank     Rank to put to.
 * @param src_rank     Rank to receive from.
 * @param num_blocks   Number of blocks in the grid.
 * @param signal_base  Base offset for signal values.
 * @param iterations   Number of iterations.
 * @param stream       CUDA stream.
 */
void launch_put_bw_kernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t total_bytes,
    int dst_rank,
    int src_rank,
    int num_blocks,
    int signal_base,
    int iterations,
    cudaStream_t stream);

} // namespace comms::pipes::ib::benchmark
