// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/core/Timeout.cuh"

namespace comms::prims {
class P2pIbgdaTransportDevice;
} // namespace comms::prims

namespace comms::prims::benchmark {

// Unidirectional ANS send/recv launchers for the IBGDA transport.
//
// These drive a VARIABLE-SIZE CopyOp (`AnsCompress`) over the transport's
// blocking send()/recv(), which is what exercises the
// `if constexpr (detail::copyop_variable_size_v<CopyOp>)` compressed path
// added in D111967119 (the plain `Memcpy` launchers only ever compile/run
// the fixed-size branch). The compressed on-wire size is data-dependent, so
// the transport reserves a worst-case staging stride per sub-chunk and the
// RDMA put is sized from `AnsCompress::send()`'s returned byte count.

/**
 * Launch unidirectional ANS (compressed) tile send. All blocks send.
 * Grid: numBlocks. Block: 256 threads (8 warps == AnsCompress NumWarps).
 */
void launch_ibgda_send_ans(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch unidirectional ANS (compressed) tile recv. All blocks receive.
 * Grid: numBlocks. Block: 256 threads (8 warps == AnsCompress NumWarps).
 */
void launch_ibgda_recv_ans(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    AbortDevice abortDevice = AbortDevice());

} // namespace comms::prims::benchmark
