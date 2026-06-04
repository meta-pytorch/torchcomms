// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/ctran/prims/Timeout.cuh"

namespace ctran::prims {
class P2pIbgdaTransportDevice;
} // namespace ctran::prims

namespace ctran::prims::benchmark {

void launch_ibgda_forward_chain(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    int my_rank,
    cudaStream_t stream,
    Timeout timeout = Timeout());

void launch_ibgda_recv_send_chain(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    int my_rank,
    cudaStream_t stream,
    Timeout timeout = Timeout());

} // namespace ctran::prims::benchmark
