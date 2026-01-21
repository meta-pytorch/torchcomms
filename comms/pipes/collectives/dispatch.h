// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives {

/**
 * Dispatch for all-to-all chunk transfer
 *
 * Handles communication with all peers in a single kernel launch using
 * round-robin tournament scheduling for deadlock-free operation with maximum
 * SM utilization. Each peer communication uses send_multiple/recv_multiple.
 *
 * OUTPUT PARAMETERS:
 * @param recvbuffs DeviceSpan of receive buffer pointers [n_ranks]
 *                  Data from peer i is written to recvbuffs[i]
 * @param output_chunk_sizes_per_rank DeviceSpan for output chunk sizes
 *                                    [n_ranks * input_chunk_sizes.size()]
 *
 * INPUT PARAMETERS:
 * @param transports DeviceSpan of Transport handles [n_ranks]
 *                   Index my_rank contains self transport
 *                   Other indices contain P2P NVL transports
 * @param my_rank This rank's ID
 * @param sendbuff_d Source buffer containing chunks to send
 * @param input_chunk_sizes DeviceSpan of all chunk sizes
 * @param input_chunk_indices_d Flattened array of chunk indices per rank
 *                              Layout: [indices_for_rank0, ...]
 * @param input_chunk_indices_count_per_rank DeviceSpan of index counts
 * [n_ranks]
 * @param stream CUDA stream for kernel execution
 * @param num_blocks Number of thread blocks to launch (default: 4)
 * @param num_threads Number of threads per block (default: 256)
 */
void dispatch(
    // Outputs
    DeviceSpan<void* const> recvbuffs,
    DeviceSpan<std::size_t> output_chunk_sizes_per_rank,
    // Inputs
    DeviceSpan<Transport> transports,
    int my_rank,
    const void* sendbuff_d,
    DeviceSpan<const std::size_t> input_chunk_sizes,
    const std::size_t* input_chunk_indices_d,
    DeviceSpan<const std::size_t> input_chunk_indices_count_per_rank,
    cudaStream_t stream,
    int num_blocks = 4,
    int num_threads = 256);

} // namespace comms::pipes::collectives
