// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives {

/**
 * Dispatch kernel for all-to-all chunk transfer
 *
 * Uses round-robin tournament scheduling for deadlock-free communication with
 * maximum SM utilization. All warps process the same peer operation at each
 * step (vertical sharding), ensuring all SMs contribute to the current
 * send/recv.
 *
 * Round-robin tournament ensures all ranks are active at every step with
 * matched send/recv pairs. For n ranks: 2*(n-1) steps, n/2 parallel pairs.
 * Within each pair, lower rank sends first on even steps, higher on odd.
 *
 * @param transports DeviceSpan of Transport objects [n_ranks]
 *                   Index my_rank contains self transport
 *                   Other indices contain P2P NVL transport
 * @param my_rank This rank's ID
 * @param sendbuff_d Source buffer containing chunks to send
 * @param recvbuffs DeviceSpan of receive buffer pointers [n_ranks]
 * @param input_chunk_sizes DeviceSpan of all chunk sizes
 * @param input_chunk_indices_d Flattened array of chunk indices per rank
 * @param input_chunk_indices_count_per_rank DeviceSpan of index counts
 * [n_ranks]
 * @param output_chunk_sizes_per_rank DeviceSpan for output chunk sizes
 *                                    [n_ranks * input_chunk_sizes.size()]
 */
__global__ void dispatchKernel(
    DeviceSpan<Transport> transports,
    int my_rank,
    const void* sendbuff_d,
    DeviceSpan<void* const> recvbuffs,
    DeviceSpan<const std::size_t> input_chunk_sizes,
    const std::size_t* input_chunk_indices_d,
    DeviceSpan<const std::size_t> input_chunk_indices_count_per_rank,
    DeviceSpan<std::size_t> output_chunk_sizes_per_rank);

} // namespace comms::pipes::collectives
