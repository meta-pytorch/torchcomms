// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims {
class P2pIbgdaTransportDevice;
} // namespace comms::prims

namespace comms::prims::test {

/**
 * Launch a chain test kernel: rank 0 sends, intermediates recv_forward,
 * last rank receives. Tests the full send → recv_forward → recv protocol.
 *
 * @param transports     Array of worldSize P2pIbgdaTransportDevice pointers
 *                       (one per peer, indexed by rank).
 * @param send_buf       Source data (only used by rank 0).
 * @param recv_buf       Destination (used by all ranks; intermediates use it
 *                       as CopyOp dst in recv_forward).
 * @param nbytes         Total bytes to transfer per block.
 * @param my_rank        This rank's global rank.
 * @param world_size     Total number of ranks.
 * @param num_blocks     CUDA grid dimension.
 * @param stream         CUDA stream.
 */
void launch_recv_forward_chain(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

/**
 * Same as above but with dst=nullptr for intermediates (forward-only mode).
 * Only the last rank writes to recv_buf.
 */
void launch_recv_forward_chain_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

/**
 * Resumable-forward variant of launch_recv_forward_chain: intermediates drive
 * init_forward_progress / progress_forward_once to completion (single lane);
 * endpoints use blocking send/recv. Used to assert byte-parity with the
 * blocking forward chain.
 */
void launch_recv_forward_chain_progress(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

/**
 * Resumable-forward, forward-only (dst=nullptr for intermediates). Only the
 * last rank writes to recv_buf.
 */
void launch_recv_forward_chain_progress_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

/**
 * Interleaved multi-lane resumable-forward chain: a single block per rank
 * drives 2 lanes (distinct group_ids) round-robin. active_blocks = 2; data is
 * split into 2 lane slices. Validates that concurrent forwards on distinct
 * group_ids multiplex without corrupting each other. Always uses dst !=
 * nullptr.
 */
void launch_recv_forward_chain_2lane_progress(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    cudaStream_t stream);

} // namespace comms::prims::test
