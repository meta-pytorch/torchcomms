// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

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
 * Same chain as launch_recv_forward_chain, but intermediates drive the
 * resumable forward (init_forward_progress / progress_forward_once) to
 * completion instead of the blocking forward. Validates the resumable forward
 * end-to-end against the blocking send/recv endpoints on the same wire
 * protocol.
 */
cudaError_t launch_recv_forward_chain_progress(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

/**
 * Resumable-forward chain with dst=nullptr for intermediates (forward-only).
 * Only the last rank writes to recv_buf.
 */
cudaError_t launch_recv_forward_chain_progress_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

/**
 * Regression for the resumable-forward slot-completion bug: rank 1 does an
 * ordinary resumable send (op A) then a reduce-forward (op B) to the same
 * `next` transport, so op B reuses op A's send slot across a pipeline-window
 * boundary (nbytes == one window). rank 0 feeds op B's recv; rank 2 receives op
 * A into recv_a and op B into recv_b. After the run: recv_a == rank 1's
 * send_buf; recv_b == rank 0's send_buf + rank 1's local_buf (float reduce).
 * Requires world_size >= 3.
 */
cudaError_t launch_send_then_reduce_forward(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    const char* local_buf,
    char* recv_a,
    char* recv_b,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream);

} // namespace comms::prims::test
