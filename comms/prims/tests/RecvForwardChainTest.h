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
 * Forward-only chain launch that records each intermediate rank's
 * post-forward() reuse-credit cursors. The group leader writes four int64_t
 * values into out: recv-slot reuseCreditStep, recv-slot nextStep, fwd-slot
 * reuseCreditStep, fwd-slot nextStep.
 */
void launch_recv_forward_reuse_credit_step(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    int64_t* out,
    cudaStream_t stream);

} // namespace comms::prims::test
