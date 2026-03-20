// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/ll128/Ll128Packet.cuh"

namespace comms::pipes::test {

/// Cross-GPU LL128 send/recv test. Launches sender on sender_gpu writing to
/// ll128_buf (on receiver_gpu) via NVLink, and receiver on receiver_gpu
/// reading from its local ll128_buf.
///
/// @param sender_gpu GPU index for the sender kernel
/// @param receiver_gpu GPU index for the receiver kernel
/// @param src_d Source buffer (on sender_gpu)
/// @param dst_d Destination buffer (on receiver_gpu)
/// @param nbytes Message size (multiple of 16)
/// @param ll128_buf LL128 packet buffer (on receiver_gpu)
/// @param buffer_num_packets Packets in ll128_buf (0 = sized to fit message)
/// @param num_steps Number of send/recv iterations
/// @param num_blocks Blocks per kernel launch
/// @param block_size Threads per block
void test_ll128_nvlink_send_recv(
    int sender_gpu,
    int receiver_gpu,
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size);

/// Cross-GPU LL128 forward test. Exercises NVLink on both hops:
///   sender_gpu → buf_a on forwarder_gpu (NVLink write)
///   forwarder_gpu reads local buf_a → writes buf_b on receiver_gpu (NVLink)
///   receiver_gpu reads local buf_b
///
/// @param sender_gpu GPU for sender
/// @param forwarder_gpu GPU for forwarder
/// @param receiver_gpu GPU for receiver
/// @param src_d Source buffer (on sender_gpu)
/// @param fwd_dst_d Forwarder's local copy destination (on forwarder_gpu)
/// @param recv_dst_d Receiver's destination (on receiver_gpu)
/// @param nbytes Message size (multiple of 16)
/// @param ll128_buf_a LL128 buffer on forwarder_gpu
/// @param ll128_buf_b LL128 buffer on receiver_gpu
/// @param num_blocks Blocks per kernel launch
/// @param block_size Threads per block
void test_ll128_nvlink_forward(
    int sender_gpu,
    int forwarder_gpu,
    int receiver_gpu,
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf_a,
    comms::pipes::Ll128Packet* ll128_buf_b,
    int num_blocks,
    int block_size,
    size_t buffer_num_packets = 0,
    int num_steps = 1);

/// Cross-GPU LL128 bidirectional test. Two independent send/recv pairs
/// running simultaneously:
///   GPU0 sends to GPU1's ll128_buf_on_gpu1 + GPU1 recvs from it
///   GPU1 sends to GPU0's ll128_buf_on_gpu0 + GPU0 recvs from it
/// All 4 kernels run concurrently on separate streams.
void test_ll128_nvlink_bidirectional(
    const char* src0_d,
    char* dst0_d,
    const char* src1_d,
    char* dst1_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf_on_gpu0,
    comms::pipes::Ll128Packet* ll128_buf_on_gpu1,
    int num_blocks,
    int block_size,
    size_t buffer_num_packets = 0,
    int num_steps = 1);

/// Cross-GPU LL128 forward test for 3 distinct GPUs.
/// All 3 roles run on separate GPUs/streams — no combined kernel needed.
///   sender_gpu writes to ll128_buf_a on forwarder_gpu
///   forwarder_gpu reads local buf_a, writes buf_b on receiver_gpu, copies to
///   fwd_dst receiver_gpu reads local buf_b to recv_dst
void test_ll128_nvlink_forward_3gpu(
    int sender_gpu,
    int forwarder_gpu,
    int receiver_gpu,
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf_a,
    comms::pipes::Ll128Packet* ll128_buf_b,
    int num_blocks,
    int block_size,
    size_t buffer_num_packets = 0,
    int num_steps = 1);

/// Cross-GPU LL128 varying-data send/recv test. Each step offsets src/dst
/// by i * nbytes to detect stale buffer contents across steps.
/// @param src_d Source buffer (num_steps * nbytes total, on sender_gpu)
/// @param dst_d Destination buffer (num_steps * nbytes total, on receiver_gpu)
/// @param buffer_num_packets Packets in ll128_buf (0 = sized to fit message)
void test_ll128_nvlink_varying_send_recv(
    int sender_gpu,
    int receiver_gpu,
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size);

} // namespace comms::pipes::test
