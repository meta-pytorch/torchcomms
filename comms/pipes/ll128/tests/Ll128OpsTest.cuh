// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/ll128/Ll128Packet.cuh"

namespace comms::pipes::test {

/// Test LL128 send/recv round-trip between two buffers on same GPU.
/// Sender writes from src to remote_ll128_buf, receiver reads from
/// local_ll128_buf to dst. (local_ll128_buf == remote_ll128_buf in P2P.)
void test_ll128_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    int num_blocks,
    int block_size);

/// Test LL128 forward: read from local LL128 buf, forward to remote, copy to
/// dst.
void test_ll128_forward(
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* local_ll128_buf,
    comms::pipes::Ll128Packet* remote_ll128_buf,
    int num_blocks,
    int block_size);

/// Test LL128 multi-step send→forward→recv pipeline.
/// Sender writes to ll128_buf_a, forwarder reads from ll128_buf_a and
/// forwards to ll128_buf_b (copying to fwd_dst), receiver reads from
/// ll128_buf_b to recv_dst.
void test_ll128_multi_step_forward(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf_a,
    comms::pipes::Ll128Packet* ll128_buf_b,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL128 multi-step send/recv: performs num_steps send/recv iterations
/// on the same buffer.
void test_ll128_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL128 chunked send/recv: buffer is smaller than the message.
void test_ll128_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_blocks,
    int block_size);

/// Test LL128 chunked multi-step send/recv.
void test_ll128_multi_step_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL128 chunked multi-step send→forward→recv pipeline.
void test_ll128_multi_step_forward_chunked(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf_a,
    comms::pipes::Ll128Packet* ll128_buf_b,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL128 windowed send/recv: buffer has fewer packets than the message,
/// exercising modular indexing for slot reuse.
/// @param max_ll128_packets  Buffer capacity in packets (must be power of 2)
void test_ll128_windowed_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    size_t max_ll128_packets,
    int num_blocks,
    int block_size);

/// Test LL128 varying-data multi-step send/recv: each step uses a different
/// offset into src/dst to detect stale buffer contents.
/// @param src_d Source buffer (num_steps * nbytes total)
/// @param dst_d Destination buffer (num_steps * nbytes total)
/// @param buffer_num_packets Packets in ll128_buf (0 = sized to fit message)
void test_ll128_varying_data_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL128 varying-data multi-step send→forward→recv pipeline.
/// @param src_d Source buffer (num_steps * nbytes total)
/// @param fwd_dst_d Forwarder destination (num_steps * nbytes total)
/// @param recv_dst_d Receiver destination (num_steps * nbytes total)
/// @param buffer_num_packets Packets in buffers (0 = sized to fit message)
void test_ll128_varying_data_multi_step_forward(
    const char* src_d,
    char* fwd_dst_d,
    char* recv_dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf_a,
    comms::pipes::Ll128Packet* ll128_buf_b,
    size_t buffer_num_packets,
    int num_steps,
    int num_blocks,
    int block_size);

} // namespace comms::pipes::test
