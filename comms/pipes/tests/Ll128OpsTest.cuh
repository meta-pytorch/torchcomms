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

/// Test LL128 multi-step send/recv: performs num_steps send/recv iterations
/// on the same buffer with incrementing flag values.
void test_ll128_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::Ll128Packet* ll128_buf,
    int64_t start_flag_value,
    int num_steps,
    int num_blocks,
    int block_size);

} // namespace comms::pipes::test
