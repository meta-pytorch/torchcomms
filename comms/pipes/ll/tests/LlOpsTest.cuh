// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/ll/LlPacket.cuh"

namespace comms::pipes::test {

/// Test LL send/recv round-trip between two buffers on same GPU.
void test_ll_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::LlLine* ll_buf,
    int num_blocks,
    int block_size);

/// Test LL multi-step send/recv on the same buffer.
void test_ll_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::LlLine* ll_buf,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL chunked send/recv: buffer is smaller than the message.
void test_ll_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_blocks,
    int block_size);

/// Test LL chunked multi-step send/recv.
void test_ll_multi_step_send_recv_chunked(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_steps,
    int num_blocks,
    int block_size);

/// Test LL varying-data multi-step send/recv.
void test_ll_varying_data_multi_step_send_recv(
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    comms::pipes::LlLine* ll_buf,
    size_t buffer_num_lines,
    int num_steps,
    int num_blocks,
    int block_size);

} // namespace comms::pipes::test
