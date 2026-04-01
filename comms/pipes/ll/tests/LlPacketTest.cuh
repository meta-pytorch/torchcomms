// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/// Test that LlLine is 16 bytes.
void test_ll_line_size(uint32_t* errorCount_d);

/// Test flag read/write round-trip via volatile helpers.
void test_ll_flag_read_write(void* line_d, uint32_t* errorCount_d);

/// Test ll_num_lines for various message sizes.
void test_ll_num_lines(uint32_t* errorCount_d);

/// Test ll_buffer_size for various message sizes.
void test_ll_buffer_size(uint32_t* errorCount_d);

/// Test ll_buffer_payload_capacity.
void test_ll_buffer_payload_capacity(uint32_t* errorCount_d);

/// Test flag initialization to kLlReadyToWrite via cudaMemset 0xFF.
void test_ll_flag_init(void* line_d, uint32_t* errorCount_d);

/// Test can_use_ll on device side.
void test_can_use_ll(const char* aligned_ptr_d, uint32_t* errorCount_d);

/// Test ll_flag function.
void test_ll_flag(uint32_t* errorCount_d);

} // namespace comms::pipes::test
