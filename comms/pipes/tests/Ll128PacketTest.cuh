// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/// Test that Ll128Packet is 128-byte aligned and sized.
void test_ll128_packet_alignment(uint32_t* errorCount_d);

/// Test flag read/write round-trip via volatile helpers.
void test_ll128_flag_read_write(void* packet_d, uint32_t* errorCount_d);

/// Test ll128_packet_payload_size for various inputs.
void test_ll128_packet_payload_size(uint32_t* errorCount_d);

/// Test ll128_num_packets for various message sizes.
void test_ll128_num_packets(uint32_t* errorCount_d);

/// Test ll128_slot_ptr returns correct addresses for each lane.
void test_ll128_slot_ptr(uint32_t* errorCount_d);

/// Test flag initialization to kLl128ReadyToWrite via cudaMemset 0xFF.
void test_ll128_flag_init(void* packet_d, uint32_t* errorCount_d);

/// Test can_use_ll128 on device side.
void test_can_use_ll128(const char* aligned_ptr_d, uint32_t* errorCount_d);

} // namespace comms::pipes::test
