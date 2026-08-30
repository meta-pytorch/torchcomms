// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::prims::test {

/// Device-side: packet geometry for both tiers (8 B / 128 B).
void test_ll_packet_geometry(uint32_t* errorCount_d);

/// Device-side: ll_slot_ptr addresses + ll_load/store_flag round-trip, both
/// tiers. `p128_d` / `p8_d` are global device buffers (>= 128 B / 8 B) — flag
/// I/O uses global volatile ops, which are illegal on shared memory.
void test_ll_packet_slot_flag(void* p128_d, void* p8_d, uint32_t* errorCount_d);

} // namespace comms::prims::test
