// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

namespace comms::prims::test {

/// Pack `nbytes` of `src_d` into `staging_d` then unpack into `dst_d`, using
/// LlxPacketGeometry (8 B packets). `errorCount_d` counts payload mismatches.
void test_ll_pack_unpack(
    const char* src_d,
    char* staging_d,
    char* dst_d,
    std::size_t nbytes,
    uint32_t* errorCount_d);

/// Device-side: LLImpl::store_flag/load_flag/is_flag_set round-trip for both
/// tiers. `p8_d` are global device buffers (>= 128 B / 8 B); flag
/// I/O uses global volatile ops, which are illegal on shared memory.
void test_ll_flag_roundtrip(void* p8_d, uint32_t* errorCount_d);

} // namespace comms::prims::test
