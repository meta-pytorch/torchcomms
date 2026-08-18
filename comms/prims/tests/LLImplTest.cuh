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

/// Reduce op selector for the unpack_reduce launchers below.
enum class ReduceKind { Sum, Max, Min };

/// pack(src) then unpack_reduce into `accum`, i.e. accum[i] = Op(accum[i],
/// src[i]). Each launcher pins one element/packet tiling of unpack_reduce:
///
///   f32  sizeof(T) == kData      -> one element per packet
///   f16  sizeof(T) <  kData      -> two elements per packet (+ partial tail)
///   i64  sizeof(T) >  kData      -> one element spanning two packets
///
/// `accum_d` is pre-seeded by the caller, so these also check that
/// unpack_reduce accumulates rather than overwriting the way unpack() does.
void test_ll_unpack_reduce_f32(
    const float* src_d,
    char* staging_d,
    float* accum_d,
    std::size_t nelems,
    ReduceKind kind);

void test_ll_unpack_reduce_f16(
    const void* src_d,
    char* staging_d,
    void* accum_d,
    std::size_t nelems);

void test_ll_unpack_reduce_i64(
    const int64_t* src_d,
    char* staging_d,
    int64_t* accum_d,
    std::size_t nelems);

} // namespace comms::prims::test
