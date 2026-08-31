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

/// pack(src) into `staging_d` under `flagVal`, then ask
/// `LLImpl::all_flags_set` whether the whole chunk has landed.
///
/// `corruptPacket >= 0` first rewrites that one packet's flag to a DIFFERENT
/// generation, which is what a not-yet-arrived packet looks like. The check
/// must then report false -- and must REPORT it, not spin: this entry point
/// exists precisely for callers that cannot block, so a hang here is the
/// failure, and the test would time out rather than assert.
///
/// Writes 1/0 to `ready_d`.
void test_ll_all_flags_set(
    const char* src_d,
    char* staging_d,
    std::size_t nbytes,
    int corruptPacket,
    uint32_t* ready_d);

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

/// pack(src) into `recvStaging_d` under the UPSTREAM generation, then relay it
/// with `repack` into `fwdStaging_d` under the DOWNSTREAM generation, decoding
/// into `dst_d` on the way.
///
/// The two generations are deliberately different: the relay must re-stamp
/// every forwarded packet, and shipping the upstream flag through is the bug
/// this pins. `repack` is byte-granular (no element type), so the axis that
/// matters is `nbytes` -- specifically `nbytes % kData`, which decides how much
/// of the final packet is the packer's zero padding.
///
/// Verification is split into two independent passes so neither failure mode
/// can mask or hang on the other:
///   1. flags: count packets in `fwdStaging_d` whose flag is not the
///      downstream one, into `flagErrors_d`. Read once, never spun on -- a
///      relay that forgot to re-stamp fails the test instead of hanging it,
///      which is what calling `unpack()` here would do.
///   2. payload: re-stamp every packet with the downstream flag, then `unpack`
///      into `packetOut_d`. That cannot hang, and it checks the data half
///      independently of the flag half.
///
/// `useDst == false` passes `dst == nullptr` (the forward-only relay shape the
/// chain's intermediate ranks use); `dst_d` is then left untouched.
void test_ll_repack(
    const char* src_d,
    char* recvStaging_d,
    char* fwdStaging_d,
    char* dst_d,
    char* packetOut_d,
    std::size_t nbytes,
    bool useDst,
    uint32_t* flagErrors_d);

} // namespace comms::prims::test
