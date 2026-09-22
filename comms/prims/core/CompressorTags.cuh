// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Compressor-family tags and the tag -> CopyOp map shared by the tile
// collectives, plus the host-callable worst-case stride query derived from
// them. Core-level: everything here depends on `core/CopyOp.cuh` and nothing
// else. The transport dispatchers that drive a compressed CopyOp over IBGDA
// live in `collectives/CompressorDispatch.cuh`, which sits above the transport
// layer and includes this header.
//
// Two visibility tiers:
//   - The `AnsCompressor` tag (with its `kMaxUncompBytes`) and
//     `ans_worst_case_chunk_stride()` are host-visible, so benchmarks and
//     tests that cannot include the device-only CopyOp.cuh can still name the
//     compressor and size a staging slot from a host `.cc`.
//   - Everything under `PIPES_ENABLE_ANS_COMPRESSION && __CUDACC__` is
//     device-only and pulls in `CopyOp.cuh` (and, through it, nvcompdx).

#pragma once

#include <cstddef>

namespace comms::prims {

// Host-visible tag selecting a compressor family. `kMaxUncompBytes` is the
// max uncompressed sub-chunk the compressor stages; a host launcher sizes the
// per-block realign aux buffer as `gridDim.x * kMaxUncompBytes`. It has to be
// a host-visible literal because the concrete CopyOp's copy of it lives inside
// the device-only `PIPES_ENABLE_ANS_COMPRESSION` region of `CopyOp.cuh`; the
// device-side `static_assert` below pins it to `AnsCompress::kMaxUncompBytes`
// so the two can never drift.
struct AnsCompressor {
  static constexpr std::size_t kMaxUncompBytes = 256ULL * 1024;
};

/**
 * Worst-case bytes one `chunk_bytes` ANS sub-chunk occupies in a transport
 * staging slot: the size-header table plus the expanded payload, padded to the
 * compressor's input alignment.
 *
 * Cheap -- a `constexpr` descriptor query, no GPU involvement. It is behind a
 * function only because the computation lives in `core/CopyOp.cuh`, which host
 * `.cc` TUs cannot include (it needs PIPES_ENABLE_ANS_COMPRESSION +
 * __CUDACC__). `core/CompressWorstCaseStride.cu` is the one nvcc TU that
 * evaluates it and re-exports it behind this plain host signature.
 *
 * Sizing a staging slot from a hand-derived approximation instead is a trap,
 * not a rounding error: when the slot cannot hold one worst-case chunk,
 * `max_safe_chunk_size_for_slot()` finds room for zero, falls back to
 * `kMaxUncompBytes`, and the transport `__trap()`s -- surfacing only as a bare
 * "unspecified launch failure" because `__trap()` swallows the diagnostic.
 *
 * Callers must round the result UP to 512 before using it to size a slot: the
 * transport floors `perBlockSlot` to the 512 B NIC burst
 * (`pipeline_chunk() & ~511`), so an unaligned stride loses up to 511 bytes of
 * the margin.
 *
 * @param num_warps  blockDim.x / 32 of the kernel that will do the compressing.
 * @param chunk_bytes  the uncompressed sub-chunk size, normally
 *                     `AnsCompressor::kMaxUncompBytes`.
 * @return the stride in bytes, or 0 if `num_warps` is unsupported.
 */
std::size_t ans_worst_case_chunk_stride(int num_warps, std::size_t chunk_bytes);

#if defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)
} // namespace comms::prims

// CopyOp.cuh is only safe to include when both PIPES_ENABLE_ANS_COMPRESSION
// AND __CUDACC__ are defined: the macro guards the nvcompdx include inside it,
// and the AnsCompress<> body uses CUDA device intrinsics. Pulled in here (out
// of the namespace) so the map below can refer to `AnsCompress<>`.
#include "comms/prims/core/CopyOp.cuh"

namespace comms::prims {

// NAMED namespace, not an anonymous one. This header is included by more than
// one collective, and `compressor_copyop` is referenced from the body of the
// external-linkage template `sendrecv_tile_compressed_kernel`: with internal
// linkage each including TU would get its own copy of the entity that an
// external-linkage template refers to, which is an ODR violation the moment
// that kernel template is instantiated from two TUs. (Contrast
// SendRecvTileCommon.cuh, where the anonymous namespace is deliberate because
// the two including TUs compile it with different macros.)
namespace detail {

// Maximum uncompressed sub-chunk size — sourced from the
// `PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES` macro defined in CopyOp.cuh, so the
// 256 KiB literal lives in a single place.
constexpr std::size_t kAnsMaxUncompBytes = PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES;

// CopyOp tag templated on `NumWarps`. Each `AnsCopyOp<N, Tag>` is a distinct
// type — instantiate exactly one per kernel so the static `__shared__` arrays
// declared inside `AnsCompress<N, ...>::send` / `::recv` only allocate space
// for the single NumWarps the kernel actually uses.
//
// `StatsTag` selects which `ans_counters::*<Tag>` the compression byte counters
// land in. Every collective that enables stats must pass its OWN tag: the
// counters are `__device__` symbols, and two independently device-linked units
// that name the same tag register the same symbol twice, which surfaces as
// `invalid device symbol` on the first `cudaMemcpyFromSymbol`. It defaults to
// `void`, which is only valid in a build without PIPES_ANS_COLLECT_STATS —
// `AnsCompress` static_asserts that.
template <int NumWarps, typename StatsTag = void>
using AnsCopyOp = AnsCompress<NumWarps, kAnsMaxUncompBytes, false, StatsTag>;

// Maps a host-visible compressor tag + NumWarps to the concrete CopyOp that a
// compressed kernel drives via `*_tile_impl<CopyOp>`. The kernel is generic
// over the tag; add a specialization here to support a new compressor family
// (and add its explicit instantiations in the collective's .cu).
template <typename Compressor, int NumWarps, typename StatsTag = void>
struct compressor_copyop;

template <int NumWarps, typename StatsTag>
struct compressor_copyop<AnsCompressor, NumWarps, StatsTag> {
  using type = AnsCopyOp<NumWarps, StatsTag>;
};

// Names an owner for the assert below only. Never launched, so it never
// defines counters -- `ans_counters::*<Tag>` are only instantiated where
// `send()`'s stats block odr-uses them, and nothing instantiates that here.
// It exists because `void` is rejected by `AnsCompress`'s stats static_assert,
// and this alias must instantiate in the stats build too.
struct MaxUncompBytesProbeTag;

// The host-visible tag constant MUST equal the concrete CopyOp's max
// uncompressed chunk (both ultimately `PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES`),
// or a host launcher would under-size the per-block realign aux buffer.
// `kMaxUncompBytes` does not depend on `StatsTag`, so checking one arbitrary
// instantiation checks them all.
static_assert(
    AnsCompressor::kMaxUncompBytes ==
        AnsCopyOp<1, MaxUncompBytesProbeTag>::kMaxUncompBytes,
    "AnsCompressor::kMaxUncompBytes must match the device "
    "AnsCompress::kMaxUncompBytes");

} // namespace detail

// Re-export into `comms::prims` so the collective headers that include this
// keep referring to these unqualified. The definitions live in `detail` purely
// to give them external linkage; this is not a second set of entities.
using detail::AnsCopyOp;
using detail::compressor_copyop;
using detail::kAnsMaxUncompBytes;

#endif // defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)

} // namespace comms::prims
