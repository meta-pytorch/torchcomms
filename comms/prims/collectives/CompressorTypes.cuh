// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Shared compressor-family abstraction for the tile collectives
// (SendRecvTile, AllToAllvTile). A compressed collective kernel is generic
// over a host-visible `Compressor` tag + `NumWarps`; it maps the pair to a
// concrete variable-size CopyOp via `compressor_copyop<Compressor,
// NumWarps>::type` and drives the transport through the
// `ibgda_{send,recv}_compressed<CopyOp>` dispatchers below. Add a new
// compressor family by adding a tag here, a `compressor_copyop`
// specialization, and the collective's explicit kernel instantiations.
//
// Two visibility tiers:
//   - The `AnsCompressor` tag (and its `kMaxUncompBytes`) is host-visible so
//     benchmarks/tests that cannot include the device-only CopyOp.cuh can
//     name the compressor at the launch site and size the per-block realign
//     aux buffer.
//   - Everything under `PIPES_ENABLE_ANS_COMPRESSION && __CUDACC__` is
//     device-only. This header must be included AFTER the transport device
//     header (for `P2pIbgdaTransportDevice`, `ThreadGroup`, `AbortDevice`) is
//     in scope — i.e. from a collective's compressed `.cuh` after its
//     `*Tile.cuh` include. `CopyOp.cuh` (AnsCompress) is pulled in here.

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

#if defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)
} // namespace comms::prims

// CopyOp.cuh is only safe to include when both PIPES_ENABLE_ANS_COMPRESSION
// AND __CUDACC__ are defined: the macro guards the nvcompdx include inside it,
// and the AnsCompress<> body uses CUDA device intrinsics. Pulled in here (out
// of the namespace) so the map and dispatchers below can refer to
// `AnsCompress<>`. `P2pIbgdaTransportDevice`, `ThreadGroup`, and `AbortDevice`
// are expected to already be in scope from the including collective header.
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

// Resolve the compressed IB path's signal granularity, and CLAMP it.
//
// The `0` sentinel means "one signal per perBlockSlot" and maps to the LARGEST
// multiple of `kAnsMaxUncompBytes` whose worst-case on-wire chunked-ANS layout
// fits within `perBlockSlot`.
//
// A non-zero value used to be passed through untouched, which made the
// documented "hint" a footgun: the transport only checks
// `max_signal_bytes < perBlockSlot` before using it, but the binding
// constraint is `worst_case_chunk_stride(value) <= perBlockSlot`, and ANS
// expands a chunk by ~1.3x plus a size-header table. So any caller value in
// between -- comfortably under perBlockSlot, but whose EXPANDED form is not --
// reached `chunkStride > perBlockSlot` and trapped, with the diagnostic printf
// swallowed by __trap() and surfacing as a bare "unspecified launch failure".
// Clamp to the largest safe chunk instead: a hint that cannot be honoured is
// reduced, never allowed to trap.
template <typename CopyOp>
__device__ __forceinline__ std::size_t default_max_signal_bytes_for_compress(
    std::size_t max_signal_bytes,
    P2pIbgdaTransportDevice& tr,
    [[maybe_unused]] int active_blocks) {
  // NOTE: the clamp below needs `perBlockSlot`, so it is applied after that is
  // computed rather than here.
  // `perBlockSlot` MUST match the transport's own per-group per-slot staging
  // region, which the variable-size send()/recv() derive from
  // `pipeline_chunk()` (perChannelBufferSize / pipelineDepth), NIC-burst (512B)
  // aligned, and carve for EVERY channel regardless of how many blocks actually
  // run. Sizing off `active_blocks` instead over-estimates the slot whenever
  // `active_blocks < max_num_channels` (e.g. AllToAllv's per-peer
  // `blocksPerPeer` is smaller than the configured channel count): the returned
  // chunk then overruns the real slot -> `chunkStride > perBlockSlot` trap
  // (whose printf is lost to __trap(), surfacing as a bare "unspecified launch
  // failure"). Callers that want the whole buffer should set `max_num_channels
  // == blocksPerPeer`.
  const std::size_t perBlockSlot = tr.pipeline_chunk() & ~511ULL;
  const std::size_t safest = CopyOp::max_safe_chunk_size_for_slot(perBlockSlot);
  if (max_signal_bytes == 0) {
    return safest;
  }
  // Honour the caller's value only while its WORST-CASE expanded layout still
  // fits the slot; otherwise fall back to the largest that does.
  return CopyOp::worst_case_chunk_stride(max_signal_bytes) <= perBlockSlot
      ? max_signal_bytes
      : safest;
}

// Compile-time-typed send dispatcher. Templated on the concrete
// `CopyOp = AnsCompress<NumWarps, MaxUncomp>` so the caller's kernel
// instantiation picks exactly one `AnsCompress<...>::send` instantiation.
template <typename CopyOp>
__device__ __forceinline__ void ibgda_send_compressed(
    P2pIbgdaTransportDevice& tr,
    ThreadGroup& group,
    void* src,
    std::size_t nbytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const AbortDevice& abortDevice,
    char* alignedAuxBuf) {
  const std::size_t effective_max_signal_bytes =
      default_max_signal_bytes_for_compress<CopyOp>(
          max_signal_bytes, tr, active_blocks);
  tr.send<CopyOp>(
      group,
      src,
      nbytes,
      effective_max_signal_bytes,
      abortDevice,
      alignedAuxBuf);
}

// Compile-time-typed recv dispatcher. Mirror of `ibgda_send_compressed`.
template <typename CopyOp>
__device__ __forceinline__ void ibgda_recv_compressed(
    P2pIbgdaTransportDevice& tr,
    ThreadGroup& group,
    void* dst,
    std::size_t nbytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const AbortDevice& abortDevice) {
  const std::size_t effective_max_signal_bytes =
      default_max_signal_bytes_for_compress<CopyOp>(
          max_signal_bytes, tr, active_blocks);
  tr.recv<CopyOp>(group, dst, nbytes, effective_max_signal_bytes, abortDevice);
}

} // namespace detail

// Re-export into `comms::prims` so the collective headers that include this
// keep referring to these unqualified. The definitions live in `detail` purely
// to give them external linkage; this is not a second set of entities.
using detail::AnsCopyOp;
using detail::compressor_copyop;
using detail::default_max_signal_bytes_for_compress;
using detail::ibgda_recv_compressed;
using detail::ibgda_send_compressed;
using detail::kAnsMaxUncompBytes;

#endif // defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)

} // namespace comms::prims
