// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// IBGDA dispatchers that drive a compressed CopyOp over the tile transport
// (SendRecvTile, AllToAllvTile), plus the signal-granularity resolution they
// share. A compressed collective kernel is generic over a host-visible
// `Compressor` tag + `NumWarps`; it maps the pair to a concrete variable-size
// CopyOp via `compressor_copyop<Compressor, NumWarps>::type` and drives the
// transport through the dispatchers below.
//
// The tags and the tag -> CopyOp map live one layer down, in
// `core/CompressorTags.cuh`, because they depend only on `core/CopyOp.cuh`.
// This header is the part that depends on the TRANSPORT, which is why it stays
// under `collectives/`: it must be included AFTER the transport device header
// (for `P2pIbgdaTransportDevice`, `ThreadGroup`, `AbortDevice`) is in scope --
// i.e. from a collective's compressed `.cuh` after its `*Tile.cuh` include.
//
// Add a new compressor family by adding a tag and a `compressor_copyop`
// specialization in `core/CompressorTags.cuh`, plus the collective's explicit
// kernel instantiations.

#pragma once

#include <cstddef>

#include "comms/prims/core/CompressorTags.cuh"

namespace comms::prims {

#if defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)

// NAMED namespace, not an anonymous one -- see the note in
// `core/CompressorTags.cuh`: these are referenced from the bodies of
// external-linkage kernel templates, so internal linkage would be an ODR
// violation once such a kernel is instantiated from two TUs.
namespace detail {

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
      abortDevice,
      effective_max_signal_bytes,
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
  tr.recv<CopyOp>(group, dst, nbytes, abortDevice, effective_max_signal_bytes);
}

} // namespace detail

// Re-export into `comms::prims` so the collective headers that include this
// keep referring to these unqualified. The definitions live in `detail` purely
// to give them external linkage; this is not a second set of entities.
using detail::default_max_signal_bytes_for_compress;
using detail::ibgda_recv_compressed;
using detail::ibgda_send_compressed;

#endif // defined(PIPES_ENABLE_ANS_COMPRESSION) && defined(__CUDACC__)

} // namespace comms::prims
