// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Public header for the ANS-compressed flavour of the sendrecv-tile
// kernel. Provided by the `:sendrecv_tile_compressed` BUCK target (which
// device-links the nvcompdx fatbin and pulls in the `:copy_op_compress`
// headers). Consumers that want compressed IBGDA chunks should depend on
// that target and `#include` this header for:
//
//   - `pick_sendrecv_tile_compressed_kernel(threads_per_block,
//     min_blocks_per_sm)` -> the matching kernel address
//   - `SendRecvAnsCompressStats` +
//     `fetch_and_reset_sendrecv_ans_compress_stats(stream)`
//
// The compressed kernel reuses `SendRecvTileArgs` from the sibling
// `SendRecvTile.cuh`. Compress requires `blockDim.x` to be one of {32,
// 64, 128, 256, 512, 1024}. NO dynamic shared memory needs to be
// reserved by the launcher: `AnsCompress::send` / `recv` declare their
// per-direction `__shared__` scratch internally, sized at compile time
// off nvcompdx's `constexpr shmem_size_group()`. Pass `0` as
// `cudaLaunchKernel`'s dynamic-shmem argument. NVL transport peers
// always use plain `Memcpy` regardless of which kernel is launched —
// compression only applies to IBGDA peers.
//
// The compressor abstraction — the host-visible `AnsCompressor` tag, the
// `compressor_copyop` tag->CopyOp map, and the device-only
// `ibgda_{send,recv}_compressed<CopyOp>` dispatchers — is shared with
// AllToAllvTile via `CompressorTypes.cuh` (included below). The
// compressed dispatchers are gated by `#ifdef PIPES_ENABLE_ANS_COMPRESSION`
// so host-only consumers (benchmarks, tests) see only the picker, the stats
// accessor and the `AnsCompressor` tag -- no CUDA syntax at all.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "comms/prims/collectives/SendRecvTile.cuh"
// CompressorTypes.cuh must follow SendRecvTile.cuh: its device-only section
// references the transport device types (`P2pIbgdaTransportDevice`,
// `ThreadGroup`, `AbortDevice`) that SendRecvTile.cuh brings into scope.
#include "comms/prims/collectives/CompressorTypes.cuh"

namespace comms::prims {

// `AnsCompressor`, `compressor_copyop`, and the compressed IBGDA
// send/recv dispatchers live in CompressorTypes.cuh (included above),
// shared with AllToAllvTile.

/**
 * Owner of this collective's ANS byte counters.
 *
 * Incomplete on purpose -- it is a name, never an object. Its only job is to
 * make `ans_counters::{uncomp,comp}_bytes<SendRecvStatsTag>` distinct from
 * every other device-link unit's. Two units that shared one counter symbol
 * would each register it, and the first `cudaMemcpyFromSymbol` would fail with
 * `invalid device symbol`; CUDA then makes every subsequent
 * `cudaLaunchKernel` fail with the same sticky error.
 *
 * Declared unconditionally (not behind `PIPES_ANS_COLLECT_STATS`) so the type
 * is nameable from a host TU that only wants to talk about the accessor.
 */
struct SendRecvStatsTag;

/**
 * Address of the compressed sendrecv-tile kernel matching the caller's
 * runtime-chosen launch shape, or `nullptr` if no such kernel exists.
 *
 * This is the ONLY supported way to get at the kernel. The raw `__global__`
 * template lives in the private `SendRecvTileCompressedKernel.cuh` and is not
 * exported by this target, for two reasons:
 *
 *  - The template accepts any `(NumWarps, MinBlocksPerSM)`, but only seven
 *    pairs are explicitly instantiated. `<AnsCompressor, 8, 4>` used to compile
 *    happily at the call site and then fail at DEVICE LINK with an undefined
 *    symbol -- a diagnostic that names the linker, not the caller's mistake.
 *    This function rejects an unsupported pair host-side, before any launch.
 *  - It keeps CUDA syntax out of the public host API. The declaration used to
 *    be duplicated behind `#ifdef __CUDACC__` with a hand-maintained host
 *    mirror; there is now one declaration and host-only consumers (benchmarks,
 *    tests) need no CUDA compiler to name it.
 *
 * Supported `threads_per_block` (= `NumWarps * 32`): 32, 64, 128, 256, 512,
 * 1024, each at `min_blocks_per_sm == 2`; plus 256 at `min_blocks_per_sm == 8`,
 * the full-occupancy variant. `threads_per_block == 2048` (64 warps) does not
 * exist and cannot: CUDA caps a block at 1024 threads.
 *
 * Launch it with `cudaLaunchKernel(ptr, grid, block, args, 0, stream)`. No
 * dynamic shared memory is required -- the codec declares its `__shared__`
 * scratch internally, sized at compile time -- so pass `0`.
 */
void* pick_sendrecv_tile_compressed_kernel(
    int threads_per_block,
    int min_blocks_per_sm);

/**
 * The compressor's own worst-case on-wire stride for a `chunk_bytes` input, as
 * seen BY THE DEVICE.
 *
 * Host callers sizing a transport staging slot need
 * `AnsCompress::worst_case_chunk_stride()`, which is derived from nvCOMPDx's
 * `max_comp_chunk_size()`. That is exact, but it is a property of the
 * compressor DESCRIPTOR, and the descriptor carries `SM<kAnsArch>` -- and
 * `kAnsArch` is resolved from `__CUDA_ARCH__`, which is undefined on the host
 * pass and falls back to 800. Calling the sizing helper from host code
 * therefore answers for sm_80 rather than for the architecture the kernel will
 * actually run on, and silently under-reports wherever the two differ.
 *
 * So this queries the device: it launches a single-thread kernel that evaluates
 * the helper under the real `__CUDA_ARCH__` and copies the answer back. One
 * launch per call, intended for setup, not a hot path.
 *
 * Returns 0 if `num_warps` has no instantiated kernel family (same supported
 * set as `pick_sendrecv_tile_compressed_kernel`).
 *
 * Sizing a slot from an approximation instead is a trap, not a rounding error:
 * when the slot cannot hold one worst-case chunk,
 * `max_safe_chunk_size_for_slot()` finds room for zero, falls back to returning
 * `kMaxUncompBytes`, and the transport `__trap()`s -- surfacing only as a bare
 * "unspecified launch failure".
 */
std::size_t sendrecv_ans_worst_case_chunk_stride(
    int num_warps,
    std::size_t chunk_bytes);

/**
 * Snapshot + reset of the running ANS compression byte counters fed by
 * `AnsCompress::send` inside the IBGDA dispatcher. Returns
 * `(uncompressed_bytes, compressed_bytes)` accumulated since the last
 * call, then resets both device counters to 0.
 *
 * ONLY DEFINED BY THE STATS-ENABLED TARGET, `:sendrecv_tile_compressed_stats`.
 * Stats are off in the production `:sendrecv_tile_compressed` because they cost
 * a block-wide barrier and two global atomics on the hot path of every
 * compressed send, for the benefit of tests and one benchmark column. The
 * declaration is left visible here so the two targets share one header, but a
 * caller that links the production target and calls this gets an undefined
 * symbol -- which is the intended failure. It does not return zeros.
 *
 * NOT atomic, despite reading as if it were: the implementation copies both
 * counters out, synchronises, then writes zeros back. Any increment landing
 * between the read and the reset is lost. Callers that need exact totals must
 * quiesce the kernels first -- see the precondition note in
 * `SendRecvTileCompressed.cu`.
 *
 * Reads `ans_counters::*<SendRecvStatsTag>`, this device-link unit's own
 * pair of counters. It previously read two process-wide `inline __device__`
 * globals shared with every other ANS consumer; that was only safe while
 * exactly one device-link unit defined them, and `AnsCopyOpBench.cc` documents
 * hitting the failure -- two units register the same symbol, the first
 * `cudaMemcpyFromSymbol` returns `invalid device symbol`, and CUDA's sticky
 * first-error semantics then fail every later launch. With per-tag ownership a
 * binary can link this collective and another compressed collective and each
 * accessor sees only its own traffic.
 */
struct SendRecvAnsCompressStats {
  uint64_t uncompressed_bytes;
  uint64_t compressed_bytes;
};

SendRecvAnsCompressStats fetch_and_reset_sendrecv_ans_compress_stats(
    cudaStream_t stream = nullptr);

} // namespace comms::prims
