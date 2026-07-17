// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/MemcpyCopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Tile.cuh"

#ifdef PIPES_ENABLE_ANS_COMPRESSION
// Pulled in only when callers opt in by defining
// PIPES_ENABLE_ANS_COMPRESSION (set on the `:copy_op_compress` BUCK
// target, transitively on `:alltoallv_tile_compressed`). Consumers of
// plain `:copy_op` never see this include and don't need to device-link
// the nvcompdx fatbin.
#include <nvcompdx.hpp>
#endif

namespace comms::prims {

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
struct TileReduce {
  // Fixed-size CopyOp policy (see AnsCompress for the variable-size one).
  static constexpr bool kVariableSize = false;
  static constexpr std::size_t kActivationThreshold = 0;
  __host__ __device__ __forceinline__ static constexpr std::size_t
  worst_case_chunk_stride(std::size_t chunkSize) {
    return chunkSize;
  }
  template <typename... Args>
  __device__ __forceinline__ static std::size_t send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    memcpy_vectorized(staging, src, nbytes, group);
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static std::size_t recv(
      char* dst,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* dst_t = reinterpret_cast<T*>(dst);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      tile_load_accumulate<T, AccumOp, kTileElems, kBlockSize>(
          acc, local_t, t, group, valid);
      tile_store<T, kTileElems, kBlockSize>(dst_t, t, acc, group, valid);
    }
#endif
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* /*dst*/,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* fwd_t = reinterpret_cast<T*>(fwd_staging);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      tile_load_accumulate<T, AccumOp, kTileElems, kBlockSize>(
          acc, local_t, t, group, valid);
      tile_store<T, kTileElems, kBlockSize>(fwd_t, t, acc, group, valid);
    }
#endif
  }
};

// Register/tile-staged reduce.
template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
struct TileReduceStaged {
  __host__ __device__ static constexpr std::size_t smem_bytes() {
    return 0;
  }

  // Fixed-size CopyOp policy (see AnsCompress for the variable-size one).
  static constexpr bool kVariableSize = false;
  static constexpr std::size_t kActivationThreshold = 0;
  __host__ __device__ __forceinline__ static constexpr std::size_t
  worst_case_chunk_stride(std::size_t chunkSize) {
    return chunkSize;
  }
  template <typename... Args>
  __device__ __forceinline__ static std::size_t send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    memcpy_vectorized(staging, src, nbytes, group);
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static std::size_t recv(
      char* dst,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* dst_t = reinterpret_cast<T*>(dst);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      auto local =
          tile_load<T, kTileElems, kBlockSize>(local_t, t, group, valid);
      tile_accumulate<T, AccumOp, kTileElems, kBlockSize>(acc, local);
      tile_store<T, kTileElems, kBlockSize>(dst_t, t, acc, group, valid);
    }
#endif
    return nbytes;
  }

  template <typename... Args>
  __device__ __forceinline__ static void forward(
      char* /*dst*/,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t byte_offset,
      const char* local_input,
      Args...) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const T* staging_t = reinterpret_cast<const T*>(staging);
    T* fwd_t = reinterpret_cast<T*>(fwd_staging);
    const T* local_t = reinterpret_cast<const T*>(local_input + byte_offset);
    std::size_t nelems = nbytes / sizeof(T);
    int ntiles = static_cast<int>((nelems + kTileElems - 1) / kTileElems);

    for (int t = 0; t < ntiles; t++) {
      std::size_t valid =
          min(static_cast<std::size_t>(kTileElems),
              nelems - static_cast<std::size_t>(t) * kTileElems);
      auto acc =
          tile_load<T, kTileElems, kBlockSize>(staging_t, t, group, valid);
      auto local =
          tile_load<T, kTileElems, kBlockSize>(local_t, t, group, valid);
      tile_accumulate<T, AccumOp, kTileElems, kBlockSize>(acc, local);
      tile_store<T, kTileElems, kBlockSize>(fwd_t, t, acc, group, valid);
    }
#endif
  }
};

#ifdef PIPES_ENABLE_ANS_COMPRESSION

// Default per-piece max uncompressed chunk size baked into the ANS
// (de)compressor type via `AnsCompress<...>`'s `MaxUncompBytes`
// template-parameter default. A macro (rather than a constexpr) so
// that downstream consumers (e.g. the AllToAllvTileCompressed
// dispatcher) can reference the same literal without instantiating
// `AnsCompress<>` solely to read its template parameter back.
//
// 256 KiB is the empirical sweet spot on H100 ANS uint8: large enough
// to amortise the per-call compressor setup, small enough that the
// `1.3 × MaxUncompBytes + 1576` worst-case compressed output (plus
// the per-piece header + 16-byte alignment padding) still fits
// inside a typical `perBlockSlot`. Override before including this
// header to pick a different default.
#ifndef PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES
#define PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES (256ULL * 1024ULL)
#endif

// nvCOMPDx requires the per-call compress/decompress input pointer
// to be 16-byte aligned. AnsCompress lays out multiple per-
// MaxUncompBytes sub-chunks back-to-back in the staging slot, so the
// start of every chunk after the first must be padded up to this
// boundary. Used by both AnsCompress::send (writer) and
// AnsCompress::recv / recv_forward (reader) and by
// AnsCompress::worst_case_chunk_stride (worst-case sizing).
constexpr std::size_t kNvcompInputAlignment = 16ULL;

__host__ __device__ __forceinline__ constexpr std::size_t alignInputBufferSize(
    std::size_t current_size) {
  return (current_size + (kNvcompInputAlignment - 1ULL)) &
      ~(kNvcompInputAlignment - 1ULL);
}

#ifdef PIPES_ANS_COLLECT_STATS
// Running totals of uncompressed/compressed bytes seen by AnsCompress::send
// across all kernel launches since the last
// fetch_and_reset_ans_compress_stats() call. Compiled in only when
// PIPES_ANS_COLLECT_STATS is set (a stats-collecting bench target, built with
// relocatable device code so the `inline __device__` external-linkage symbol
// has a single definition across TUs). The default `:copy_op_compress`
// consumers are compiled in whole-program mode (-rdc=false), where an inline
// __device__ variable with external linkage is rejected, so keep both the
// definition and its atomicAdd write sites behind this macro.
inline __device__ unsigned long long g_pipes_ans_total_uncomp_bytes = 0;
inline __device__ unsigned long long g_pipes_ans_total_comp_bytes = 0;
#endif

/**
 * AnsCompress — drop-in `CopyOp` policy that ANS-compresses the per-
 * sub-chunk staging copy on the sender side and decompresses on the
 * receiver. Same three static methods as `Memcpy` so it slots straight
 * into `tr.send<AnsCompress<NumWarps, MaxUncompBytes>>(…)` /
 * `tr.recv<…>(…)` (and `tr.forward<…>(…)` once D103888256 lands on top).
 *
 * Chunked staging layout
 * ----------------------
 * `send()` no longer requires `nbytes <= MaxUncompBytes`. When
 * `nbytes > MaxUncompBytes` the input is split into
 * `numChunks = ceil(nbytes / MaxUncompBytes)` ANS-compressed pieces
 * laid out back-to-back in the staging slot:
 *
 *   [staging + 0                  .. + numChunks * sizeof(size_t))
 *       per-piece compressed-size header table (one uint64 per chunk)
 *   [staging + alignInputBufferSize(numChunks*8)
 *                                  .. + comp[0])      // chunk[0] payload
 *   [pad to kNvcompInputAlignment]                    // chunk[1] start is
 *                                                     //
 * `kNvcompInputAlignment`-
 *                                                     // aligned for nvCOMPDx
 *                                                     // decompress
 *   [chunk[1] payload]
 *   ...
 *   [chunk[N-1] payload]
 *   [tail pad to kNvcompInputAlignment]               // included in
 *                                                     // copyResult so the
 *                                                     // RDMA put length
 *                                                     // stays aligned
 *
 * The whole region is then rounded up to `kNicBurstAlignment` (512) in
 * `worst_case_chunk_stride()` so consecutive sub-chunk staging offsets
 * `subStep * chunkStride` keep the NIC RDMA engine on its preferred
 * PCIe burst boundary.
 *
 * `send()` returns the total staging bytes written; the transport's
 * leader uses that value directly as the RDMA put length (no in-staging
 * header read-back). `recv()` walks the same layout: reads each
 * `compSize` from the header table via `__ldcv`, decompresses the
 * corresponding payload region into `dst + i * MaxUncompBytes`, and
 * advances by `alignInputBufferSize(compSize)`.
 *
 * Correctness notes for the chunked path:
 *   - send() / recv() / recv_forward() issue an inter-piece
 *     `group.sync()` after each `execute()` ONLY when there is a
 *     next iteration (i.e. `i + 1 < numChunks`). The barrier protects
 *     the shared `s_compress_shmem` / `s_decompress_shmem` scratch
 *     and any in-flight writes from racing with the next iteration's
 *     reuse. The final iteration's sync is omitted: the caller
 *     (`P2pIbgdaTransportDevice::send` / `recv` / `forward`) issues
 *     its own `group.sync()` immediately after these methods return,
 *     which doubles as the publish-barrier for the last piece's
 *     `sizeHeaders[numChunks-1]` (in send) and any post-loop reads.
 *     Saves one block-wide barrier per call, and the entire barrier
 *     when `numChunks == 1` (the common single-piece path).
 *
 * Caller sizing requirement: the IBGDA `dataBufferSize` must satisfy
 *   perBlockSlot >= worst_case_chunk_stride(chunkSize)
 * The transport traps when this fails. See
 * `max_safe_chunk_size_for_slot()` for the dispatcher-side helper that
 * derives the largest safe `chunkSize` from a given `perBlockSlot`.
 *
 * Build / launch requirements:
 *   1. Define PIPES_ENABLE_ANS_COMPRESSION at compile time. This is
 *      done automatically by depending on `:copy_op_compress` (or the
 *      transitive `:alltoallv_tile_compressed` target).
 *   2. Compile the consuming .cu with cuda_compile_style = "mono" +
 *      `--device-c` so nvcompdx's relocatable device symbols can be
 *      device-linked. See the `:alltoallv_tile_compressed_obj` target.
 *   3. Device-link the resulting object archive against the nvcompdx
 *      LTO fatbin via the `nvcompdx_device_link_target` rule (see
 *      `fbcode/comms/prims/collectives/defs.bzl`).
 *   4. Launch the consuming kernel with blockDim.x in
 *      {32, 64, 128, 256, 512, 1024} (so NumWarps = blockDim.x / 32
 *      is one of the supported template instantiations 1, 2, 4, 8,
 *      16, 32). NO dynamic shared memory needs to be reserved by the
 *      launcher: `send` / `recv` declare their per-direction
 *      `__shared__` scratch internally, sized at compile time off
 *      `AnsCompressorType::shmem_size_group()` /
 *      `AnsDecompressorType::shmem_size_group()` (both `constexpr`
 *      in nvcompdx). Pass `0` as `cudaLaunchKernel`'s dynamic-shmem
 *      argument.
 */
// `kSrcAligned`: compile-time promise that every caller passes a 16-byte
// aligned `src` to `send()`. When true, the runtime alignment check and
// the realign-via-alignedAuxBuf path are elided, dropping a uniform
// branch + several registers from the per-chunk send hot path. The
// alignedAuxBuf argument is then ignored (callers may pass nullptr).
//
// Defaults to `false` for safety — flip per call-site once the caller
// can prove its `src` pointer is always 16-byte aligned (e.g. raw
// `cudaMalloc` pointers, or sub-pointers stepped by multiples of 16).
template <
    int NumWarps,
    std::size_t MaxUncompBytes = PIPES_ANS_DEFAULT_MAX_UNCOMP_BYTES,
    bool kSrcAligned = false>
struct AnsCompress {
 public:
  // Each per-piece compress feeds nvcompdx `src + i * MaxUncompBytes`; the
  // single up-front 16-byte alignment check on the base `src` only stays valid
  // for every sub-piece if MaxUncompBytes is itself a multiple of the required
  // input alignment. Enforce by the type so arbitrary template overrides can't
  // silently break the per-piece alignment invariant.
  static_assert(
      MaxUncompBytes % kNvcompInputAlignment == 0,
      "AnsCompress: MaxUncompBytes must be a multiple of kNvcompInputAlignment "
      "(16) so each src + i*MaxUncompBytes sub-piece keeps src's 16-byte alignment");

  // ANS produces a variable-length payload, so the transport sizes the RDMA
  // put from the value `send()` returns (the total staging bytes written)
  // rather than from the uncompressed chunk size — there is no in-staging
  // header read-back on the transport side. See
  // P2pIbgdaTransportDevice::send().
  static constexpr bool kVariableSize = true;

  // Minimum per-tile byte count at which the compressed CopyOp is worth
  // activating. Below this threshold the per-tile dispatcher
  // (`send_to_peer` in AllToAllvTileCommon.cuh) should fall back to the
  // plain Memcpy IB path: small tiles don't pay back the per-chunk ANS
  // compress + decompress cost, so the wire-bandwidth savings from
  // compression are dominated by the kernel-side compression latency
  // and the extra pipeline machinery. 4 MiB is the empirical
  // crossover point on the H100 IB sweep — tiles smaller than this
  // are bandwidth-comparable on plain Memcpy and avoid the extra
  // shmem / aux-buf overhead.
  static constexpr std::size_t kActivationThreshold = 4ULL * 1024 * 1024;

 private:
  // ===========================================================================
  // Compile-time SM picker. nvCOMPDx instantiates a fully-typed
  // (de)compressor at compile time including the target arch via SM<>.
  // For fbcode's typical single-arch nvcc invocations we floor
  // __CUDA_ARCH__ to the canonical major (sm_80 / sm_90 / sm_100) that
  // nvCOMPDx ships shaders for. SM<900a> / SM<100a> are accepted via
  // SM<900> / SM<1000>. Host-only parses (where __CUDA_ARCH__ is
  // undefined) get an arbitrary valid arch so the type aliases below
  // remain well-formed; nothing on the host path actually instantiates
  // the (de)compressor.
  // ===========================================================================
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 1000
  static constexpr unsigned int kAnsArch = 1000;
#elif __CUDA_ARCH__ >= 900
  static constexpr unsigned int kAnsArch = 900;
#elif __CUDA_ARCH__ >= 800
  static constexpr unsigned int kAnsArch = 800;
#else
#error "PIPES_ENABLE_ANS_COMPRESSION requires sm_80 or higher"
#endif
#else
  static constexpr unsigned int kAnsArch = 800;
#endif

  // DataType is uint8 (opaque-bytes view) so tmp_size_group() == 0 — no
  // global scratch buffer is needed beyond the internal __shared__
  // scratch declared in send/recv.
  using AnsCompressorType =
      decltype(nvcompdx::Algorithm<nvcompdx::algorithm::ans>() + nvcompdx::DataType<nvcompdx::datatype::uint8>() + nvcompdx::Direction<nvcompdx::direction::compress>() + nvcompdx::MaxUncompChunkSize<MaxUncompBytes>() + nvcompdx::Block() + nvcompdx::BlockWarp<NumWarps, true>() + nvcompdx::SM<kAnsArch>());

  using AnsDecompressorType = decltype(nvcompdx::Algorithm<nvcompdx::algorithm::ans>() + nvcompdx::DataType<nvcompdx::datatype::uint8>() + nvcompdx::Direction<nvcompdx::direction::decompress>() + nvcompdx::MaxUncompChunkSize<MaxUncompBytes>() + nvcompdx::Block() + nvcompdx::BlockWarp<NumWarps, true>() + nvcompdx::SM<kAnsArch>());

#if defined(__CUDA_ARCH__)
  // Fail loudly if the SM<> picker above did not land the descriptor's arch
  // exactly on __CUDA_ARCH__. nvcompdx's `NVCOMPDX_SKIP_IF_NOT_APPLICABLE`
  // (used in send/recv/recv_forward) expands to
  // `if constexpr (sm_of_v<T> != __CUDA_ARCH__) return;` — a bare `return;`
  // that would be an invalid void return in these std::size_t methods if it
  // were ever selected. This static_assert guarantees `sm_of_v ==
  // __CUDA_ARCH__` for every compiled arch, so that skip branch is provably
  // dead. If it fires, extend the SM<> picker to map the current arch to a
  // canonical nvcompdx SM (80/90/100) rather than compiling for an arch
  // nvcompdx ships no shaders for.
  static_assert(
      nvcompdx::sm_of_v<AnsCompressorType> == __CUDA_ARCH__ &&
          nvcompdx::sm_of_v<AnsDecompressorType> == __CUDA_ARCH__,
      "AnsCompress: nvcompdx descriptor SM must equal __CUDA_ARCH__; extend the "
      "SM<> picker to cover this arch (nvcompdx ships shaders for sm_80/90/100)");
#endif

 public:
 private:
  // NIC RDMA burst alignment. The IBGDA NIC's RDMA engine prefers
  // source/destination addresses laid out on 512-byte PCIe burst
  // boundaries; mis-aligned sub-chunk offsets cause sporadic
  // illegal-access faults on the second-and-later sub-chunks within
  // a block's staging slot (because consecutive bursts straddle the
  // boundary into a neighbouring sub-chunk's slot region). We round
  // `worst_case_chunk_stride()` up to this so consecutive sub-chunk
  // staging addresses (`subStep * worst_case_chunk_stride`) stay
  // 512-byte aligned, matching the historical pre-chunking
  // `kSlotAlignment` and pairing with the `& ~511ULL` perBlockSlot
  // mask used by some transport flavours.
  static constexpr std::size_t kNicBurstAlignment = 512ULL;

  // Compile-time max of the per-direction nvcompdx scratch
  // requirements. Used by `get_shared_scratch()` below to size a
  // SINGLE per-(NumWarps, MaxUncompBytes, kSrcAligned) static
  // `__shared__` array that both `send()` and `recv()` (and
  // `recv_forward()`) share. nvcompdx's `shmem_size_group()` is
  // `constexpr` so this is resolved at compile time and the
  // unused-direction storage drops out without any runtime cost.
  //
  // SAFETY PAD (1024 B): nvcompdx's `shmem_size_group()` (queried on
  // the `BlockWarp<NumWarps, true>` cooperative-block descriptors used
  // here) empirically under-reports the actual `__shared__` write
  // footprint by a small constant on H100 (compute-sanitizer memcheck
  // flagged ~28 bytes of `coalesce_subchunks` writes past the reported
  // end at NumWarps=8 — see internal task / D-history on
  // AllToAllvTileCompressed). Without the pad the OOB writes corrupt
  // neighbouring `__shared__` and surface much later as an opaque
  // `cudaErrorIllegalAddress` from a DeviceBuffer destructor. The pad
  // costs 1 KiB of per-kernel static shmem (well under the 48 KiB
  // H100 cap budgeted in the BUCK comment) and absorbs the
  // under-report across all `NumWarps ∈ {1, 2, 4, 8, 16, 32}`
  // instantiations. DO NOT remove without confirming nvcompdx
  // upstream has fixed `shmem_size_group()` and running
  // compute-sanitizer over the full IbSweepCompressed sweep.
  inline static constexpr std::size_t kSharedScratchPadBytes = 1024ULL;
  inline static constexpr std::size_t kSharedScratchBytes =
      (AnsCompressorType::shmem_size_group() >
               AnsDecompressorType::shmem_size_group()
           ? AnsCompressorType::shmem_size_group() + kSharedScratchPadBytes
           : AnsDecompressorType::shmem_size_group() + kSharedScratchPadBytes);

  // Compile-time max of the per-direction nvcompdx shared-memory
  // alignment requirements (queried via the constexpr
  // `shmem_alignment()` accessor each (de)compressor descriptor
  // exposes — see `comp_execution.hpp::shmem_alignment()`). Because
  // the SAME `s_ans_shared_scratch` allocation is fed to both
  // `compressor.execute(..., shmem, ...)` and
  // `decompressor.execute(..., shmem, ...)` (via
  // `get_shared_scratch()`), it must satisfy the stricter of the
  // two alignments — using only one direction's value would
  // silently mis-align the buffer for the other direction. Both
  // are `constexpr` so this folds to a literal at compile time.
  inline static constexpr std::size_t kSharedScratchAlignment =
      (AnsCompressorType::shmem_alignment() >
               AnsDecompressorType::shmem_alignment()
           ? AnsCompressorType::shmem_alignment()
           : AnsDecompressorType::shmem_alignment());

  // Single per-instantiation `__shared__` scratch shared between
  // `send()` and `recv()`. CUDA semantics: a `static __shared__`
  // declared inside an `__forceinline__ static __device__` helper
  // gets ONE allocation per (kernel, declaring function
  // instantiation) — every inlined call site from the same kernel
  // resolves to the same physical bytes. So a kernel that calls
  // BOTH `AnsCompress<N>::send` AND `AnsCompress<N>::recv` (e.g.
  // the production `alltoallv_tile_*_compressed_kernel<N>` whose
  // blocks are split into sender / receiver roles via
  // `partition_interleaved(2)`) reserves
  // `max(send_shmem, recv_shmem)` bytes per kernel instead of
  // `send_shmem + recv_shmem` — typically a ~30-50% per-kernel
  // static-shmem reduction.
  //
  // SAFETY CONTRACT — callers MUST ensure that `send()` and
  // `recv()` are never executed concurrently within the same block.
  // `alltoallv_tile_impl` enforces this via its block-role
  // partition; any future caller that interleaves the two
  // directions in the same block must instead use two separate
  // scratch buffers (or two separate `AnsCompress` instantiations
  // with different template parameters). The
  // standalone `ans_copy_op_{compress,decompress}_bench_kernel`
  // pair launches each direction in its own kernel so the contract
  // trivially holds.
  __device__ __forceinline__ static uint8_t* get_shared_scratch() {
    __shared__ __align__(kSharedScratchAlignment)
        uint8_t s_ans_shared_scratch[kSharedScratchBytes];
    return s_ans_shared_scratch;
  }

 public:
  /**
   * Worst-case ANS compressed-output stride per transport sub-chunk, in
   * bytes.
   *
   * The runtime `chunkSize` (per-sub-chunk size requested by the
   * transport) is no longer constrained to <= `MaxUncompBytes`. When
   * `chunkSize > MaxUncompBytes`, AnsCompress::send internally splits
   * the sub-chunk into `numChunks = ceil(chunkSize / MaxUncompBytes)`
   * ANS-compressed pieces. The wire layout written to the staging slot
   * is:
   *
   *   [ size_t * numChunks ]            // per-piece compressed-size headers
   *   [ pad to kNvcompInputAlignment ]  // so chunk[0] starts 16-byte aligned
   *   [ chunk[0] compressed payload ]
   *   [ pad to kNvcompInputAlignment ]  // so chunk[1] starts 16-byte aligned
   *   [ chunk[1] compressed payload ]
   *   ...
   *   [ chunk[N-1] compressed payload ] // also padded for stride uniformity
   *
   * So the worst-case wire size (incompressible input) is:
   *
   *   alignInputBufferSize(numChunks * sizeof(size_t))
   *   + numChunks * alignInputBufferSize(max_comp_chunk_size())
   *
   * which we then round up to `kNicBurstAlignment` (512) so consecutive
   * sub-chunk staging addresses stay 512-byte aligned for the IBGDA
   * NIC's RDMA burst engine. (Per-piece compressed payload starts
   * stay `kNvcompInputAlignment`-aligned via the `alignInputBufferSize`
   * helper used to compute the layout; the final 512-byte round-up
   * here is purely for inter-sub-chunk NIC burst alignment.)
   *
   * Caller must size the IBGDA `dataBufferSize` such that
   *   perBlockSlot >= worst_case_chunk_stride(chunkSize)
   * (the transport traps when this fails).
   *
   * The per-piece worst-case padded payload size
   * (`alignInputBufferSize(max_comp_chunk_size())`) is provided by the
   * `__forceinline__` `worst_comp_padded_per_chunk()` accessor so the
   * launcher doesn't reconstruct the compressor tag type and re-call
   * `max_comp_chunk_size()` on every per-chunk dispatch.
   */
  __host__ __device__ __forceinline__ static std::size_t
  worst_case_chunk_stride(std::size_t chunkSize) {
    if (chunkSize == 0) {
      return 0;
    }
    const std::size_t numChunks =
        (chunkSize + MaxUncompBytes - 1ULL) / MaxUncompBytes;
    const std::size_t headerBytes = numChunks * sizeof(std::size_t);
    // The per-chunk size header table is written here as `size_t` and read
    // back on the receiver via `unsigned long long` (`__ldcv` in recv /
    // recv_forward); require identical width so the two views agree. Holds on
    // all 64-bit CUDA targets in scope; fails loudly if that ever changes.
    static_assert(
        sizeof(std::size_t) == sizeof(unsigned long long),
        "ANS size-header table assumes sizeof(size_t) == sizeof(unsigned long long)");
    const std::size_t headerPadded = alignInputBufferSize(headerBytes);
    const std::size_t worstTotal =
        headerPadded + numChunks * worst_comp_padded_per_chunk();
    const std::size_t aligned = (worstTotal + (kNicBurstAlignment - 1ULL)) &
        ~(kNicBurstAlignment - 1ULL);
    // Keep the historical `max(chunkSize, ...)` behaviour so callers
    // that hand in a raw chunkSize > the compressed worst-case still
    // get a slot large enough to hold the raw bytes (defensive — the
    // compressed layout above is strictly larger for any non-zero
    // chunkSize, but the max is essentially free).
    return chunkSize > aligned ? chunkSize : aligned;
  }

  /**
   * Largest multiple of `MaxUncompBytes` for which the on-wire
   * chunked-ANS layout (header table + padded compressed pieces,
   * rounded up to `kNicBurstAlignment`) is guaranteed to fit within
   * `perBlockSlot`. Used by the dispatcher in
   * AllToAllvTileCompressed.cuh as a safe default when the caller
   * passes the `max_signal_bytes == 0` sentinel: we substitute the
   * largest chunk size that the transport's
   * `chunkStride > perBlockSlot` check accepts, instead of letting
   * `chunkSize` default to the full `perBlockSlot` and trapping
   * (per-piece compressed worst case is ~1.3x the uncompressed
   * input, so a chunkSize == perBlockSlot can never fit in
   * perBlockSlot of staging).
   *
   * Walks the candidate piece count downwards from the slot-size /
   * per-piece-worst-case upper bound, terminating as soon as
   * `worst_case_chunk_stride(N * MaxUncompBytes) <= perBlockSlot`.
   * The loop runs at most a couple of iterations in practice
   * (header padding only changes the total by ~256 bytes per
   * boundary).
   *
   * Returns 0 if `perBlockSlot == 0` (no usable slot). Returns
   * `MaxUncompBytes` as a best-effort fallback if even a single
   * worst-case piece cannot fit — the transport's
   * `chunkStride > perBlockSlot` trap will then surface the
   * misconfiguration with a clear message.
   */
  __host__ __device__ __forceinline__ static std::size_t
  max_safe_chunk_size_for_slot(std::size_t perBlockSlot) {
    if (perBlockSlot == 0) {
      return 0;
    }
    std::size_t numChunks = perBlockSlot / worst_comp_padded_per_chunk();
    if (numChunks == 0) {
      return MaxUncompBytes;
    }
    while (numChunks > 0 &&
           worst_case_chunk_stride(numChunks * MaxUncompBytes) > perBlockSlot) {
      --numChunks;
    }
    if (numChunks == 0) {
      return MaxUncompBytes;
    }
    return numChunks * MaxUncompBytes;
  }

 private:
  // Worst-case per-piece padded payload size (compressed worst-case payload,
  // rounded up to `kNvcompInputAlignment` so that chunk[i+1] starts at a
  // 16-byte-aligned offset for nvCOMPDx's decompress input-alignment
  // requirement). This is a `__host__ __device__` accessor rather than an
  // `inline static const` with a lambda initializer: nvcompdx's
  // `max_comp_chunk_size()` isn't guaranteed `constexpr` across versions, so a
  // dynamically-initialized static would be a host-only symbol that is illegal
  // to read from the `__host__ __device__` sizing helpers above (and rejected
  // outright by some CUDA toolchains). The `__forceinline__` folds the call
  // away at each site, so there is no per-dispatch recomputation cost.
  __host__ __device__ __forceinline__ static std::size_t
  worst_comp_padded_per_chunk() {
    return alignInputBufferSize(AnsCompressorType().max_comp_chunk_size());
  }

 public:
  /**
   * Block-cooperative ANS compression of one transport sub-chunk into
   * the sender staging slot. Mirrors `Memcpy::send` signature so it slots
   * straight into `tr.send<>()`. Now accepts arbitrary `nbytes` — when
   * `nbytes > MaxUncompBytes`, internally splits into
   * `numChunks = ceil(nbytes / MaxUncompBytes)` per-piece ANS compress
   * calls and lays them out back-to-back in the staging slot with a
   * leading per-piece size header table. See
   * `worst_case_chunk_stride()` for the on-wire layout description.
   *
   * Returns the total number of staging bytes written
   * (`alignInputBufferSize(numChunks * 8) + sum_i
   * alignInputBufferSize(comp_i)`). The transport leader uses this directly to
   * size the RDMA put, so no in-staging header read-back is needed.
   *
   * The compressor's per-block shared-memory scratch is declared
   * inline as a static `__shared__` array, sized at compile time off
   * `AnsCompressorType::shmem_size_group()` (constexpr in nvcompdx).
   * The same scratch region is reused across the inner per-piece
   * compress calls.
   *
   * `char* alignedAuxBuf` is a per-block 16-byte-aligned global-memory
   * scratch region (size >= MaxUncompBytes) used ONLY when the per-call
   * `src` pointer is not 16-byte aligned. nvcompdx requires the input
   * pointer to be 16-byte aligned (see
   * `InputAlignment<G,datatype,ans,compress>::execute()`), but neither
   * checks nor traps at runtime — a misaligned input silently corrupts
   * the compressed stream or hangs in warp-cooperative code paths. We
   * detect this at runtime and, when needed, cooperatively memcpy each
   * sub-piece of `src` into `alignedAuxBuf` (one piece at a time,
   * MaxUncompBytes max) and feed the aligned copy to
   * `compressor.execute`. Pass `nullptr` if every caller is guaranteed
   * to provide a 16-byte-aligned `src`; the fast-path is identical to
   * the previous behaviour.
   */
  template <typename... Args>
  __device__ __forceinline__ static std::size_t send(
      char* staging,
      const char* src,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      char* alignedAuxBuf,
      Args...) {
    // Per-block compressor scratch. Sized at compile time from
    // nvcompdx's constexpr `shmem_size_group()`, but overlaid with
    // the matching `recv` `s_decompress_shmem` allocation via the
    // shared `get_shared_scratch()` helper (see its doc-comment for
    // the mutual-exclusion contract). Kernels that only call
    // `send()` still pay only `max(send, recv)` bytes rather than
    // both — the unused-direction delta is negligible.
    uint8_t* s_compress_shmem = get_shared_scratch();
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(AnsCompressorType);

    if (nbytes == 0) {
      return 0;
    }

    // Number of MaxUncompBytes-sized sub-pieces needed to cover nbytes.
    // Each sub-piece is ANS-compressed independently into the staging
    // slot. The leading header table stores the compressed size of
    // each sub-piece so the receiver can walk the layout.
    const std::size_t numChunks =
        (nbytes + MaxUncompBytes - 1ULL) / MaxUncompBytes;
    const std::size_t headerBytes = numChunks * sizeof(std::size_t);
    // Padded header section so chunk[0] starts at a 16-byte aligned
    // offset (nvCOMPDx decompress input alignment requirement).
    std::size_t writeOffset = alignInputBufferSize(headerBytes);

    size_t* const sizeHeaders = reinterpret_cast<size_t*>(staging);

    // Misalignment detection — done once for the base src pointer.
    // For typical MaxUncompBytes that is a multiple of 16, every
    // sub-piece start (`src + i*MaxUncompBytes`) inherits the same
    // 16-byte alignment as `src` itself, so a single up-front check
    // captures the misalignment state for the whole loop. The required
    // input alignment is `kNvcompInputAlignment` (16) — nvcompdx's
    // documented input-pointer requirement, and the same value the
    // staging layout is padded to via `alignInputBufferSize`.
    bool needRealign = false;
    if constexpr (!kSrcAligned) {
      needRealign = (reinterpret_cast<uintptr_t>(src) &
                     (kNvcompInputAlignment - 1ULL)) != 0ULL;
      if (needRealign) {
        if (alignedAuxBuf == nullptr) {
          if (group.is_leader()) {
            printf(
                "[PIPES] FATAL: AnsCompress::send src=%p is not "
                "16-byte aligned but alignedAuxBuf is null "
                "block=(%u,%u,%u). Caller must allocate a per-block "
                "aligned aux buffer (size >= MaxUncompBytes) when "
                "feeding sub-pointers with sub-16-byte alignment to "
                "the ANS-compressed path, or instantiate "
                "AnsCompress<..., kSrcAligned=true>.\n",
                src,
                blockIdx.x,
                blockIdx.y,
                blockIdx.z);
          }
          __trap();
        }
        if ((reinterpret_cast<uintptr_t>(alignedAuxBuf) &
             (kNvcompInputAlignment - 1ULL)) != 0ULL) {
          if (group.is_leader()) {
            printf(
                "[PIPES] FATAL: AnsCompress::send alignedAuxBuf=%p is "
                "not 16-byte aligned block=(%u,%u,%u). The default "
                "cudaMalloc alignment is >=256 bytes, so this should "
                "only happen if the caller derived alignedAuxBuf from "
                "a sub-pointer with sub-16-byte alignment.\n",
                alignedAuxBuf,
                blockIdx.x,
                blockIdx.y,
                blockIdx.z);
          }
          __trap();
        }
      }
    } else {
      // kSrcAligned == true: caller has promised src is 16-byte aligned.
      // alignedAuxBuf is unused on this path; suppress the unused-parameter
      // warning without emitting any SASS.
      (void)alignedAuxBuf;
    }

    // Inner loop: compress each sub-piece in serial, advancing writeOffset by
    // the (16-byte-aligned) per-piece compressed size. Every thread iterates
    // identically; for all but the last piece the `group.sync()` below makes
    // that piece's `sizeHeaders[i]` write visible before the read. On the last
    // piece the sync is skipped, so a non-leader thread's returned
    // `writeOffset` may lag by that piece's size until execute()'s own
    // completion propagates — but the transport consumes only the *leader*'s
    // returned value to size the RDMA put, so the leader's value is the
    // authoritative contract.
    for (std::size_t i = 0; i < numChunks; ++i) {
      const std::size_t pieceOffset = i * MaxUncompBytes;
      const std::size_t pieceBytes = (pieceOffset + MaxUncompBytes <= nbytes)
          ? MaxUncompBytes
          : (nbytes - pieceOffset);

      const char* effectiveSrc = src + pieceOffset;
      if constexpr (!kSrcAligned) {
        if (needRealign) {
          // Realign this piece into alignedAuxBuf so nvcompdx sees a
          // 16-byte-aligned input pointer. memcpy_vectorized handles
          // unaligned src by falling back to byte loads if needed.
          memcpy_vectorized(
              alignedAuxBuf, src + pieceOffset, pieceBytes, group);
          effectiveSrc = alignedAuxBuf;
          group.sync();
        }
      }

      AnsCompressorType{}.execute(
          effectiveSrc,
          staging + writeOffset,
          pieceBytes,
          &sizeHeaders[i],
          s_compress_shmem,
          static_cast<uint8_t*>(nullptr));

      // Skip the inter-piece barrier on the last iteration: there's no
      // next iteration to race against on `s_compress_shmem`, and the
      // caller (`P2pIbgdaTransportDevice::send`) issues its own
      // `group.sync()` immediately after `CopyOp::send` returns, which
      // also publishes `sizeHeaders[numChunks-1]` for any post-loop
      // consumers (e.g. the `PIPES_ANS_COLLECT_STATS` totaliser
      // below). Saves one block-wide barrier per `send()` call.
      if (i + 1 < numChunks) {
        group.sync();
      }
      const std::size_t compSize = sizeHeaders[i];
      writeOffset += alignInputBufferSize(compSize);
    }

#ifdef PIPES_ANS_COLLECT_STATS
    // Per-send leader-only atomicAdd into the global compression-ratio
    // counters. Gated behind `PIPES_ANS_COLLECT_STATS` (set on
    // `:alltoallv_tile_compressed_obj` for the bench's compRatio
    // column). The loop above skips its inter-piece barrier on the final
    // iteration, so the last piece's `sizeHeaders[numChunks-1]` write is not
    // yet guaranteed visible to the leader here — sync before totalising.
    group.sync();
    if (group.is_leader()) {
      std::size_t totalCompPayload = 0;
      for (std::size_t i = 0; i < numChunks; ++i) {
        totalCompPayload += sizeHeaders[i];
      }
      atomicAdd(
          &g_pipes_ans_total_uncomp_bytes,
          static_cast<unsigned long long>(nbytes));
      atomicAdd(
          &g_pipes_ans_total_comp_bytes,
          static_cast<unsigned long long>(totalCompPayload));
    }
#endif

    // No per-thread release fence here: the transport
    // (`P2pIbgdaTransportDevice::send`) already calls
    // `__threadfence_system()` from every cooperating thread
    // immediately after `CopyOp::send` returns and before the
    // leader-issued RDMA put.
    return writeOffset;
  }

  /**
   * Block-cooperative ANS decompression of one transport sub-chunk from
   * the receiver staging slot into `dst`. The leading
   * `numChunks * sizeof(size_t)` bytes of `staging` are the per-piece
   * compressed-size headers written by the sender, where
   * `numChunks = ceil(nbytes / MaxUncompBytes)`. See
   * `worst_case_chunk_stride()` for the full on-wire layout.
   *
   * Returns the total number of staging bytes consumed, matching the
   * value returned by the paired `send()` call. The transport doesn't
   * currently use this on the recv side (recv-side put sizing happens
   * pre-transmit on the sender), but it's surfaced for symmetry with
   * `send` and for callers that want to advance their own staging
   * cursors.
   */
  template <typename... Args>
  __device__ __forceinline__ static std::size_t recv(
      char* dst,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(AnsDecompressorType);
    // Per-block decompressor scratch. Overlaid with the matching
    // `send` `s_compress_shmem` allocation via the shared
    // `get_shared_scratch()` helper (see its doc-comment); both
    // directions must NEVER execute concurrently within the same
    // block.
    uint8_t* s_decompress_shmem = get_shared_scratch();

    if (nbytes == 0) {
      return 0;
    }

    const std::size_t numChunks =
        (nbytes + MaxUncompBytes - 1ULL) / MaxUncompBytes;
    const std::size_t headerBytes = numChunks * sizeof(std::size_t);
    // Chunk[0] begins immediately after the (16-byte-padded) header
    // section; subsequent chunks each advance by
    // `alignInputBufferSize(comp_i)` so their start offsets remain
    // 16-byte aligned (nvCOMPDx decompress input requirement).
    std::size_t readOffset = alignInputBufferSize(headerBytes);

    __shared__ size_t ans_out_size;
    for (std::size_t i = 0; i < numChunks; ++i) {
      const std::size_t pieceOffset = i * MaxUncompBytes;
      const std::size_t pieceBytes = (pieceOffset + MaxUncompBytes <= nbytes)
          ? MaxUncompBytes
          : (nbytes - pieceOffset);

      // The transport's wait_signal upstream of this call synchronises
      // with the inbound NIC payload (DATA_READY is fenced after the
      // put), establishing acquire ordering on the staging cacheline
      // for every thread on this rank. Once the ack is observed, the
      // header table is fully committed in L2 and a per-thread
      // cache-volatile load (`__ldcv`) returns the same value to every
      // thread without a block-wide barrier — saving one `group.sync()`
      // per recv on the hot path. `__ldcv` bypasses L1 and forces an L2
      // read so SMs with stale L1 lines still observe the committed
      // bytes.
      const std::size_t compSize = static_cast<std::size_t>(
          __ldcv(reinterpret_cast<const unsigned long long*>(staging) + i));

#ifdef DEBUG
      // Bringup-only defensive bounds check on the in-staging header,
      // compiled in only with `-DDEBUG`. If it is bogus, surface it
      // explicitly instead of crashing inside the decompressor.
      if (group.is_leader()) {
        const std::size_t maxValidCompSize =
            AnsCompressorType().max_comp_chunk_size();
        if (compSize == 0 || compSize > maxValidCompSize) {
          printf(
              "[PIPES] FATAL: AnsCompress::recv invalid compSize=%llu "
              "(max=%llu) piece=%llu/%llu nbytes=%llu block=(%u,%u,%u)\n",
              (unsigned long long)compSize,
              (unsigned long long)maxValidCompSize,
              (unsigned long long)i,
              (unsigned long long)numChunks,
              (unsigned long long)nbytes,
              blockIdx.x,
              blockIdx.y,
              blockIdx.z);
          __trap();
        }
      }
#endif

      AnsDecompressorType{}.execute(
          staging + readOffset,
          dst + pieceOffset,
          compSize,
          &ans_out_size,
          s_decompress_shmem,
          static_cast<uint8_t*>(nullptr));

      // Skip the inter-piece barrier on the last iteration: there's
      // no next iteration to race against on `s_decompress_shmem` /
      // `ans_out_size`, and the caller
      // (`P2pIbgdaTransportDevice::recv`) issues its own
      // `group.sync()` immediately after `CopyOp::recv` returns.
      // Saves one block-wide barrier per `recv()` call, plus the
      // entire barrier when numChunks == 1 (the common single-piece
      // path).
      if (i + 1 < numChunks) {
        group.sync();
      }

#ifdef DEBUG
      if (group.is_leader()) {
        assert(ans_out_size == pieceBytes);
      }
#else
      (void)pieceBytes;
#endif
      readOffset += alignInputBufferSize(compSize);
    }
    // No trailing group.sync() here: the caller
    // (`P2pIbgdaTransportDevice::recv`) does its own `group.sync()`
    // immediately after `CopyOp::recv` returns.
    return readOffset;
  }

  /**
   * Decompresses `staging` → `dst` AND forwards the still-compressed
   * staging region (header table + padded compressed pieces) to
   * `fwd_staging` in one pass. This intentionally diverges from
   * `Memcpy::recv_forward` (which forwards the *decompressed* `dst`) —
   * for ANS it saves bandwidth on the next hop to forward the
   * compressed bytes unchanged. When `dst == nullptr` the local rank
   * is just relaying; we skip decompression entirely.
   *
   * Returns the total compressed-region size (matching the value
   * returned by the paired `send()` call) so the caller can size the
   * next-hop RDMA put without re-reading the staging.
   *
   * Not currently called by anything in this stack — D103888256
   * (`forward()` on P2pIbgdaTransportDevice) hasn't landed yet —
   * but defined so it's ready to drop in the day it does.
   */
  template <typename... Args>
  __device__ __forceinline__ static std::size_t recv_forward(
      char* dst,
      char* fwd_staging,
      const char* staging,
      std::size_t nbytes,
      ThreadGroup& group,
      std::size_t /*byte_offset*/,
      Args...) {
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(AnsDecompressorType);
    // Shared per-direction scratch, overlaid with `send`'s — see
    // `get_shared_scratch()` doc-comment. `recv_forward` is itself
    // a recv-side flow (it only decompresses, then forwards the
    // already-compressed staging verbatim) so the same
    // mutual-exclusion contract applies.
    uint8_t* s_decompress_shmem = get_shared_scratch();

    if (nbytes == 0) {
      return 0;
    }

    const std::size_t numChunks =
        (nbytes + MaxUncompBytes - 1ULL) / MaxUncompBytes;
    const std::size_t headerBytes = numChunks * sizeof(std::size_t);
    const std::size_t firstChunkOffset = alignInputBufferSize(headerBytes);

    // Walk the header table first so we know the total compressed
    // region size for the verbatim forward. Per-thread `__ldcv` loads —
    // see `recv()` for the rationale (the upstream wait_signal
    // provides the acquire fence; `__ldcv` bypasses L1 so all SMs
    // observe the committed L2 line).
    std::size_t totalCompBytes = firstChunkOffset;
    for (std::size_t i = 0; i < numChunks; ++i) {
      const std::size_t compSize = static_cast<std::size_t>(
          __ldcv(reinterpret_cast<const unsigned long long*>(staging) + i));
      totalCompBytes += alignInputBufferSize(compSize);
    }

    // Forward the compressed region (headers + padded pieces) verbatim
    // into fwd_staging so downstream hops can decompress without
    // re-paying the compression cost.
    memcpy_vectorized(fwd_staging, staging, totalCompBytes, group);

    // If we're the terminal hop on this rank, also produce the
    // decompressed output. Relay-only ranks (dst == nullptr) skip it.
    if (dst != nullptr) {
      __shared__ size_t ans_fwd_out_size;
      std::size_t readOffset = firstChunkOffset;
      for (std::size_t i = 0; i < numChunks; ++i) {
        const std::size_t pieceOffset = i * MaxUncompBytes;
        const std::size_t compSize = static_cast<std::size_t>(
            __ldcv(reinterpret_cast<const unsigned long long*>(staging) + i));
        AnsDecompressorType{}.execute(
            staging + readOffset,
            dst + pieceOffset,
            compSize,
            &ans_fwd_out_size,
            s_decompress_shmem,
            static_cast<uint8_t*>(nullptr));
        // Skip the inter-piece barrier on the last iteration: same
        // rationale as recv() — no next iteration to race against
        // and the caller (`P2pIbgdaTransportDevice::forward`) syncs
        // immediately after we return.
        if (i + 1 < numChunks) {
          group.sync();
        }
        readOffset += alignInputBufferSize(compSize);
      }
      // No trailing group.sync() here: the caller
      // (`P2pIbgdaTransportDevice::forward`) does its own `group.sync()`
      // immediately after `CopyOp::recv_forward` returns.
    }

    return totalCompBytes;
  }
};

#endif // PIPES_ENABLE_ANS_COMPRESSION

} // namespace comms::prims
