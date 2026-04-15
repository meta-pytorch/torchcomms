// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device Window - GIN-specific NVLink-optimized operations
//
// This file contains NVLink-optimized put operations that bypass the generic
// TorchCommDeviceWindow API for maximum performance. These are GIN-specific
// and use ncclGetPeerPointer / ncclTeamLsa directly.
//
// For generic (backend-agnostic) operations, see device_window.cu.
//
// Functions in this file:
//   - torchcomms_put_block_direct: NVLink inline PTX put with GIN fallback
//   - torchcomms_put_warp_chunked_direct: Warp-distributed chunked variant
//
// Design:
//   NVLink peers: inline PTX memcpy (zero allocas, zero register spills)
//   GIN peers: __noinline__ fallback to win->put() to isolate register pressure

#include <cuda_runtime.h>

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

using namespace torchcomms::device;
using torch::comms::RegisteredBuffer;

using DeviceWindow = TorchCommDeviceWindow<NCCLGinBackend>;

extern "C" {

// =============================================================================
// Inline PTX memcpy — zero allocas, zero register spills
//
// Replaces pipes::memcpy_vectorized which uses VecType v[kUnroll] arrays
// that LLVM lowers to alloca [8 x uint4] (128 bytes = 32 registers).
// When multiple memcpy_vectorized instantiations exist in the same
// compilation unit (even in separate functions), Triton's LLVM→PTX
// lowering inlines everything into a single kernel entry, causing
// all allocas to coexist and generating 42+ register spills.
//
// This inline PTX approach uses only 4 registers per thread for the
// copy (val.x, val.y, val.z, val.w) — no alloca, no spills.  The
// PTX instructions are emitted directly by clang into the bitcode
// and pass through to the final PTX unchanged.
//
// Two variants with different unroll factors:
//   nvl_memcpy_ptx_u1: 1 uint4 per iteration (4 regs, zero spills)
//   nvl_memcpy_ptx_u2: 2 uint4 per iteration (8 regs, zero spills,
//                       better ILP from overlapping load/store)
// =============================================================================

// Single-uint4 loop: 4 registers, zero spills, max simplicity.
__device__ __forceinline__ void nvl_memcpy_ptx(
    char* __restrict__ dst,
    const char* __restrict__ src,
    size_t bytes,
    int tid,
    int nthreads) {
  // Each thread copies 16 bytes (one uint4) per iteration, strided by nthreads.
  // This gives perfect coalescing: 32 threads × 16 bytes = 512 bytes per warp.
  size_t stride = static_cast<size_t>(nthreads) * 16;
  size_t aligned_bytes = (bytes / stride) * stride;

  // Main aligned loop: uint4 (128-bit) loads and stores
  for (size_t off = static_cast<size_t>(tid) * 16; off < aligned_bytes;
       off += stride) {
    unsigned int v0, v1, v2, v3;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                 : "l"(src + off));
    asm volatile("st.global.v4.u32 [%4], {%0,%1,%2,%3};"
                 :
                 : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(dst + off));
  }

  // Remainder: handle tail bytes not aligned to stride.
  // First handle full uint4 chunks, then byte-level for the final < 16 bytes.
  size_t uint4_end = (bytes / 16) * 16;
  for (size_t off = aligned_bytes + static_cast<size_t>(tid) * 16;
       off < uint4_end;
       off += stride) {
    unsigned int v0, v1, v2, v3;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                 : "l"(src + off));
    asm volatile("st.global.v4.u32 [%4], {%0,%1,%2,%3};"
                 :
                 : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(dst + off));
  }

  // Byte-level tail: copy the final bytes that don't fill a uint4 (< 16 bytes).
  // This handles minimum-size messages (e.g., 4 bytes = 1 float).
  for (size_t off = uint4_end + static_cast<size_t>(tid); off < bytes;
       off += static_cast<size_t>(nthreads)) {
    dst[off] = src[off];
  }
}

// =============================================================================
// GIN (RDMA) fallback — __noinline__ to prevent its alloca from polluting
// the NVLink hot path's register allocation.
//
// When Triton inlines all functions into one PTX kernel entry, any
// memcpy_vectorized alloca [8 x uint4] from the GIN path would coexist
// with the NVLink inline PTX path, causing register spills.  By marking
// the GIN fallback __noinline__, its alloca stays in a separate function
// and doesn't affect the NVLink path's register budget.
// =============================================================================

__device__ __noinline__ int gin_put_fallback(
    DeviceWindow* win,
    size_t dst_offset,
    const RegisteredBuffer& src_buf,
    size_t src_offset,
    int dst_rank,
    size_t bytes) {
  return win->put(
      dst_offset,
      src_buf,
      src_offset,
      dst_rank,
      bytes,
      -1,
      -1,
      CoopScope::BLOCK);
}

__device__ __noinline__ int gin_put_warp_fallback(
    DeviceWindow* win,
    size_t dst_offset,
    const RegisteredBuffer& src_buf,
    size_t src_offset,
    int dst_rank,
    size_t bytes) {
  return win->put(
      dst_offset,
      src_buf,
      src_offset,
      dst_rank,
      bytes,
      -1,
      -1,
      CoopScope::WARP);
}

// =============================================================================
// NVLink-optimized put with GIN fallback
//
// These functions check if the peer is on the LSA (NVLink) team:
//   - NVLink: uses inline PTX memcpy (zero allocas, zero spills)
//   - GIN: falls back to win->put() via __noinline__ helper
//
// The two functions are intentionally SEPARATE to avoid having two
// memcpy instantiations in the same function (see register pressure
// analysis in the plan document).
// =============================================================================

__device__ int torchcomms_put_block_direct(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_registered_buf_ptr,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  auto* src_buf =
      reinterpret_cast<const RegisteredBuffer*>(src_registered_buf_ptr);
  const ncclDevComm& dev_comm = win->comm();

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), dst_rank)) {
    // NVLink path: use base_ptr directly from RegisteredBuffer.
    // base_ptr == ncclGetLocalPointer(backend_window, 0), avoiding the
    // indirection through ncclWindow_t.
    ncclWindow_t dst_win = win->window();
    char* dst_base =
        static_cast<char*>(ncclGetPeerPointer(dst_win, 0, dst_rank));
    char* src_base = static_cast<char*>(src_buf->base_ptr);

    nvl_memcpy_ptx(
        dst_base + static_cast<size_t>(dst_offset),
        src_base + static_cast<size_t>(src_offset),
        static_cast<size_t>(bytes),
        threadIdx.x,
        blockDim.x);
  } else {
    // GIN (RDMA) fallback: pass the full RegisteredBuffer.
    gin_put_fallback(
        win,
        static_cast<size_t>(dst_offset),
        *src_buf,
        static_cast<size_t>(src_offset),
        dst_rank,
        static_cast<size_t>(bytes));
  }

  __syncthreads();
  return 0;
}

__device__ int torchcomms_put_warp_chunked_direct(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_registered_buf_ptr,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long total_bytes,
    unsigned long long chunk_size) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  auto* src_buf =
      reinterpret_cast<const RegisteredBuffer*>(src_registered_buf_ptr);
  const ncclDevComm& dev_comm = win->comm();

  auto total = static_cast<size_t>(total_bytes);
  auto chunk = static_cast<size_t>(chunk_size);
  auto num_chunks = (total + chunk - 1) / chunk;

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), dst_rank)) {
    // NVLink path: use base_ptr directly from RegisteredBuffer.
    ncclWindow_t dst_win = win->window();
    char* dst_base =
        static_cast<char*>(ncclGetPeerPointer(dst_win, 0, dst_rank));
    char* src_base = static_cast<char*>(src_buf->base_ptr);

    auto warp_id = threadIdx.x / 32;
    auto num_warps = blockDim.x / 32;

    for (size_t c = warp_id; c < num_chunks; c += num_warps) {
      auto off = c * chunk;
      auto len = (off + chunk <= total) ? chunk : (total - off);
      nvl_memcpy_ptx(
          dst_base + static_cast<size_t>(dst_offset) + off,
          src_base + static_cast<size_t>(src_offset) + off,
          len,
          threadIdx.x % 32,
          32);
    }
  } else {
    // GIN (RDMA) fallback: pass the full RegisteredBuffer.
    auto warp_id = threadIdx.x / 32;
    auto num_warps = blockDim.x / 32;

    for (size_t c = warp_id; c < num_chunks; c += num_warps) {
      auto off = c * chunk;
      auto len = (off + chunk <= total) ? chunk : (total - off);
      gin_put_warp_fallback(
          win,
          static_cast<size_t>(dst_offset) + off,
          *src_buf,
          static_cast<size_t>(src_offset) + off,
          dst_rank,
          len);
    }
  }

  __syncthreads();
  return 0;
}

} // extern "C"
