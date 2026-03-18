// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device Window - nvcc-compiled implementations
//
// This file provides extern "C" wrappers around TorchCommDeviceWindow methods.
// Compiled with nvcc to support NCCLX GIN templates.
// Linked at compile time into Triton kernels via extern_libs bitcode.
//
// Design:
//   - Block-scope ops (put_block, signal_block, flush_block, barrier_block):
//     All 128 block threads call these functions simultaneously (Triton
//     extern_elementwise invokes each extern once per thread). The caller must
//     invoke these convergently (no divergent control flow before the call
//     site).
//
//     put_block / signal_block:
//       Delegate to win->put() / win->signal() with CoopScope::BLOCK, which
//       handles both paths internally:
//       - LSA (NVLink): all threads cooperate on memcpy_vectorized; signal()
//         uses atom.release.sys to order prior stores before the signal write.
//       - GIN (RDMA): CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__
//         before/after posting WQE. Safe because all threads enter
//         convergently.
//
//     flush_block / barrier_block:
//       Cannot use CoopScope::BLOCK — two problems:
//       (a) __syncthreads__: ncclGin::flush and ncclBarrierSession both emit
//           __syncthreads__ (via ncclCoopCta::sync()), risking deadlock if any
//           thread diverges before the call.
//       (b) barrier semantic: each thread independently signals peers in the
//           GIN barrier protocol, so 128 threads would each send 128x the
//           intended signal increments, corrupting epoch tracking.
//       Fix: threadIdx.x == 0 guard + CoopScope::THREAD (ncclCoopTile<1>
//       whose sync() is a compile-time no-op). Other threads return
//       immediately.
//
//   - Thread-scope ops (wait_signal, fence, read/reset, rank, etc.):
//     Idempotent w.r.t. thread count — spin-polls, PTX fences, and atomic
//     reads produce the same result whether called from 1 or 128 threads.
//     All 128 threads do call these (extern_elementwise), which is harmless.

#include <cuda_runtime.h>

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

using namespace torchcomms::device;

using DeviceWindow = TorchCommDeviceWindow<NCCLGinBackend>;

extern "C" {

// =============================================================================
// Block-scope RMA Operations
//
// All 128 block threads call these functions simultaneously (Triton
// extern_elementwise invokes each extern once per thread). The caller must
// invoke convergently (no divergent control flow before the call site).
// See the file-level design comment for details per function.
// =============================================================================

// torchcomms_self_copy_block: block-cooperative local memory copy for
// self-send (peer == my_rank) in alltoallv.
//
// Performs a direct memory copy from send_buf to recv_buf using all threads
// in the block.  This replaces the Triton tl.load/tl.store self-copy loop,
// which generated 85+ comparison/mask SSA values in the LLVM IR and added
// significant register pressure to the main alltoallv kernel.
//
// By moving self-copy to a CUDA extern, the Triton kernel's IR is smaller
// and the register allocator has fewer live values to manage in the hot
// memcpy path.
__device__ int torchcomms_self_copy_block(
    void* dst_ptr,
    unsigned long long dst_offset,
    void* src_ptr,
    unsigned long long src_offset,
    unsigned long long bytes) {
  auto* dst = reinterpret_cast<char*>(dst_ptr) + dst_offset;
  auto* src = reinterpret_cast<const char*>(src_ptr) + src_offset;
  auto group = detail::make_thread_group(CoopScope::BLOCK);
  comms::pipes::memcpy_vectorized(dst, src, static_cast<size_t>(bytes), group);
  return 0;
}

// torchcomms_put_block: block-cooperative data transfer.
//
// win->put(CoopScope::BLOCK) handles LSA and GIN internally:
//   - LSA (NVLink): all threads cooperate on memcpy_vectorized; signal() uses
//     atom.release.sys which orders all prior stores before the signal write.
//   - GIN (RDMA): CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__ before/
//     after posting WQE. Safe because all threads enter convergently.
//
// src buffer specified by its components (base_ptr, size, nccl_win) rather
// than a pointer to a RegisteredBuffer struct to avoid GPU memory allocation
// conflicts with NCCLX's cuMemMap-based memory management.
__device__ int torchcomms_put_block(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_base_ptr,
    unsigned long long src_size,
    void* src_nccl_win,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);

  RegisteredBuffer src_buf;
  src_buf.base_ptr = src_base_ptr;
  src_buf.size = static_cast<size_t>(src_size);
  src_buf.backend_window = src_nccl_win;

  return win->put(
      static_cast<size_t>(dst_offset),
      src_buf,
      static_cast<size_t>(src_offset),
      dst_rank,
      static_cast<size_t>(bytes),
      signal_id,
      counter_id,
      CoopScope::BLOCK);
}

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
    void* src_base_ptr,
    size_t src_size,
    void* src_nccl_win,
    size_t src_offset,
    int dst_rank,
    size_t bytes) {
  RegisteredBuffer src_buf;
  src_buf.base_ptr = src_base_ptr;
  src_buf.size = src_size;
  src_buf.backend_window = src_nccl_win;
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
    void* src_base_ptr,
    size_t src_size,
    void* src_nccl_win,
    size_t src_offset,
    int dst_rank,
    size_t bytes) {
  RegisteredBuffer src_buf;
  src_buf.base_ptr = src_base_ptr;
  src_buf.size = src_size;
  src_buf.backend_window = src_nccl_win;
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
    void* src_nccl_win,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  const ncclDevComm& dev_comm = win->comm();

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), dst_rank)) {
    // NVLink path: inline PTX memcpy, zero allocas, zero spills.
    ncclWindow_t dst_win = win->window();
    ncclWindow_t src_win = static_cast<ncclWindow_t>(src_nccl_win);
    char* dst_base =
        static_cast<char*>(ncclGetPeerPointer(dst_win, 0, dst_rank));
    char* src_base = static_cast<char*>(ncclGetLocalPointer(src_win, 0));

    nvl_memcpy_ptx(
        dst_base + static_cast<size_t>(dst_offset),
        src_base + static_cast<size_t>(src_offset),
        static_cast<size_t>(bytes),
        threadIdx.x,
        blockDim.x);
  } else {
    // GIN (RDMA) fallback: __noinline__ to isolate register pressure.
    gin_put_fallback(
        win,
        static_cast<size_t>(dst_offset),
        nullptr,
        0,
        src_nccl_win,
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
    void* src_nccl_win,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long total_bytes,
    unsigned long long chunk_size) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  const ncclDevComm& dev_comm = win->comm();

  auto total = static_cast<size_t>(total_bytes);
  auto chunk = static_cast<size_t>(chunk_size);
  auto num_chunks = (total + chunk - 1) / chunk;

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), dst_rank)) {
    // NVLink path: inline PTX memcpy, zero allocas, zero spills.
    ncclWindow_t dst_win = win->window();
    ncclWindow_t src_win = static_cast<ncclWindow_t>(src_nccl_win);
    char* dst_base =
        static_cast<char*>(ncclGetPeerPointer(dst_win, 0, dst_rank));
    char* src_base = static_cast<char*>(ncclGetLocalPointer(src_win, 0));

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
    // GIN (RDMA) fallback: __noinline__ to isolate register pressure.
    auto warp_id = threadIdx.x / 32;
    auto num_warps = blockDim.x / 32;

    for (size_t c = warp_id; c < num_chunks; c += num_warps) {
      auto off = c * chunk;
      auto len = (off + chunk <= total) ? chunk : (total - off);
      gin_put_warp_fallback(
          win,
          static_cast<size_t>(dst_offset) + off,
          nullptr,
          0,
          src_nccl_win,
          static_cast<size_t>(src_offset) + off,
          dst_rank,
          len);
    }
  }

  __syncthreads();
  return 0;
}

__device__ int torchcomms_signal_block(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long value) {
  // LSA: signal() guards with thread_id_in_group==0 internally
  // (CoopScope::BLOCK). GIN: CoopScope::BLOCK → ncclCoopCta{} → __syncthreads__
  // before/after posting the atomic WQE. Safe when all block threads enter this
  // function convergently.
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->signal(peer, signal_id, SignalOp::ADD, value, CoopScope::BLOCK);
}

__device__ int torchcomms_flush_block(void* win_ptr) {
  // Cannot call flush(CoopScope::BLOCK) from all threads.
  //
  // gin.flush(ncclCoopCta{}) emits __syncthreads__ unconditionally (twice:
  // before and after the peer-poll loop in ncclGin_BackendMask::flush). If
  // any thread reaches that barrier without the others (e.g. due to divergent
  // control flow in the caller), the block hangs forever.
  //
  // Fix: only thread 0 runs flush(CoopScope::THREAD), which uses
  // ncclCoopTile<1> whose sync() is a compile-time no-op. Other threads return
  // immediately.
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->flush(CoopScope::THREAD);
}

__device__ int torchcomms_barrier_block(void* win_ptr, int barrier_id) {
  // Cannot call barrier(CoopScope::BLOCK) from all threads — two problems:
  //
  // 1. __syncthreads__: ncclBarrierSession emits __syncthreads__ multiple
  //    times (LSA arrive/wait, GIN sync×2, destructors). Same deadlock risk
  //    as flush_block if any thread diverges before the call.
  //
  // 2. Semantic correctness: the GIN barrier's signal loop runs independently
  //    per thread — each thread signals (nRanks-1) peers. With 128 threads,
  //    each peer would receive 128× the intended signal increments, corrupting
  //    epoch tracking and firing the barrier protocol 128 times.
  //
  // Fix: only thread 0 runs barrier(CoopScope::THREAD). Others return
  // immediately.
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->barrier(barrier_id, CoopScope::THREAD);
}

// =============================================================================
// Signal Operations (Remote Notification)
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// =============================================================================

__device__ int torchcomms_wait_signal(
    void* win_ptr,
    int signal_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal(signal_id, CmpOp::GE, expected_value);
}

// Wait for signal from a specific peer to reach expected value.
// Used for per-peer synchronization in alltoallv and similar patterns.
// Thread-scope (idempotent) — all 128 threads can call; same result.
__device__ int torchcomms_wait_signal_from(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal_from(peer, signal_id, CmpOp::GE, expected_value);
}

__device__ unsigned long long torchcomms_read_signal(
    void* win_ptr,
    int signal_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->read_signal(signal_id);
}

__device__ void torchcomms_reset_signal(void* win_ptr, int signal_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  win->reset_signal(signal_id);
}

// =============================================================================
// Counter Operations (Local Completion)
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// (wait_local/read_counter/reset_counter use ncclCoopThread{} internally.)
// =============================================================================

__device__ int torchcomms_wait_local(
    void* win_ptr,
    int counter_id,
    unsigned long long expected_value) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_local(counter_id, CmpOp::GE, expected_value);
}

__device__ unsigned long long torchcomms_read_counter(
    void* win_ptr,
    int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->read_counter(counter_id);
}

__device__ void torchcomms_reset_counter(void* win_ptr, int counter_id) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  win->reset_counter(counter_id);
}

// =============================================================================
// Synchronization & Completion
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// =============================================================================

__device__ int torchcomms_fence(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->fence();
}

// =============================================================================
// Window Properties
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. Spin-polls and atomic reads are thread-safe.
// =============================================================================

__device__ int torchcomms_rank(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->rank();
}

__device__ int torchcomms_num_ranks(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->num_ranks();
}

__device__ void* torchcomms_base(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->base();
}

__device__ unsigned long long torchcomms_size(void* win_ptr) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return static_cast<unsigned long long>(win->size());
}

// =============================================================================
// NVLink Address Query
// Thread-scope (idempotent) — all 128 threads call these; result is the same
// as if only one thread called. ncclGetPeerPointer computes a flat LSA address
// from constant fields in the window struct — no side effects, no atomics.
// =============================================================================

__device__ void* torchcomms_get_nvlink_address(void* win_ptr, int peer) {
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return ncclGetPeerPointer(win->window(), 0, peer);
}

} // extern "C"
