// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Device-side constants used across the dispatch / combine kernels.
//
// subset that actually shows up in the kernel bodies. The host-side constants
// (NUM_WORKSPACE_BYTES, NUM_MAX_NVL_PEERS, NUM_BUFFER_ALIGNMENT_BYTES) live
// in `cpp/Config.h` so host code can use them without pulling in CUDA/HIP
// headers.
//
// NVSHMEM / rocshmem includes are intentionally omitted — pipes'
// MultiPeerNvlTransport replaces NVSHMEM for NVLink signaling, and pipes'
// MultipeerIbgdaTransport replaces NVSHMEM for RDMA.

namespace comms::prims::moe_ep::kernels {

// Maximum number of FIFO slots used by the `task_fifo_ptrs` barrier. We
// don't use task_fifo (we use pipes' barrier_sync instead), but the kernels
// still reserve workspace for them in the layout helpers.
inline constexpr int NUM_MAX_FIFO_SLOTS = 32768;

inline constexpr int NUM_MAX_LOCAL_EXPERTS = 1024;

// Tag value used by combine's grid_barrier to mark "I am done."
inline constexpr int FINISHED_SUM_TAG = 1024;

// CPU-side timeout: 100 seconds before throwing.
inline constexpr int NUM_CPU_TIMEOUT_SECS = 100;

// GPU-side spin-loop timeout. NVIDIA: ~100s on 2 GHz `clock64`. AMD's
// `__builtin_amdgcn_s_memrealtime` ticks at ~100 MHz (15x slower), so use a
// proportionally smaller bound so the timeout actually fires within the
// test harness window.
#ifdef __HIP_PLATFORM_AMD__
inline constexpr long long NUM_TIMEOUT_CYCLES = 3000000000ll; // ~30s @ 100MHz
#else
inline constexpr long long NUM_TIMEOUT_CYCLES = 200000000000ll;
#endif

inline constexpr int NUM_WAIT_NANOSECONDS = 500;

#ifdef __HIP_PLATFORM_AMD__
// On AMD, additional cycle budget per spin iteration since AMD wave size
// (64) doubles the iteration cost vs NVIDIA warp (32).
inline constexpr int NUM_WAIT_CYCLES_TIMES_64 = 16;
#endif

// Warp / wave size — 32 on NVIDIA, 64 on AMD.
#ifdef __HIP_PLATFORM_AMD__
inline constexpr int kWarpSize = 64;
inline constexpr int kEmulatedWarpSize = kWarpSize / 2; // 32
inline constexpr std::uint64_t kFullWarpMask = 0xffffffffffffffffULL;
inline constexpr std::uint64_t kFirstHalfMask = 0x00000000ffffffffULL;
inline constexpr std::uint64_t kSecondHalfMask = 0xffffffff00000000ULL;
#else
inline constexpr int kWarpSize = 32;
inline constexpr int kEmulatedWarpSize = kWarpSize;
inline constexpr std::uint32_t kFullWarpMask = 0xffffffffU;
#endif

// Topk index dtype — we standardize on int64 (TOPK_IDX_BITS=64).
using topk_idx_t = std::int64_t;

} // namespace comms::prims::moe_ep::kernels
