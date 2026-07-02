// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

//
// Three small device-only RAII helpers used to slice the flat workspace
// passed to dispatch / combine kernels into typed sub-buffers + per-channel
// + per-rank stripes. They:
//   - take a `void*&` cursor by reference and bump it forward by `total_bytes`
//   - cast the slice to a typed pointer so kernels can index without manual
//     pointer arithmetic
//   - support the three layouts dispatch / combine need:
//       Buffer        — flat per-rank scratch
//       AsymBuffer    — per-channel × per-rank scratch (asymmetric — sender
//                       writes to recv_ptr, recv reads from send_ptr; same
//                       memory)
//       SymBuffer     — per-channel × per-rank send + recv scratch (when
//                       decoupled, send and recv are different memory
//                       regions; when not, they alias and direction is
//                       determined by the kernel)

namespace comms::prims::moe_ep::kernels {

template <typename DType>
struct Buffer {
 private:
  std::uint8_t* ptr;

 public:
  int total_bytes;

  __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

  __device__ __forceinline__
  Buffer(void*& gbl_ptr, int num_elems, int offset = 0) {
    total_bytes = num_elems * sizeof(DType);
    ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) + offset * sizeof(DType);
    gbl_ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) + total_bytes;
  }

  __device__ __forceinline__ Buffer advance_also(void*& gbl_ptr) {
    gbl_ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) + total_bytes;
    return *this;
  }

  __device__ __forceinline__ DType* buffer() {
    return reinterpret_cast<DType*>(ptr);
  }

  __device__ __forceinline__ DType& operator[](int idx) {
    return buffer()[idx];
  }
};

template <typename DType, int kNumRanks = 1>
struct AsymBuffer {
 private:
  std::uint8_t* ptrs[kNumRanks];
  int num_bytes;

 public:
  int total_bytes;

  // Single-rank ctor — used in dispatch's per-channel local-receiver buffer.
  __device__ __forceinline__ AsymBuffer(
      void*& gbl_ptr,
      int num_elems,
      int num_ranks,
      int sm_id = 0,
      int num_sms = 1,
      int offset = 0) {
    static_assert(kNumRanks == 1, "Single-rank ctor requires kNumRanks == 1");
    num_bytes = num_elems * sizeof(DType);

    const int per_channel_bytes = num_bytes * num_ranks;
    total_bytes = per_channel_bytes * num_sms;
    ptrs[0] = reinterpret_cast<std::uint8_t*>(gbl_ptr) +
        per_channel_bytes * sm_id + num_bytes * offset;
    gbl_ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) + total_bytes;
  }

  // Multi-rank ctor — used in dispatch's per-channel sender buffer (one
  // per peer NVLink target).
  __device__ __forceinline__ AsymBuffer(
      void** gbl_ptrs,
      int num_elems,
      int num_ranks,
      int sm_id = 0,
      int num_sms = 1,
      int offset = 0) {
    static_assert(kNumRanks > 1, "Multi-rank ctor requires kNumRanks > 1");
    num_bytes = num_elems * sizeof(DType);

    const int per_channel_bytes = num_bytes * num_ranks;
    total_bytes = per_channel_bytes * num_sms;
#pragma unroll
    for (int i = 0; i < kNumRanks; ++i) {
      ptrs[i] = reinterpret_cast<std::uint8_t*>(gbl_ptrs[i]) +
          per_channel_bytes * sm_id + num_bytes * offset;
      gbl_ptrs[i] = reinterpret_cast<std::uint8_t*>(gbl_ptrs[i]) + total_bytes;
    }
  }

  __device__ __forceinline__ void advance(int shift) {
#pragma unroll
    for (int i = 0; i < kNumRanks; ++i) {
      ptrs[i] = ptrs[i] + shift * sizeof(DType);
    }
  }

  __device__ __forceinline__ AsymBuffer advance_also(void*& gbl_ptr) {
    gbl_ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) + total_bytes;
    return *this;
  }

  template <int kNumAlsoRanks>
  __device__ __forceinline__ AsymBuffer advance_also(void** gbl_ptrs) {
#pragma unroll
    for (int i = 0; i < kNumAlsoRanks; ++i) {
      gbl_ptrs[i] = reinterpret_cast<std::uint8_t*>(gbl_ptrs[i]) + total_bytes;
    }
    return *this;
  }

  __device__ __forceinline__ DType* buffer(int idx = 0) {
    static_assert(kNumRanks == 1, "`buffer` is only available for single rank");
    return reinterpret_cast<DType*>(ptrs[0] + num_bytes * idx);
  }

  __device__ __forceinline__ DType* buffer_by(int rank_idx, int idx = 0) {
    static_assert(
        kNumRanks > 1, "`buffer_by` is only available for multi-rank");
    return reinterpret_cast<DType*>(ptrs[rank_idx] + num_bytes * idx);
  }
};

template <typename DType, bool kDecoupled = true>
struct SymBuffer {
 private:
  // NOTE: for non-decoupled case `recv_ptr` is unused.
  std::uint8_t* send_ptr;
  std::uint8_t* recv_ptr;
  int num_bytes;

 public:
  int total_bytes;

  __device__ __forceinline__ SymBuffer(
      void*& gbl_ptr,
      int num_elems,
      int num_ranks,
      int sm_id = 0,
      int num_sms = 1) {
    num_bytes = num_elems * sizeof(DType);

    const int per_channel_bytes = num_bytes * num_ranks;
    total_bytes =
        per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
    send_ptr =
        reinterpret_cast<std::uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;
    recv_ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) +
        per_channel_bytes * (sm_id + num_sms);
    gbl_ptr = reinterpret_cast<std::uint8_t*>(gbl_ptr) + total_bytes;
  }

  __device__ __forceinline__ DType* send_buffer(int idx = 0) {
    static_assert(
        kDecoupled, "`send_buffer` is only available for the decoupled case");
    return reinterpret_cast<DType*>(send_ptr + num_bytes * idx);
  }

  __device__ __forceinline__ DType* recv_buffer(int idx = 0) {
    static_assert(
        kDecoupled, "`recv_buffer` is only available for the decoupled case");
    return reinterpret_cast<DType*>(recv_ptr + num_bytes * idx);
  }

  __device__ __forceinline__ DType* buffer(int idx = 0) {
    static_assert(
        !kDecoupled, "`buffer` is only available for the non-decoupled case");
    return reinterpret_cast<DType*>(send_ptr + num_bytes * idx);
  }
};

} // namespace comms::prims::moe_ep::kernels
