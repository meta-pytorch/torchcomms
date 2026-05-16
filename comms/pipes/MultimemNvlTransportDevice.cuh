// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comms/common/AtomicUtils.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {

namespace detail {

__device__ __forceinline__ uint8_t load_u8_unaligned(const char* src) {
  return *reinterpret_cast<const uint8_t*>(src);
}

__device__ __forceinline__ uint16_t load_u16_unaligned(const char* src) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(src);
  return static_cast<uint16_t>(bytes[0]) |
      (static_cast<uint16_t>(bytes[1]) << 8);
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const char* src) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(src);
  return static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) |
      (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ uint64_t load_u64_unaligned(const char* src) {
  return static_cast<uint64_t>(load_u32_unaligned(src)) |
      (static_cast<uint64_t>(load_u32_unaligned(src + 4)) << 32);
}

__device__ __forceinline__ uint4 load_u4_unaligned(const char* src) {
  return uint4{
      load_u32_unaligned(src),
      load_u32_unaligned(src + 4),
      load_u32_unaligned(src + 8),
      load_u32_unaligned(src + 12)};
}

// PTX exposes multimem stores for 4/8/16-byte widths. NCCL-style byte-tail
// paths use regular global byte/halfword stores on the multimem VA.
__device__ __forceinline__ void multimem_store_u8(uint8_t* dst, uint8_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("st.global.b8 [%0], %1;"
               :
               : "l"(dst), "r"(static_cast<uint32_t>(v))
               : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_u16(uint16_t* dst, uint16_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("st.global.b16 [%0], %1;" : : "l"(dst), "h"(v) : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_u32(uint32_t* dst, uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.global.b32 [%0], %1;"
               :
               : "l"(dst), "r"(v)
               : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_u64(uint64_t* dst, uint64_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.global.b64 [%0], %1;"
               :
               : "l"(dst), "l"(v)
               : "memory");
#else
  *dst = v;
#endif
}

__device__ __forceinline__ void multimem_store_release_sys_u64(
    uint64_t* dst,
    uint64_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(dst), "l"(v)
               : "memory");
#else
  comms::device::st_release_sys_global(dst, v);
#endif
}

__device__ __forceinline__ void multimem_red_release_sys_add_u64(
    uint64_t* dst,
    uint64_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;"
               :
               : "l"(dst), "l"(v)
               : "memory");
#else
  comms::device::atomic_fetch_add_release_sys_global(dst, v);
#endif
}

__device__ __forceinline__ void multimem_store_v4_u32(uint4* dst, uint4 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
#else
  *dst = v;
#endif
}

template <int kUnroll>
__device__ __forceinline__ void strided_multimem_store_aligned(
    ThreadGroup& group,
    uint4* dstVec,
    const uint4* srcVec,
    std::size_t vecCount) {
  static_assert(kUnroll > 0);
  constexpr std::size_t unroll = static_cast<std::size_t>(kUnroll);
  const std::size_t loopStride =
      static_cast<std::size_t>(group.group_size) * unroll;
  const std::size_t alignedVecCount = (vecCount / loopStride) * loopStride;

  for (std::size_t i = group.thread_id_in_group; i < alignedVecCount;
       i += loopStride) {
    uint4 vals[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      vals[j] = srcVec[i + static_cast<std::size_t>(j) * group.group_size];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const std::size_t offset =
          i + static_cast<std::size_t>(j) * group.group_size;
      multimem_store_v4_u32(dstVec + offset, vals[j]);
    }
  }

  for (std::size_t i = alignedVecCount + group.thread_id_in_group; i < vecCount;
       i += group.group_size) {
    multimem_store_v4_u32(dstVec + i, srcVec[i]);
  }
}

__device__ __forceinline__ void
multimem_store_bytes(char* dst, const char* src, std::size_t bytes) {
  while (bytes > 0) {
    const auto dstAddr = reinterpret_cast<uintptr_t>(dst);
    if (bytes >= sizeof(uint4) && (dstAddr & 0xF) == 0) {
      multimem_store_v4_u32(
          reinterpret_cast<uint4*>(dst), load_u4_unaligned(src));
      dst += sizeof(uint4);
      src += sizeof(uint4);
      bytes -= sizeof(uint4);
    } else if (bytes >= sizeof(uint64_t) && (dstAddr & 0x7) == 0) {
      multimem_store_u64(
          reinterpret_cast<uint64_t*>(dst), load_u64_unaligned(src));
      dst += sizeof(uint64_t);
      src += sizeof(uint64_t);
      bytes -= sizeof(uint64_t);
    } else if (bytes >= sizeof(uint32_t) && (dstAddr & 0x3) == 0) {
      multimem_store_u32(
          reinterpret_cast<uint32_t*>(dst), load_u32_unaligned(src));
      dst += sizeof(uint32_t);
      src += sizeof(uint32_t);
      bytes -= sizeof(uint32_t);
    } else if (bytes >= sizeof(uint16_t) && (dstAddr & 0x1) == 0) {
      multimem_store_u16(
          reinterpret_cast<uint16_t*>(dst), load_u16_unaligned(src));
      dst += sizeof(uint16_t);
      src += sizeof(uint16_t);
      bytes -= sizeof(uint16_t);
    } else {
      multimem_store_u8(
          reinterpret_cast<uint8_t*>(dst), load_u8_unaligned(src));
      ++dst;
      ++src;
      --bytes;
    }
  }
}

__device__ __forceinline__ void strided_multimem_store_unaligned(
    ThreadGroup& group,
    char* dst,
    const char* src,
    std::size_t bytes) {
  constexpr std::size_t kChunkBytes = sizeof(uint4);
  for (std::size_t offset =
           static_cast<std::size_t>(group.thread_id_in_group) * kChunkBytes;
       offset < bytes;
       offset += static_cast<std::size_t>(group.group_size) * kChunkBytes) {
    const std::size_t remaining = bytes - offset;
    multimem_store_bytes(
        dst + offset,
        src + offset,
        remaining < kChunkBytes ? remaining : kChunkBytes);
  }
}

} // namespace detail

/**
 * Device-side handle for one multicast staging window.
 *
 * `multimemData` and multimem signal spans are multicast VAs. Writes through
 * `store()` preserve multicast semantics; `localData` and local signal spans
 * are this rank's backing memory after multicast replication.
 */
struct MultimemNvlTransportDevice {
  char* localData{nullptr};
  char* multimemData{nullptr};
  DeviceSpan<SignalState> userLocalSignals{};
  DeviceSpan<SignalState> userMultimemSignals{};
  DeviceSpan<SignalState> internalLocalSignals{};
  DeviceSpan<SignalState> internalMultimemSignals{};
  std::size_t dataBufferSize{0};

  __device__ __forceinline__ char* local_data_at(std::size_t offset) const {
    return localData + offset;
  }

  __device__ __forceinline__ char* multimem_data_at(std::size_t offset) const {
    return multimemData + offset;
  }

  template <int kUnroll = 1>
  __device__ __forceinline__ void store(
      ThreadGroup& group,
      std::size_t dst_offset,
      const void* src,
      std::size_t bytes) const {
    static_assert(kUnroll > 0);
    auto* dst = multimem_data_at(dst_offset);
    const auto srcAddr = reinterpret_cast<uintptr_t>(src);
    const auto dstAddr = reinterpret_cast<uintptr_t>(dst);

    if (((srcAddr | dstAddr) & 0xF) == 0) {
      const std::size_t alignedBytes = bytes & ~static_cast<std::size_t>(0xF);
      auto* srcVec = reinterpret_cast<const uint4*>(src);
      auto* dstVec = reinterpret_cast<uint4*>(dst);
      const std::size_t vecCount = alignedBytes / sizeof(uint4);
      detail::strided_multimem_store_aligned<kUnroll>(
          group, dstVec, srcVec, vecCount);
      if (alignedBytes != bytes) {
        detail::strided_multimem_store_unaligned(
            group,
            dst + alignedBytes,
            static_cast<const char*>(src) + alignedBytes,
            bytes - alignedBytes);
      }
    } else {
      detail::strided_multimem_store_unaligned(
          group, dst, static_cast<const char*>(src), bytes);
    }
  }

  __device__ __forceinline__ void signal(
      ThreadGroup& group,
      uint64_t signal_id,
      SignalOp op,
      uint64_t value) const {
    signal_at(group, user_multimem_signal_ptr(signal_id), op, value);
  }

  __device__ __forceinline__ uint64_t read_signal(uint64_t signal_id) const {
    return user_local_signal_at(signal_id)->load();
  }

  __device__ __forceinline__ void wait_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    user_local_signal_at(signal_id)->wait_until(group, op, expected, timeout);
  }

  __device__ __forceinline__ void signal_internal(
      ThreadGroup& group,
      uint64_t signal_id,
      SignalOp op,
      uint64_t value) const {
    signal_at(group, internal_multimem_signal_ptr(signal_id), op, value);
  }

  __device__ __forceinline__ uint64_t
  read_internal_signal(uint64_t signal_id) const {
    return internal_local_signal_at(signal_id)->load();
  }

  __device__ __forceinline__ void wait_internal_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    internal_local_signal_at(signal_id)->wait_until(
        group, op, expected, timeout);
  }

  __device__ __forceinline__ void multicast_signal_set(
      ThreadGroup& group,
      uint32_t signalIndex,
      uint64_t value) const {
    signal(group, signalIndex, SignalOp::SIGNAL_SET, value);
  }

  __device__ __forceinline__ void wait_local_signal(
      ThreadGroup& group,
      uint32_t signalIndex,
      CmpOp op,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    wait_signal_until(group, signalIndex, op, expected, timeout);
  }

 private:
  __device__ __forceinline__ SignalState* signal_ptr_at(
      DeviceSpan<SignalState> signals,
      uint64_t signal_id,
      const char* kind) const {
#if defined(__CUDA_ARCH__)
    if (signal_id >= signals.size()) {
      printf(
          "MultimemNvlTransportDevice: %s signal_id=%llu out of range "
          "(count=%u)\n",
          kind,
          static_cast<unsigned long long>(signal_id),
          static_cast<unsigned>(signals.size()));
      __trap();
    }
#endif
    return signals.data() + signal_id;
  }

  __device__ __forceinline__ SignalState* user_local_signal_at(
      uint64_t signal_id) const {
    return signal_ptr_at(userLocalSignals, signal_id, "user local");
  }

  __device__ __forceinline__ SignalState* user_multimem_signal_ptr(
      uint64_t signal_id) const {
    return signal_ptr_at(userMultimemSignals, signal_id, "user multimem");
  }

  __device__ __forceinline__ SignalState* internal_local_signal_at(
      uint64_t signal_id) const {
    return signal_ptr_at(internalLocalSignals, signal_id, "internal local");
  }

  __device__ __forceinline__ SignalState* internal_multimem_signal_ptr(
      uint64_t signal_id) const {
    return signal_ptr_at(
        internalMultimemSignals, signal_id, "internal multimem");
  }

  __device__ __forceinline__ void signal_at(
      ThreadGroup& group,
      SignalState* signal,
      SignalOp op,
      uint64_t value) const {
    comms::device::fence_acq_rel_sys();
    group.sync();
    if (group.is_leader()) {
      switch (op) {
        case SignalOp::SIGNAL_SET:
          detail::multimem_store_release_sys_u64(&signal->signal_, value);
          break;
        case SignalOp::SIGNAL_ADD:
          detail::multimem_red_release_sys_add_u64(&signal->signal_, value);
          break;
      }
    }
  }
};

} // namespace comms::pipes
