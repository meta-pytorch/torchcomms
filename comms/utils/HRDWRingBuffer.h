// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h> // @manual

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda/atomic> // @manual
#endif

// needed for US_TICK_TIMESTAMP_SHIFT
#include "comms/utils/GpuClockCalibration.h"

#include <algorithm>
#if __cplusplus >= 202002L
#include <bit>
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// EMPTY epoch sentinel: a freshly initialized slot has epoch 0. Real
// writes encode epoch = uint32_t(slot >> shift) + 1, so 0 is unused.
#define HRDW_RINGBUFFER_SLOT_EMPTY 0u

namespace meta::comms::colltrace {

// ==========================================================================
// HRDWRingBuffer — Host-Read Device-Write ring buffer for GPU-to-CPU
// event transfer with lock-free, torn-read-safe polling.
//
// OVERVIEW
//   GPU kernels claim a slot via atomicAdd on a mapped writeIndex, then
//   emit a single 128-bit system-scope atomic exchange of {timestamp,
//   epoch, data}. The CPU polls each slot with a single 16B atomic load,
//   which the writer's slot-atomic publication guarantees observes either
//   the fully-old or fully-new state — no torn reads, no retries.
//
// USAGE
//   1. Create a ring buffer. DataT must fit in 8 bytes (the entry packs
//      into exactly 16 bytes for atomic 128b stores). For non-trivially-
//      destructible DataT, the destructor IS invoked on the displaced
//      occupant when a slot is overwritten — the writer reclaims the
//      previous bytes via atom.exch.b128 and runs ~DataT() on them, so
//      refcounted DataT types stay balanced across overwrites.
//
//        HRDWRingBuffer<MyEvent> buf(4096);
//
//   2. Write events. Each write claims a slot, packs timestamp + epoch +
//      data into 16 bytes, and emits a single 128-bit store:
//
//        buf.write(stream, myEvent);
//
//   3. On the CPU, create a reader and poll periodically. Convert the
//      packed timestamp via GlobaltimerCalibration::toWallClock(uint32_t):
//
//        HRDWRingBufferReader<MyEvent> reader(buf);
//        reader.poll([&](const auto& entry, uint64_t slot) {
//          const MyEvent& evt = entry.data;
//          auto wall = cal.toWallClock(entry.timestamp);
//        });
//
// LAYOUT (16 bytes, 16-byte aligned)
//   - timestamp: uint32_t — globaltimer_ns >> 10 (~1024ns ticks). Wraps
//                every 2^42 ns ≈ 73 min; the reader reconstructs full
//                precision against system_clock::now() via
//                GlobaltimerCalibration::toWallClock(uint32_t), so callers
//                must poll often enough that no slot lingers past the
//                ~73 min wrap window. Older entries alias to a more
//                recent time and cannot be recovered from 32 bits alone.
//   - epoch:     uint32_t — uint32_t(slot >> log2(size_)) + 1.
//                Zero is the EMPTY sentinel; the +1 keeps slot 0's epoch
//                non-zero so a never-written slot is always distinguishable
//                from any valid write.
//   - data:      DataT, ≤ 8 bytes (padded to 16 if smaller).
//
// MEMORY ORDERING
//   - The writer issues a single 128b atomic exchange at system scope via
//     cuda::atomic_ref<unsigned __int128, thread_scope_system>::exchange
//     (libcu++). Lowers to atom.exch.b128 / atom.cas.b128 on sm_90+,
//     which is uncontended in practice because the preceding writeIndex
//     atomicAdd already serialized us against other writers. The
//     exchange is slot-atomic in hardware — the reader never sees a torn
//     entry from a concurrent writer.
//   - On unsupported architectures (HIP, pre-Hopper CUDA),
//     hrdwRingBufferWrite compiles to a no-op so fat binaries link
//     cleanly; only sm_90+ cubins carry the real trace path.
//   - The displaced bytes are reinterpreted as HRDWEntry<DataT> and
//     ~DataT() is invoked on the previous occupant when the slot was
//     non-empty, so refcounted DataT types stay balanced across slot
//     overwrites.
//   - Release ordering w.r.t. the writeIndex atomicAdd is unnecessary:
//     the reader's per-slot epoch check + writeIndex bound is
//     self-correcting regardless of the order in which the slot write
//     and writeIndex bump become visible to the host.
//   - Reader pattern: a single 16B atomic relaxed-load of ring[idx]
//     (lowers to LDP on aarch64 / LSE2, cmpxchg16b on x86_64). Because
//     the writer's atom.exch.b128 publishes the slot atomically, the
//     host load sees either the fully-old or fully-new state. The reader
//     then compares the loaded epoch against `(slot >> log2(size_)) + 1`
//     to classify the entry as kSuccess / kOverwritten / kNotReady.
//
// RING SIZING
//   The size is rounded up to a power of 2. Larger rings reduce loss
//   from lapping; overwritten entries are reported as entriesLost.
// ==========================================================================

// 16-byte ring buffer entry. Layout is fixed for atomic 128b stores.
template <typename DataT>
struct alignas(16) HRDWEntry {
  uint32_t timestamp;
  uint32_t epoch;
  DataT data;
};

// Device-side inline write into a ring buffer. Performs a single 128b
// atomic exchange of {timestamp, epoch, data} after claiming a slot via
// atomicAdd. Compiled to a no-op on unsupported architectures (HIP, and
// pre-Hopper CUDA) so fat binaries build cleanly — only sm_90+ cubins
// carry the real trace path.
#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename DataT>
__device__ __forceinline__ void hrdwRingBufferWrite(
    HRDWEntry<DataT>* ring,
    uint64_t* writeIndex,
    uint32_t mask,
    uint32_t shift,
    DataT data) {
  static_assert(
      sizeof(HRDWEntry<DataT>) == 16,
      "HRDWEntry must be exactly 16 bytes (DataT must be <= 8 bytes)");
  static_assert(
      alignof(HRDWEntry<DataT>) == 16, "HRDWEntry must be 16-byte aligned");

#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
  // Unsupported arch — atom.exch.b128 / atom.cas.b128 require sm_90+,
  // and HIP doesn't yet have a wired-up 128b atomic system-scope store.
  // Log and trap so the kernel fails loudly — the CUDA runtime surfaces
  // this as an error on the next API call, rather than silently dropping
  // events.
  (void)ring;
  (void)writeIndex;
  (void)mask;
  (void)shift;
  (void)data;
  printf(
      "[HRDWRingBuffer] hrdwRingBufferWrite unsupported on this GPU "
      "(requires sm_90+)\n");
#if defined(__HIPCC__)
  abort();
#else
  __trap();
#endif
#else
  // Relaxed system-scope fetch-add: the only invariant we need from this
  // is "every writer gets a unique slot number"; ordering with the
  // subsequent slot publication is unnecessary (the reader's epoch check
  // is self-correcting if the writeIndex bump and the slot bytes become
  // host-visible in either order).
  uint64_t slot =
      cuda::atomic_ref<uint64_t, cuda::thread_scope_system>(*writeIndex)
          .fetch_add(1ULL, cuda::memory_order_relaxed);
  uint64_t idx = slot & mask;

  HRDWEntry<DataT> packed{};
  packed.timestamp =
      static_cast<uint32_t>(readGlobaltimer() >> US_TICK_TIMESTAMP_SHIFT);
  packed.epoch = static_cast<uint32_t>(slot >> shift) + 1u;
  packed.data = data;

  // Single 128b atomic exchange at system scope via raw PTX. We use
  // atom.exch.relaxed.sys.b128 for ALL DataT (not st.relaxed.sys.b128
  // even for trivially-destructible types), because the RMW form is the
  // only one hardware guarantees as atomic across the GPU↔host boundary
  // on systems without negotiated PCIe AtomicOp support — st.b128 can
  // decompose into two 64b transactions, and the host's 16B atomic load
  // would observe a torn write whose halves landed in a stale order.
  // (Going through libcu++'s cuda::atomic_ref<T> for 16B T also
  // miscompiled on sm_100a in practice, so we emit the instruction
  // directly.)
  //
  // For non-trivially-destructible DataT we run ~DataT() on the
  // displaced occupant; for trivially-destructible DataT the compiler
  // DCEs the prev_lo/prev_hi outputs.
  uint64_t packed_lo, packed_hi;
  __builtin_memcpy(&packed_lo, &packed, sizeof(packed_lo));
  __builtin_memcpy(
      &packed_hi,
      reinterpret_cast<const char*>(&packed) + sizeof(packed_lo),
      sizeof(packed_hi));

  [[maybe_unused]] uint64_t prev_lo, prev_hi;
  asm volatile(
      "{ .reg .b128 _src, _dst;\n\t"
      "  mov.b128 _src, {%2, %3};\n\t"
      "  atom.exch.relaxed.sys.b128 _dst, [%4], _src;\n\t"
      "  mov.b128 {%0, %1}, _dst; }"
      : "=l"(prev_lo), "=l"(prev_hi)
      : "l"(packed_lo), "l"(packed_hi), "l"(&ring[idx])
      : "memory");

  if constexpr (!std::is_trivially_destructible_v<DataT>) {
    // Hold the displaced bytes in a raw char buffer (not a
    // HRDWEntry<DataT>) so the compiler never auto-destructs them at
    // scope exit — that would double-call ~DataT() after our explicit
    // destruction below.
    alignas(HRDWEntry<DataT>) char prev_buf[sizeof(HRDWEntry<DataT>)];
    __builtin_memcpy(prev_buf, &prev_lo, sizeof(prev_lo));
    __builtin_memcpy(prev_buf + sizeof(prev_lo), &prev_hi, sizeof(prev_hi));
    auto* prev = reinterpret_cast<HRDWEntry<DataT>*>(prev_buf);
    if (prev->epoch != HRDW_RINGBUFFER_SLOT_EMPTY) {
      prev->data.~DataT();
    }
  }
#endif
}
#endif

// Lightweight, trivially-copyable handle for writing into an HRDWRingBuffer
// from device code. Pass by value to GPU kernels. See
// HRDWRingBufferDeviceHandle.cuh for full documentation and usage examples.
template <typename DataT>
struct HRDWRingBufferDeviceHandle {
  HRDWEntry<DataT>* ring;
  uint64_t* writeIndex;
  uint32_t mask;
  uint32_t shift;

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ void write(DataT data) {
    hrdwRingBufferWrite(ring, writeIndex, mask, shift, data);
  }
#endif
};

// Templated kernel launch for host-side write(). The template definition
// is available in .cu compilation units; .cc files link against explicit
// instantiations provided by consumers (see
// comms/utils/colltrace/HRDWRingBufferInstantiations.cu for examples).
#if defined(__CUDACC__) || defined(__HIPCC__)
namespace detail {
template <typename DataT>
__global__ void ringBufferWriteKernel(
    HRDWEntry<DataT>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data) {
  hrdwRingBufferWrite(ring, writeIdx, mask, shift, data);
}
} // namespace detail

template <typename DataT>
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    HRDWEntry<DataT>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  detail::ringBufferWriteKernel<<<1, 1, 0, stream>>>(
      ring, writeIdx, mask, shift, data);
  return cudaGetLastError();
}
#else
// Declaration only — .cc files link against explicit instantiations.
template <typename DataT>
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    HRDWEntry<DataT>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    uint32_t shift,
    DataT data);
#endif

// Host-Read Device-Write (HRDW) ring buffer. Device kernels atomically claim
// slots via writeIndex, then emit a single 128b atomic exchange of
// {timestamp, epoch, data}. The host reader uses writeIndex to bound scans
// and validates each entry via the per-entry epoch field.
//
// DataT must fit in 8 bytes so the entry packs into exactly 16 bytes for
// atomic 128b writes. For non-trivially-destructible DataT, the writer
// runs ~DataT() on the displaced occupant when a slot is overwritten, so
// refcounted DataT types stay balanced across overwrites.
//
// Owns the ring buffer (mapped pinned) and write index (mapped pinned).
// Move-only — CUDA resources are not copyable.
template <typename DataT>
class HRDWRingBuffer {
  static_assert(
      sizeof(HRDWEntry<DataT>) == 16,
      "HRDWEntry must be exactly 16 bytes (DataT must be <= 8 bytes)");
  static_assert(
      alignof(HRDWEntry<DataT>) == 16,
      "HRDWEntry must be 16-byte aligned");

 public:
  using Entry = HRDWEntry<DataT>;

  explicit HRDWRingBuffer(uint32_t size) {
    // Round up to next power of 2 if needed.
    if (size == 0) {
      size = 1;
    }
    if (size > (1u << 31)) {
      fprintf(
          stderr,
          "HRDWRingBuffer: size %u exceeds maximum power-of-2 size\n",
          size);
      return; // valid() will return false
    }
    if ((size & (size - 1)) != 0) {
#if __cplusplus >= 202002L
      uint32_t nextPowerOf2 = 1U << (std::bit_width(size - 1));
#else
      // C++17 fallback for std::bit_width
      uint32_t nextPowerOf2 = 1;
      while (nextPowerOf2 < size) {
        nextPowerOf2 <<= 1;
      }
#endif
      fprintf(
          stderr,
          "HRDWRingBuffer: size %u is not a power of 2, rounding up to %u\n",
          size,
          nextPowerOf2);
      size = nextPowerOf2;
    }

    // x & mask == x % size
    // but is just a single bitwise op
    size_ = size;
    mask_ = size - 1;
#if __cplusplus >= 202002L
    shift_ = static_cast<uint32_t>(std::countr_zero(size));
#else
    shift_ = 0;
    for (uint32_t s = size; s > 1; s >>= 1) {
      ++shift_;
    }
#endif

    allocatePinnedMapped();
  }

  ~HRDWRingBuffer() {
    destructLiveEntries();
    if (ring_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      (void)cudaFreeHost(ring_);
    }
    if (writeIndex_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      (void)cudaFreeHost(writeIndex_);
    }
  }

  // Move-only.
  HRDWRingBuffer(const HRDWRingBuffer&) = delete;
  HRDWRingBuffer& operator=(const HRDWRingBuffer&) = delete;

  HRDWRingBuffer(HRDWRingBuffer&& other) noexcept
      : ring_(std::exchange(other.ring_, nullptr)),
        writeIndex_(std::exchange(other.writeIndex_, nullptr)),
        size_(other.size_),
        mask_(other.mask_),
        shift_(other.shift_) {}

  HRDWRingBuffer& operator=(HRDWRingBuffer&& other) noexcept {
    if (this != &other) {
      destructLiveEntries();
      if (ring_) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        (void)cudaFreeHost(ring_);
      }
      if (writeIndex_) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        (void)cudaFreeHost(writeIndex_);
      }
      ring_ = std::exchange(other.ring_, nullptr);
      writeIndex_ = std::exchange(other.writeIndex_, nullptr);
      size_ = other.size_;
      mask_ = other.mask_;
      shift_ = other.shift_;
    }
    return *this;
  }

  uint32_t size() const {
    return size_;
  }

  // Returns true if all allocations succeeded.
  bool valid() const {
    return ring_ != nullptr && writeIndex_ != nullptr;
  }

  // Launch a single-thread kernel that atomically claims a slot, packs
  // timestamp + epoch + data, and emits a single 128b store.
  cudaError_t write(cudaStream_t stream, DataT data) {
    assert(valid());
    return launchRingBufferWrite<DataT>(
        stream, ring_, writeIndex_, mask_, shift_, data);
  }

  // Return a lightweight, trivially-copyable handle that can be passed by
  // value to GPU kernels for inline device-side writes. See
  // HRDWRingBufferDeviceHandle.cuh for usage.
  HRDWRingBufferDeviceHandle<DataT> deviceHandle() const {
    assert(valid());
    return {ring_, writeIndex_, mask_, shift_};
  }

 private:
  template <typename>
  friend class HRDWRingBufferReader;
  template <typename>
  friend class HRDWRingBufferTestAccessor;

  void destructLiveEntries() {
    if constexpr (!std::is_trivially_destructible_v<DataT>) {
      if (ring_ && writeIndex_) {
        uint64_t written = *writeIndex_;
        uint64_t count = std::min(written, static_cast<uint64_t>(size_));
        for (uint64_t i = 0; i < count; ++i) {
          ring_[i].data.~DataT();
        }
      }
    }
  }

  void allocatePinnedMapped() {
    void* ringPtr = nullptr;
    auto ringErr =
        cudaHostAlloc(&ringPtr, sizeof(Entry) * size_, cudaHostAllocDefault);
    if (ringErr == cudaSuccess) {
      ring_ = static_cast<Entry*>(ringPtr);
      // Stamp HRDW_RINGBUFFER_SLOT_EMPTY (0) into every slot's epoch.
      // The other bytes are raw storage from cudaHostAlloc; the first
      // writer overwrites the full 16B slot via atom.exch.b128, and
      // skips ~DataT() on the displaced bytes because epoch==0 means
      // the slot had no prior occupant.
      for (uint32_t i = 0; i < size_; ++i) {
        ring_[i].epoch = HRDW_RINGBUFFER_SLOT_EMPTY;
      }
    } else {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate ring buffer: %s\n",
          cudaGetErrorString(ringErr));
      return;
    }
    void* writeIdxPtr = nullptr;
    auto idxErr =
        cudaHostAlloc(&writeIdxPtr, sizeof(uint64_t), cudaHostAllocDefault);
    if (idxErr == cudaSuccess) {
      writeIndex_ = static_cast<uint64_t*>(writeIdxPtr);
      *writeIndex_ = 0;
    } else {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate write index: %s\n",
          cudaGetErrorString(idxErr));
    }
  }

  Entry* ring_{nullptr};
  uint64_t* writeIndex_{nullptr};
  uint32_t size_{0};
  // size_ - 1. Cached so `slot & mask_` is one bitwise op per write
  // instead of `slot % size_`.
  uint32_t mask_{0};
  // log2(size_). Cached so the per-slot epoch is computed as
  // `(slot >> shift_) + 1` — a shift instead of a divide.
  uint32_t shift_{0};
};

} // namespace meta::comms::colltrace
