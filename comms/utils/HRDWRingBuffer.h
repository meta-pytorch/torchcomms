// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h> // @manual

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "comms/utils/GpuClockCalibration.h"
#endif

#include <algorithm>
#if __cplusplus >= 202002L
#include <bit>
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HRDW_RINGBUFFER_SLOT_EMPTY UINT64_MAX

namespace meta::comms::colltrace {

namespace detail {
template <std::size_t kAlign>
constexpr std::size_t alignTo(std::size_t bytes) {
  return (bytes + kAlign - 1) & ~(kAlign - 1);
}

template <std::size_t kAlign, typename T>
static inline __attribute__((__always_inline__)) std::size_t allocAligned(
    T*& ptr,
    std::size_t num = 1) {
  auto bytes = detail::alignTo<kAlign>(sizeof(T) * num);
  ptr = static_cast<T*>(std::aligned_alloc(kAlign, bytes));
  return ptr ? bytes : 0;
}
} // namespace detail

// Memory allocation strategy for the ring buffer.
enum class HRDWMemoryStrategy {
  // cudaHostAlloc pinned mapped memory — works everywhere.
  PinnedMapped,
  // Plain malloc + cudaMemAdvise(SetAccessedBy) — on Grace Hopper (GB200),
  // NVLink-C2C makes this as fast as device memory. Falls back to
  // PinnedMapped on non-ATS systems.
  AtsPageable,
  // Automatically select the best strategy for the current platform.
  Auto,
};

// Returns true if the current GPU supports hardware-backed ATS (Address
// Translation Services) for pageable host memory access. True on Grace
// Hopper (GB200) with NVLink-C2C.
//
// Two attributes must both be set: pageable memory access must be
// available, AND it must be backed by hardware page tables rather than
// software emulation. The latter distinguishes true ATS (NVLink-C2C) from
// HMM-style software-managed access, which lacks the coherence guarantees
// we rely on for the device-scope fence in hrdwRingBufferWrite.
inline bool isAtsSupported() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }
  int pageableMemoryAccess = 0;
  if (cudaDeviceGetAttribute(
          &pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, device) !=
      cudaSuccess) {
    return false;
  }
  int usesHostPageTables = 0;
  if (cudaDeviceGetAttribute(
          &usesHostPageTables,
          cudaDevAttrPageableMemoryAccessUsesHostPageTables,
          device) != cudaSuccess) {
    return false;
  }
  return pageableMemoryAccess != 0 && usesHostPageTables != 0;
}

// ==========================================================================
// HRDWRingBuffer — Host-Read Device-Write ring buffer for GPU-to-CPU
// event transfer with lock-free, torn-read-safe polling.
//
// OVERVIEW
//   GPU kernels claim ring buffer slots via atomicAdd on a mapped write
//   index, write a globaltimer timestamp and a user data value, issue
//   __threadfence_system(), then stamp sequence = slot. A CPU thread
//   polls the ring using a seqlock protocol on the per-entry sequence.
//
// USAGE
//   1. Create a ring buffer:
//
//        HRDWRingBuffer<MyEvent> buf(4096);
//
//   2. Write events. Each write claims a slot, records globaltimer()
//      as timestamp_ns, copies your data by value, and stamps sequence:
//
//        buf.write(stream, myEvent);
//
//   3. On the CPU, create a reader and poll periodically:
//
//        HRDWRingBufferReader<MyEvent> reader(buf);
//        reader.poll([](const auto& entry, uint64_t slot) {
//          const MyEvent& evt = entry.data;
//          auto ts = entry.timestamp_ns;
//        });
//
// MEMORY ORDERING
//   - The write kernel uses __threadfence_system() between writing data
//     fields and stamping sequence. This ensures the CPU sees consistent
//     data when it observes a valid sequence value.
//   - The reader uses a seqlock protocol: read sequence (acquire), copy
//     entry, re-read sequence. If the sequence changed, the entry was
//     overwritten during the copy and is discarded.
//   - The reader uses writeIndex (mapped memory) to bound the scan range
//     and detect lapping. Per-entry sequence validates each entry.
//
// RING SIZING
//   Larger rings reduce data loss from lapping. If the ring is smaller
//   than the number of in-flight events, overwritten entries are counted
//   as entriesLost... This won't result in data corruption, just data loss.
//   The size is automatically rounded up to the next power of 2.
//
// TIMEOUT
//   If the writer continuously outpaces the reader, the reader will keep
//   jumping to the tail and processing overwritten entries indefinitely. Pass a
//   timeout to poll() to bound how long the reader spends catching up
//   before returning partial results.
//
// ENTRY LIFECYCLE
//   slot claimed (atomicAdd on writeIndex) → timestamp_ns + data written
//   → __threadfence_system() → sequence = slot → CPU can read
// ==========================================================================

// Entry stored in the ring buffer. The GPU kernel writes timestamp_ns and
// data (copied by value), issues __threadfence_system(), then stamps
// sequence = slot.
//
template <typename DataT>
struct HRDWEntry {
  uint64_t timestamp_ns;
  uint64_t sequence;
  DataT data;
};

// Device-side inline write into a ring buffer. Shared by
// ringBufferWriteKernel and HRDWRingBufferDeviceHandle::write().
#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename DataT>
__device__ __forceinline__ void hrdwRingBufferWrite(
    HRDWEntry<DataT>* ring,
    uint64_t* writeIndex,
    uint32_t mask,
    DataT data,
    HRDWMemoryStrategy strategy) {
  uint64_t slot =
      atomicAdd(reinterpret_cast<unsigned long long*>(writeIndex), 1ULL);
  uint64_t idx = slot & mask;
  ring[idx].timestamp_ns = readGlobaltimer();
  // Only destruct if the slot was previously written (sequence != EMPTY).
  // First write to a slot hits uninitialized memory — skip destructor.
  if constexpr (!std::is_trivially_destructible_v<DataT>) {
    if (ring[idx].sequence != HRDW_RINGBUFFER_SLOT_EMPTY) {
      ring[idx].data.~DataT();
    }
  }
  if constexpr (std::is_move_constructible_v<DataT>) {
    new (&ring[idx].data) DataT(static_cast<DataT&&>(data));
  } else {
    new (&ring[idx].data) DataT(data);
  }
  // ATS (NVLink-C2C) provides hardware coherence between GPU and CPU,
  // so a device-scope fence is sufficient. Non-coherent interconnects
  // (PCIe) require a system-scope fence.
  if (strategy == HRDWMemoryStrategy::AtsPageable) {
    __threadfence();
  } else {
    __threadfence_system();
  }
  ring[idx].sequence = slot;
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
  HRDWMemoryStrategy strategy;

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ void write(DataT data) {
    hrdwRingBufferWrite(ring, writeIndex, mask, data, strategy);
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
    DataT data,
    HRDWMemoryStrategy strategy) {
  hrdwRingBufferWrite(ring, writeIdx, mask, data, strategy);
}
} // namespace detail

template <typename DataT>
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    HRDWEntry<DataT>* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    DataT data,
    HRDWMemoryStrategy strategy) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  detail::ringBufferWriteKernel<<<1, 1, 0, stream>>>(
      ring, writeIdx, mask, data, strategy);
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
    DataT data,
    HRDWMemoryStrategy strategy);
#endif

// Host-Read Device-Write (HRDW) ring buffer. Device kernels atomically claim
// slots via writeIndex, write data with __threadfence_system(), then stamp
// the per-entry sequence. The host reader uses writeIndex to bound scans
// and per-entry sequence for seqlock validation.
//
// DataT is the user-defined type stored by value in each entry.
// Access it via entry.data (typed as DataT, no cast needed).
//
// Owns the ring buffer (mapped pinned) and write index (device memory).
// Move-only — CUDA resources are not copyable.
template <typename DataT>
class HRDWRingBuffer {
 public:
  using Entry = HRDWEntry<DataT>;

  explicit HRDWRingBuffer(
      uint32_t size,
      HRDWMemoryStrategy strategy = HRDWMemoryStrategy::Auto) {
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

    // Resolve Auto strategy.
    if (strategy == HRDWMemoryStrategy::Auto) {
      strategy = isAtsSupported() ? HRDWMemoryStrategy::AtsPageable
                                  : HRDWMemoryStrategy::PinnedMapped;
    }
    // Fallback if ATS requested but not supported.
    if (strategy == HRDWMemoryStrategy::AtsPageable && !isAtsSupported()) {
      strategy = HRDWMemoryStrategy::PinnedMapped;
    }
    strategy_ = strategy;

    // x & mask == x % size
    // but is just a single bitwise op
    size_ = size;
    mask_ = size - 1;

    if (strategy_ == HRDWMemoryStrategy::AtsPageable) {
      allocateAtsPageable();
    } else {
      allocatePinnedMapped();
    }
  }

  ~HRDWRingBuffer() {
    destructLiveEntries();
    if (strategy_ == HRDWMemoryStrategy::AtsPageable) {
      std::free(ring_);
      std::free(writeIndex_);
    } else {
      if (ring_) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        (void)cudaFreeHost(ring_);
      }
      if (writeIndex_) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        (void)cudaFreeHost(writeIndex_);
      }
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
        strategy_(other.strategy_) {}

  HRDWRingBuffer& operator=(HRDWRingBuffer&& other) noexcept {
    if (this != &other) {
      destructLiveEntries();
      // Free current resources using the current strategy.
      if (strategy_ == HRDWMemoryStrategy::AtsPageable) {
        std::free(ring_);
        std::free(writeIndex_);
      } else {
        if (ring_) {
          // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
          (void)cudaFreeHost(ring_);
        }
        if (writeIndex_) {
          // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
          (void)cudaFreeHost(writeIndex_);
        }
      }
      ring_ = std::exchange(other.ring_, nullptr);
      writeIndex_ = std::exchange(other.writeIndex_, nullptr);
      size_ = other.size_;
      mask_ = other.mask_;
      strategy_ = other.strategy_;
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

  // Launch a single-thread kernel that atomically claims a slot, records
  // globaltimer() as timestamp_ns, copies data by value, and stamps sequence.
  cudaError_t write(cudaStream_t stream, DataT data) {
    assert(valid());
    return launchRingBufferWrite<DataT>(
        stream, ring_, writeIndex_, mask_, data, strategy_);
  }

  // Return a lightweight, trivially-copyable handle that can be passed by
  // value to GPU kernels for inline device-side writes. See
  // HRDWRingBufferDeviceHandle.cuh for usage.
  HRDWRingBufferDeviceHandle<DataT> deviceHandle() const {
    assert(valid());
    return {ring_, writeIndex_, mask_, strategy_};
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

  void allocateAtsPageable() {
    int device = 0;
    cudaGetDevice(&device);

    constexpr std::size_t kAlign = 128;
    {
      auto bytesAllocated = detail::allocAligned<kAlign, Entry>(ring_, size_);
      if (!bytesAllocated) {
        fprintf(stderr, "HRDWRingBuffer: allocAligned failed for ring\n");
        return;
      }
      initRing();
#if CUDART_VERSION >= 13000
      cudaMemLocation location{cudaMemLocationTypeDevice, device};
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaMemAdvise(
          ring_, bytesAllocated, cudaMemAdviseSetAccessedBy, location);
#else
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaMemAdvise(ring_, bytesAllocated, cudaMemAdviseSetAccessedBy, device);
#endif
    }

    {
      auto bytesAllocated = detail::allocAligned<kAlign, uint64_t>(writeIndex_);
      if (!bytesAllocated) {
        fprintf(stderr, "HRDWRingBuffer: allocAligned failed for writeIndex\n");
        return;
      }
      *writeIndex_ = 0;
#if CUDART_VERSION >= 13000
      cudaMemLocation location{cudaMemLocationTypeDevice, device};
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaMemAdvise(
          writeIndex_, bytesAllocated, cudaMemAdviseSetAccessedBy, location);
#else
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaMemAdvise(
          writeIndex_, bytesAllocated, cudaMemAdviseSetAccessedBy, device);
#endif
    }
  }

  void allocatePinnedMapped() {
    void* ringPtr = nullptr;
    auto ringErr =
        cudaHostAlloc(&ringPtr, sizeof(Entry) * size_, cudaHostAllocDefault);
    if (ringErr == cudaSuccess) {
      ring_ = static_cast<Entry*>(ringPtr);
      initRing();
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

  void initRing() {
    // Only initialize the metadata fields (timestamp_ns, sequence) — leave
    // the DataT storage uninitialized. No Entry objects have been constructed
    // yet (the memory comes from cudaHostAlloc, not new[]). The GPU kernel
    // checks sequence != SLOT_EMPTY before calling ~DataT().
    for (uint32_t i = 0; i < size_; ++i) {
      ring_[i].timestamp_ns = 0;
      ring_[i].sequence = HRDW_RINGBUFFER_SLOT_EMPTY;
    }
  }

  Entry* ring_{nullptr};
  uint64_t* writeIndex_{nullptr};
  uint32_t size_{0};
  uint32_t mask_{0};
  HRDWMemoryStrategy strategy_{HRDWMemoryStrategy::PinnedMapped};
};

} // namespace meta::comms::colltrace
