// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

#define HRDW_RINGBUFFER_SLOT_EMPTY UINT64_MAX

namespace meta::comms::colltrace {

// ==========================================================================
// HRDWRingBuffer — Host-Read Device-Write ring buffer for GPU-to-CPU
// event transfer with lock-free, torn-read-safe polling.
//
// OVERVIEW
//   GPU kernels claim ring buffer slots via atomicAdd on a mapped write
//   index, write a globaltimer timestamp and a user data pointer, issue
//   __threadfence_system(), then stamp sequence = slot. A CPU thread
//   polls the ring using a seqlock protocol on the per-entry sequence.
//
// USAGE
//   1. Create a ring buffer:
//
//        HRDWRingBuffer<MyContext*> buf(4096);
//
//   2. Write events. Each write claims a slot, records globaltimer()
//      as timestamp_ns, stores your data pointer, and stamps sequence:
//
//        buf.write(stream, myContextPtr);
//
//   3. On the CPU, create a reader and poll periodically:
//
//        HRDWRingBufferReader<MyContext*> reader(buf);
//        reader.poll([](const auto& entry, uint64_t slot) {
//          MyContext* ctx = entry.data_as();
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
// data, issues __threadfence_system(), then stamps sequence = slot.
struct HRDWEntry {
  uint64_t timestamp_ns;
  uint64_t sequence;
  void* data;
};

// Forward declaration for the free function used by the member function.
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data);

// Host-Read Device-Write (HRDW) ring buffer. Device kernels atomically claim
// slots via writeIndex, write data with __threadfence_system(), then stamp
// the per-entry sequence. The host reader uses writeIndex to bound scans
// and per-entry sequence for seqlock validation.
//
// DataT is the user-defined type stored in each entry's data pointer.
// Access it via entry.data_as() which returns DataT& without requiring
// a template argument.
//
// Owns the ring buffer (mapped pinned) and write index (device memory).
// Move-only — CUDA resources are not copyable.
template <typename DataT>
class HRDWRingBuffer {
 public:
  // Typed entry — extends HRDWEntry with a data_as() accessor that
  // returns DataT& directly.
  struct Entry : HRDWEntry {
    __attribute__((always_inline)) DataT& data_as() {
      return *static_cast<DataT*>(data);
    }
    __attribute__((always_inline)) const DataT& data_as() const {
      return *static_cast<const DataT*>(data);
    }
  };

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
      uint32_t nextPowerOf2 = 1U << (std::bit_width(size - 1));
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

    void* ringPtr = nullptr;
    auto ringErr = cudaHostAlloc(
        &ringPtr, sizeof(HRDWEntry) * size_, cudaHostAllocDefault);
    if (ringErr == cudaSuccess) {
      ring_ = static_cast<HRDWEntry*>(ringPtr);
      for (uint32_t i = 0; i < size_; ++i) {
        ring_[i] = {};
        ring_[i].sequence = HRDW_RINGBUFFER_SLOT_EMPTY;
      }
    } else {
      fprintf(
          stderr,
          "HRDWRingBuffer: Failed to allocate ring buffer: %s\n",
          cudaGetErrorString(ringErr));
      return; // valid() will return false
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

  ~HRDWRingBuffer() {
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
        mask_(other.mask_) {}

  HRDWRingBuffer& operator=(HRDWRingBuffer&& other) noexcept {
    if (this != &other) {
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
  // globaltimer() as timestamp_ns, stores data, and stamps sequence.
  cudaError_t write(cudaStream_t stream, void* data) {
    assert(valid());
    return launchRingBufferWrite(stream, ring_, writeIndex_, mask_, data);
  }

 private:
  template <typename>
  friend class HRDWRingBufferReader;
  friend class HRDWRingBufferTestAccessor;
  HRDWEntry* ring_{nullptr};
  uint64_t* writeIndex_{nullptr};
  uint32_t size_{0};
  uint32_t mask_{0};
};

// Launch a single-thread kernel that atomically claims a slot via
// atomicAdd(*writeIdx), writes globaltimer() to timestamp_ns, writes data,
// issues __threadfence_system(), then stamps sequence = slot.
cudaError_t launchRingBufferWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data);

} // namespace meta::comms::colltrace
