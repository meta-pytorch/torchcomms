// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

#define HRDW_RINGBUFFER_WRITE_PENDING UINT64_MAX

namespace meta::comms::colltrace {

// ==========================================================================
// HRDWRingBuffer — Host-Read Device-Write ring buffer for GPU-to-CPU
// timestamp transfer with lock-free, torn-read-safe polling.
//
// OVERVIEW
//   GPU kernels claim ring buffer slots via atomic increment, write
//   timestamps and user data, then stamp a sequence number. A CPU thread
//   polls the ring buffer using a seqlock protocol that guarantees it
//   never observes partially-overwritten entries.
//
// USAGE
//   1. Create a ring buffer, specifying the size and a DataT for typed
//      access to the per-entry user data pointer:
//
//        HRDWRingBuffer<MyContext*> buf(4096);
//
//   2. For each operation, launch the start and end kernels on a CUDA
//      stream. The start kernel claims a slot, records globaltimer() as
//      start_ns, stores your data pointer, and writes the slot index to
//      slotStorage. The end kernel records end_ns and stamps the sequence:
//
//        launchRingBufferStartWrite(stream, buf.ring(), buf.writeIndex(),
//                                   buf.mask(), myContextPtr, slotStorage);
//        // ... collective / kernel work ...
//        launchRingBufferEndWrite(stream, buf.ring(), slotStorage,
//                                 buf.mask());
//
//   3. On the CPU, create a reader and poll periodically. The callback
//      receives a typed Entry (with data_as() accessor) and the slot:
//
//        HRDWRingBufferReader<MyContext*> reader(buf);
//        reader.poll([](const auto& entry, uint64_t slot) {
//          MyContext* ctx = entry.data_as();
//          auto start = entry.start_ns;
//          auto end = entry.end_ns;
//        });
//
// MEMORY ORDERING
//   - The end kernel uses __threadfence_system() between writing data
//     fields and stamping the sequence, ensuring the CPU sees consistent
//     data when it observes a valid sequence.
//   - The reader uses a seqlock protocol: it reads the sequence with
//     acquire semantics BEFORE copying the entry, then re-reads after.
//     If the sequence changed, the entry is discarded.
//   - A post-read writeIndex check detects entries that were lapped
//     during the read pass (torn reads). Affected entries are discarded
//     and counted as lost.
//
// RING SIZING
//   INVARIANT: The number of concurrent in-flight entries (start kernel
//   fired, end kernel not yet stamped) must be less than ringSize / 2.
//   If the ring wraps while an end kernel is still pending, the stale
//   end kernel can corrupt a newer entry's data — this race cannot be
//   prevented without making the entire start+end sequence atomic.
//   CollTrace enforces this by sizing the ring to at least 2x the max
//   number of concurrent collectives.
//
//   Entries that are overwritten after completion (the reader fell behind)
//   are safely detected and reported as lost (PollResult::entriesLost).
//   The size is automatically rounded up to the next power of 2.
//
// ENTRY LIFECYCLE
//   slot claimed (start kernel) → sequence = WRITE_PENDING → data written
//   → end kernel → end_ns written → __threadfence_system → sequence = slot
//   → CPU can read
// ==========================================================================

// Entry stored in the ring buffer. The GPU kernels write start_ns, end_ns,
// sequence, and data; the CPU reader polls and validates via the sequence
// field (seqlock protocol).
//
// The `data` pointer is opaque to the ring buffer — callers can store
// whatever context they need and access it via Entry::data_as() in the
// poll callback.
// Entries with sequence == HRDW_RINGBUFFER_WRITE_PENDING have a start kernel
// that ran but whose end kernel hasn't stamped the final sequence yet.
struct HRDWEntry {
  uint64_t start_ns;
  uint64_t end_ns;
  uint64_t sequence;
  void* data;
};

// Host-Read Device-Write (HRDW) ring buffer. Device kernels atomically claim
// slots and stamp timestamps; the host polls with a seqlock mechanism to
// protect against torn reads.
//
// DataT is the user-defined type stored in each entry's data pointer.
// Access it via entry.data_as() which returns DataT& without requiring
// a template argument.
//
// Owns the ring buffer and write index in mapped pinned memory (GPU-writable,
// CPU-readable). Move-only — CUDA resources are not copyable.
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
        ring_[i].sequence = HRDW_RINGBUFFER_WRITE_PENDING;
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
      cudaFreeHost(ring_);
    }
    if (writeIndex_) {
      // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
      cudaFreeHost(writeIndex_);
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
        cudaFreeHost(ring_);
      }
      if (writeIndex_) {
        // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
        cudaFreeHost(writeIndex_);
      }
      ring_ = std::exchange(other.ring_, nullptr);
      writeIndex_ = std::exchange(other.writeIndex_, nullptr);
      size_ = other.size_;
      mask_ = other.mask_;
    }
    return *this;
  }

  Entry* ring() const {
    return static_cast<Entry*>(ring_);
  }
  uint64_t* writeIndex() const {
    return writeIndex_;
  }
  uint32_t size() const {
    return size_;
  }
  uint32_t mask() const {
    return mask_;
  }

  // Returns true if all allocations succeeded.
  bool valid() const {
    return ring_ != nullptr && writeIndex_ != nullptr;
  }

 private:
  HRDWEntry* ring_{nullptr};
  uint64_t* writeIndex_{nullptr};
  uint32_t size_{0};
  uint32_t mask_{0};
};

// Launch a single-thread kernel that atomically claims a slot in the ring
// buffer via *writeIdx, writes globaltimer() to the entry's start_ns, writes
// data to the entry's data field, sets sequence =
// HRDW_RINGBUFFER_WRITE_PENDING (write pending), and stores the raw slot index
// to *slotOut.
cudaError_t launchRingBufferStartWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* writeIdx,
    uint32_t mask,
    void* data,
    uint64_t* slotOut);

// Launch a single-thread kernel that reads the slot from *slotIn, writes
// globaltimer() to ring[slot & mask].end_ns, and stamps sequence = slot.
// If completionCounter is non-null, atomically increments it after stamping.
cudaError_t launchRingBufferEndWrite(
    cudaStream_t stream,
    void* ring,
    uint64_t* slotIn,
    uint32_t mask,
    uint64_t* completionCounter = nullptr);

} // namespace meta::comms::colltrace
