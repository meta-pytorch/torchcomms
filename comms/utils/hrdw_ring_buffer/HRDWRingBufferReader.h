// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace hrdw_ring_buffer {

// Relaxed atomic load from GPU-mapped pinned memory. The reader doesn't
// need acquire ordering: the writer publishes writeIndex and ring[idx]
// independently (each as its own atomic), and the per-slot epoch check
// in tryRead classifies any stale state as kNotReady / kOverwritten if
// the writeIndex bump becomes visible before the slot publication.
//
// For the 16-byte ring entry, both paths lower to lock cmpxchg16b
// (x86_64 with -mcx16) or to a libcall to __atomic_load_16 otherwise.
// The v2_29/v2_30 makefiles and the github CMakeLists link -latomic
// unconditionally so the libcall always resolves at link time.
template <typename T>
__attribute__((always_inline)) inline T relaxedLoad(const T* ptr) {
#if defined(__cpp_lib_atomic_ref)
  return std::atomic_ref<T>(*const_cast<T*>(ptr))
      .load(std::memory_order_relaxed);
#else
  return __atomic_load_n(ptr, __ATOMIC_RELAXED);
#endif
}

// Result of a single poll() call, specialized per MemoryCoherenceScope.
// Both specializations carry entriesRead and entriesLost; each adds the
// scope-specific fields its poll() implementation can populate.
template <MemoryCoherenceScope C>
struct PollResult;

template <>
struct PollResult<MemoryCoherenceScope::System> {
  uint64_t entriesRead{0};
  uint64_t entriesLost{0};
  // True if poll() exited because the timeout elapsed before any new
  // entries arrived.
  bool timedOut{false};
  // Blocking rings only: true if the ring was saturated at poll() entry, i.e.
  // the device writer is (or just was) backpressured in write_blocking()
  // waiting for the consumer to free a slot. Always false for Overwrite rings.
  // The ring stays logging-free — consumers decide whether/how to surface it.
  bool writerThrottled{false};
};

template <>
struct PollResult<MemoryCoherenceScope::Device> {
  uint64_t entriesRead{0};
  uint64_t entriesLost{0};
  // Error from cudaStreamSynchronize / cudaMemcpy during the drain.
  cudaError_t error{cudaSuccess};
};

// CPU-side consumer for an HRDWRingBuffer. Both scopes expose the same
// public method — poll(...) — but the implementation differs per
// MemoryCoherenceScope:
//
//   System: poll() reads pinned mapped memory directly and validates
//           each entry via the per-slot epoch field. Lock-free and
//           concurrent with the writer. Takes a callback and an
//           optional timeout.
//
//   Device: poll() synchronizes the stream, cudaMemcpys the ring + write
//           index to host, then iterates the host-side copy. Takes a
//           stream and a callback. The internal read cursor advances
//           on every poll, so successive calls only return entries
//           written since the previous one.
//
// Non-owning — the HRDWRingBuffer must outlive this reader. Single-
// threaded (only one thread should call into the reader).
template <
    typename DataT,
    MemoryCoherenceScope C = MemoryCoherenceScope::System,
    WritePolicy W = WritePolicy::Overwrite>
class HRDWRingBufferReader;

// State + ctor shared by the System and Device readers. The handle
// fields (ring_, writeIndex_, size_, mask_, shift_) and the read
// cursor are identical across scopes; only poll() differs.
template <typename DataT, MemoryCoherenceScope C, WritePolicy W>
class HRDWRingBufferReaderBase {
 public:
  uint64_t lastReadIndex() const {
    return lastReadIndex_;
  }

 protected:
  using EntryT = HRDWEntry<DataT, C>;

  explicit HRDWRingBufferReaderBase(const HRDWRingBuffer<DataT, C, W>& buffer)
      : ring_(buffer.ring_),
        writeIndex_(buffer.writeIndex_),
        readIndex_(readIndexOf(buffer)),
        size_(buffer.size_),
        mask_(buffer.mask_),
        shift_(buffer.shift_) {
    assert(buffer.valid());
  }

  static uint64_t* readIndexOf(const HRDWRingBuffer<DataT, C, W>& buffer) {
    if constexpr (W == WritePolicy::Blocking) {
      return buffer.readIndex_;
    } else {
      return nullptr;
    }
  }

  EntryT* ring_;
  uint64_t* writeIndex_;
  // Consumer cursor for Blocking-ring backpressure; absent (zero bytes) on
  // Overwrite readers via the same [[no_unique_address]]/define_if pattern the
  // buffer and device handle use.
  [[no_unique_address]] detail::define_if<W == WritePolicy::Blocking, uint64_t*>
      readIndex_{};
  uint32_t size_;
  // size_ - 1. Cached so `slot & mask_` is one bitwise op per read
  // instead of `slot % size_`.
  uint32_t mask_;
  // log2(size_). Cached so the per-slot expected epoch is computed as
  // `(slot >> shift_) + 1` — a shift instead of a divide.
  uint32_t shift_;
  uint64_t lastReadIndex_{0};
};

template <typename DataT, WritePolicy W>
class HRDWRingBufferReader<DataT, MemoryCoherenceScope::System, W>
    : public HRDWRingBufferReaderBase<DataT, MemoryCoherenceScope::System, W> {
  using Base = HRDWRingBufferReaderBase<DataT, MemoryCoherenceScope::System, W>;
  using Base::lastReadIndex_;
  using Base::mask_;
  using Base::readIndex_;
  using Base::ring_;
  using Base::shift_;
  using Base::size_;
  using Base::writeIndex_;
  using typename Base::EntryT;

 public:
  using Result = PollResult<MemoryCoherenceScope::System>;

  explicit HRDWRingBufferReader(
      const HRDWRingBuffer<DataT, MemoryCoherenceScope::System, W>& buffer)
      : Base(buffer) {}

  enum class ReadResult { kSuccess, kOverwritten, kNotReady };

  // Try to read a single entry at the given slot. Returns:
  //   kSuccess:     entry copied into dest, valid
  //   kOverwritten: entry was overwritten by a newer writer, lost
  //   kNotReady:    entry not yet written, retry later
  //
  // The slot is published by the writer as a single atom.exch.b128, so a
  // 16B atomic relaxed load on the host side observes either the fully-
  // old or fully-new state — no torn reads, no re-check needed.
  // Relaxed (rather than acquire) is sufficient because we don't rely on
  // ring[idx] writes being ordered with the writeIndex bump: the epoch
  // check below classifies any stale state as kNotReady / kOverwritten.
  ReadResult tryRead(uint64_t slot, EntryT& dest) const {
    uint64_t idx = slot & mask_;
    uint32_t expected = static_cast<uint32_t>(slot >> shift_) + 1u;

    static_assert(
        sizeof(EntryT) == sizeof(unsigned __int128),
        "Entry must be exactly 16 bytes for the atomic load");
    unsigned __int128 raw =
        relaxedLoad(reinterpret_cast<const unsigned __int128*>(&ring_[idx]));
    __builtin_memcpy(&dest, &raw, sizeof(dest));

    if (dest.epoch == expected) {
      return ReadResult::kSuccess;
    }

    if constexpr (W == WritePolicy::Blocking) {
      // Backpressure guarantees the writer never laps a slot the reader still
      // holds the cursor at, so it can never be overwritten out from under us.
      // A non-matching epoch is therefore only ever EMPTY or a stale prior-lap
      // epoch — i.e. "not published yet", never overwritten.
      return ReadResult::kNotReady;
    } else {
      // Reader stays within size_ of head, so |actual - expected| ≤ 1
      // (modular). Signed delta disambiguates earlier vs later writes and
      // wraps correctly.
      int32_t delta = static_cast<int32_t>(dest.epoch - expected);
      if (dest.epoch != HRDW_RINGBUFFER_SLOT_EMPTY && delta > 0) {
        return ReadResult::kOverwritten;
      }
      return ReadResult::kNotReady;
    }
  }

  // Poll for new entries. Calls callback(entry, slot) for each valid entry.
  // Accumulates all readable entries first, then delivers via callback.
  //
  // If timeout is non-zero, waits up to that duration for the first entry
  // before returning empty.
  //
  // Returns a Result with counts of entries read and lost.
  template <typename Callback>
  Result poll(
      Callback&& callback,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    Result result;

    [[maybe_unused]] auto deadline = std::chrono::steady_clock::now() + timeout;

    // jump to the oldest valid entry if the reader fell behind by more than
    // size_ and returns the number of entries skipped (lost). also updates
    // head to the newest-read writeIndex_
    auto jumpToTail = [&](uint64_t& head) -> uint64_t {
      head = relaxedLoad(writeIndex_);
      // A Blocking ring backpressures the writer, so the reader can never fall
      // behind the published data by more than the ring size — it is lossless
      // by construction. head may still sit one slot past that: a writer claims
      // its slot with fetch_add(writeIndex) *before* blocking on backpressure,
      // so a single blocked writer leaves writeIndex at readIndex + size + 1
      // while the slot it is parked on stays unpublished (read as kNotReady,
      // never lost). Applying the Overwrite skip here would drop that
      // still-live entry and then release the writer to overwrite it — so never
      // skip.
      if constexpr (W == WritePolicy::Blocking) {
        return 0;
      }
      if (head > lastReadIndex_ && head - lastReadIndex_ > size_) {
        uint64_t lost = head - lastReadIndex_ - size_;
        lastReadIndex_ = head - size_;
        return lost;
      }
      return 0;
    };

    auto head = relaxedLoad(writeIndex_);
    if constexpr (W == WritePolicy::Blocking) {
      result.writerThrottled = (head - lastReadIndex_) >= size_;
    }
    if (head <= lastReadIndex_ /* no new entries since last read */) {
      return result;
    }

    result.entriesLost += jumpToTail(head);

    validEntries_.clear();

    while (lastReadIndex_ < head) {
      EntryT entry;
      auto readResult = tryRead(lastReadIndex_, entry);

      switch (readResult) {
        case ReadResult::kSuccess:
          validEntries_.emplace_back(entry, lastReadIndex_);
          ++lastReadIndex_;
          maybePublishConsumed(lastReadIndex_);
          break;
        case ReadResult::kOverwritten: {
          if constexpr (W == WritePolicy::Overwrite) {
            auto lost = jumpToTail(head);
            if (lost ==
                0 /* not lapped by full ring - just lost this entry */) {
              ++lastReadIndex_;
              lost = 1;
            }
            result.entriesLost += lost;
            if (timeout.count() > 0 &&
                std::chrono::steady_clock::now() > deadline) {
              result.timedOut = true;
              goto done;
            }
          } else {
            assert(
                false &&
                "Blocking ring must never observe an overwritten slot");
          }
          break;
        }
        case ReadResult::kNotReady:
          goto done;
      }
    }
  done:

    for (const auto& [entry, slot] : validEntries_) {
      callback(entry, slot);
    }
    result.entriesRead = validEntries_.size();

    return result;
  }

 private:
  // Publish the consumer cursor so a Blocking ring's device writer can tell
  // which slots are safe to reuse. Release-store pairs with the acquire load in
  // write_blocking(). No-op for Overwrite rings, which never allocate
  // readIndex_.
  __attribute__((always_inline)) inline void maybePublishConsumed(
      uint64_t consumedUpTo) const {
    if constexpr (W == WritePolicy::Blocking) {
#if defined(__cpp_lib_atomic_ref)
      std::atomic_ref<uint64_t>(*readIndex_)
          .store(consumedUpTo, std::memory_order_release);
#else
      __atomic_store_n(readIndex_, consumedUpTo, __ATOMIC_RELEASE);
#endif
    }
  }

  // Reusable buffer for accumulated entries — avoids allocation per poll().
  std::vector<std::pair<EntryT, uint64_t>> validEntries_;
};

// Device-scope reader. Single-consumer, synchronous: poll() pulls a
// cudaMemcpy snapshot of the ring after a stream sync, then invokes the
// callback for each fresh entry. The internal read cursor advances on
// every poll, so successive calls only return entries written since
// the previous one.
template <typename DataT, WritePolicy W>
class HRDWRingBufferReader<DataT, MemoryCoherenceScope::Device, W>
    : public HRDWRingBufferReaderBase<DataT, MemoryCoherenceScope::Device, W> {
  using Base = HRDWRingBufferReaderBase<DataT, MemoryCoherenceScope::Device, W>;
  using Base::lastReadIndex_;
  using Base::mask_;
  using Base::ring_;
  using Base::shift_;
  using Base::size_;
  using Base::writeIndex_;
  using typename Base::EntryT;

 public:
  using Result = PollResult<MemoryCoherenceScope::Device>;

  explicit HRDWRingBufferReader(
      const HRDWRingBuffer<DataT, MemoryCoherenceScope::Device, W>& buffer)
      : Base(buffer) {}

  // Poll for new entries since the last poll(). Synchronizes the stream,
  // copies the ring and writeIndex from device to host, then invokes
  // callback(entry, slot) for each valid entry. Entries that were lapped
  // before this poll are counted in entriesLost.
  template <typename Callback>
  Result poll(cudaStream_t stream, Callback&& callback) {
    Result result;

    result.error = cudaStreamSynchronize(stream);
    if (result.error != cudaSuccess) {
      return result;
    }

    uint64_t head = 0;
    result.error =
        cudaMemcpy(&head, writeIndex_, sizeof(head), cudaMemcpyDeviceToHost);
    if (result.error != cudaSuccess) {
      return result;
    }

    if (head <= lastReadIndex_) {
      return result;
    }

    uint64_t tail = lastReadIndex_;
    if (head - tail > size_) {
      result.entriesLost = head - tail - size_;
      tail = head - size_;
    }

    uint64_t count = head - tail;
    drainBuf_.resize(count);

    // The entries may wrap around the ring. Copy in up to two segments.
    uint64_t startIdx = tail & mask_;
    uint64_t firstChunk =
        std::min(count, static_cast<uint64_t>(size_) - startIdx);
    result.error = cudaMemcpy(
        drainBuf_.data(),
        ring_ + startIdx,
        firstChunk * sizeof(EntryT),
        cudaMemcpyDeviceToHost);
    if (result.error != cudaSuccess) {
      return result;
    }
    if (firstChunk < count) {
      result.error = cudaMemcpy(
          drainBuf_.data() + firstChunk,
          ring_,
          (count - firstChunk) * sizeof(EntryT),
          cudaMemcpyDeviceToHost);
      if (result.error != cudaSuccess) {
        return result;
      }
    }

    // Validate epoch per slot. Two writers can claim slots that map to
    // the same idx (slot and slot + size_); their atom.exch.b128 stores
    // are non-torn but order isn't determined by claim order, so an
    // older-generation entry can be observed at an idx whose current-
    // generation entry was claimed first. The epoch check classifies
    // such stale slots as lost rather than delivering wrong-generation
    // data to the callback.
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t slot = tail + i;
      uint32_t expected = static_cast<uint32_t>(slot >> shift_) + 1u;
      if (drainBuf_[i].epoch == expected) {
        callback(drainBuf_[i], slot);
        ++result.entriesRead;
      } else {
        ++result.entriesLost;
      }
    }
    lastReadIndex_ = head;

    return result;
  }

 private:
  std::vector<EntryT> drainBuf_;
};

} // namespace hrdw_ring_buffer
