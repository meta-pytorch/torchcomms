// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

#include "comms/utils/HRDWRingBuffer.h"

namespace meta::comms::colltrace {

// Relaxed atomic load from GPU-mapped pinned memory. The reader doesn't
// need acquire ordering: the writer publishes writeIndex and ring[idx]
// independently (each as its own atomic), and the per-slot epoch check
// in tryRead classifies any stale state as kNotReady / kOverwritten if
// the writeIndex bump becomes visible before the slot publication.
template <typename T>
__attribute__((always_inline)) inline T relaxedLoad(const T* ptr) {
  return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}

// 16-byte specialization. __atomic_load_n on __int128 emits a libcall to
// __atomic_load_16 (libatomic) — even with -mcx16, GCC doesn't reliably
// inline cmpxchg16b for it. Some build images don't ship libatomic, so on
// x86_64 we sidestep the libcall via SSE2 MOVDQA, which is single-copy
// atomic for naturally-aligned 16-byte addresses on every production
// x86_64 CPU. The slot is alignas(16), so this is always safe.
#if defined(__x86_64__)
template <>
__attribute__((always_inline)) inline unsigned __int128 relaxedLoad(
    const unsigned __int128* ptr) {
  __m128i v = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
  unsigned __int128 result;
  __builtin_memcpy(&result, &v, sizeof(result));
  return result;
}
#endif

// Result of a single poll() call.
struct PollResult {
  uint64_t entriesRead{0};
  uint64_t entriesLost{0};
  bool timedOut{false};
};

// CPU-side consumer for a HRDWRingBuffer. Uses the mapped writeIndex to
// know how far ahead writers have gone, then validates each entry via
// the per-entry epoch field. If the reader falls behind by more than
// ringSize, it jumps to the tail and counts skipped entries as lost.
//
// Non-owning — the HRDWRingBuffer must outlive this reader. Single-threaded
// (only the poll thread should call poll()).
template <typename DataT>
class HRDWRingBufferReader {
  using Entry = typename HRDWRingBuffer<DataT>::Entry;

 public:
  explicit HRDWRingBufferReader(const HRDWRingBuffer<DataT>& buffer)
      : ring_(buffer.ring_),
        writeIndex_(buffer.writeIndex_),
        size_(buffer.size_),
        mask_(buffer.mask_),
        shift_(buffer.shift_) {
    assert(buffer.valid());
  }

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
  ReadResult tryRead(uint64_t slot, Entry& dest) const {
    uint64_t idx = slot & mask_;
    uint32_t expected = static_cast<uint32_t>(slot >> shift_) + 1u;

    static_assert(
        sizeof(Entry) == sizeof(unsigned __int128),
        "Entry must be exactly 16 bytes for the atomic load");
    unsigned __int128 raw =
        relaxedLoad(reinterpret_cast<const unsigned __int128*>(&ring_[idx]));
    __builtin_memcpy(&dest, &raw, sizeof(dest));

    if (dest.epoch == expected) {
      return ReadResult::kSuccess;
    }

    // Reader stays within size_ of head, so |actual - expected| ≤ 1
    // (modular). Signed delta disambiguates earlier vs later writes and
    // wraps correctly.
    int32_t delta = static_cast<int32_t>(dest.epoch - expected);
    if (dest.epoch != HRDW_RINGBUFFER_SLOT_EMPTY && delta > 0) {
      return ReadResult::kOverwritten;
    }
    return ReadResult::kNotReady;
  }

  // Poll for new entries. Calls callback(entry, slot) for each valid entry.
  // Accumulates all readable entries first, then delivers via callback.
  //
  // If timeout is non-zero, waits up to that duration for the first entry
  // before returning empty.
  //
  // Returns a PollResult with counts of entries read and lost.
  template <typename Callback>
  PollResult poll(
      Callback&& callback,
      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
    PollResult result;

    auto deadline = std::chrono::steady_clock::now() + timeout;

    // jump to the oldest valid entry if the reader fell behind by more than
    // size_ and returns the number of entries skipped (lost). also updates
    // head to the newest-read writeIndex_
    auto jumpToTail = [&](uint64_t& head) -> uint64_t {
      head = relaxedLoad(writeIndex_);
      if (head > lastReadIndex_ && head - lastReadIndex_ > size_) {
        uint64_t lost = head - lastReadIndex_ - size_;
        lastReadIndex_ = head - size_;
        return lost;
      }
      return 0;
    };

    auto head = relaxedLoad(writeIndex_);
    if (head <= lastReadIndex_ /* no new entries since last read */) {
      return result;
    }

    result.entriesLost += jumpToTail(head);

    validEntries_.clear();

    while (lastReadIndex_ < head) {
      Entry entry;
      auto readResult = tryRead(lastReadIndex_, entry);

      switch (readResult) {
        case ReadResult::kSuccess:
          validEntries_.emplace_back(entry, lastReadIndex_);
          ++lastReadIndex_;
          break;
        case ReadResult::kOverwritten: {
          auto lost = jumpToTail(head);
          if (lost == 0 /* not lapped by full ring - just lost this entry */) {
            ++lastReadIndex_;
            lost = 1;
          }
          result.entriesLost += lost;
          if (timeout.count() > 0 &&
              std::chrono::steady_clock::now() > deadline) {
            result.timedOut = true;
            goto done;
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

  uint64_t lastReadIndex() const {
    return lastReadIndex_;
  }

 private:
  Entry* ring_;
  uint64_t* writeIndex_;
  uint32_t size_;
  // size_ - 1. Cached so `slot & mask_` is one bitwise op per read
  // instead of `slot % size_`.
  uint32_t mask_;
  // log2(size_). Cached so the per-slot expected epoch is computed as
  // `(slot >> shift_) + 1` — a shift instead of a divide.
  uint32_t shift_;
  uint64_t lastReadIndex_{0};

  // Reusable buffer for accumulated entries — avoids allocation per poll().
  std::vector<std::pair<Entry, uint64_t>> validEntries_;
};

} // namespace meta::comms::colltrace
