// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include "comms/utils/HRDWRingBuffer.h"

namespace meta::comms::colltrace {

// Acquire-semantics load from GPU-mapped pinned memory. On x86 this compiles
// to a plain load (TSO provides acquire semantics); on ARM it emits ldar.
__attribute__((always_inline)) inline uint64_t acquireLoad(
    const uint64_t* ptr) {
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

// Result of a single poll() call.
struct PollResult {
  uint64_t entriesRead{0};
  uint64_t entriesLost{0};
  bool timedOut{false};
};

// CPU-side consumer for a HRDWRingBuffer. Uses the mapped writeIndex to
// know how far ahead writers have gone, then validates each entry via
// per-entry sequence (seqlock). If the reader falls behind by more than
// ringSize, it jumps to the tail and counts skipped entries as lost.
//
// Non-owning — the HRDWRingBuffer must outlive this reader. Single-threaded
// (only the poll thread should call poll()).
template <typename DataT>
class HRDWRingBufferReader {
  using Entry = typename HRDWRingBuffer<DataT>::Entry;

 public:
  explicit HRDWRingBufferReader(const HRDWRingBuffer<DataT>& buffer)
      : ring_(static_cast<Entry*>(buffer.ring_)),
        writeIndex_(buffer.writeIndex_),
        size_(buffer.size_),
        mask_(buffer.mask_) {
    assert(buffer.valid());
  }

  enum class ReadResult { kSuccess, kOverwritten, kNotReady };

  // Try to read a single entry at the given slot. Returns:
  //   kSuccess:     entry copied into dest, valid
  //   kOverwritten: entry was overwritten by a newer writer, lost
  //   kNotReady:    entry not yet written, retry later
  ReadResult tryRead(uint64_t slot, Entry& dest) const {
    uint64_t idx = slot & mask_;
    auto preSeq = acquireLoad(&ring_[idx].sequence);

    if (preSeq == slot) {
      dest = ring_[idx];
      if (acquireLoad(&ring_[idx].sequence) != preSeq) {
        return ReadResult::kOverwritten;
      }
      return ReadResult::kSuccess;
    }

    if (preSeq != HRDW_RINGBUFFER_SLOT_EMPTY && preSeq > slot) {
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
      head = acquireLoad(writeIndex_);
      if (head > lastReadIndex_ && head - lastReadIndex_ > size_) {
        uint64_t lost = head - lastReadIndex_ - size_;
        lastReadIndex_ = head - size_;
        return lost;
      }
      return 0;
    };

    auto head = acquireLoad(writeIndex_);
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
  uint32_t mask_;
  uint64_t lastReadIndex_{0};

  // Reusable buffer for accumulated entries — avoids allocation per poll().
  std::vector<std::pair<Entry, uint64_t>> validEntries_;
};

} // namespace meta::comms::colltrace
