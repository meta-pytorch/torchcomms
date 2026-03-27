// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

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
  uint64_t entriesInFlight{0};
  bool tornReadDetected{false};
};

// CPU-side snapshot-and-validate consumer for a HRDWRingBuffer. Tracks the
// last-read index and implements the standard pattern:
//   1. Snapshot writeIndex via acquire load
//   2. Skip ahead if behind by more than buffer size (returns lost count)
//   3. For each entry: seqlock pre-read, snapshot, seqlock post-read
//   4. Post-read: re-read writeIndex, discard if lapped (torn read)
//   5. Only invoke callbacks after validation passes
//
// Non-owning — the HRDWRingBuffer must outlive this reader. Single-threaded
// (only the poll thread should call poll()).
template <typename DataT>
class HRDWRingBufferReader {
  using Entry = typename HRDWRingBuffer<DataT>::Entry;

 public:
  explicit HRDWRingBufferReader(const HRDWRingBuffer<DataT>& buffer)
      : ring_(buffer.ring()),
        writeIndex_(buffer.writeIndex()),
        size_(buffer.size()),
        mask_(buffer.mask()) {}

  // Poll for new entries. Calls callback(entry, slot) for each valid entry.
  // The entry's data_as() method returns the typed data pointer.
  // Callbacks are only invoked after the torn-read check passes — they never
  // see partially-overwritten data.
  //
  // When scanIncomplete is true, the reader continues scanning past the
  // first WRITE_PENDING entry and delivers write-pending entries through
  // the same callback. The callback can distinguish them by checking
  // entry.sequence == HRDW_RINGBUFFER_WRITE_PENDING. Only start_ns and
  // data are valid on write-pending entries (end_ns is not yet stamped).
  // When false (default), the reader breaks at the first WRITE_PENDING.
  //
  // Returns a PollResult with counts of entries read, lost, and whether a
  // torn read was detected (in which case no callbacks are invoked).
  template <typename Callback>
  PollResult poll(Callback&& callback, bool scanIncomplete = false) {
    PollResult result;

    if (ring_ == nullptr || writeIndex_ == nullptr) {
      return result;
    }

    auto snapshotWriteIdx = acquireLoad(writeIndex_);

    if (snapshotWriteIdx <= lastReadIndex_) {
      return result;
    }

    uint64_t readStart = lastReadIndex_;

    // If we fell behind by more than the ring size, some entries are lost.
    if (snapshotWriteIdx - readStart > size_) {
      result.entriesLost = snapshotWriteIdx - readStart - size_;
      readStart = snapshotWriteIdx - size_;
    }

    uint64_t newEntries = snapshotWriteIdx - readStart;

    // Snapshot valid entries into a local buffer. Callbacks are deferred
    // until after the post-read torn-read check so they never observe
    // partially-overwritten data.
    //
    // Write-pending entries (sequence == HRDW_RINGBUFFER_WRITE_PENDING) are
    // not lost — they just haven't been stamped by the end kernel yet. We
    // don't advance lastReadIndex_ past the first one, so it will be
    // retried on the next poll. When scanIncomplete is true, we continue
    // scanning and snapshot write-pending entries into validEntries_ so
    // the callback can identify which collectives are in-flight.
    // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
    struct ClearGuard {
      std::vector<std::pair<Entry, uint64_t>>& v;
      ~ClearGuard() {
        v.clear();
      }
    } guard{validEntries_};

    bool hitFirstIncomplete = false;
    for (uint64_t i = 0; i < newEntries; ++i) {
      uint64_t slot = readStart + i;
      uint64_t idx = slot & mask_;

      // Seqlock: read sequence BEFORE data to establish ordering.
      // The acquire load ensures subsequent reads see at least the
      // memory state from when sequence was written.
      auto preSeq = acquireLoad(&ring_[idx].sequence);

      if (preSeq == HRDW_RINGBUFFER_WRITE_PENDING) {
        // In-flight: start kernel ran, end kernel hasn't stamped yet.
        // Don't advance past the first write-pending slot — retry on
        // next poll.
        if (!hitFirstIncomplete) {
          hitFirstIncomplete = true;
          result.entriesInFlight = snapshotWriteIdx - slot;
          snapshotWriteIdx = slot;
          if (!scanIncomplete) {
            break;
          }
        }

        // Snapshot start_ns and data (guaranteed visible by the start
        // kernel's __threadfence_system()).
        Entry entryCopy;
        entryCopy.start_ns = ring_[idx].start_ns;
        entryCopy.data = ring_[idx].data;
        entryCopy.sequence = HRDW_RINGBUFFER_WRITE_PENDING;
        entryCopy.end_ns = 0;

        // Re-read sequence to confirm still WRITE_PENDING — if the end
        // kernel stamped between our reads, skip (it'll be picked up
        // as a completed entry on the next poll).
        auto confirmSeq = acquireLoad(&ring_[idx].sequence);
        if (confirmSeq == HRDW_RINGBUFFER_WRITE_PENDING) {
          validEntries_.emplace_back(std::move(entryCopy), slot);
        }
        continue;
      }

      // Once past the first incomplete, don't collect completed entries
      // — they'll be picked up after the incomplete one finishes.
      if (hitFirstIncomplete) {
        continue;
      }

      if (preSeq != slot) {
        ++result.entriesLost;
        continue;
      }

      // Copy data fields. The acquire fence above ensures we see data
      // at least as new as when sequence was stamped.
      Entry entryCopy = ring_[idx];

      // Post-read: re-check sequence. If it changed, the entry was
      // overwritten during our copy — discard.
      auto postSeq = acquireLoad(&ring_[idx].sequence);
      if (postSeq != preSeq) {
        if (postSeq == HRDW_RINGBUFFER_WRITE_PENDING) {
          if (!hitFirstIncomplete) {
            hitFirstIncomplete = true;
            result.entriesInFlight = snapshotWriteIdx - slot;
            snapshotWriteIdx = slot;
            if (!scanIncomplete) {
              break;
            }
          }
          Entry pendingCopy;
          pendingCopy.start_ns = ring_[idx].start_ns;
          pendingCopy.data = ring_[idx].data;
          pendingCopy.sequence = HRDW_RINGBUFFER_WRITE_PENDING;
          pendingCopy.end_ns = 0;
          auto confirmSeq = acquireLoad(&ring_[idx].sequence);
          if (confirmSeq == HRDW_RINGBUFFER_WRITE_PENDING) {
            validEntries_.emplace_back(std::move(pendingCopy), slot);
          }
          continue;
        }
        ++result.entriesLost;
        continue;
      }

      validEntries_.emplace_back(std::move(entryCopy), slot);
    }

    // Re-read writeIndex after the read pass. If writers advanced by
    // more than bufferSize since the earliest slot we read, that slot has
    // been overwritten (torn read). Discard everything from this pass.
    if (auto newWriteIdx = acquireLoad(writeIndex_);
        newWriteIdx - readStart > size_) {
      result.tornReadDetected = true;
      result.entriesRead = 0;
      result.entriesLost = newWriteIdx - lastReadIndex_;
      lastReadIndex_ = newWriteIdx;
    } else {
      // Validation passed — invoke callbacks with snapshotted entries.
      for (const auto& [entry, slot] : validEntries_) {
        callback(entry, slot);
      }
      result.entriesRead = validEntries_.size();
      lastReadIndex_ = snapshotWriteIdx;
    }

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

  // Reusable buffer for snapshotted entries — avoids allocation per poll().
  std::vector<std::pair<Entry, uint64_t>> validEntries_;
};

} // namespace meta::comms::colltrace
