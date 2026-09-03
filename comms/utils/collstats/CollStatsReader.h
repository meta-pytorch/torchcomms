// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsKeyRegistry.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// Host-side readout of one window from a device-resident stats block. The
// aggregate is double-buffered: finalizers write the current-epoch bank, and a
// window is read by flipping the epoch (so new finalizers move to the other
// bank), then copying and zeroing the now-retired bank. Because each bank holds
// exactly one window, the snapshot needs no host delta subtraction.
//
// Cross-stream gating — recording events on the instrumented streams so
// old-epoch finalizers have retired before the copy, and gating new-window
// launches on the flip — is optional: pass a CollStatsReadGating to get it,
// or a null one to leave the flip unordered against producers. Run on a
// dedicated reader stream, never the training stream.

namespace meta::comms::collstats {

// One window's worth of per-key aggregates, copied to the host.
//
// `values` and `keys` are both indexed by the dense id the host key registry
// assigned, so values[i] belongs to keys[i]. `values` carries one extra
// trailing entry, values[numKeys], holding everything that resolved to the
// catch-all; it has no entry in `keys` because it is not one key.
//
// Only the occupied prefix is transferred. Ids are handed out densely and never
// recycled, so numKeys is the registry's size at the moment the window was
// issued, not the bank's capacity.
struct CollStatSnapshot {
  uint32_t numKeys{0};
  uint64_t windowEpoch{0}; // pre-flip epoch value; a monotonic window sequence
  uint64_t catchAllCount{0}; // cumulative, from the host registry
  // Wall-clock bounds of the window, unix nanoseconds, or 0 when the producer
  // did not stamp them. Wall rather than monotonic because their purpose is
  // lining a window up against events outside this process; durations come
  // from the device clock and never from here, so a clock step can misplace a
  // boundary but cannot corrupt a measurement.
  //
  // The span also makes the window's duty cycle computable: the per-key
  // duration sums over this wall interval is the fraction of the period the
  // rank spent inside collectives.
  uint64_t windowStartUnixNs{0};
  uint64_t windowEndUnixNs{0};
  /* The bucketing this window was produced under. Exported alongside the
   * buckets so a consumer never has to assume the defaults, and so a window
   * recorded before a retune stays interpretable. Defaulted rather than
   * zero-initialized: a zero geometry has numBuckets 0, and the readout's
   * `numBuckets - 1` overflow index would wrap. */
  CollStatHistGeometry hist{collStatDefaultHistGeometry()};
  uint64_t thresholdsNs[kMaxThresholds]{
      kDefaultThresholdsNs[0],
      kDefaultThresholdsNs[1],
      kDefaultThresholdsNs[2],
      kDefaultThresholdsNs[3]};
  uint32_t numThresholds{kMaxThresholds};
  /* The size-class edges this window was produced under, carried for the same
   * reason as `hist`: a key's sizeClass is an index into these, so without them
   * an exported row cannot be turned back into byte bounds off-box. */
  CollStatSizeClasses sizeClasses{collStatDefaultSizeClasses()};
  std::vector<CollStatKey> keys; // [numKeys]
  std::vector<CollStatValue> values; // [numKeys + 1]
};

// Cross-stream gating for the epoch flip. The caller (holding the per-comm
// boundary lock, so no collective enqueues mid-flip) supplies the instrumented
// streams the comm launched collectives on, plus reusable events: one per
// stream and one flip event. The reader stream waits each instrumented stream's
// event so every old-epoch finalizer has retired before the copy, and each
// instrumented stream waits the flip event so no new-window collective launches
// until the flip is visible and thus writes the new bank. Without gating a
// concurrent producer could write the bank being copied.
struct CollStatsReadGating {
  const cudaStream_t* instrumentedStreams;
  const cudaEvent_t* streamEvents; // [numStreams], reusable, caller-owned
  uint32_t numStreams;
  cudaEvent_t flipEvent; // reusable, caller-owned
};

// Flip the epoch on `readerStream`, then copy and zero the retired bank, all
// stream-ordered on `readerStream`. `keys` supplies both how much of the bank
// is occupied and the identity of each slot. When `gating` is non-null, the
// cross-stream event sequence brackets the flip (wait old finalizers -> flip ->
// gate new launches -> copy -> zero). Synchronizes `readerStream` and returns
// the retired window's snapshot. On a null handle or a CUDA failure the
// returned snapshot has numKeys == 0.
CollStatSnapshot collStatsReadWindow(
    const CollStatsDeviceBlockHandle& handle,
    cudaStream_t readerStream,
    const CollStatsKeyRegistry& keys,
    const CollStatsReadGating* gating = nullptr);

// Pinned host destination for one window's device-to-host copy.
//
// This is not an optimization. A device-to-host cudaMemcpyAsync whose
// destination is pageable memory blocks the calling thread until the stream
// drains, so copying straight into CollStatSnapshot's std::vectors would make
// the "async" issue path stall the enqueue thread for as long as the GPU is
// behind — exactly what the pipelined design exists to avoid. Page-locked
// memory is what makes the copy genuinely asynchronous.
//
// Allocated once per driver and reused across windows; the snapshot handed to
// the sink is filled by a plain host memcpy in publish() after the copy-done
// event fires, so CollStatSnapshot stays cheap to copy and queue.
class CollStatsPinnedStaging {
 public:
  CollStatsPinnedStaging() = default;
  ~CollStatsPinnedStaging();

  CollStatsPinnedStaging(const CollStatsPinnedStaging&) = delete;
  CollStatsPinnedStaging& operator=(const CollStatsPinnedStaging&) = delete;
  // Non-movable as well as non-copyable: the driver owns one staging buffer for
  // its lifetime, and a moved-from buffer whose pinned allocation had already
  // been handed to an in-flight copy would free it under the copy.
  CollStatsPinnedStaging(CollStatsPinnedStaging&&) = delete;
  CollStatsPinnedStaging& operator=(CollStatsPinnedStaging&&) = delete;

  // Page-locks room for a bank of `capacity` key slots plus the catch-all.
  // Returns false and leaves the object empty on failure, so the caller can
  // fail open.
  bool allocate(uint32_t capacity);

  bool valid() const {
    return values_ != nullptr;
  }
  uint32_t capacity() const {
    return capacity_;
  }

  // Number of key slots the last issue actually copied. Recorded at issue time
  // rather than read from the registry at publish time, because the registry
  // may have handed out further ids while the copy was in flight.
  uint32_t staged() const {
    return staged_;
  }
  void setStaged(uint32_t n) {
    staged_ = n;
  }

  // Copy the staged window into `out`, resizing it, and attribute each slot
  // from `keys`. Host-to-host, so only call once the copy-done event has
  // completed.
  void publish(
      uint64_t windowEpoch,
      const CollStatsKeyRegistry& keys,
      const CollStatsBlockConfig& cfg,
      CollStatSnapshot& out) const;

  // Device-to-host copy destination, for the reader only. Holds
  // capacity + 1 slots; the reader fills [0, staged) plus the trailing
  // catch-all at index staged.
  CollStatValue* values() const {
    return values_;
  }

 private:
  void release();

  uint32_t capacity_{0};
  uint32_t staged_{0};
  CollStatValue* values_{nullptr}; // [capacity + 1]
};

// Async issue of one window's readout, for the pipelined driver that must never
// sync on the producer thread. Enqueues the gated flip and the copy of the
// retired bank's occupied prefix plus its catch-all slot into `staging` on
// `readerStream`, zeroes the retired bank, and records `copyDoneEvent` after
// the copies. Does NOT synchronize. The
// caller tracks `currentEpoch` (the device epoch before this flip; only this
// path flips it, so a local counter stays in lockstep) and must not read
// `staging` until `copyDoneEvent` has completed, at which point
// CollStatsPinnedStaging::publish turns it into a snapshot. `staging` must
// already be allocated for the handle's key capacity. Returns the first CUDA
// error, else cudaSuccess.
//
// Takes the registry rather than a key count: how many slots are live has to be
// sampled after the gate is installed, which only this function can do. A count
// passed in from the caller is necessarily read before the flip and can miss a
// collective that still lands in the retired bank.
cudaError_t collStatsIssueReadWindow(
    const CollStatsDeviceBlockHandle& handle,
    cudaStream_t readerStream,
    const CollStatsReadGating* gating,
    uint64_t currentEpoch,
    cudaEvent_t copyDoneEvent,
    CollStatsPinnedStaging& staging,
    const CollStatsKeyRegistry& keys);

} // namespace meta::comms::collstats
