// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <functional>

#include <cuda_runtime.h>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsReader.h"

// Per-communicator pipelined readout driver. Called from the producer thread
// after each instrumented collective, it runs the window readout inline without
// ever synchronizing: every `cadence` collectives it harvests the previous
// window's already-completed copy (a non-blocking event query) and issues the
// next window's copy on a dedicated reader stream. Because it is driven from
// the same thread that enqueues collectives, the flip sequence needs no
// boundary lock — no collective can enqueue mid-flip on that thread.
//
// It owns the reader stream, the reusable events, one page-locked staging
// buffer, and one snapshot, all reused across windows. The staging buffer must
// be page-locked for the non-blocking claim above to hold: a device-to-host
// cudaMemcpyAsync into pageable memory blocks the calling thread until the
// stream drains. Page-locking is necessary but has not proven sufficient
// everywhere — one tick behind a spin kernel measures ~0ms on a devgpu H100 and
// ~2005ms under remote execution, cause not yet identified (T282705070). A CUDA
// failure disables the driver (telemetry stops) rather than risk the workload;
// every dropped or lost window is counted.
//
// Not thread-safe: onCollective must be called from the comm's enqueue thread,
// the same serialization the ctran launch path already assumes.

namespace meta::comms::collstats {

class CollStatsReadoutDriver {
 public:
  using Sink = std::function<void(const CollStatSnapshot&)>;

  // Creates a dedicated non-blocking reader stream and reusable events. If any
  // CUDA resource fails the driver is born disabled (a no-op). `handle.dev`
  // must be non-null; `cadence` is the number of collectives between readouts
  // (>= 1). `sink` receives each completed window synchronously and must
  // consume it before returning (the staging buffer is reused). `keys` is the
  // registry that assigned the value slots; it must outlive the driver, and it
  // supplies both how much of the bank to read and the identity of each slot.
  //
  // `sink` should not throw. It is invoked from the destructor's final flush,
  // where an escaping exception would call std::terminate and take the job down
  // over telemetry. Exceptions are caught and counted (sinkExceptions()) rather
  // than propagated, but a sink that throws still loses that window.
  CollStatsReadoutDriver(
      const CollStatsDeviceBlockHandle& handle,
      uint32_t cadence,
      Sink sink,
      const CollStatsKeyRegistry& keys);
  ~CollStatsReadoutDriver();

  CollStatsReadoutDriver(const CollStatsReadoutDriver&) = delete;
  CollStatsReadoutDriver& operator=(const CollStatsReadoutDriver&) = delete;
  // Non-movable as well: the driver owns a reader stream, events and a pinned
  // staging buffer that an in-flight copy is already addressed at, so a
  // moved-from driver's destructor would tear them down under that copy.
  CollStatsReadoutDriver(CollStatsReadoutDriver&&) = delete;
  CollStatsReadoutDriver& operator=(CollStatsReadoutDriver&&) = delete;

  // Called after each instrumented collective launch on `instrumentedStream`.
  // Every `cadence` calls: harvest the previous window if its copy has
  // completed (a non-blocking event query), then issue the next. Intended not
  // to block the caller; see T282705070 for where that does not yet hold.
  void onCollective(cudaStream_t instrumentedStream);

  // Synchronize the reader stream and harvest an already-issued window.
  //
  // This recovers the last window that reached a cadence boundary, not the
  // collectives accumulated since — those are still in the live device bank and
  // were never staged. Use flushFinal() at teardown to get both.
  void flush();

  // flush(), then issue and harvest one extra window for the collectives
  // accumulated since the last cadence boundary. Called from the destructor, so
  // the up to `cadence - 1` collectives at the end of a run are reported rather
  // than freed unread with the device bank.
  //
  // Blocks, unlike onCollective. The extra window is issued ungated, after a
  // device synchronize: the usual cross-stream gating records an event on the
  // instrumented stream, and a destructor cannot assume a stream it does not
  // own still exists. The device sync gives the same guarantee — every
  // finalizer has retired before the bank is copied — without touching a
  // borrowed handle.
  //
  // The one ordering requirement is that the device block outlive the driver.
  // CtranAlgo declares the driver last, so it is destroyed first.
  void flushFinal();

  uint64_t windowsExported() const {
    return windowsExported_;
  }
  uint64_t windowsDropped() const {
    return windowsDropped_;
  }
  // Harvest attempts that found the copy still in flight. Counts attempts, not
  // windows: a deferred window stays pending and still lands in exactly one of
  // exported/dropped, so one stalled copy can bump this many times.
  uint64_t harvestRetries() const {
    return harvestRetries_;
  }
  // Windows whose sink threw. Counted apart from windowsDropped_: the readout
  // itself succeeded and the window was complete, so this attributes the loss
  // to the consumer rather than to the device path.
  uint64_t sinkExceptions() const {
    return sinkExceptions_;
  }
  bool disabled() const {
    return disabled_;
  }

 private:
  void harvestIfReady();
  // Enqueues one window's readout, gated when `gating` is non-null. Marks the
  // driver disabled and counts a drop on failure.
  void issue(const CollStatsReadGating* gating);

  CollStatsDeviceBlockHandle handle_;
  uint32_t cadence_;
  Sink sink_;
  const CollStatsKeyRegistry* keys_;

  // The device current at construction, where reader_, the events and the
  // bank live. flushFinal's device sync only covers the calling thread's
  // current device, and teardown can run on a thread that never selected ours.
  int device_{-1};

  cudaStream_t reader_{nullptr};
  cudaEvent_t streamEvent_{nullptr};
  cudaEvent_t flipEvent_{nullptr};
  cudaEvent_t copyDone_{nullptr};

  // The device-to-host copy lands in page-locked memory (otherwise the copy
  // would be synchronous and the issue path would block the producer thread);
  // published into `snapshot_` by a host memcpy once copyDone_ fires.
  CollStatsPinnedStaging pinned_;
  CollStatSnapshot snapshot_;
  uint64_t localEpoch_{0}; // in lockstep with the device epoch (only we flip)
  uint64_t pendingEpoch_{0}; // epoch of the window currently in flight
  // Wall-clock bounds, captured at the flip rather than at the harvest: the
  // readout is pipelined, so the harvest happens an arbitrary time later and
  // does not bound when the collectives it describes actually ran. Advanced
  // only when an issue succeeds, so a dropped window leaves the next one
  // correctly spanning the gap as well as its own period.
  uint64_t windowOpenNs_{0};
  uint64_t pendingOpenNs_{0};
  uint64_t pendingCloseNs_{0};
  // Collectives against the current bank. Cleared only when an issue succeeds
  // and the epoch actually flips, so a skipped issue leaves them counted.
  uint32_t sinceReadout_{0};
  bool pending_{false}; // a copy into pinned_ is in flight
  bool disabled_{false};

  uint64_t windowsExported_{0};
  // Windows genuinely lost: a CUDA error on the query, a failed stream sync, or
  // a failed issue. With windowsExported_ this partitions windows.
  uint64_t windowsDropped_{0};
  uint64_t harvestRetries_{0};
  uint64_t sinkExceptions_{0};
};

} // namespace meta::comms::collstats
