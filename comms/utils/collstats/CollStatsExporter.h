// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "comms/utils/collstats/CollStatsIdentity.h"
#include "comms/utils/collstats/CollStatsSnapshot.h"

// Out-of-band, asynchronous exporter for readout windows. The producer thread
// hands a snapshot to submit() — a cheap bounded-queue enqueue that never
// blocks — and a background thread serializes it (tagging it with the export
// identity) and passes the blob to a pluggable sink. Serialization and I/O
// therefore never touch the producer/training thread. The background thread
// does no CUDA, so it carries none of the poll-thread CUDA hazards.
//
// The queue is bounded and lossy: if the sink cannot keep up, snapshots are
// dropped and counted rather than stalling training — every export hop is async
// and droppable, and every drop is counted.
//
// Both what a window turns into and where it goes are the caller's choice: this
// class owns the thread, the bounded queue and the loss accounting, and knows
// nothing about the wire format. That is what lets a second reporting backend
// reuse the transport instead of reimplementing it.

namespace meta::comms::collstats {

class CollStatsExporter {
 public:
  // Receives one window's serialized blob. Called only on the background
  // thread.
  using JsonSink = std::function<void(const std::string&)>;

  // Turns one window plus its identity into the blob handed to the sink.
  // Called only on the background thread, so it may be as expensive as the
  // format requires. A throw is caught and counted as a failed window.
  using Serializer = std::function<
      std::string(const CollStatsExportIdentity&, const CollStatSnapshot&)>;

  CollStatsExporter(
      CollStatsExportIdentity identity,
      Serializer serializer,
      JsonSink sink,
      std::size_t queueCapacity = 16);
  ~CollStatsExporter();

  CollStatsExporter(const CollStatsExporter&) = delete;
  CollStatsExporter& operator=(const CollStatsExporter&) = delete;
  // Non-movable as well: the background thread captures `this`, so moving the
  // object would leave that thread reading a moved-from queue and mutex.
  CollStatsExporter(CollStatsExporter&&) = delete;
  CollStatsExporter& operator=(CollStatsExporter&&) = delete;

  // Enqueue a window for export. Non-blocking; drops and counts if the queue is
  // full. The snapshot is moved in.
  //
  // Throws only if the enqueue itself cannot allocate, which is deliberate: the
  // caller is the training thread, and a telemetry path that swallows
  // std::bad_alloc there would be hiding an out-of-memory condition the job
  // cannot continue through anyway. Every other loss is counted, not thrown.
  void submit(CollStatSnapshot snapshot);

  // Stop accepting windows, drain what is queued within `budget`, and join the
  // background thread. Idempotent.
  //
  // Call this before reading the counters. Windows abandoned at the deadline
  // are charged to dropped() here, so only a caller that shuts down explicitly
  // can observe the teardown loss -- the destructor's own call runs after the
  // last chance to read, which is where a teardown drop would go unreported.
  //
  // `budget` bounds how many slow sink calls teardown waits through, not how
  // long any one of them may take; see kDrainBudget.
  void shutdown(std::chrono::steady_clock::duration budget = kDrainBudget);

  uint64_t exported() const {
    return exported_.load(std::memory_order_relaxed);
  }
  uint64_t dropped() const {
    return dropped_.load(std::memory_order_relaxed);
  }
  // Windows that reached the exporter thread but produced no output: the sink
  // or the serializer threw, or either is absent. Distinct from dropped(),
  // which counts windows that never got that far.
  uint64_t failed() const {
    return failed_.load(std::memory_order_relaxed);
  }

 private:
  void run();

  CollStatsExportIdentity identity_;
  Serializer serializer_;
  JsonSink sink_;
  const std::size_t capacity_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<CollStatSnapshot> queue_; // guarded by mu_
  bool stop_{false}; // guarded by mu_

  std::atomic<uint64_t> exported_{0};
  std::atomic<uint64_t> dropped_{0};
  std::atomic<uint64_t> failed_{0};

  // How long the destructor lets the background thread keep pushing queued
  // windows through the sink before abandoning the rest. The destructor joins
  // that thread, so an unbounded drain would let a wedged sink stall teardown.
  static constexpr std::chrono::seconds kDrainBudget{2};
  std::chrono::steady_clock::time_point drainDeadline_{}; // guarded by mu_

  std::thread thread_;
};

// A JSON sink that appends each blob as one line (JSON Lines) to `path`. The
// file is opened once and shared with the returned callable. Returns an empty
// (falsy) std::function if the open fails — the caller must test it and report
// the failure, because an exporter built on an empty sink writes nothing and
// would otherwise be indistinguishable from one that had no windows to write.
// Intended default sink for file-per-rank export.
CollStatsExporter::JsonSink collStatsMakeFileSink(const std::string& path);

} // namespace meta::comms::collstats
