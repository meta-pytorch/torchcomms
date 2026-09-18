// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsExporter.h"

#include <fstream>
#include <ios>
#include <memory>
#include <utility>

namespace meta::comms::collstats {

CollStatsExporter::CollStatsExporter(
    CollStatsExportIdentity identity,
    Serializer serializer,
    JsonSink sink,
    std::size_t queueCapacity)
    : identity_(std::move(identity)),
      serializer_(std::move(serializer)),
      sink_(std::move(sink)),
      capacity_(queueCapacity < 1 ? 1 : queueCapacity) {
  thread_ = std::thread([this]() { run(); });
}

CollStatsExporter::~CollStatsExporter() {
  shutdown();
}

void CollStatsExporter::shutdown(std::chrono::steady_clock::duration budget) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    drainDeadline_ = std::chrono::steady_clock::now() + budget;
  }
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void CollStatsExporter::submit(CollStatSnapshot snapshot) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_ || queue_.size() >= capacity_) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    queue_.push(std::move(snapshot));
  }
  cv_.notify_one();
}

void CollStatsExporter::run() {
  for (;;) {
    CollStatSnapshot snapshot;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this]() { return stop_ || !queue_.empty(); });
      if (queue_.empty()) {
        // Only woken for stop with nothing left to drain.
        return;
      }
      if (stop_ && std::chrono::steady_clock::now() > drainDeadline_) {
        // The teardown drain is time-boxed: the destroying thread reaches this
        // queue through join(), so a slow sink would otherwise stall comm
        // teardown for as long as it likes.
        //
        // The bound is on the queue, not on one call: the deadline is only
        // consulted here, between items, so a single wedged sink call still
        // blocks join() for as long as it runs. Bounding that would need the
        // sink on its own abandonable thread, which is a bigger contract than
        // the queue's -- so the budget caps how many slow calls teardown waits
        // through, not how long any one of them may take.
        dropped_.fetch_add(queue_.size(), std::memory_order_relaxed);
        // Popped rather than swapped against a fresh queue: constructing one
        // allocates, and this runs outside the try below, so a throw here would
        // escape the thread body -- the std::terminate route this function
        // exists to close.
        while (!queue_.empty()) {
          queue_.pop();
        }
        return;
      }
      snapshot = std::move(queue_.front());
      queue_.pop();
    }
    // Serialize and hand off outside the lock so submit() never blocks on I/O.
    // Telemetry must never take the job down: without the catch, a throw from
    // a caller-supplied serializer or sink escapes the thread body and calls
    // std::terminate.
    if (!serializer_ || !sink_) {
      failed_.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    try {
      sink_(serializer_(identity_, snapshot));
      exported_.fetch_add(1, std::memory_order_relaxed);
    } catch (...) {
      failed_.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

CollStatsExporter::JsonSink collStatsMakeFileSink(const std::string& path) {
  auto file = std::make_shared<std::ofstream>(path, std::ios::app);
  if (!file->is_open()) {
    // Empty sink, so the caller can tell the open failed. Returning a
    // silently-discarding sink instead would make a bad output path
    // indistinguishable from a working one that produced no windows.
    return {};
  }
  return [file](const std::string& json) {
    (*file) << json << '\n';
    file->flush();
    // std::ofstream does not throw on a failed write by default: it sets
    // fail/bad and every subsequent write silently no-ops. Unchecked, a disk
    // that fills after the open turns every remaining window into a counted
    // export that reached nothing. Throwing hands it to the exporter's catch,
    // which counts it under failed() -- a visible loss rather than a silent
    // one. No is_open() guard: the factory returns an empty sink when the open
    // failed, and this lambda's shared_ptr is the only owner, so the stream
    // cannot be closed underneath it.
    if (!file->good()) {
      // Cleared before throwing: the fail/bad bits latch, so leaving them set
      // would make every later window throw on this stream's state rather than
      // on its own write, turning one transient failure into a whole run of
      // failed() windows.
      file->clear();
      throw std::ios_base::failure("collstats file sink write failed");
    }
  };
}

} // namespace meta::comms::collstats
