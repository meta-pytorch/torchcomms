// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "comms/utils/colltrace/v2/CollTraceV2.h"

namespace meta::comms::colltracev2 {

struct CommState;

// Owns the collected events and the thread that collects them: every
// COLLTRACE_DRAIN_INTERVAL_MS it pulls each watched comm's ring into that
// comm's host event buffer (oldest dropped past COLLTRACE_MAX_BUFFERED_EVENTS).
// collectEvents() pulls immediately and hands the buffer over, so a caller
// never waits on the timer.
class CollEvents {
 public:
  CollEvents();
  ~CollEvents();

  CollEvents(const CollEvents&) = delete;
  CollEvents& operator=(const CollEvents&) = delete;
  CollEvents(CollEvents&&) = delete;
  CollEvents& operator=(CollEvents&&) = delete;

  void watch(std::shared_ptr<CommState> state);
  void unwatch(uint32_t commIndex);

  std::vector<Event> collectEvents(CommState& state);

 private:
  void drainLocked(CommState& state);
  void loop();

  const std::chrono::milliseconds interval_;
  const std::size_t maxBufferedEvents_;
  std::chrono::steady_clock::time_point lastLossLog_{};

  std::mutex watchedMu_;
  std::unordered_map<uint32_t, std::shared_ptr<CommState>> watched_;

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_{false};
  std::thread thread_;
};

} // namespace meta::comms::colltracev2
