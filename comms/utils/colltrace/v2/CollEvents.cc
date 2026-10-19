// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/v2/CollEvents.h"

#include <pthread.h>

#include "comms/utils/colltrace/v2/CommState.h"
#include "comms/utils/colltrace/v2/Env.h"
#include "comms/utils/colltrace/v2/Log.h"
#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"

namespace meta::comms::colltracev2 {

CollEvents::CollEvents()
    : interval_(envUint("COLLTRACE_DRAIN_INTERVAL_MS", 100)),
      maxBufferedEvents_(
          static_cast<std::size_t>(
              envUint("COLLTRACE_MAX_BUFFERED_EVENTS", uint64_t{1} << 20))) {
  thread_ = std::thread(&CollEvents::loop, this);
}

CollEvents::~CollEvents() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

void CollEvents::watch(std::shared_ptr<CommState> state) {
  std::lock_guard<std::mutex> lock(watchedMu_);
  const auto index = state->index;
  watched_.emplace(index, std::move(state));
}

void CollEvents::unwatch(uint32_t commIndex) {
  std::lock_guard<std::mutex> lock(watchedMu_);
  watched_.erase(commIndex);
}

std::vector<Event> CollEvents::collectEvents(CommState& state) {
  std::lock_guard<std::mutex> lock(state.mu);
  drainLocked(state);
  std::vector<Event> out(state.events.begin(), state.events.end());
  state.events.clear();
  return out;
}

void CollEvents::drainLocked(CommState& state) {
  const auto& cal = ::hrdw_ring_buffer::GlobaltimerCalibration::get();
  // The ring carries only the low 32 bits of the per-comm counter; the rest
  // comes from the host counter. Ids are minted before any kernel can write
  // them and replays re-emit captured ids, so a drained entry is never ahead of
  // the counter; the second condition keeps the subtraction from wrapping if
  // one ever is.
  const auto restoreLocal = [&state](uint32_t low32) {
    uint64_t local = (state.nextLocalCollId & ~uint64_t{0xffffffff}) | low32;
    if (local >= state.nextLocalCollId && local >= (uint64_t{1} << 32)) {
      local -= uint64_t{1} << 32;
    }
    return local;
  };
  const auto droppedBefore = state.dropped;
  const auto result = state.reader.poll([&](const auto& entry, uint64_t) {
    state.events.push_back(
        Event{
            .collId = makeCollId(state.index, restoreLocal(entry.data.collId)),
            .kind = entry.data.phase == colltrace::GraphCollTracePhase::kStart
                ? EventKind::kCollStartEvent
                : EventKind::kCollEndEvent,
            .ts = cal.toWallClock(entry.timestamp),
        });
    if (state.events.size() > maxBufferedEvents_) {
      state.events.pop_front();
      ++state.dropped;
    }
  });
  if (droppedBefore == 0 && state.dropped > 0) {
    COLLTRACE_LOG_WARN() << "commDesc '" << state.commDesc
                         << "' exceeded COLLTRACE_MAX_BUFFERED_EVENTS="
                         << maxBufferedEvents_
                         << "; dropping oldest events until getCollEvents() "
                            "is called";
  }

  if (result.entriesLost > 0) {
    state.lost += result.entriesLost;
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto last = lastLossLog_.load(std::memory_order_relaxed);
    constexpr auto kThrottle =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::seconds{5})
            .count();
    if (now.count() - last >= kThrottle &&
        lastLossLog_.compare_exchange_strong(
            last, now.count(), std::memory_order_relaxed)) {
      COLLTRACE_LOG_WARN() << "commDesc '" << state.commDesc << "' lost "
                           << result.entriesLost << " ring entries (total "
                           << state.lost
                           << "); drain is too slow or the ring is too small";
    }
  }
}

void CollEvents::loop() {
  pthread_setname_np(pthread_self(), "ColltraceDrain");
  std::unique_lock<std::mutex> lock(mu_);
  while (!stop_) {
    cv_.wait_for(lock, interval_, [this] { return stop_; });
    if (stop_) {
      return;
    }
    lock.unlock();

    std::vector<std::shared_ptr<CommState>> states;
    {
      std::lock_guard<std::mutex> watchedLock(watchedMu_);
      states.reserve(watched_.size());
      for (const auto& [_, state] : watched_) {
        states.push_back(state);
      }
    }
    for (const auto& state : states) {
      std::lock_guard<std::mutex> stateLock(state->mu);
      drainLocked(*state);
    }

    lock.lock();
  }
}

} // namespace meta::comms::colltracev2
