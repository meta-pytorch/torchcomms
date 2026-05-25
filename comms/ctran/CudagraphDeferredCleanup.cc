// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CudagraphDeferredCleanup.h"

#include <utility>

#include "comms/utils/logger/LogUtils.h"

void CudagraphDeferredCleanup::add(
    std::function<void()> fn,
    CleanupReadyFlag cleanupReady) {
  entries_.wlock()->push_back(
      Entry{
          .fn = std::move(fn),
          .cleanupReady = std::move(cleanupReady),
      });
}

void CudagraphDeferredCleanup::run(bool force) {
  std::vector<Entry> entriesToRun;
  size_t forcedNotReadyCount = 0;
  {
    auto entries = entries_.wlock();
    std::vector<Entry> pending;
    pending.reserve(entries->size());

    for (auto& entry : *entries) {
      const bool hasReadinessFlag = entry.cleanupReady != nullptr;
      const bool cleanupReady = hasReadinessFlag &&
          entry.cleanupReady->load(std::memory_order_acquire);
      if (cleanupReady || force) {
        if (force && hasReadinessFlag && !cleanupReady) {
          forcedNotReadyCount++;
        }
        entriesToRun.push_back(std::move(entry));
      } else {
        pending.push_back(std::move(entry));
      }
    }
    *entries = std::move(pending);
  }

  if (forcedNotReadyCount > 0) {
    CLOGF(
        WARN,
        "CTRAN-CUDAGRAPH: forced cleanup of {} resources that were not marked cleanup-ready",
        forcedNotReadyCount);
  }

  for (auto& entry : entriesToRun) {
    entry.fn();
  }
}
