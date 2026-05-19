// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include <folly/Synchronized.h>

// Deferred cleanup for CUDA graph resources. CUDA user-object destructor
// callbacks cannot call CUDA APIs, so cleanup is enqueued here and executed
// later where CUDA APIs are safe.
class CudagraphDeferredCleanup {
 public:
  using CleanupReadyFlag = std::shared_ptr<std::atomic<bool>>;

  void add(std::function<void()> fn, CleanupReadyFlag cleanupReady = nullptr);

  // Runs cleanup entries whose CUDA graph owner has been destroyed. When
  // force=true, runs every pending entry and warns for entries with an explicit
  // cleanup flag that was not marked before communicator destruction.
  void run(bool force = false);

 private:
  struct Entry {
    std::function<void()> fn;
    CleanupReadyFlag cleanupReady;
  };

  folly::Synchronized<std::vector<Entry>> entries_;
};
