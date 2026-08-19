// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

namespace comms::prims {

class PipesTrace {
 public:
  using Buffer =
      ::hrdw_ring_buffer::HRDWRingBuffer<comms::prims::PipesTraceEvent>;
  using Entry = typename Buffer::Entry;
  using Reader =
      ::hrdw_ring_buffer::HRDWRingBufferReader<comms::prims::PipesTraceEvent>;
  // May run on the internal poll thread; captured state must remain valid for
  // the lifetime of this object and support concurrent invocation. Exceptions
  // are contained and reported to stderr rather than escaping a worker thread
  // or destructor.
  using WarningCallback = std::function<void(std::string_view message)>;

  explicit PipesTrace(WarningCallback warningCallback);
  ~PipesTrace();
  PipesTrace(const PipesTrace&) = delete;
  PipesTrace& operator=(const PipesTrace&) = delete;
  PipesTrace(PipesTrace&&) = delete;
  PipesTrace& operator=(PipesTrace&&) = delete;

  static uint32_t normalizeRingSize(
      uint64_t ringSize,
      const WarningCallback& warningCallback);

  using EventCallback =
      std::function<void(const PipesTraceEvent& event, uint64_t slot)>;

  // Allocate the ring (if needed) and start the background poll thread.
  void ensure(
      uint32_t ringSize,
      std::chrono::milliseconds pollInterval,
      EventCallback eventCallback = nullptr,
      uint32_t rank = 0);

  // Device-side handle into the ring.
  //
  // Lifetime contract: traced kernels write directly through this handle, so
  // the ring it points at must outlive every kernel that was handed it. This is
  // the same lifetime contract as other CTRAN comm-owned device resources:
  // callers must not destroy the communicator until all CTRAN work that may
  // reference those resources has completed. ~PipesTrace() follows that
  // contract and does not synchronize the device itself.
  PipesTraceHandle deviceHandle() const;

 private:
  struct PendingLogEntry {
    Entry entry;
    uint64_t slot;
  };

  struct PendingLogBatch {
    std::vector<PendingLogEntry> entries;
    uint64_t entriesLost{0};
    uint32_t rank{0};
  };

  void logBatch(const PendingLogBatch& batch) const;
  void drain();
  void pollLoop();
  void startPollThread();
  void stopPollThread();
  void warn(std::string_view message) const noexcept;

  WarningCallback warningCallback_;
  std::unique_ptr<Buffer> buffer_;
  std::unique_ptr<Reader> reader_;
  mutable std::mutex drainMutex_;
  std::thread pollThread_;
  std::mutex pollMutex_;
  std::condition_variable pollWake_;
  std::chrono::milliseconds pollInterval_{0};
  EventCallback eventCallback_;
  uint64_t sessionId_{0};
  uint32_t rank_{0};
  bool stopPolling_{false};
};

} // namespace comms::prims
