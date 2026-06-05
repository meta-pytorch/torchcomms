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

  PipesTrace() = default;
  ~PipesTrace();
  PipesTrace(const PipesTrace&) = delete;
  PipesTrace& operator=(const PipesTrace&) = delete;
  PipesTrace(PipesTrace&&) = delete;
  PipesTrace& operator=(PipesTrace&&) = delete;

  static uint32_t normalizeRingSize(uint64_t ringSize);

  using EventCallback =
      std::function<void(const PipesTraceEvent& event, uint64_t slot)>;

  // Allocate the ring (if needed) and start the background poll thread.
  void ensure(
      uint32_t ringSize,
      std::chrono::milliseconds pollInterval,
      EventCallback eventCallback = nullptr);

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
  };

  void logBatch(const PendingLogBatch& batch) const;
  void drain();
  void pollLoop();
  void startPollThread();
  void stopPollThread();

  std::unique_ptr<Buffer> buffer_;
  std::unique_ptr<Reader> reader_;
  mutable std::mutex drainMutex_;
  std::thread pollThread_;
  std::mutex pollMutex_;
  std::condition_variable pollWake_;
  std::chrono::milliseconds pollInterval_{0};
  EventCallback eventCallback_;
  bool stopPolling_{false};
};

} // namespace comms::prims
