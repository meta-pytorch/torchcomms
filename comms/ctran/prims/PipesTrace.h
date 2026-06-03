// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "comms/ctran/prims/PipesTraceTypes.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBufferReader.h"

namespace ctran::prims {

class PipesTrace {
 public:
  using Buffer =
      ::hrdw_ring_buffer::HRDWRingBuffer<ctran::prims::PipesTraceEvent>;
  using Entry = typename Buffer::Entry;
  using Reader =
      ::hrdw_ring_buffer::HRDWRingBufferReader<ctran::prims::PipesTraceEvent>;

  PipesTrace() = default;
  ~PipesTrace();
  PipesTrace(const PipesTrace&) = delete;
  PipesTrace& operator=(const PipesTrace&) = delete;
  PipesTrace(PipesTrace&&) = delete;
  PipesTrace& operator=(PipesTrace&&) = delete;

  static uint32_t normalizeRingSize(uint64_t ringSize);

  void ensure(uint32_t ringSize);
  PipesTraceHandle deviceHandle() const;
  void drain();
  void enqueueDrain(cudaStream_t stream);
  void drainFromCallback();

 private:
  struct PendingLogEntry {
    Entry entry;
    uint64_t slot;
  };

  struct PendingLogBatch {
    std::vector<PendingLogEntry> entries;
    uint64_t entriesRead{0};
    uint64_t entriesLost{0};
    uint64_t lastRead{0};
  };

  void enqueueLogBatch(PendingLogBatch batch);
  void logBatch(const PendingLogBatch& batch) const;
  void logLoop();
  void startLogThread();
  void stopLogThread();
  bool beginDrainCallback();
  void finishDrainCallback();

  std::unique_ptr<Buffer> buffer_;
  std::unique_ptr<Reader> reader_;
  mutable std::mutex drainMutex_;
  std::mutex logMutex_;
  std::condition_variable logAvailable_;
  std::deque<PendingLogBatch> pendingLogBatches_;
  std::thread logThread_;
  std::mutex callbacksMutex_;
  std::condition_variable callbacksDone_;
  std::size_t pendingCallbacks_{0};
  bool stopLogging_{false};
  bool shuttingDown_{false};
};

} // namespace ctran::prims
