// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/trace/PipesTrace.h"

#include <pthread.h>

#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"
#include "comms/utils/logger/LogUtils.h"

namespace comms::prims {
namespace {

const char* pipesTraceEventTypeName(uint8_t type) {
  using Type = PipesTraceEventType;
  switch (static_cast<Type>(type)) {
    case Type::kUnknown:
      return "unknown";
    case Type::kHierAgIbChunkBegin:
      return "hier_ag_ib_chunk_begin";
    case Type::kHierAgIbChunkReady:
      return "hier_ag_ib_chunk_ready";
    case Type::kHierAgNvlWaitBegin:
      return "hier_ag_nvl_wait_begin";
    case Type::kHierAgNvlChunkReady:
      return "hier_ag_nvl_chunk_ready";
    case Type::kHierAgNvlTaskDone:
      return "hier_ag_nvl_task_done";
    case Type::kIbSendBegin:
      return "ib_send_begin";
    case Type::kIbSendEnd:
      return "ib_send_end";
    case Type::kIbRecvBegin:
      return "ib_recv_begin";
    case Type::kIbRecvEnd:
      return "ib_recv_end";
    case Type::kIbForwardBegin:
      return "ib_forward_begin";
    case Type::kIbForwardEnd:
      return "ib_forward_end";
  }
  return "unknown";
}

} // namespace

PipesTrace::~PipesTrace() {
  // CTRAN teardown relies on the same lifetime contract as other comm-owned
  // device resources; see deviceHandle(). Teardown here only stops the poller
  // and drains, without synchronizing the device itself.
  stopPollThread();
  // The poller is stopped, so this is now the only reader: flush and log
  // whatever the kernels wrote since the last poll tick before the ring is
  // torn down.
  try {
    drain();
  } catch (const std::exception& ex) {
    CLOGF(WARN, "Prims trace final drain failed: {}", ex.what());
  } catch (...) {
    CLOGF(WARN, "Prims trace final drain failed with unknown exception");
  }
}

uint32_t PipesTrace::normalizeRingSize(uint64_t ringSize) {
  if (ringSize == 0) {
    return 0;
  }

  constexpr uint64_t kMaxRingEntries = 1ULL << 31;
  if (ringSize > kMaxRingEntries) {
    CLOGF(
        WARN,
        "Prims trace clamps ring size {} to {}",
        ringSize,
        kMaxRingEntries);
    return static_cast<uint32_t>(kMaxRingEntries);
  }
  return static_cast<uint32_t>(ringSize);
}

void PipesTrace::ensure(
    uint32_t ringSize,
    std::chrono::milliseconds pollInterval,
    EventCallback eventCallback) {
  if (ringSize == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(drainMutex_);
  if (buffer_ != nullptr && reader_ != nullptr) {
    if (buffer_->size() < ringSize) {
      CLOGF(
          WARN,
          "Prims trace keeps existing ring_size={} despite requested_ring_size={} because device handles may be in flight",
          buffer_->size(),
          ringSize);
    }
    return;
  }

  reader_.reset();
  buffer_ = std::make_unique<Buffer>(ringSize);
  if (!buffer_->valid()) {
    CLOGF(
        WARN, "Prims trace failed to allocate ring with {} entries", ringSize);
    buffer_.reset();
    return;
  }

  reader_ = std::make_unique<Reader>(*buffer_);
  ::hrdw_ring_buffer::GlobaltimerCalibration::get();
  // Set before starting the poller; the poll thread reads pollInterval_ only
  // after this store and it is never mutated again for this ring.
  pollInterval_ = pollInterval;
  eventCallback_ = std::move(eventCallback);
  startPollThread();
  CLOGF(
      INFO,
      "Prims trace buffer ready ring_size={} poll_interval_ms={}",
      buffer_->size(),
      static_cast<long long>(pollInterval_.count()));
}

PipesTraceHandle PipesTrace::deviceHandle() const {
  std::lock_guard<std::mutex> lock(drainMutex_);
  if (buffer_ == nullptr) {
    return {};
  }
  auto handle = buffer_->deviceHandle();
  return PipesTraceHandle{
      .ring = reinterpret_cast<PipesTraceEntry*>(handle.ring),
      .writeIndex = handle.writeIndex,
      .mask = handle.mask,
      .shift = handle.shift};
}

void PipesTrace::drain() {
  PendingLogBatch batch;
  {
    std::lock_guard<std::mutex> lock(drainMutex_);
    if (reader_ == nullptr) {
      return;
    }

    // Poll and copy entries under the lock, then format and log them outside
    // it. The trace is low-rate and the poll thread is our own (it blocks no
    // CUDA stream), so logging inline is cheap and keeps the design single-
    // threaded.
    auto result = reader_->poll([&](const auto& entry, uint64_t slot) {
      batch.entries.push_back(PendingLogEntry{entry, slot});
    });
    batch.entriesLost = result.entriesLost;
  }
  logBatch(batch);
}

void PipesTrace::logBatch(const PendingLogBatch& batch) const {
  auto& calibration = ::hrdw_ring_buffer::GlobaltimerCalibration::get();
  for (const auto& pendingEntry : batch.entries) {
    const auto& entry = pendingEntry.entry;
    const auto wallTime = calibration.toWallClock(entry.timestamp);
    const auto wallTimeNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            wallTime.time_since_epoch())
            .count();
    const auto& event = entry.data;
    CLOGF(
        INFO,
        "Prims trace event={} step={} rank={} detail={} slot={} wall_time_ns={}",
        pipesTraceEventTypeName(event.type),
        event.step,
        static_cast<int>(event.rank),
        event.detail,
        pendingEntry.slot,
        wallTimeNs);
    if (eventCallback_ != nullptr) {
      eventCallback_(event, pendingEntry.slot);
    }
  }

  if (batch.entriesLost != 0) {
    CLOGF(WARN, "Prims trace lost {} entries", batch.entriesLost);
  }
}

void PipesTrace::pollLoop() {
  pthread_setname_np(pthread_self(), "PrimsTracePoll");
  while (true) {
    {
      std::unique_lock<std::mutex> lock(pollMutex_);
      // Wake early on shutdown; otherwise drain once per interval.
      if (pollWake_.wait_for(
              lock, pollInterval_, [&] { return stopPolling_; })) {
        return;
      }
    }
    try {
      drain();
    } catch (const std::exception& ex) {
      CLOGF(WARN, "Prims trace poll drain failed: {}", ex.what());
    } catch (...) {
      CLOGF(WARN, "Prims trace poll drain failed with unknown exception");
    }
  }
}

void PipesTrace::startPollThread() {
  if (pollInterval_.count() <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(pollMutex_);
  if (pollThread_.joinable()) {
    return;
  }
  stopPolling_ = false;
  pollThread_ = std::thread([this] { pollLoop(); });
}

void PipesTrace::stopPollThread() {
  {
    std::lock_guard<std::mutex> lock(pollMutex_);
    stopPolling_ = true;
  }
  pollWake_.notify_one();
  if (pollThread_.joinable()) {
    pollThread_.join();
  }
}

} // namespace comms::prims
