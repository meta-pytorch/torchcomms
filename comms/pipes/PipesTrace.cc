// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/PipesTrace.h"

#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

#include "comms/utils/GpuClockCalibration.h"
#include "comms/utils/logger/LogUtils.h"

namespace comms::pipes {
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
  }
  return "unknown";
}

void CUDART_CB drainPipesTraceCallback(void* userData) {
  static_cast<PipesTrace*>(userData)->drainFromCallback();
}

} // namespace

PipesTrace::~PipesTrace() {
  {
    std::unique_lock<std::mutex> lock(callbacksMutex_);
    shuttingDown_ = true;
    // Drain callbacks are stream-ordered after traced kernels, so this keeps
    // the ring alive until outstanding kernel writes and callback users finish.
    callbacksDone_.wait(lock, [&] { return pendingCallbacks_ == 0; });
  }
  {
    std::lock_guard<std::mutex> lock(drainMutex_);
  }
  stopLogThread();
}

uint32_t PipesTrace::normalizeRingSize(uint64_t ringSize) {
  if (ringSize == 0) {
    return 0;
  }

  constexpr uint64_t kMaxRingEntries = 1ULL << 31;
  if (ringSize > kMaxRingEntries) {
    CLOGF(
        WARN,
        "Pipes trace clamps ring size {} to {}",
        ringSize,
        kMaxRingEntries);
    return static_cast<uint32_t>(kMaxRingEntries);
  }
  return static_cast<uint32_t>(ringSize);
}

void PipesTrace::ensure(uint32_t ringSize) {
  if (ringSize == 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(callbacksMutex_);
    if (shuttingDown_) {
      return;
    }
  }

  std::lock_guard<std::mutex> lock(drainMutex_);
  if (buffer_ != nullptr && reader_ != nullptr) {
    if (buffer_->size() < ringSize) {
      CLOGF(
          WARN,
          "Pipes trace keeps existing ring_size={} despite requested_ring_size={} because device handles may be in flight",
          buffer_->size(),
          ringSize);
    }
    return;
  }

  reader_.reset();
  buffer_ = std::make_unique<Buffer>(ringSize);
  if (!buffer_->valid()) {
    CLOGF(
        WARN, "Pipes trace failed to allocate ring with {} entries", ringSize);
    buffer_.reset();
    return;
  }

  reader_ = std::make_unique<Reader>(*buffer_);
  ::meta::comms::colltrace::GlobaltimerCalibration::get();
  startLogThread();
  CLOGF(INFO, "Pipes trace buffer ready ring_size={}", buffer_->size());
}

PipesTraceHandle PipesTrace::deviceHandle() const {
  std::lock_guard<std::mutex> lock(drainMutex_);
  if (buffer_ == nullptr) {
    return {};
  }
  return buffer_->deviceHandle();
}

void PipesTrace::drain() {
  PendingLogBatch batch;
  {
    std::lock_guard<std::mutex> lock(drainMutex_);
    if (reader_ == nullptr) {
      return;
    }

    // Keep CUDA host callbacks short: poll and copy entries here, then let the
    // logger thread do per-entry timestamp conversion and formatting.
    auto result = reader_->poll([&](const auto& entry, uint64_t slot) {
      batch.entries.push_back(PendingLogEntry{entry, slot});
    });
    batch.entriesRead = result.entriesRead;
    batch.entriesLost = result.entriesLost;
    batch.lastRead = reader_->lastReadIndex();
  }
  enqueueLogBatch(std::move(batch));
}

void PipesTrace::enqueueLogBatch(PendingLogBatch batch) {
  {
    std::lock_guard<std::mutex> lock(logMutex_);
    if (stopLogging_) {
      return;
    }
    pendingLogBatches_.push_back(std::move(batch));
  }
  logAvailable_.notify_one();
}

void PipesTrace::logBatch(const PendingLogBatch& batch) const {
  auto& calibration = ::meta::comms::colltrace::GlobaltimerCalibration::get();
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
        "Pipes trace event={} step={} rank={} detail={} slot={} wall_time_ns={}",
        pipesTraceEventTypeName(event.type),
        event.step,
        static_cast<int>(event.rank),
        event.detail,
        pendingEntry.slot,
        wallTimeNs);
  }

  if (batch.entriesLost != 0) {
    CLOGF(WARN, "Pipes trace lost {} entries", batch.entriesLost);
  }
  CLOGF(
      INFO,
      "Pipes trace drain entries_read={} entries_lost={} last_read={}",
      batch.entriesRead,
      batch.entriesLost,
      batch.lastRead);
}

void PipesTrace::logLoop() {
  while (true) {
    PendingLogBatch batch;
    {
      std::unique_lock<std::mutex> lock(logMutex_);
      logAvailable_.wait(
          lock, [&] { return stopLogging_ || !pendingLogBatches_.empty(); });
      if (pendingLogBatches_.empty()) {
        if (stopLogging_) {
          return;
        }
        continue;
      }
      batch = std::move(pendingLogBatches_.front());
      pendingLogBatches_.pop_front();
    }
    logBatch(batch);
  }
}

void PipesTrace::startLogThread() {
  std::lock_guard<std::mutex> lock(logMutex_);
  if (logThread_.joinable()) {
    return;
  }
  stopLogging_ = false;
  logThread_ = std::thread([this] { logLoop(); });
}

void PipesTrace::stopLogThread() {
  {
    std::lock_guard<std::mutex> lock(logMutex_);
    stopLogging_ = true;
  }
  logAvailable_.notify_one();
  if (logThread_.joinable()) {
    logThread_.join();
  }
}

void PipesTrace::enqueueDrain(cudaStream_t stream) {
  {
    std::lock_guard<std::mutex> lock(drainMutex_);
    if (reader_ == nullptr) {
      return;
    }
  }

  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  auto captureErr = cudaStreamIsCapturing(stream, &captureStatus);
  if (captureErr != cudaSuccess) {
    CLOGF(
        WARN,
        "Pipes trace failed to query stream capture status: {}",
        cudaGetErrorString(captureErr));
    return;
  }
  if (captureStatus != cudaStreamCaptureStatusNone) {
    return;
  }

  if (!beginDrainCallback()) {
    return;
  }
  auto launchErr = cudaLaunchHostFunc(stream, drainPipesTraceCallback, this);
  if (launchErr != cudaSuccess) {
    finishDrainCallback();
    CLOGF(
        WARN,
        "Pipes trace failed to enqueue drain callback: {}",
        cudaGetErrorString(launchErr));
  }
}

void PipesTrace::drainFromCallback() {
  try {
    drain();
  } catch (const std::exception& ex) {
    CLOGF(WARN, "Pipes trace callback drain failed: {}", ex.what());
  } catch (...) {
    CLOGF(WARN, "Pipes trace callback drain failed with unknown exception");
  }
  finishDrainCallback();
}

bool PipesTrace::beginDrainCallback() {
  std::lock_guard<std::mutex> lock(callbacksMutex_);
  if (shuttingDown_) {
    return false;
  }
  ++pendingCallbacks_;
  return true;
}

void PipesTrace::finishDrainCallback() {
  std::lock_guard<std::mutex> lock(callbacksMutex_);
  --pendingCallbacks_;
  callbacksDone_.notify_all();
}

} // namespace comms::pipes
