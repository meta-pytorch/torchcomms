// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/trace/PipesTrace.h"

#include <pthread.h>

#include <chrono>
#include <cstdio>
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
    case Type::kAllReducePhase1Begin:
      return "allreduce_phase1_begin";
    case Type::kAllReducePhase1End:
      return "allreduce_phase1_end";
    case Type::kAllReducePhase2Begin:
      return "allreduce_phase2_begin";
    case Type::kAllReducePhase2End:
      return "allreduce_phase2_end";
    case Type::kAllReducePhase3Begin:
      return "allreduce_phase3_begin";
    case Type::kAllReducePhase3End:
      return "allreduce_phase3_end";
    case Type::kAllReduceRingReduceScatterBegin:
      return "allreduce_ring_rs_begin";
    case Type::kAllReduceRingReduceScatterEnd:
      return "allreduce_ring_rs_end";
    case Type::kAllReduceRingAllGatherBegin:
      return "allreduce_ring_ag_begin";
    case Type::kAllReduceRingAllGatherEnd:
      return "allreduce_ring_ag_end";
    case Type::kAllReduceSendSyncBegin:
      return "allreduce_send_sync_begin";
    case Type::kAllReduceSendSyncEnd:
      return "allreduce_send_sync_end";
    case Type::kAllReduceSlotPrepareBegin:
      return "allreduce_slot_prepare_begin";
    case Type::kAllReduceSlotPrepareEnd:
      return "allreduce_slot_prepare_end";
    case Type::kAllReduceWqeSubmitBegin:
      return "allreduce_wqe_submit_begin";
    case Type::kAllReduceWqeSubmitEnd:
      return "allreduce_wqe_submit_end";
    case Type::kAllReduceDataReadyWaitBegin:
      return "allreduce_data_ready_wait_begin";
    case Type::kAllReduceDataReadyWaitEnd:
      return "allreduce_data_ready_wait_end";
    case Type::kAllReduceReduceCopyBegin:
      return "allreduce_reduce_copy_begin";
    case Type::kAllReduceReduceCopyEnd:
      return "allreduce_reduce_copy_end";
    case Type::kAllReduceDrainBegin:
      return "allreduce_drain_begin";
    case Type::kAllReduceDrainEnd:
      return "allreduce_drain_end";
    case Type::kAllReduceBookkeepingBegin:
      return "allreduce_bookkeeping_begin";
    case Type::kAllReduceBookkeepingEnd:
      return "allreduce_bookkeeping_end";
    case Type::kAllReduceLocalCompletionWaitBegin:
      return "allreduce_local_completion_wait_begin";
    case Type::kAllReduceLocalCompletionWaitEnd:
      return "allreduce_local_completion_wait_end";
    case Type::kAllReduceRemoteSlotFreeWaitBegin:
      return "allreduce_remote_slot_free_wait_begin";
    case Type::kAllReduceRemoteSlotFreeWaitEnd:
      return "allreduce_remote_slot_free_wait_end";
    case Type::kAllReduceStageCopyBegin:
      return "allreduce_stage_copy_begin";
    case Type::kAllReduceStageCopyEnd:
      return "allreduce_stage_copy_end";
    case Type::kAllReducePathStaged:
      return "allreduce_path_staged";
    case Type::kAllReducePathRegisteredProgress:
      return "allreduce_path_registered_progress";
  }
  return "unknown";
}

bool isFineAllReduceEvent(uint8_t type) {
  using Type = PipesTraceEventType;
  return type >= static_cast<uint8_t>(Type::kAllReduceSendSyncBegin) &&
      type <= static_cast<uint8_t>(Type::kAllReducePathRegisteredProgress);
}

const char* allReducePhaseName(uint32_t phase) {
  using Phase = PipesTraceAllReducePhase;
  switch (static_cast<Phase>(phase)) {
    case Phase::RingReduceScatter:
      return "ring_reduce_scatter";
    case Phase::RingAllGather:
      return "ring_all_gather";
    case Phase::TreeReduce:
      return "tree_reduce";
    case Phase::TreeBroadcast:
      return "tree_broadcast";
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
  std::fprintf(
      stderr,
      "Prims trace buffer ready ring_size=%u poll_interval_ms=%lld\n",
      static_cast<unsigned int>(buffer_->size()),
      static_cast<long long>(pollInterval_.count()));
  std::fflush(stderr);
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
    if (isFineAllReduceEvent(event.type)) {
      const uint32_t packed = event.step;
      const uint32_t phase =
          (packed >> kPipesTracePhaseShift) & kPipesTracePhaseMask;
      std::fprintf(
          stderr,
          "Prims fine trace event=%s rank=%u phase=%s dependency_step=%u stripe=%u lane=%u peer=%u qp_lane=%u bytes=%u slot=%llu wall_time_ns=%lld\n",
          pipesTraceEventTypeName(event.type),
          static_cast<unsigned int>(event.rank),
          allReducePhaseName(phase),
          (packed >> kPipesTraceDependencyStepShift) &
              kPipesTraceDependencyStepMask,
          (packed >> kPipesTraceStripeShift) & kPipesTraceStripeMask,
          (packed >> kPipesTraceLaneShift) & kPipesTraceLaneMask,
          (packed >> kPipesTracePeerShift) & kPipesTracePeerMask,
          (packed >> kPipesTraceQpLaneShift) & kPipesTraceQpLaneMask,
          static_cast<unsigned int>(event.detail) * kPipesTraceBytesQuantum,
          static_cast<unsigned long long>(pendingEntry.slot),
          static_cast<long long>(wallTimeNs));
      if (eventCallback_ != nullptr) {
        eventCallback_(event, pendingEntry.slot);
      }
      continue;
    }
    std::fprintf(
        stderr,
        "Prims trace event=%s step=%u rank=%u detail=%u slot=%llu wall_time_ns=%lld\n",
        pipesTraceEventTypeName(event.type),
        event.step,
        static_cast<unsigned int>(event.rank),
        event.detail,
        static_cast<unsigned long long>(pendingEntry.slot),
        static_cast<long long>(wallTimeNs));
    if (eventCallback_ != nullptr) {
      eventCallback_(event, pendingEntry.slot);
    }
  }

  if (batch.entriesLost != 0) {
    CLOGF(WARN, "Prims trace lost {} entries", batch.entriesLost);
  }
  if (!batch.entries.empty()) {
    std::fflush(stderr);
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
