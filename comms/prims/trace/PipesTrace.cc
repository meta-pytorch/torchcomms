// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/trace/PipesTrace.h"

#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

#include <fmt/format.h>

#include "comms/utils/hrdw_ring_buffer/GpuClockCalibration.h"

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
    case Type::kAllReduceTreeSchedulerIdleBegin:
      return "allreduce_tree_scheduler_idle_begin";
    case Type::kAllReduceTreeSchedulerIdleEnd:
      return "allreduce_tree_scheduler_idle_end";
  }
  return "unknown";
}

bool isFineAllReduceEvent(uint8_t type) {
  using Type = PipesTraceEventType;
  return type >= static_cast<uint8_t>(Type::kAllReduceRingReduceScatterBegin) &&
      type <= static_cast<uint8_t>(Type::kAllReduceTreeSchedulerIdleEnd);
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

const char* allReduceRoleName(uint32_t role) {
  using Role = PipesTraceAllReduceRole;
  switch (static_cast<Role>(role)) {
    case Role::Send:
      return "send";
    case Role::RecvCopy:
      return "recv_copy";
    case Role::RecvReduce:
      return "recv_reduce";
    case Role::ForwardCopy:
      return "forward_copy";
    case Role::ForwardReduce:
      return "forward_reduce";
    case Role::Scheduler:
      return "scheduler";
    case Role::Envelope:
      return "envelope";
    case Role::Reserved:
      return "reserved";
  }
  return "unknown";
}

uint64_t allocatePipesTraceSessionId() {
  static std::atomic<uint64_t> nextPipesTraceSessionId{1};
  return nextPipesTraceSessionId.fetch_add(1, std::memory_order_relaxed);
}

void emitWarning(
    const PipesTrace::WarningCallback& warningCallback,
    std::string_view message) noexcept {
  try {
    if (warningCallback) {
      warningCallback(message);
      return;
    }
  } catch (...) {
    /*
     * The callback may have emitted before throwing. Falling back can
     * duplicate that warning, but suppressing the fallback could lose the
     * diagnostic.
     */
  }

  flockfile(stderr);
  std::fputs("Prims trace warning: ", stderr);
  std::fwrite(message.data(), sizeof(char), message.size(), stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  funlockfile(stderr);
}

size_t utf8CompletePrefixSize(std::string_view text) noexcept {
  if (text.empty()) {
    return 0;
  }

  size_t sequenceStart = text.size() - 1;
  while (sequenceStart > 0 &&
         (static_cast<unsigned char>(text[sequenceStart]) & 0xC0) == 0x80) {
    --sequenceStart;
  }

  const auto lead = static_cast<unsigned char>(text[sequenceStart]);
  size_t sequenceSize = 1;
  if ((lead & 0xE0) == 0xC0) {
    sequenceSize = 2;
  } else if ((lead & 0xF0) == 0xE0) {
    sequenceSize = 3;
  } else if ((lead & 0xF8) == 0xF0) {
    sequenceSize = 4;
  }
  return sequenceStart + sequenceSize <= text.size() ? text.size()
                                                     : sequenceStart;
}

template <typename... Args>
void emitFormattedWarning(
    const PipesTrace::WarningCallback& warningCallback,
    fmt::format_string<Args...> format,
    Args&&... args) noexcept {
  constexpr size_t kWarningBufferSize = 2048;
  constexpr std::string_view kTruncationMarker = " [truncated]";
  constexpr size_t kPayloadSize = kWarningBufferSize - kTruncationMarker.size();
  std::array<char, kWarningBufferSize> buffer{};
  try {
    const auto result = fmt::format_to_n(
        buffer.data(), kPayloadSize, format, std::forward<Args>(args)...);
    size_t messageSize = std::min(result.size, kPayloadSize);
    if (result.size > kPayloadSize) {
      messageSize =
          utf8CompletePrefixSize(std::string_view{buffer.data(), messageSize});
      std::copy(
          kTruncationMarker.begin(),
          kTruncationMarker.end(),
          buffer.begin() + messageSize);
      messageSize += kTruncationMarker.size();
    }
    emitWarning(warningCallback, std::string_view{buffer.data(), messageSize});
  } catch (...) {
    emitWarning(warningCallback, "Prims trace warning formatting failed");
  }
}

void emitExceptionWarning(
    const PipesTrace::WarningCallback& warningCallback,
    const char* prefix,
    const std::exception& ex) noexcept {
  emitFormattedWarning(warningCallback, "{}: {}", prefix, ex.what());
}

} // namespace

PipesTrace::PipesTrace(WarningCallback warningCallback)
    : warningCallback_(std::move(warningCallback)),
      sessionId_(allocatePipesTraceSessionId()) {}

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
    emitExceptionWarning(
        warningCallback_, "Prims trace final drain failed", ex);
  } catch (...) {
    emitWarning(
        warningCallback_,
        "Prims trace final drain failed with unknown exception");
  }
}

uint32_t PipesTrace::normalizeRingSize(
    uint64_t ringSize,
    const WarningCallback& warningCallback) {
  if (ringSize == 0) {
    return 0;
  }

  constexpr uint64_t kMaxRingEntries = 1ULL << 31;
  if (ringSize > kMaxRingEntries) {
    emitFormattedWarning(
        warningCallback,
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
    EventCallback eventCallback,
    uint32_t rank) {
  if (ringSize == 0) {
    return;
  }

  enum class EnsureOutcome {
    NoChange,
    Initialized,
    ExistingRingTooSmall,
    AllocationFailure,
  };
  struct EnsureResult {
    EnsureOutcome outcome{EnsureOutcome::NoChange};
    uint32_t ringSize{0};
  };
  EnsureResult result;
  {
    std::lock_guard<std::mutex> lock(drainMutex_);
    rank_ = rank;
    if (buffer_ != nullptr && reader_ != nullptr) {
      if (buffer_->size() < ringSize) {
        result = {
            .outcome = EnsureOutcome::ExistingRingTooSmall,
            .ringSize = buffer_->size()};
      }
    } else {
      reader_.reset();
      buffer_ = std::make_unique<Buffer>(ringSize);
      if (!buffer_->valid()) {
        buffer_.reset();
        result.outcome = EnsureOutcome::AllocationFailure;
      } else {
        reader_ = std::make_unique<Reader>(*buffer_);
        ::hrdw_ring_buffer::GlobaltimerCalibration::get();
        // Set before starting the poller; the poll thread reads pollInterval_
        // only after this store and it is never mutated again for this ring.
        pollInterval_ = pollInterval;
        eventCallback_ = std::move(eventCallback);
        startPollThread();
        result = {
            .outcome = EnsureOutcome::Initialized, .ringSize = buffer_->size()};
      }
    }
  }

  switch (result.outcome) {
    case EnsureOutcome::NoChange:
      return;
    case EnsureOutcome::Initialized:
      /*
       * Successful initialization and trace records remain an unsuppressed,
       * parseable stderr stream; the injected callback is warning-only.
       */
      flockfile(stderr);
      std::fprintf(
          stderr,
          "Prims trace buffer ready trace_session=%llu rank=%u ring_size=%u poll_interval_ms=%lld\n",
          static_cast<unsigned long long>(sessionId_),
          static_cast<unsigned int>(rank),
          static_cast<unsigned int>(result.ringSize),
          static_cast<long long>(pollInterval.count()));
      std::fflush(stderr);
      funlockfile(stderr);
      return;
    case EnsureOutcome::ExistingRingTooSmall:
      emitFormattedWarning(
          warningCallback_,
          "Prims trace keeps existing ring_size={} despite requested_ring_size={} because device handles may be in flight",
          result.ringSize,
          ringSize);
      return;
    case EnsureOutcome::AllocationFailure:
      emitFormattedWarning(
          warningCallback_,
          "Prims trace failed to allocate ring with {} entries",
          ringSize);
      return;
  }
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
    batch.rank = rank_;
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
      const uint32_t role =
          (packed >> kPipesTraceRoleShift) & kPipesTraceRoleMask;
      std::fprintf(
          stderr,
          "Prims fine trace schema_version=%u trace_session=%llu event=%s rank=%u op_tag=%u phase=%s dependency_step=%u block=%u lane=%u chunk_tag=%u role=%s peer=%u qp_lane=%u bytes=%u slot=%llu wall_time_ns=%lld\n",
          kPipesTraceFineSchemaVersion,
          static_cast<unsigned long long>(sessionId_),
          pipesTraceEventTypeName(event.type),
          static_cast<unsigned int>(batch.rank),
          (packed >> kPipesTraceOpTagShift) & kPipesTraceOpTagMask,
          allReducePhaseName(phase),
          (packed >> kPipesTraceDependencyStepShift) &
              kPipesTraceDependencyStepMask,
          (packed >> kPipesTraceBlockShift) & kPipesTraceBlockMask,
          (packed >> kPipesTraceLaneShift) & kPipesTraceLaneMask,
          (packed >> kPipesTraceChunkShift) & kPipesTraceChunkMask,
          allReduceRoleName(role),
          static_cast<unsigned int>(event.rank),
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
    emitFormattedWarning(
        warningCallback_, "Prims trace lost {} entries", batch.entriesLost);
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
      emitExceptionWarning(
          warningCallback_, "Prims trace poll drain failed", ex);
    } catch (...) {
      emitWarning(
          warningCallback_,
          "Prims trace poll drain failed with unknown exception");
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
