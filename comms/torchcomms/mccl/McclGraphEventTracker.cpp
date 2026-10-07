// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/mccl/McclGraphEventTracker.hpp"

#include <atomic>
#include <cstdlib>
#include <list>
#include <string>
#include <string_view>

#include <folly/ScopeGuard.h>

#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

namespace {

torch::comms::McclSharedCallbackState* allocateCallbackState() {
  static std::mutex mutex;
  static std::list<torch::comms::McclSharedCallbackState> pool;
  std::lock_guard<std::mutex> lock(mutex);
  pool.emplace_back();
  return &pool.back();
}

// Cached tri-state: -1 = unresolved, 0 = disabled, 1 = enabled. Wrapped in a
// function-local static so it is not a mutable namespace-scope global.
std::atomic<int>& mcclGraphTimeoutMonitoringState() {
  static std::atomic<int> state{-1};
  return state;
}

} // namespace

namespace torch::comms {

bool isMcclGraphTimeoutMonitoringEnabled() {
  int state = mcclGraphTimeoutMonitoringState().load(std::memory_order_relaxed);
  if (state < 0) {
    const char* env = std::getenv("TORCHCOMM_MCCL_GRAPH_TIMEOUT_MONITORING");
    bool enabled = true;
    if (env != nullptr) {
      std::string val(env);
      enabled = (val != "0" && val != "false");
    }
    state = enabled ? 1 : 0;
    mcclGraphTimeoutMonitoringState().store(state, std::memory_order_relaxed);
  }
  return state == 1;
}

void resetMcclGraphTimeoutMonitoringCacheForTest() {
  mcclGraphTimeoutMonitoringState().store(-1, std::memory_order_relaxed);
}

McclGraphEventTracker::McclGraphEventTracker(TorchCommMCCL* comm)
    : comm_(comm) {}

McclGraphEventTracker::~McclGraphEventTracker() {
  destroyAll();
  CudaApi* api = comm_->getCudaApi();
  for (cudaEvent_t event : event_pool_) {
    (void)api->eventDestroy(event);
  }
}

bool McclGraphEventTracker::initOnGraphStart(cudaStream_t stream) {
  // Self-gate: never accumulate per-graph state when monitoring is off, even if
  // a future caller forgets to gate. addEntry() requires a prior successful
  // initOnGraphStart(), so gating here covers both entry points.
  if (!isMcclGraphTimeoutMonitoringEnabled()) {
    return false;
  }
  CudaApi* api = comm_->getCudaApi();

  // Get CUDA stream capture info; no-op if this stream is not being captured.
  cudaStreamCaptureStatus capture_status;
  unsigned long long graph_id;
  cudaGraph_t graph;
  CUDA_CHECK(
      api,
      api->streamGetCaptureInfo_v2(
          stream, &capture_status, &graph_id, &graph, nullptr, nullptr),
      "Failed to get CUDA stream capture info");
  if (capture_status != cudaStreamCaptureStatusActive) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  // Reused at subsequent addEntry().
  current_graph_id_ = graph_id;
  maybeInitGraphState(stream, graph_id, graph);
  return true;
}

void McclGraphEventTracker::maybeInitGraphState(
    cudaStream_t stream,
    unsigned long long graph_id,
    cudaGraph_t graph) {
  // One-time initialization per graph.
  auto [it, inserted] = graphs_.try_emplace(graph_id);
  if (!inserted) {
    return;
  }
  auto& state = it->second;
  state.event_pool_ = &event_pool_;
  state.counter_pool_ = &counter_pool_;

  CudaApi* api = comm_->getCudaApi();

  McclSharedCallbackState* shared = allocateCallbackState();
  state.shared_ = shared;

  // Replay counter kernel node — fires on each replay. Only installed when
  // monitoring is enabled; the cleanup callback is always installed for
  // GraphState lifecycle management.
  if (isMcclGraphTimeoutMonitoringEnabled()) {
    CUDA_CHECK(
        api,
        acquireCounter(state.replay_counter),
        "Failed to acquire replay counter");
    // Route the increment through the comm's graph-monitor side stream so the
    // captured kernel rejoins the graph (avoids an unjoined fork at
    // capture_end) and stays off the main stream's critical path.
    DeviceCounter* counter = state.replay_counter.get();
    cudaError_t increment_err = cudaSuccess;
    CUDA_CHECK(
        api,
        comm_->forkGraphMonitorSideStream(
            stream,
            [counter, &increment_err](cudaStream_t s) {
              increment_err = counter->increment(s);
            }),
        "Failed to fork side stream for replay counter increment");
    CUDA_CHECK(api, increment_err, "Failed to record replay counter increment");
  }

  // Deferred cleanup via a CUDA user object — when the graph is destroyed, the
  // callback sets the released flag; the watchdog's next checkAll() destroys
  // the owned events.
  cudaUserObject_t user_object;
  CUDA_CHECK(
      api,
      api->userObjectCreate(
          &user_object,
          &shared->released,
          cleanupCallback,
          1,
          cudaUserObjectNoDestructorSync),
      "Failed to create user object");

  auto user_obj_guard = folly::makeGuard(
      [api, user_object] { (void)api->userObjectRelease(user_object, 1); });

  CUDA_CHECK(
      api,
      api->graphRetainUserObject(
          graph, user_object, 1, cudaGraphUserObjectMove),
      "Failed to retain user object");

  // graphRetainUserObject succeeded — graph now owns user_object.
  user_obj_guard.dismiss();
}

void McclGraphEventTracker::addEntry(
    cudaStream_t stream,
    cudaEvent_t start_event,
    cudaEvent_t end_event,
    std::chrono::milliseconds timeout) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto [it, inserted] = graphs_.try_emplace(current_graph_id_);

  // The tracker owns the start/end events from here on; they are returned to
  // event_pool_ when the graph state is destroyed.
  if (start_event && end_event) {
    it->second.stream_entries[stream].emplace_back(
        start_event, end_event, timeout);
  }
}

void McclGraphEventTracker::notifyReplayProgress(
    unsigned long long graph_id,
    void* stream,
    size_t collective_index,
    McclGraphWork& entry,
    uint64_t current_replay,
    int current_event) {
  if (comm_->graphReplayHooks_.empty()) {
    return;
  }

  // Per-replay clog event tokens, indexed by event ordinal (kEventS=0,
  // kEventE=1, Wait=2): S = collective start, E = collective end, W = wait.
  // Only S and E are observable during graph replay (W is not), matching the
  // eager-mode clog logging convention.
  static constexpr std::string_view kEvents[] = {"S", "E", "W"};
  // Aborted variants (S!/E!/W!), emitted when the comm's error/abort flag is
  // set at log time.
  static constexpr std::string_view kEventsAborted[] = {"S!", "E!", "W!"};

  if (current_replay < entry.notified_through_replay ||
      (current_replay == entry.notified_through_replay &&
       current_event <= entry.notified_through_event)) {
    return;
  }

  uint64_t r = entry.notified_through_replay;
  int e = entry.notified_through_event + 1;

  if (e > kEventE) {
    r++;
    e = kEventS;
  }

  // Mark events with the aborted variant ("E!" etc.) when the comm's error /
  // abort flag is set at log time (MCCL sets it on a collective self-timeout,
  // after which subsequent collectives on the comm fast-fail), so the clog
  // shows the collective completed under an aborted comm.
  const auto& events = comm_->isAborted() ? kEventsAborted : kEvents;

  for (; r <= current_replay; r++) {
    int end_e = (r == current_replay) ? current_event : kEventE;
    for (; e <= end_e; e++) {
      comm_->fireGraphReplayHook(
          graph_id, r, stream, collective_index, events[e]);
    }
    e = kEventS;
  }

  entry.notified_through_replay = current_replay;
  entry.notified_through_event = current_event;
}

McclGraphEventTracker::CheckResult McclGraphEventTracker::checkAll() {
  if (!isMcclGraphTimeoutMonitoringEnabled()) {
    return CheckResult::OK;
  }

  CudaApi* api = comm_->getCudaApi();
  std::lock_guard<std::mutex> lock(mutex_);

  // Queries a CUDA event; returns false (and logs) on an unexpected CUDA error,
  // in which case the caller returns CheckResult::ERROR. On success `out` is
  // cudaSuccess or cudaErrorNotReady. graph_id/i are passed explicitly so the
  // dependencies are visible at the call site.
  auto query_event = [this, api](
                         cudaError_t call_result,
                         cudaError_t& out,
                         std::string_view event_desc,
                         unsigned long long graph_id,
                         size_t i) -> bool {
    out = call_result;
    if (out != cudaSuccess && out != cudaErrorNotReady) {
      TC_LOG(ERROR, comm_) << "Graph monitor: CUDA error during " << event_desc
                           << " for graph " << graph_id << " collective " << i
                           << ": " << api->getErrorString(out) << " (" << out
                           << ")";
      return false;
    }
    return true;
  };

  // Cleanup released graphs — delivers final replay events before erasing.
  cleanupReleasedGraphs();

  for (auto& [graph_id, graph_state] : graphs_) {
    if (!graph_state.replay_counter) {
      TC_LOG(ERROR, comm_) << "Graph monitor: replay counter is null for graph "
                           << graph_id
                           << " -- expected counter when monitoring is enabled";
      return CheckResult::ERROR;
    }
    uint64_t current_replay = graph_state.replay_counter->read();

    // Collectives are ordered per stream — within each stream, if collective i
    // has not completed, collective i+1 cannot have started, so we can skip the
    // rest of the stream once we find the first incomplete one.
    for (auto& [stream, entries] : graph_state.stream_entries) {
      for (size_t i = 0; i < entries.size(); ++i) {
        auto& entry = entries[i];

        // Detect a new replay — reset timer to avoid a false timeout spanning
        // multiple replays.
        if (current_replay != entry.last_seen_replay) {
          entry.start_completed_time.reset();
          entry.last_seen_replay = current_replay;
        }

        cudaError_t start_status, end_status;
        if (!query_event(
                api->eventQuery(entry.start_event),
                start_status,
                "start event query",
                graph_id,
                i) ||
            !query_event(
                api->eventQuery(entry.end_event),
                end_status,
                "end event query",
                graph_id,
                i)) {
          return CheckResult::ERROR;
        }

        if (end_status == cudaSuccess) {
          entry.start_completed_time.reset();
          notifyReplayProgress(
              graph_id, stream, i, entry, current_replay, kEventE);
          continue;
        }

        // end is notReady — first incomplete collective on this stream.
        if (start_status == cudaSuccess) {
          notifyReplayProgress(
              graph_id, stream, i, entry, current_replay, kEventS);
        }

        // Observability-only by default: entry.timeout is recorded but not
        // enforced (MCCL self-times-out on-device). The next collective on this
        // stream cannot have started, so stop scanning here.
        if (!enforce_timeout_) {
          break;
        }

        if (!entry.start_completed_time.has_value()) {
          if (start_status == cudaSuccess) {
            entry.start_completed_time = std::chrono::steady_clock::now();
          } else {
            // collective NEVER started — skip timer logic
            break;
          }
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() -
            entry.start_completed_time.value());
        if (entry.timeout.count() >= 0 && elapsed > entry.timeout) {
          TC_LOG(ERROR, comm_)
              << "Graph monitor: collective TIMED OUT for graph " << graph_id
              << " collective " << i << " on rank " << comm_->getRank()
              << " - elapsed " << elapsed.count() << "ms > timeout "
              << entry.timeout.count() << "ms";
          return CheckResult::TIMEOUT;
        }
        break;
      }
    }
  }
  return CheckResult::OK;
}

void McclGraphEventTracker::cleanupReleasedGraphs() {
  CudaApi* api = comm_->getCudaApi();

  for (auto it = graphs_.begin(); it != graphs_.end();) {
    if (!it->second.shared_->released.load(std::memory_order_relaxed)) {
      ++it;
      continue;
    }

    // Deliver final replay events before erasing. The graph is destroyed so no
    // more replays will occur; sweep all completed events not yet reported.
    auto& graph_state = it->second;
    if (graph_state.replay_counter) {
      uint64_t current_replay = graph_state.replay_counter->read();
      for (auto& [stream, entries] : graph_state.stream_entries) {
        for (size_t i = 0; i < entries.size(); ++i) {
          auto& entry = entries[i];
          cudaError_t end_status = api->eventQuery(entry.end_event);
          if (end_status == cudaSuccess) {
            notifyReplayProgress(
                it->first, stream, i, entry, current_replay, kEventE);
          } else if (end_status != cudaErrorNotReady) {
            // Best-effort cleanup still erases the graph below, but surface an
            // unexpected CUDA error rather than dropping it silently (mirrors
            // checkAll()'s EVENT_QUERY_CHECK discipline).
            TC_LOG(ERROR, comm_)
                << "Graph monitor: CUDA error during cleanup end event query "
                << "for graph " << it->first << " collective " << i << ": "
                << api->getErrorString(end_status) << " (" << end_status << ")";
          }
        }
      }
    }

    it = graphs_.erase(it);
  }
}

cudaError_t McclGraphEventTracker::acquireCounter(
    std::unique_ptr<DeviceCounter>& out) {
  // Caller must hold mutex_: this touches the shared counter_pool_ unguarded
  // (unlike acquireEvent, which locks). The only caller, maybeInitGraphState,
  // runs under the lock taken in initOnGraphStart; locking here would deadlock.
  if (!counter_pool_.empty()) {
    out = std::move(counter_pool_.back());
    counter_pool_.pop_back();
    out->reset();
    return cudaSuccess;
  }
  return DeviceCounter::create(comm_->getCudaApi(), out);
}

cudaError_t McclGraphEventTracker::acquireEvent(cudaEvent_t& out) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!event_pool_.empty()) {
      out = event_pool_.back();
      event_pool_.pop_back();
      return cudaSuccess;
    }
  }
  // Pool empty: create outside the lock (touches no shared state).
  return comm_->getCudaApi()->eventCreateWithFlags(
      &out, cudaEventDisableTiming);
}

void McclGraphEventTracker::releaseEvent(cudaEvent_t event) {
  if (event == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  event_pool_.push_back(event);
}

void McclGraphEventTracker::destroyAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Deliver final replay events for any released graphs before destroying
  // state (covers graphs destroyed after the watchdog has exited).
  cleanupReleasedGraphs();
  graphs_.clear();
}

// Static callback — fires when the graph is destroyed to set the released flag.
void CUDART_CB McclGraphEventTracker::cleanupCallback(void* userData) {
  static_cast<std::atomic_bool*>(userData)->store(
      true, std::memory_order_relaxed);
}

} // namespace torch::comms
