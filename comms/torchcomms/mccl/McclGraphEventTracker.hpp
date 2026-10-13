// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/device/cuda/DeviceCounter.h"

namespace torch::comms {

// Forward declarations
class TorchCommMCCL;
class TorchWorkMCCL;

// Returns true if graph timeout monitoring (and the per-replay clog event
// firing that piggybacks on it) is enabled for the MCCL backend. Controlled by
// the env var TORCHCOMM_MCCL_GRAPH_TIMEOUT_MONITORING (default: enabled).
bool isMcclGraphTimeoutMonitoringEnabled();
void resetMcclGraphTimeoutMonitoringCacheForTest();

// Tracks a single graph-captured collective for clog replay-event firing (and
// records its timeout for observability). Mirrors the NCCLX GraphEventTracker
// design.
struct McclGraphWork {
  cudaEvent_t start_event; // OWNED — stashed in tracker event_pool_ on cleanup
  cudaEvent_t end_event; // OWNED — stashed in tracker event_pool_ on cleanup
  // The effective timeout the MCCL collective runs with (the comm-level
  // mccl_comm_->getTimeout() at capture; -1 ms if unset). Recorded for
  // observability only — MCCL self-times-out on-device, so checkAll() does not
  // enforce this (gated by McclGraphEventTracker::enforce_timeout_).
  std::chrono::milliseconds timeout;
  std::optional<std::chrono::steady_clock::time_point> start_completed_time;
  uint64_t last_seen_replay{0};
  // Replay event logging state. Events are ordered: S=0, E=1 (W is not
  // observable during replay). Initialized to (replay=0, event=E): the counter
  // starts at 0 and the first g.replay() increments it to 1, which is the first
  // replay execution that should be reported.
  uint64_t notified_through_replay{0};
  int notified_through_event{1};

  McclGraphWork(cudaEvent_t start, cudaEvent_t end, std::chrono::milliseconds t)
      : start_event(start), end_event(end), timeout(t) {}
};

// Shared state for the graph-release flag, read by the tracker's watchdog.
// Allocated from a static pool so the cleanup callback only performs an atomic
// store, avoiding mutex acquisition or CUDA API calls inside the callback
// (which violates CUDA docs). The resource is released at process exit, so
// graph destruction and comm finalization can occur in any order.
struct McclSharedCallbackState {
  std::atomic_bool released{false};
};

// Per-graph state. Holds the CUDA events / replay counter / CPU tensors for all
// captured collectives. The destructor returns owned events to the tracker's
// event_pool_ and the counter to counter_pool_, so neither cudaEventDestroy nor
// cudaFreeHost runs inline on the watchdog thread.
struct McclGraphState {
  // Entries grouped by stream — collectives are only ordered within a stream,
  // so per-stream grouping enables early-exit in checkAll().
  std::unordered_map<cudaStream_t, std::vector<McclGraphWork>> stream_entries;
  McclSharedCallbackState* shared_{nullptr};
  std::unique_ptr<DeviceCounter> replay_counter;
  std::vector<cudaEvent_t>* event_pool_{nullptr};
  std::vector<std::unique_ptr<DeviceCounter>>* counter_pool_{nullptr};

  McclGraphState() = default;
  // Move-only: ownership of the owned events/counter transfers with the move,
  // and the moved-from object's destructor becomes a no-op (empty containers,
  // null counter). Copying would double-return resources to the pools.
  McclGraphState(McclGraphState&&) = default;
  McclGraphState& operator=(McclGraphState&&) = default;
  McclGraphState(const McclGraphState&) = delete;
  McclGraphState& operator=(const McclGraphState&) = delete;

  ~McclGraphState() {
    if (counter_pool_ && replay_counter) {
      counter_pool_->push_back(std::move(replay_counter));
    }
    if (!event_pool_) {
      return;
    }
    for (auto& [_, entries] : stream_entries) {
      for (auto& entry : entries) {
        event_pool_->push_back(entry.start_event);
        event_pool_->push_back(entry.end_event);
      }
    }
  }
};

// Monitors graph-captured MCCL collectives for timeout/error after graph
// launch, and fires per-replay clog events via comm->fireGraphReplayHook().
//
// CUDA graph capture turns each collective into a recorded node; the normal
// eager-mode watchdog cannot monitor them. This tracker takes ownership of each
// collective's start/end CUDA events (recorded with cudaEventRecordExternal so
// they stay host-queryable during replay) and polls them from the watchdog
// thread. A single-thread GPU kernel node atomically increments a per-graph
// counter in mapped pinned memory on every replay, so the watchdog can
// distinguish "not yet replayed" from "stuck during a replay" and assign a
// replay id to each fired event.
//
// Cleanup: a CUDA user-object callback sets a released flag when the graph is
// destroyed; the watchdog's next checkAll() sees the flag and destroys the
// owned events (deferred cleanup — callbacks never call CUDA APIs directly).
class McclGraphEventTracker {
 public:
  enum class CheckResult { OK, TIMEOUT, ERROR };

  explicit McclGraphEventTracker(TorchCommMCCL* comm);
  ~McclGraphEventTracker();

  // Non-copyable, non-movable (contains mutex)
  McclGraphEventTracker(const McclGraphEventTracker&) = delete;
  McclGraphEventTracker& operator=(const McclGraphEventTracker&) = delete;
  McclGraphEventTracker(McclGraphEventTracker&&) = delete;
  McclGraphEventTracker& operator=(McclGraphEventTracker&&) = delete;

  // One-time initialization per graph during capture. Checks graph capture
  // mode internally. Returns true if `stream` is actively being captured (graph
  // state ready / current_graph_id_ set), false otherwise (caller should skip
  // recording). Must be called before the collective's events are recorded.
  bool initOnGraphStart(cudaStream_t stream);
  // Add a new entry for a captured collective. Takes ownership of the supplied
  // start/end events (returned to the event pool on graph cleanup). Must be
  // called after initOnGraphStart(). Unlike the NCCLX tracker (which reads the
  // work object), the MCCL backend tracks completion via the MCCL work handle,
  // so the bracketing CUDA events are created/recorded by the comm and passed
  // in explicitly.
  void addEntry(
      cudaStream_t stream,
      cudaEvent_t start_event,
      cudaEvent_t end_event,
      std::chrono::milliseconds timeout);
  // Check all entries for timeout or error and fire replay events. Called from
  // the watchdog thread.
  CheckResult checkAll();
  // Destroy all owned events and replay counters. Called from finalize().
  void destroyAll();

  // Acquire/release a bracketing CUDA event from a reused pool: acquireEvent()
  // pops a pooled event (creating one only when the pool is empty);
  // releaseEvent() returns it. The pool is also refilled when graph state is
  // destroyed (~McclGraphState) and drained at ~McclGraphEventTracker, so
  // events are reused across captures and cudaEventDestroy never runs on the
  // watchdog thread. Both lock mutex_, so they are safe to call from the comm's
  // main thread while the watchdog returns events.
  cudaError_t acquireEvent(cudaEvent_t& out);
  void releaseEvent(cudaEvent_t event);

 private:
  static void CUDART_CB cleanupCallback(void* userData);
  // One-time per-graph setup: replay counter kernel + cleanup user object.
  // Must be called with mutex_ held.
  void maybeInitGraphState(
      cudaStream_t stream,
      unsigned long long graph_id,
      cudaGraph_t graph);
  void cleanupReleasedGraphs();

  // Must be called with mutex_ held (touches counter_pool_ unguarded).
  cudaError_t acquireCounter(std::unique_ptr<DeviceCounter>& out);

  // Fire graph replay hooks for all events from the entry's last notified
  // position up to (current_replay, current_event). Events are ordered S=0,
  // E=1. Catches up missed replays automatically.
  static constexpr int kEventS = 0;
  static constexpr int kEventE = 1;

  void notifyReplayProgress(
      unsigned long long graph_id,
      void* stream,
      size_t collective_index,
      McclGraphWork& entry,
      uint64_t current_replay,
      int current_event);

  TorchCommMCCL* comm_; // raw pointer — parent owns this tracker
  // MCCL collectives self-timeout on-device and set a comm-level abort flag, so
  // the tracker never enforces a host-side graph timeout —
  // McclGraphWork.timeout is recorded for observability only. Kept as a field
  // (not a constant) so the NCCLX-style enforcement path in checkAll() can be
  // re-enabled if needed.
  bool enforce_timeout_{false};
  std::mutex mutex_;
  unsigned long long current_graph_id_{0};
  // Pinned-memory counter pool, reused across captures. ~McclGraphState returns
  // counters here; drained at destruction (cudaFreeHost is safe at shutdown).
  std::vector<std::unique_ptr<DeviceCounter>> counter_pool_;
  // CUDA event pool from completed graphs, destroyed at ~McclGraphEventTracker.
  // cudaEventDestroy is synchronizing — running it on the watchdog thread can
  // deadlock with the eager warmup after a PAFT recapture.
  std::vector<cudaEvent_t> event_pool_;
  // Declared last so it is destroyed FIRST: ~McclGraphState returns its owned
  // events/counters to counter_pool_/event_pool_ above, which must still be
  // alive. This keeps teardown correct independently of the explicit
  // destroyAll() in ~McclGraphEventTracker.
  std::unordered_map<unsigned long long, McclGraphState> graphs_;
};

} // namespace torch::comms
