// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <ATen/ATen.h>

#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/device/cuda/DeviceCounter.h"

namespace torch::comms {

// Forward declarations
class TorchCommNCCLX;
class TorchWorkNCCLX;

// Tracks a single graph-captured collective for timeout detection.
struct GraphWork {
  cudaEvent_t start_event; // OWNED — stashed in tracker event_pool_ on cleanup
  cudaEvent_t end_event; // OWNED — stashed in tracker event_pool_ on cleanup
  std::chrono::milliseconds timeout;
  std::optional<std::chrono::steady_clock::time_point> start_completed_time;
  uint64_t last_seen_replay{0};
  // Replay event logging state.
  // Events are ordered: S=0, E=1 (W is not observable during replay).
  // Initialized to (replay=0, event=E) — the counter starts at 0
  // and the first g.replay() increments it to 1, which is the first
  // replay execution that should be reported.
  uint64_t notified_through_replay{0};
  int notified_through_event{1};

  GraphWork(cudaEvent_t start, cudaEvent_t end, std::chrono::milliseconds t)
      : start_event(start), end_event(end), timeout(t) {}
};

// Shared state for the graph-release flag, read by the tracker's watchdog.
// Allocated from a static pool so that the cleanup callback only performs
// an atomic store, avoiding mutex acquisition or CUDA API calls inside
// the callback (which violates CUDA docs). Resource is automatically
// released when process exits, so graph destruction and comm finalization
// can occur in any order.
struct SharedCallbackState {
  std::atomic_bool released{false};
};

// Per-graph state. Holds the CUDA events / dependency tensors
// for all captured collectives. The destructor returns owned events to the
// tracker's event_pool_; the counter is explicitly moved to counter_pool_
// before destruction. This ensures neither cudaEventDestroy nor cudaFreeHost
// runs inline on the watchdog thread.
struct GraphState {
  // Entries grouped by stream — collectives are only ordered within a stream,
  // so per-stream grouping enables early-exit optimization in checkAll().
  std::unordered_map<cudaStream_t, std::vector<GraphWork>> stream_entries;
  SharedCallbackState* shared_{nullptr};
  std::unique_ptr<DeviceCounter> replay_counter;
  // CPU tensors that must be kept alive for the graph's lifetime.
  // These tensors are moved from work objects during graph capture and
  // remain valid until the graph is destroyed.
  std::vector<at::Tensor> cpu_tensors;
  // Set by GraphEventTracker::maybeInitGraphState to point at the tracker's
  // pools. Null for GraphState objects that were never fully initialized.
  std::vector<cudaEvent_t>* event_pool_{nullptr};
  std::vector<std::unique_ptr<DeviceCounter>>* counter_pool_{nullptr};

  ~GraphState() {
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

// Monitors graph-captured collectives for timeout/error after graph launch.
//
// CUDA graph capture turns each collective into a recorded node; the normal
// eager-mode watchdog cannot monitor them.  GraphEventTracker solves this by
// taking ownership of each collective's start/end CUDA events and polling
// them from the watchdog thread.
//
// Replay detection: a single-thread GPU kernel node atomically increments a
// per-graph counter in mapped pinned memory on every replay, so the watchdog
// can distinguish "not yet replayed" from "stuck during a replay" without
// the CPU round-trip cost of a host-function callback.
//
// Cleanup: a CUDA user-object callback sets a released flag when the graph
// is destroyed.  The watchdog's next checkAll() call sees the flag and
// destroys the owned events (deferred cleanup model — callbacks never call
// CUDA APIs directly).
//
// Timeout-detection state machine (per collective, per replay):
//
//   start_event  end_event   → state
//   ─────────────────────────────────────────────────
//   COMPLETED    COMPLETED   → OK (reset timer)
//   NOT REACHED  NOT REACHED → no replay in progress (reset timer)
//   COMPLETED    NOT REACHED → IN PROGRESS (start / continue timer)
//   NOT REACHED  COMPLETED   → impossible (would indicate a bug)
//
// The timer is also reset whenever a new replay is detected, preventing
// false timeouts that would span multiple replays.
class GraphEventTracker {
 public:
  enum class CheckResult { OK, TIMEOUT, ERROR };

  explicit GraphEventTracker(TorchCommNCCLX* comm);
  ~GraphEventTracker();

  // Non-copyable, non-movable (contains mutex)
  GraphEventTracker(const GraphEventTracker&) = delete;
  GraphEventTracker& operator=(const GraphEventTracker&) = delete;
  GraphEventTracker(GraphEventTracker&&) = delete;
  GraphEventTracker& operator=(GraphEventTracker&&) = delete;

  // One-time initialization per graph during capture. Checks graph capture
  // mode internally; no-op if not capturing. Must be called before the
  // first collective's start_event is recorded on the stream.
  void initOnGraphStart(cudaStream_t stream);
  // Add a new entry for a captured collective. Takes ownership of the work's
  // start/end events. Must be called after initOnGraphStart().
  void addEntry(TorchWorkNCCLX* work);
  // Check all entries for timeout or error. Called from the watchdog thread.
  CheckResult checkAll();
  // Destroy all owned events and replay counters. Called from finalize().
  void destroyAll();

 private:
  // Static callback for CUDA user object cleanup — sets released flag
  static void CUDART_CB cleanupCallback(void* userData);
  // One-time per-graph setup: replay counter kernel + cleanup user object.
  // Must be called with mutex_ held.
  void maybeInitGraphState(
      cudaStream_t stream,
      unsigned long long graph_id,
      cudaGraph_t graph);
  void cleanupReleasedGraphs();

  // Counter pool. Acquire returns a counter from the pool (resetting it to
  // zero) or creates a new one if empty. ~GraphState handles returning
  // counters to the pool. Pooling avoids the synchronizing cudaFreeHost
  // in ~DeviceCounter on the watchdog thread — see cleanupReleasedGraphs()
  // for context.
  cudaError_t acquireCounter(std::unique_ptr<DeviceCounter>& out);

  // Event pool. Acquire returns an event from the pool, creating a new one
  // if empty. ~GraphState handles returning events to the pool.
  // Pooling avoids the synchronizing cudaEventDestroy on the watchdog
  // thread — see cleanupReleasedGraphs() for context.
  cudaError_t acquireEvent(cudaEvent_t& out);

  // Fire graph replay hooks for all events from the entry's last notified
  // position up to (current_replay, current_event). Events are ordered
  // S=0, E=1. Catches up missed replays automatically.
  static constexpr int kEventS = 0;
  static constexpr int kEventE = 1;

  void notifyReplayProgress(
      unsigned long long graph_id,
      void* stream,
      size_t collective_index,
      GraphWork& entry,
      uint64_t current_replay,
      int current_event);

  TorchCommNCCLX* comm_; // raw pointer — parent owns this tracker
  std::mutex mutex_;
  // cached at initOnGraphStart() to be reused in addEntry() for each collective
  unsigned long long current_graph_id_{0};
  std::unordered_map<unsigned long long, GraphState> graphs_;
  // Pool of pinned-memory counter regions, reused across graph captures.
  // Guarded by mutex_. Drained at destruction (~DeviceCounter calls
  // cudaFreeHost — safe at comm shutdown when device is quiescent).
  std::vector<std::unique_ptr<DeviceCounter>> counter_pool_;
  // Pool of CUDA events from completed graphs, deferred for destruction
  // until ~GraphEventTracker. cudaEventDestroy is a synchronizing call —
  // running it on the watchdog thread can deadlock with the eager warmup
  // that follows a PAFT recapture. Guarded by mutex_.
  std::vector<cudaEvent_t> event_pool_;
};

} // namespace torch::comms
