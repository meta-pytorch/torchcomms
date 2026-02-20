// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/torchcomms/device/cuda/CudaApi.hpp"

namespace torch::comms {

// Forward declarations
class TorchCommNCCLX;
class TorchWorkNCCLX;

// Tracks a single graph-captured collective for timeout detection.
struct GraphWork {
  cudaEvent_t start_event; // OWNED — ad-hoc created, destroyed on cleanup
  cudaEvent_t end_event; // OWNED — ad-hoc created, destroyed on cleanup
  std::chrono::milliseconds timeout;
  std::optional<std::chrono::steady_clock::time_point> start_completed_time;
  uint64_t last_seen_replay{0};

  GraphWork(cudaEvent_t start, cudaEvent_t end, std::chrono::milliseconds t)
      : start_event(start), end_event(end), timeout(t) {}

  void destroyEvents(CudaApi* api) {
    (void)api->eventDestroy(start_event);
    (void)api->eventDestroy(end_event);
  }
};

// Shared state between CUDA callbacks and the tracker. Allocated from a static
// pool so that callbacks only perform atomic stores, avoiding mutex acquisition
// or CUDA API calls inside the callback (which violates CUDA docs). Resource is
// automatically released when prcoess exits, so graph destruction and comm
// finalization can occur in any order.
struct SharedCallbackState {
  std::atomic<uint64_t> replay_counter{0};
  std::atomic<bool> released{false};
};

// Monitors graph-captured collectives for timeout/error after graph launch.
//
// CUDA graph capture turns each collective into a recorded node; the normal
// eager-mode watchdog cannot monitor them.  GraphEventTracker solves this by
// taking ownership of each collective's start/end CUDA events and polling
// them from the watchdog thread.
//
// Replay detection: a host-node callback increments a per-graph atomic
// counter on every replay so the watchdog can distinguish "not yet replayed"
// from "stuck during a replay."
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
  ~GraphEventTracker() = default;

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
  // Static callback for replay detection (fires on each graph replay)
  static void CUDART_CB replayCallback(void* userData);
  // One-time per-graph setup: replay counter + cleanup user object.
  // Must be called with mutex_ held.
  void maybeInitGraphState(
      cudaStream_t stream,
      unsigned long long graph_id,
      cudaGraph_t graph);
  void cleanupReleasedGraphs();

  struct GraphState {
    std::vector<GraphWork> entries;
    SharedCallbackState* shared_{nullptr};
  };

  TorchCommNCCLX* comm_; // raw pointer — parent owns this tracker
  std::mutex mutex_;
  // cached at initOnGraphStart() to be reused in addEntry() for each collective
  unsigned long long current_graph_id_{0};
  std::unordered_map<unsigned long long, GraphState> graphs_;
};

} // namespace torch::comms
