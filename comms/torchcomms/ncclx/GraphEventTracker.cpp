// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/torchcomms/ncclx/GraphEventTracker.hpp"

#include <list>
#include <stdexcept>

#include <folly/ScopeGuard.h>

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

namespace {

torch::comms::SharedCallbackState* allocateCallbackState() {
  static std::mutex mutex;
  static std::list<torch::comms::SharedCallbackState> pool;
  std::lock_guard<std::mutex> lock(mutex);
  pool.emplace_back();
  return &pool.back();
}

} // namespace

namespace torch::comms {

GraphEventTracker::GraphEventTracker(TorchCommNCCLX* comm) : comm_(comm) {}

void GraphEventTracker::initOnGraphStart(cudaStream_t stream) {
  // No op if not in graph capture mode
  if (!comm_->getGraphCaptureMode()) {
    return;
  }

  CudaApi* api = comm_->getCudaApi();

  // Get CUDA stream capture info
  cudaStreamCaptureStatus capture_status;
  unsigned long long graph_id;
  cudaGraph_t graph;
  CUDA_CHECK(
      api,
      api->streamGetCaptureInfo_v2(
          stream, &capture_status, &graph_id, &graph, nullptr, nullptr),
      "Failed to get CUDA stream capture info");
  if (capture_status != cudaStreamCaptureStatusActive) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  // reuse at subsequent enqueueWork->addEntry()
  current_graph_id_ = graph_id;
  maybeInitGraphState(stream, graph_id, graph);
}

void GraphEventTracker::maybeInitGraphState(
    cudaStream_t stream,
    unsigned long long graph_id,
    cudaGraph_t graph) {
  // One-time initialization per graph
  auto [it, inserted] = graphs_.try_emplace(graph_id);
  // No op if already initialized
  if (!inserted) {
    return;
  }
  auto& state = it->second;
  CudaApi* api = comm_->getCudaApi();

  SharedCallbackState* shared = allocateCallbackState();
  state.shared_ = shared;

  // Set up replay counter — host node fires on each replay
  CUDA_CHECK(
      api,
      api->launchHostFunc(stream, replayCallback, &shared->replay_counter),
      "Failed to launch replay counter host func");

  // Set up deferred cleanup via a CUDA user object — when the graph is
  // destroyed, the callback sets the released flag; the watchdog's next
  // checkAll() will destroy the owned events.
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

  // graphRetainUserObject succeeded — graph now owns user_object
  user_obj_guard.dismiss();
}

void GraphEventTracker::addEntry(TorchWorkNCCLX* work) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Transfer start/end event ownership from the work object
  auto [it, inserted] = graphs_.try_emplace(current_graph_id_);
  it->second.entries.emplace_back(
      work->start_event_, work->end_event_, work->timeout_ms_);
  work->start_event_ = nullptr;
  work->end_event_ = nullptr;

  // Transfer CPU tensors from the work object to the graph state.
  // These tensors (e.g., CPU pointer tensors used by alltoallv_dynamic_dispatch
  // operations) must remain alive for the graph's lifetime to avoid
  // use-after-free during graph replay.
  auto& cpu_tensors = it->second.cpu_tensors;
  cpu_tensors.insert(
      cpu_tensors.end(),
      std::make_move_iterator(work->cpuTensors_.begin()),
      std::make_move_iterator(work->cpuTensors_.end()));
  work->cpuTensors_.clear();
}

// Queries a CUDA event and returns CheckResult::ERROR on unexpected CUDA
// errors. After the macro, `ret` is either cudaSuccess or cudaErrorNotReady.
#define EVENT_QUERY_CHECK(call, ret, event_desc)                              \
  do {                                                                        \
    ret = (call);                                                             \
    if (ret != cudaSuccess && ret != cudaErrorNotReady) {                     \
      TC_LOG(ERROR, comm_) << "Graph monitor: CUDA error during "             \
                           << (event_desc) << " for graph " << graph_id       \
                           << " collective " << i << ": "                     \
                           << api->getErrorString(ret) << " (" << ret << ")"; \
      return CheckResult::ERROR;                                              \
    }                                                                         \
  } while (0)

// Timeout detection for graph-captured collectives.
//
// During a single replay, the GPU executes nodes in order:
//   host_node (counter++) → start_event record → NCCL collective → end_event
//   record
//
// The watchdog may poll at any point. Possible observations:
//
//   (1) Between replays (previous end=success, counter unchanged):
//       end=success → reset timer. Correct: collective is done.
//
//   (2) New replay started, GPU past host_node but before event records:
//       counter changed → replay detection resets timer.
//       end still=success from previous replay → reset timer (redundant,
//       harmless).
//
//   (3) GPU past event records, collective in progress:
//       start=success, end=notReady → start/continue timer. Correct.
//
//   (4) GPU past host_node but before this collective's start_event:
//       both notReady → reset timer. Correct: collective hasn't started.
//
// Without the replay counter, case (2) could be missed if the watchdog
// never observes end=success between consecutive replays, causing the
// timer to span N replays and false-trigger a timeout.
GraphEventTracker::CheckResult GraphEventTracker::checkAll() {
  CudaApi* api = comm_->getCudaApi();
  std::lock_guard<std::mutex> lock(mutex_);

  // Cleanup released graphs
  cleanupReleasedGraphs();

  // Traverse remaining active graphs
  for (auto& [graph_id, graph_state] : graphs_) {
    uint64_t current_replay =
        graph_state.shared_->replay_counter.load(std::memory_order_acquire);

    for (size_t i = 0; i < graph_state.entries.size(); ++i) {
      auto& entry = graph_state.entries[i];

      // Detect new replay — reset timer to avoid false timeout spanning
      // multiple replays
      if (current_replay != entry.last_seen_replay) {
        entry.start_completed_time.reset();
        entry.last_seen_replay = current_replay;
      }

      cudaError_t start_status, end_status;
      EVENT_QUERY_CHECK(
          api->eventQuery(entry.start_event),
          start_status,
          "start event query");
      EVENT_QUERY_CHECK(
          api->eventQuery(entry.end_event), end_status, "end event query");

      if (end_status == cudaSuccess) {
        // Collective completed or no replay in progress
        entry.start_completed_time.reset();
      } else if (start_status == cudaSuccess) {
        // Collective in progress — start or continue timing
        if (!entry.start_completed_time.has_value()) {
          entry.start_completed_time = std::chrono::steady_clock::now();
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
      } else {
        // Both notReady — replay started but haven't reached this collective
        entry.start_completed_time.reset();
      }
    }
  }
  return CheckResult::OK;
}

#undef EVENT_QUERY_CHECK

void GraphEventTracker::cleanupReleasedGraphs() {
  CudaApi* api = comm_->getCudaApi();
  for (auto it = graphs_.begin(); it != graphs_.end();) {
    if (it->second.shared_->released.load(std::memory_order_relaxed)) {
      for (auto& entry : it->second.entries) {
        entry.destroyEvents(api);
      }
      it = graphs_.erase(it);
    } else {
      ++it;
    }
  }
}

void GraphEventTracker::destroyAll() {
  CudaApi* api = comm_->getCudaApi();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [graph_id, graph_state] : graphs_) {
    for (auto& entry : graph_state.entries) {
      entry.destroyEvents(api);
    }
  }
  graphs_.clear();
}

// Static callback — fires on each graph replay to increment counter
void CUDART_CB GraphEventTracker::replayCallback(void* userData) {
  auto* counter = static_cast<std::atomic<uint64_t>*>(userData);
  counter->fetch_add(1, std::memory_order_release);
}

// Static callback — fires when graph is destroyed to set released flag
void CUDART_CB GraphEventTracker::cleanupCallback(void* userData) {
  static_cast<std::atomic<bool>*>(userData)->store(
      true, std::memory_order_relaxed);
}

} // namespace torch::comms
