// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <latch>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <folly/MPMCQueue.h>
#include <folly/concurrency/ConcurrentHashMap.h>

#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/HRDWRingBufferReader.h"
#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/colltrace/CollTracePlugin.h"
#include "comms/utils/colltrace/GraphCollTraceEvent.h"
#include "comms/utils/commSpecs.h"

struct CUstream_st;

namespace meta::comms::colltrace {

struct GraphCollTraceState;
struct GraphCollectiveEntry;
class GraphCudaWaitEvent;

struct CollTraceConfig {
  static constexpr ::size_t kDefaultMaxPendingQueueSize{1024};
  // The max time CollTrace thread will be waiting before it can respond to
  // a cancellation request. This is to ensure that we don't wait forever and
  // will respond reasonably quickly during teardown. Default to 1 second.
  std::chrono::milliseconds maxCheckCancelInterval{1000};

  // If the collective takes more than this to finish, we will consider it as
  // a timeout and trigger whenCollKernelHang() function of plugins.
  std::chrono::seconds collTraceCollStuckTimeout{30};

  // Tuning parameters for the size of the pending queue.
  // If we have too many pending events, we will start dropping events.
  // 1024 should be enough for most of the cases.
  ::size_t maxPendingQueueSize{kDefaultMaxPendingQueueSize};
};

// Action types for the unified polling pipeline.
enum class PendingActionType {
  kScheduleAndStart, // Graph replay started — fire scheduled + start plugins
  kStart, // Eager collective started — fire start plugin
  kProgressing, // Heartbeat — fire progressing plugin (watchdog)
  kEnd, // Collective ended — fire end plugin
};

struct PendingAction {
  CollTraceEvent* event{nullptr};
  PendingActionType type{};
  std::chrono::system_clock::time_point timestamp{};

  // ordered by timestamp, or by action type if timestamp is the same
  // which will ensure that for a given collective, we always fire the
  // the start plugin before the progressing plugin, and the progressing plugin
  // before the end plugin.
  bool operator<(const PendingAction& other) const {
    if (timestamp != other.timestamp) {
      return timestamp < other.timestamp;
    }
    return type < other.type;
  }
};

class CollTrace : public ICollTrace {
 public:
  // Create a thread to collect traces (In the future we should use singelton or
  // coroutines)
  CollTrace(
      CollTraceConfig config,
      CommLogData logMetaData,
      std::function<CommsMaybeVoid(void)> threadSetupFunc,
      std::vector<std::unique_ptr<ICollTracePlugin>> plugins);

  // Stop the CollTrace thread and wait for it to finish. This will also
  // invalidate all the handles that are still valid.
  ~CollTrace() override;

  // Delete copy constructor and copy assignment operator
  CollTrace(const CollTrace&) = delete;
  CollTrace& operator=(const CollTrace&) = delete;
  // Delete move constructor and move assignment operator
  CollTrace(CollTrace&&) = delete;
  CollTrace& operator=(CollTrace&&) = delete;

  // Enqueue a collective event to be traced. Note that now this will first in
  // the pending enqueue map. Then it will move the queue for pending execution
  // once the collective is scheduled.
  CommsMaybe<std::shared_ptr<ICollTraceHandle>> recordCollective(
      std::unique_ptr<ICollMetadata> metadata,
      std::unique_ptr<ICollWaitEvent> waitEvent) noexcept override;

  // Returns the plugin with specific name. Please note there is NO GUARANTEE
  // that the Colltrace thread might not be using the plugin at the time.
  ICollTracePlugin* getPluginByName(std::string name) noexcept override;

  CommsMaybeVoid triggerEventState(
      CollTraceEvent& collEvent,
      CollTraceHandleTriggerState state) noexcept override;

 private:
  // Internal impl for graph-captured collectives, called when
  // recordCollective detects a GraphCudaWaitEvent.
  CommsMaybe<std::shared_ptr<ICollTraceHandle>> recordGraphCollectiveImpl(
      std::unique_ptr<ICollMetadata> metadata,
      std::unique_ptr<ICollWaitEvent> waitEvent) noexcept;

  /****************************************************************************
   * Start of Private Methods for CollTrace thread. All of these methods should
   * be called from the CollTrace thread only. And they should all support
   * cancellation.
   ***************************************************************************/
  bool isThreadCancelled() const noexcept;

  void collTraceThread(
      const std::function<CommsMaybeVoid(void)>& threadSetupFunc);

  // Poll all active graph-captured events (non-blocking), appending completed
  // or progressing actions to the provided vector.
  void pollGraphEvents(std::multiset<PendingAction>& actions) noexcept;

  // Get or create the per-graph state for a stream currently in graph capture.
  // Installs a cleanup user object on first call per graph so we know when the
  // graph is destroyed.
  std::shared_ptr<GraphCollTraceState> getOrCreateGraphState(
      CUstream_st* stream);

  // Non-blocking drain of MPMC queue and poll of in-flight eager events,
  // appending completed or progressing actions to the provided vector.
  void pollEagerEvents(std::multiset<PendingAction>& actions) noexcept;
  // for all temporarly-ordered pending actions fire plugins and clean
  // up completed eager events.
  void processCompletedEvents(std::multiset<PendingAction>& actions) noexcept;

  // **** Start of Private Member Variables ****
  CollTraceConfig config_;

  // Represents the metadata of the communicator that is being traced.
  // This is used for logging purposes.
  CommLogData logMetaData_;
  std::string logPrefix_;

  // Represends the collectives that has been enqueued but not yet executed.
  // Producer: Enqueing Thread
  // Consumer: CollTrace Thread (should be single consumer)
  folly::MPMCQueue<std::unique_ptr<CollTraceEvent>> pendingTraceColls_;

  // The next event to enqueue. Currently we only support one pending enqueue
  // event. This is okay as enqueueing multiple events at the same time is not
  // going to work for any of the CCLs we support anyway.
  std::unique_ptr<CollTraceEvent> pendingEnqueueColl_;

  // Represents the currently active handles. The handle corresponds to a
  // collective event should be invalidated and removed from this map once the
  // CollTraceEvent is destroyed.
  // Insert by: Enqueing Thread
  // Remove by: CollTrace Thread
  folly::ConcurrentHashMap<CollTraceEvent*, std::shared_ptr<CollTraceHandle>>
      eventToHandleMap_;

  std::unordered_map<std::string, ICollTracePlugin&> pluginByName_;
  std::vector<std::unique_ptr<ICollTracePlugin>> plugins_;

  // CollTrace internal collective id. Should always increment monotonically.
  // uint32_t to match GraphCollTraceEvent.collId without truncation.
  std::atomic<uint32_t> collId_{0};

  // Should only be accessed from CollTrace thread
  std::optional<ICollWaitEvent::system_clock_time_point> lastCollEndTime_;

  // Per-graph state map — maps CUDA graph IDs to their GraphCollTraceState.
  // Accessed from capture threads (recordCollective) and the poll thread
  // (pollGraphEvents). Protected by graphStatesMutex_.
  std::mutex graphStateMutex_;
  std::unordered_map<unsigned long long, std::shared_ptr<GraphCollTraceState>>
      graphStateMap_;

  // Single shared ring buffer for ALL cuda graphs. RAII-managed via
  // HRDWRingBuffer (mapped pinned memory, GPU-writable, CPU-readable).
  std::optional<HRDWRingBuffer<GraphCollTraceEvent>> ringBuffer_;
  // CPU-side reader for the shared ring buffer (poll thread only).
  std::optional<HRDWRingBufferReader<GraphCollTraceEvent>> ringReader_;
  // Map from collId to GraphCollectiveEntry* for fast lookup during polling.
  // Only accessed from the colltrace thread (no mutex needed). Entries are
  // added under graphStatesMutex_ and then inserted here on first poll.
  std::unordered_map<uint32_t, GraphCollectiveEntry*> collIdMap_;
  // CollIds that have seen a start event but not yet a matching end event.
  // Used to emit periodic kProgressing actions so the watchdog plugin can
  // detect hung graph collectives.
  std::unordered_set<uint32_t> progressingGraphCollectives_;

  // Eager events being polled by the colltrace thread. Only accessed from
  // the colltrace thread — no mutex needed.
  std::vector<std::unique_ptr<CollTraceEvent>> eagerEvents_;

  std::atomic_flag threadShouldStop_;
  // Signaled by the poll thread after its first loop iteration completes.
  // The constructor waits on this so the poll thread is actively polling
  // before any graph replays can fire.
  std::latch threadStarted_{1};
  // Always keep this thread as the last member variable to ensure that it is
  // initialized after all the other member variables.
  std::thread traceCollThread_;
};

} // namespace meta::comms::colltrace
