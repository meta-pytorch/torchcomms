// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <thread>

#include <folly/MPMCQueue.h>
#include <folly/concurrency/ConcurrentHashMap.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/colltrace/CollTracePlugin.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

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
  CommsMaybe<std::shared_ptr<CollTraceHandle>> recordCollective(
      std::unique_ptr<ICollMetadata> metadata,
      std::unique_ptr<ICollWaitEvent> waitEvent) noexcept override;

  // Returns the plugin with specific name. Please note there is NO GUARANTEE
  // that the Colltrace thread might not be using the plugin at the time.
  ICollTracePlugin* getPluginByName(std::string name) noexcept override;

  CommsMaybeVoid triggerEventState(
      CollTraceEvent& collEvent,
      CollTraceHandleTriggerState state) noexcept override;

 private:
  /****************************************************************************
   * Start of Private Methods for CollTrace thread. All of these methods should
   * be called from the CollTrace thread only. And they should all support
   * cancellation.
   ***************************************************************************/
  bool isThreadCancelled() const noexcept;

  CommsMaybe<std::unique_ptr<CollTraceEvent>> getNextPendingEvent() noexcept;

  CommsMaybeVoid waitCollStart(CollTraceEvent& event) noexcept;

  CommsMaybeVoid waitCollEnd(CollTraceEvent& event) noexcept;

  void collTraceThread(
      const std::function<CommsMaybeVoid(void)>& threadSetupFunc);

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

  // CollTrace internal collective id. Should always increment monotonically
  std::atomic<int64_t> collId_{0};

  // Should only be accessed from CollTrace thread
  std::optional<ICollWaitEvent::system_clock_time_point> lastCollEndTime_;

  std::atomic_flag threadShouldStop_;
  // Always keep this thread as the last member variable to ensure that it is
  // initialized after all the other member variables.
  std::thread traceCollThread_;
};

} // namespace meta::comms::colltrace
