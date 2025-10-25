// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CollTrace.h"

#include <fmt/core.h>
#include <folly/logging/xlog.h>
#include <folly/stop_watch.h>

#include "comms/utils/CommsMaybeChecks.h"

// Helper macro to return from a function once cancellation is requested
#define SAFE_RETURN_POINT(shouldCancel) \
  if (shouldCancel) {                   \
    return;                             \
  }

#define SAFE_RETURN_POINT_WITH_VAL(shouldCancel, returnVal) \
  if (shouldCancel) {                                       \
    return returnVal;                                       \
  }

namespace meta::comms::colltrace {

namespace {
template <auto Method>
void triggerPlugins(
    std::vector<std::unique_ptr<ICollTracePlugin>>& plugins,
    CollTraceEvent& curEvent) noexcept {
  for (auto& plugin : plugins) {
    CommsMaybeVoid res = ((*plugin).*Method)(curEvent);
    if (res.hasError()) {
      XLOG_FIRST_N(
          ERR,
          10,
          "Exception thrown in plugin {} when calling method {}: {}",
          plugin->getName(),
          typeid(Method).name(),
          res.error().message);
    }
  }
}
} // namespace

CollTrace::CollTrace(
    CollTraceConfig config,
    CommLogData logMetaData,
    std::function<CommsMaybeVoid(void)> threadSetupFunc,
    std::vector<std::unique_ptr<ICollTracePlugin>> plugins)
    : config_(std::move(config)),
      logMetaData_(std::move(logMetaData)),
      logPrefix_(fmt::format(
          "commHash {:#x} commDesc {} rank {}",
          logMetaData_.commHash,
          logMetaData_.commDesc,
          logMetaData_.rank)),
      pendingTraceColls_(folly::MPMCQueue<std::unique_ptr<CollTraceEvent>>{
          config_.maxPendingQueueSize}),
      plugins_(std::move(plugins)),
      traceCollThread_(
          std::thread(&CollTrace::collTraceThread, this, threadSetupFunc)) {
  // pluginByName_ is not used in the colltrace thread. It is okay to initialize
  // it after the colltrace thread starts
  for (const auto& plugin : plugins_) {
    XLOG(DBG0) << "Registering plugin " << plugin->getName();
    pluginByName_.emplace(plugin->getName(), *plugin);
  }
}

CollTrace::~CollTrace() {
  // Set the cancellation flag
  threadShouldStop_.test_and_set();
  // Invalidate all the handles
  for (auto& [_, handle] : eventToHandleMap_) {
    handle->invalidate();
  }
  // Wait for the thread to finish
  if (traceCollThread_.joinable()) {
    traceCollThread_.join();
  }
}

CommsMaybe<std::shared_ptr<CollTraceHandle>> CollTrace::recordCollective(
    std::unique_ptr<ICollMetadata> metadata,
    std::unique_ptr<ICollWaitEvent> waitEvent) noexcept {
  pendingEnqueueColl_ = std::make_unique<CollTraceEvent>(
      std::make_shared<CollRecord>(collId_.fetch_add(1), std::move(metadata)),
      std::move(waitEvent));
  auto handle =
      std::make_shared<CollTraceHandle>(this, pendingEnqueueColl_.get());
  eventToHandleMap_.emplace(pendingEnqueueColl_.get(), handle);
  return handle;
}

ICollTracePlugin* CollTrace::getPluginByName(std::string name) noexcept {
  return folly::get_ptr(pluginByName_, name);
}

CommsMaybeVoid CollTrace::triggerEventState(
    CollTraceEvent& collEvent,
    CollTraceHandleTriggerState state) noexcept {
  switch (state) {
    case CollTraceHandleTriggerState::BeforeEnqueueKernel: {
      if (&collEvent != pendingEnqueueColl_.get()) {
        return folly::makeUnexpected(CommsError(
            "Only pendingEnqueueColl_ can be triggered in BeforeEnqueueKernel state",
            commInvalidUsage));
      }
      auto beforeKernelRes = collEvent.waitEvent->beforeCollKernelScheduled();
      triggerPlugins<&ICollTracePlugin::beforeCollKernelScheduled>(
          plugins_, collEvent); // Trigger plugins after calling waitEvent
      EXPECT_CHECK_ALWAYS_RETURN(beforeKernelRes);
    }
    case CollTraceHandleTriggerState::AfterEnqueueKernel: {
      if (&collEvent != pendingEnqueueColl_.get()) {
        return folly::makeUnexpected(CommsError(
            "Only pendingEnqueueColl_ can be triggered in AfterEnqueueKernel state",
            commInvalidUsage));
      }
      triggerPlugins<&ICollTracePlugin::afterCollKernelScheduled>(
          plugins_, collEvent); // Trigger plugins before calling waitEvent
      EXPECT_CHECK(collEvent.waitEvent->afterCollKernelScheduled());
      collEvent.collRecord->getTimingInfo().setCollEnqueueTs(
          std::chrono::system_clock::now());
      if (pendingTraceColls_.write(std::move(pendingEnqueueColl_))) {
        return folly::unit;
        // If the write fails, pendingEnqueueColl_ will not be moved. Do a
        // check for nullptr as sanity check
      } else if (pendingEnqueueColl_ != nullptr) {
        // TODO: This is not safe. But I could not find a better way to do it
        // as the caller of triggerEventState (which is CollTraceHandle itself)
        // holds its write lock and calling invalidate here will cause deadlock.
        eventToHandleMap_.at(pendingEnqueueColl_.get())->invalidateUnsafe();
        eventToHandleMap_.erase(pendingEnqueueColl_.get());
        pendingEnqueueColl_ = nullptr;
        return folly::makeUnexpected(CommsError(
            "Failed to write to pendingTraceColls_ queue", commInternalError));
      } else {
        // This code should not be reached
        XLOG_FIRST_N(
            DBG,
            1,
            "pendingEnqueueColl_ is nullptr after write to queue failed");
        return folly::makeUnexpected(CommsError(
            "pendingEnqueueColl_ is nullptr after write to pendingTraceColls_ queue",
            commInternalError));
      }
    }
    case CollTraceHandleTriggerState::KernelStarted: {
      EXPECT_CHECK_ALWAYS_RETURN(collEvent.waitEvent->signalCollStart());
    }
    case CollTraceHandleTriggerState::KernelFinished: {
      EXPECT_CHECK_ALWAYS_RETURN(collEvent.waitEvent->signalCollEnd());
    }
    default:
      return folly::makeUnexpected(CommsError(
          fmt::format(
              "Invalid state {} received when calling triggerEventState",
              triggerStateToStr(state)),
          commInternalError));
  }
  return folly::makeUnexpected(
      CommsError("Unexpected return path", commInternalError));
}

bool CollTrace::isThreadCancelled() const noexcept {
  return threadShouldStop_.test(std::memory_order_relaxed);
}

CommsMaybe<std::unique_ptr<CollTraceEvent>>
CollTrace::getNextPendingEvent() noexcept {
  std::unique_ptr<CollTraceEvent> event;
  auto curTime = std::chrono::steady_clock::now();
  while (!pendingTraceColls_.tryReadUntil(
      curTime + config_.maxCheckCancelInterval, event)) {
    SAFE_RETURN_POINT_WITH_VAL(
        isThreadCancelled(),
        folly::makeUnexpected(CommsError("Got cancellation", commInProgress)));
    curTime = std::chrono::steady_clock::now();
  }
  if (event == nullptr) {
    XLOG_FIRST_N(
        ERR, 2, logPrefix_, ": Got null event from pendingTrace queue");
    return folly::makeUnexpected(
        CommsError("Got null event from queue", commInternalError));
  }
  return event;
}

CommsMaybeVoid CollTrace::waitCollStart(CollTraceEvent& event) noexcept {
  while (!isThreadCancelled()) {
    auto res = event.waitEvent->waitCollStart(config_.maxCheckCancelInterval);
    triggerPlugins<&ICollTracePlugin::collEventProgressing>(plugins_, event);
    EXPECT_CHECK_RES(res);
    if (res.value()) {
      return folly::unit;
    }
  }
  return folly::unit;
}

CommsMaybeVoid CollTrace::waitCollEnd(CollTraceEvent& event) noexcept {
  // Check for
  while (!isThreadCancelled()) {
    auto res = event.waitEvent->waitCollEnd(config_.maxCheckCancelInterval);
    triggerPlugins<&ICollTracePlugin::collEventProgressing>(plugins_, event);
    EXPECT_CHECK_RES(res);
    if (res.value()) {
      return folly::unit;
    }
  }
  return folly::unit;
}

void CollTrace::collTraceThread(
    const std::function<CommsMaybeVoid(void)>& threadSetupFunc) {
  XLOGF(INFO, "{}: Colltrace thread INIT", logPrefix_);
  auto res = threadSetupFunc();
  if (res.hasError()) {
    XLOGF(
        ERR,
        "{}: Error in calling colltrace thread setup function: {}",
        logPrefix_,
        res.error().message);
    return;
  }

  XLOGF(INFO, "{}: CollTrace thread STARTED", logPrefix_);

  while (!isThreadCancelled()) {
    // ----- Get the next pending event from the MPMC Queue -----
    auto eventMaybe = getNextPendingEvent();
    SAFE_RETURN_POINT(isThreadCancelled());

    if (eventMaybe.hasError()) {
      XLOG_FIRST_N(
          ERR,
          2,
          logPrefix_,
          ": Got error from pendingTrace queue: ",
          eventMaybe.error().message);
      continue;
    }

    // ----- Get the event, Setup guard for handle -----
    auto event = std::move(eventMaybe.value());
    // Always invalidate the handle to the event to ensure we don't reference
    // to it after the event is destroyed
    auto handleReleaseGuard = folly::makeGuard([&event, this]() {
      auto it = eventToHandleMap_.find(event.get());
      if (it != eventToHandleMap_.end()) {
        it->second->invalidate();
        eventToHandleMap_.erase(it);
      }
    });

    if (lastCollEndTime_.has_value()) [[likely]] {
      event->collRecord->getTimingInfo().setPreviousCollEndTs(
          lastCollEndTime_.value());
    }

    XLOGF(DBG2, "Start tracking collId {}", event->collRecord->getCollId());

    // ----- Track the start of the event -----
    EXPECT_CHECK_CONTINUE_LOG_FIRST_N(waitCollStart(*event), 2);
    XLOGF(DBG2, "CollId {} started kernel", event->collRecord->getCollId());
    SAFE_RETURN_POINT(isThreadCancelled());

    event->collRecord->getTimingInfo().setCollStartTs(
        std::chrono::system_clock::now());

    triggerPlugins<&ICollTracePlugin::afterCollKernelStart>(plugins_, *event);

    // ----- Track the end of the event -----
    EXPECT_CHECK_CONTINUE_LOG_FIRST_N(waitCollEnd(*event), 2);
    XLOGF(DBG2, "CollId {} finished kernel", event->collRecord->getCollId());
    SAFE_RETURN_POINT(isThreadCancelled());

    auto endTs = std::chrono::system_clock::now();
    event->collRecord->getTimingInfo().setCollEndTs(endTs);
    lastCollEndTime_ = endTs; // Update the last coll end time

    triggerPlugins<&ICollTracePlugin::afterCollKernelEnd>(plugins_, *event);
  }
}

} // namespace meta::comms::colltrace
