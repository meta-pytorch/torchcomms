// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CollTrace.h"

#include <algorithm>

#include <fmt/core.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>
#include <folly/stop_watch.h>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/utils/CommsMaybeChecks.h"
#include "comms/utils/colltrace/GpuClockCalibration.h"
#include "comms/utils/colltrace/GraphCollTraceHandle.h"
#include "comms/utils/colltrace/GraphCollTraceState.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace meta::comms::colltrace {

namespace {

// fires when graph is destroyed to set released flag s.t., the associated
// GraphCollTraceState will stop being used.
void
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    CUDART_CB
#endif
    graphCleanupCallback(void* userData) {
  auto* sp = static_cast<std::shared_ptr<GraphCollTraceState>*>(userData);
  (*sp)->graph_destructed.store(true, std::memory_order_relaxed);
  delete sp;
}

} // namespace

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

CollTrace::CollTrace(
    CollTraceConfig config,
    CommLogData logMetaData,
    std::function<CommsMaybeVoid(void)> threadSetupFunc,
    std::vector<std::unique_ptr<ICollTracePlugin>> plugins)
    : config_(std::move(config)),
      logMetaData_(std::move(logMetaData)),
      logPrefix_(
          fmt::format(
              "commHash {:#x} commDesc {} rank {}",
              logMetaData_.commHash,
              logMetaData_.commDesc,
              logMetaData_.rank)),
      pendingTraceColls_(
          folly::MPMCQueue<std::unique_ptr<CollTraceEvent>>{
              config_.maxPendingQueueSize}),
      plugins_(std::move(plugins)) {
  if (NCCL_COLLTRACE_TRACE_CUDA_GRAPH) {
    // 2x the max plugin retention ensures the ring holds at least the last
    // max_retention collective entries (each collective has 1x start + 1x end
    // event). The ring ctor rounds up to the next power of 2.
    int64_t maxRetention = 0;
    for (const auto& plugin : plugins_) {
      maxRetention = std::max(maxRetention, plugin->maxEventRetention());
    }
    uint32_t ringSize =
        std::max(kDefaultRingSize, static_cast<uint32_t>(maxRetention) * 2);
    XLOGF(
        DBG0,
        "{}: graph ring buffer sized to {} entries (max plugin retention={}, "
        "default={})",
        logPrefix_,
        ringSize,
        maxRetention,
        kDefaultRingSize);

    ringBuffer_.emplace(ringSize);
    if (ringBuffer_->valid()) {
      ringReader_.emplace(*ringBuffer_);
    } else {
      XLOG_FIRST_N(ERR, 1) << logPrefix_
                           << ": Failed to allocate shared ring buffer";
      ringBuffer_.reset();
    }
  }

  // pluginByName_ is not used in the colltrace thread. It is okay to initialize
  // it after the colltrace thread starts
  for (const auto& plugin : plugins_) {
    XLOG(DBG0) << "Registering plugin " << plugin->getName();
    pluginByName_.emplace(plugin->getName(), *plugin);
  }

  // Start the poll thread after ring buffer initialization to avoid a data
  // race: the thread reads ringBuffer_.has_value() on its first iteration,
  // and std::optional is not thread-safe for concurrent read/write.
  traceCollThread_ =
      std::thread(&CollTrace::collTraceThread, this, threadSetupFunc);

  threadStarted_.wait();
}

CollTrace::~CollTrace() {
  // Set the cancellation flag
  threadShouldStop_.test_and_set();
  // Invalidate all eager handles.
  for (auto& [_, handle] : eventToHandleMap_) {
    handle->invalidate();
  }
  // Invalidate all graph handles.
  for (auto& [_, state] : graphStateMap_) {
    for (auto& [_, collEntry] : state->collectives) {
      if (auto h = collEntry.handle.lock()) {
        h->invalidate();
      }
    }
  }
  // Wait for the thread to finish
  if (traceCollThread_.joinable()) {
    traceCollThread_.join();
  }
}

std::shared_ptr<GraphCollTraceState> CollTrace::getOrCreateGraphState(
    cudaStream_t stream) {
  cudaStreamCaptureStatus captureStatus;
  unsigned long long graphId = 0;
  cudaGraph_t graph = nullptr;

#if CUDART_VERSION >= 13000
  auto res = cudaStreamGetCaptureInfo(
      stream, &captureStatus, &graphId, &graph, nullptr, nullptr, nullptr);
#else
  auto res = cudaStreamGetCaptureInfo_v2(
      stream, &captureStatus, &graphId, &graph, nullptr, nullptr);
#endif
  if (res != cudaSuccess || captureStatus != cudaStreamCaptureStatusActive ||
      graph == nullptr) {
    return nullptr;
  }

  // Hold the lock for the entire create-or-lookup path. This is a capture-time
  // path so contention is not a concern, and it avoids the race where two
  // threads both see an empty map and create duplicate state / leak CUDA user
  // objects.
  std::lock_guard<std::mutex> lock(graphStateMutex_);

  auto it = graphStateMap_.find(graphId);
  if (it != graphStateMap_.end()) {
    return it->second;
  }

  auto state = std::make_shared<GraphCollTraceState>();

  // heap alloc a copy s.t., the graph will keep the state alive until its
  // dtor flips the flag. the readers will see this and stop using
  // the state, which will drop the refcount to 0.
  auto* prevent_free = new std::shared_ptr<GraphCollTraceState>(state);

  cudaUserObject_t userObject;
  auto createRes = cudaUserObjectCreate(
      &userObject,
      prevent_free,
      graphCleanupCallback,
      1,
      cudaUserObjectNoDestructorSync);

  if (createRes != cudaSuccess) {
    delete prevent_free;
    XLOG_FIRST_N(WARN, 1) << "Failed to create graph cleanup user object: "
                          << cudaGetErrorString(createRes);
    return nullptr;
  }

  auto retainRes =
      cudaGraphRetainUserObject(graph, userObject, 1, cudaGraphUserObjectMove);
  if (retainRes != cudaSuccess) {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaUserObjectRelease(userObject, 1);
    XLOG_FIRST_N(WARN, 1) << "Failed to retain graph user object: "
                          << cudaGetErrorString(retainRes);
    return nullptr;
  }

  graphStateMap_.emplace(graphId, state);

  return state;
}

CommsMaybe<std::shared_ptr<ICollTraceHandle>> CollTrace::recordCollective(
    std::unique_ptr<ICollMetadata> metadata,
    std::unique_ptr<ICollWaitEvent> waitEvent) noexcept {
  if (metadata == nullptr) {
    return folly::makeUnexpected(CommsError(
        "Received nullptr for metadata during recordCollective",
        commInternalError));
  }
  if (waitEvent == nullptr) {
    return folly::makeUnexpected(CommsError(
        "Received nullptr for waitEvent during recordCollective",
        commInternalError));
  }

  if (dynamic_cast<GraphCudaWaitEvent*>(waitEvent.get()) != nullptr) {
    return recordGraphCollectiveImpl(std::move(metadata), std::move(waitEvent));
  }

  if (pendingEnqueueColl_ != nullptr) {
    XLOG_FIRST_N(
        ERR,
        1,
        fmt::format(
            "{}: Got another collective enqueued when a previous one haven't finished, colltrace result would be inaccurate. Previous: {}, Next:{}",
            logPrefix_,
            folly::toJson(pendingEnqueueColl_->collRecord->toDynamic()),
            folly::toJson(metadata->toDynamic())));
    auto handlePtr = eventToHandleMap_.find(pendingEnqueueColl_.get());
    if (handlePtr != eventToHandleMap_.end()) {
      handlePtr->second->invalidate();
      eventToHandleMap_.erase(pendingEnqueueColl_.get());
    }
  }

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

CommsMaybe<std::shared_ptr<ICollTraceHandle>>
CollTrace::recordGraphCollectiveImpl(
    std::unique_ptr<ICollMetadata> metadata,
    std::unique_ptr<ICollWaitEvent> waitEvent) noexcept {
  // we know this will be non-null because we checked it in the caller
  auto* rawWaitEvent = dynamic_cast<GraphCudaWaitEvent*>(waitEvent.get());

  // Assign a unique collId for this collective — used to tag entries in
  // the shared ring buffer so the poll thread can dispatch by collective.
  auto collIdVal = collId_.fetch_add(1);
  rawWaitEvent->setCollId(collIdVal);

  if (!ringBuffer_.has_value()) {
    return folly::makeUnexpected(CommsError(
        "Ringbuffer not initialized during recordGraphCollective",
        commInternalError));
  }

  // Get or create per-graph state (shared ring buffer, cleanup callback).
  auto graphState = getOrCreateGraphState(rawWaitEvent->getStream());

  if (graphState == nullptr) {
    return folly::makeUnexpected(CommsError(
        "Failed to initialize graph state for capturing stream",
        commInternalError));
  }

  rawWaitEvent->attachRingBuffer(&*ringBuffer_);

  auto collRecord =
      std::make_shared<CollRecord>(collIdVal, std::move(metadata));
  auto recordPtr = std::static_pointer_cast<ICollRecord>(collRecord);

  auto collEvent = std::make_unique<CollTraceEvent>(CollTraceEvent{
      .collRecord = std::move(collRecord),
      .waitEvent = std::move(waitEvent),
  });

  auto handle = std::make_shared<GraphCollTraceHandle>(
      rawWaitEvent, std::move(recordPtr));

  uint32_t collId = rawWaitEvent->getCollId();

  GraphCollectiveEntry collectiveEntry{
      .graphWaitEvent = rawWaitEvent,
      .event = std::move(collEvent),
      .handle = handle,
  };

  // afterCollKernelScheduled is fired per replay via kScheduleAndStart on the
  // poll thread — nothing to schedule at capture time.

  {
    std::lock_guard<std::mutex> lock(graphStateMutex_);
    graphState->collectives.emplace(collId, std::move(collectiveEntry));
  }

  return handle;
}

bool CollTrace::isThreadCancelled() const noexcept {
  return threadShouldStop_.test(std::memory_order_relaxed);
}

void CollTrace::pollGraphEvents(
    std::multiset<PendingAction>& actions) noexcept {
  if (!ringBuffer_.has_value() || !ringReader_.has_value()) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(graphStateMutex_);

    // check to see if any graphs have been destroyed.
    // if so, remove stop tracking associated state.
    std::erase_if(graphStateMap_, [this](const auto& entry) {
      const auto& state = entry.second;
      if (state->graph_destructed.load(std::memory_order_relaxed)) {
        for (const auto& [collId, collEntry] : state->collectives) {
          // Invalidate the handle to prevent use-after-free of the
          // raw GraphCudaWaitEvent pointer once we destroy the entry.
          if (auto h = collEntry.handle.lock()) {
            h->invalidate();
          }
          collIdMap_.erase(collId);
          progressingGraphCollectives_.erase(collId);
        }
        return true;
      }
      return false;
    });

    // add new collectives that aren't in collIdMap_ yet.
    for (auto& [_, state] : graphStateMap_) {
      for (auto& [collId, collEntry] : state->collectives) {
        if (collIdMap_.find(collId) == collIdMap_.end()) {
          collIdMap_[collId] = &collEntry;
        }
      }
    }
  }

  if (collIdMap_.empty()) {
    return;
  }

  const auto& cal = GlobaltimerCalibration::get();

  auto pollResult = ringReader_->poll(
      [&](const auto& entry, uint64_t /*slot*/) {
        auto collId = entry.data.collId;
        bool isStartEvent = entry.data.phase == GraphCollTracePhase::kStart;

        auto it = collIdMap_.find(collId);
        if (it == collIdMap_.end()) {
          return;
        }

        auto& collEntry = *(it->second);
        auto timestamp = cal.toWallClock(entry.timestamp_ns);

        if (isStartEvent) {
          // if we see a new start event before we observed this collectives
          // last end, the end event was likely overwritten (ring too small).
          // the watchdog detects this via the changed start timestamp and
          // resets its timer automatically but we should log it anyway
          if (auto [_, inserted] = progressingGraphCollectives_.insert(collId);
              !inserted) {
            XLOG_EVERY_MS(WARN, 5000)
                << logPrefix_ << ": graph collective " << collId
                << " saw a new start event before the previous end event"
                   " — the end event was likely overwritten (ring too small)";
          }
          collEntry.event->collRecord->getTimingInfo().setCollStartTs(
              timestamp);
          actions.insert(
              {collEntry.event.get(),
               PendingActionType::kScheduleAndStart,
               timestamp});
        } else /* isEndEvent */ {
          collEntry.event->collRecord->getTimingInfo().setCollEndTs(timestamp);
          progressingGraphCollectives_.erase(collId);
          actions.insert(
              {collEntry.event.get(), PendingActionType::kEnd, timestamp});
        }
      },
      /*timeout=*/config_.maxCheckCancelInterval);

  if (pollResult.entriesLost > 0) {
    XLOG_EVERY_MS(WARN, 5000)
        << logPrefix_ << ": missed " << pollResult.entriesLost
        << " graph replay timestamp(s) (overwritten)";
  }

  // Emit kProgressing for every pending start so the watchdog plugin's
  // timer accumulates. Without this, graph collectives would never trigger
  // the watchdog timeout because collEventProgressing is only called on
  // discrete ring buffer events.
  auto now = std::chrono::system_clock::now();
  for (auto collId : progressingGraphCollectives_) {
    auto it = collIdMap_.find(collId);
    if (it != collIdMap_.end()) {
      actions.insert(
          {it->second->event.get(), PendingActionType::kProgressing, now});
    }
  }
}

void CollTrace::pollEagerEvents(
    std::multiset<PendingAction>& actions) noexcept {
  {
    // drain pendingTraceColls_ into eagerEvents_
    std::unique_ptr<CollTraceEvent> event;
    while (pendingTraceColls_.read(event)) {
      if (event == nullptr) {
        XLOG_FIRST_N(
            ERR, 2, logPrefix_, ": Got null event from pendingTrace queue");
        continue;
      }
      eagerEvents_.push_back(std::move(event));
    }
  }

  static constexpr auto kEpoch = ICollWaitEvent::system_clock_time_point{};

  for (auto& event : eagerEvents_) {
    if (event == nullptr) {
      continue;
    }

    auto& timing = event->collRecord->getTimingInfo();
    bool started = timing.getCollStartTs() != kEpoch;
    bool ended = timing.getCollEndTs() != kEpoch;

    if (!started) {
      if (event->waitEvent->waitCollStart(config_.maxCheckCancelInterval)
              .value_or(false)) {
        auto now = std::chrono::system_clock::now();
        auto startTimeRes = event->waitEvent->getCollStartTime();
        auto startTs = startTimeRes.hasValue() ? startTimeRes.value() : now;
        timing.setCollStartTs(startTs);
        actions.insert({event.get(), PendingActionType::kStart, startTs});
        started = true;
      } else {
        // Fire progressing even before start so the watchdog plugin can
        // detect pre-start timeouts and async errors.
        actions.insert(
            {event.get(),
             PendingActionType::kProgressing,
             std::chrono::system_clock::now()});
      }
    }

    if (started && !ended) {
      if (event->waitEvent->waitCollEnd(config_.maxCheckCancelInterval)
              .value_or(false)) {
        auto now = std::chrono::system_clock::now();
        auto endTimeRes = event->waitEvent->getCollEndTime();
        auto endTs = endTimeRes.hasValue() ? endTimeRes.value() : now;
        timing.setCollEndTs(endTs);
        actions.insert({event.get(), PendingActionType::kEnd, endTs});
      } else {
        actions.insert(
            {event.get(),
             PendingActionType::kProgressing,
             std::chrono::system_clock::now()});
      }
    }
  }
}

void CollTrace::processCompletedEvents(
    std::multiset<PendingAction>& actions) noexcept {
  for (auto& action : actions) {
    switch (action.type) {
      case PendingActionType::kScheduleAndStart: {
        // for graph collectives, there is no actual scheduling
        // so we just fire the before/after callbacks here
        // in-case plugins depend on them
        triggerPlugins<&ICollTracePlugin::beforeCollKernelScheduled>(
            plugins_, *action.event);
        triggerPlugins<&ICollTracePlugin::afterCollKernelScheduled>(
            plugins_, *action.event);
        [[fallthrough]];
      }
      case PendingActionType::kStart: {
        if (lastCollEndTime_.has_value()) [[likely]] {
          action.event->collRecord->getTimingInfo().setPreviousCollEndTs(
              lastCollEndTime_.value());
        }
        triggerPlugins<&ICollTracePlugin::afterCollKernelStart>(
            plugins_, *action.event);
        break;
      }
      case PendingActionType::kEnd: {
        // Fire progressing one last time before end so plugins (e.g.
        // watchdog) can check async errors even for fast-completing
        // collectives. This matches the old behavior where
        // collEventProgressing was called unconditionally inside the
        // waitCollEnd loop.
        triggerPlugins<&ICollTracePlugin::collEventProgressing>(
            plugins_, *action.event);
        triggerPlugins<&ICollTracePlugin::afterCollKernelEnd>(
            plugins_, *action.event);
        lastCollEndTime_ =
            action.event->collRecord->getTimingInfo().getCollEndTs();
        break;
      }
      case PendingActionType::kProgressing: {
        triggerPlugins<&ICollTracePlugin::collEventProgressing>(
            plugins_, *action.event);
        break;
      }
    }
  }

  // clean up completed eager events...
  static constexpr auto kEpoch = ICollWaitEvent::system_clock_time_point{};
  std::erase_if(
      eagerEvents_, [this](const std::unique_ptr<CollTraceEvent>& event) {
        if (event == nullptr) {
          return true;
        }
        if (event->collRecord->getTimingInfo().getCollEndTs() == kEpoch) {
          return false;
        }
        auto it = eventToHandleMap_.find(event.get());
        if (it != eventToHandleMap_.end()) {
          it->second->invalidate();
          eventToHandleMap_.erase(it);
        }
        return true;
      });
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
    // Unblock the constructor so it doesn't hang forever.
    threadStarted_.count_down();
    return;
  }

  XLOGF(INFO, "{}: CollTrace thread STARTED", logPrefix_);

  bool initialized = false;

  while (!isThreadCancelled()) {
    std::multiset<PendingAction> actions;

    pollGraphEvents(actions);
    pollEagerEvents(actions);
    processCompletedEvents(actions);

    if (!initialized) {
      threadStarted_.count_down();
      initialized = true;
    }

    if (actions.empty()) {
      // No active events. Block on the MPMC queue so incoming eager events
      // wake the thread immediately instead of sleeping for the full
      // interval. Graph events are polled at the top of the next iteration.
      std::unique_ptr<CollTraceEvent> event;
      auto deadline =
          std::chrono::steady_clock::now() + config_.maxCheckCancelInterval;
      if (pendingTraceColls_.tryReadUntil(deadline, event)) {
        if (event != nullptr) {
          eagerEvents_.push_back(std::move(event));
        }
      }
    }
  }
}

} // namespace meta::comms::colltrace
