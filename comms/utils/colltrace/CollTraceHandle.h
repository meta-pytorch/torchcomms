// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <string_view>

#include <folly/Synchronized.h>
#include <folly/dynamic.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

enum class CollTraceHandleTriggerState {
  BeforeEnqueueKernel,
  AfterEnqueueKernel,
  KernelStarted,
  KernelFinished,
  NumTriggerStates
};

std::string_view triggerStateToStr(CollTraceHandleTriggerState state);

class ICollTrace; // Declear CollTrace to avoid circular dependency

// Define the interface so that we can use it to handle legacy colltrace
class ICollTraceHandle {
 public:
  virtual ~ICollTraceHandle() = default;
  virtual CommsMaybeVoid trigger(
      CollTraceHandleTriggerState state) noexcept = 0;
  virtual CommsMaybeVoid triggerPlugin(
      std::string pluginName,
      folly::dynamic params) noexcept = 0;
  virtual CommsMaybe<std::shared_ptr<ICollRecord>> getCollRecord() noexcept = 0;
  virtual CommsMaybeVoid invalidate() noexcept = 0;
};

// Handle to be returned to the user for triggering stages for the collective.
class CollTraceHandle : public ICollTraceHandle {
 public:
  CollTraceHandle(ICollTrace* collTrace, CollTraceEvent* event);

  CommsMaybeVoid trigger(CollTraceHandleTriggerState state) noexcept override;

  CommsMaybeVoid triggerPlugin(
      std::string pluginName,
      folly::dynamic params) noexcept override;

  CommsMaybe<std::shared_ptr<ICollRecord>> getCollRecord() noexcept override;

  CommsMaybeVoid invalidate() noexcept override;

  // This is not safe! It should only be used in the case where we are sure
  // that the current thread is holding the **write** lock of state_.
  // Currently it is being used in the case of
  // trigger handle -> colltrace -> invalidate
  // In this case, we are already holding the write lock of state_ when we
  // call trigger handle, so it is safe to call invalidateUnsafe.
  void invalidateUnsafe() noexcept;

  // Delete copy constructor and copy assignment operator
  CollTraceHandle(const CollTraceHandle&) = delete;
  CollTraceHandle& operator=(const CollTraceHandle&) = delete;

  // Delete move constructor and move assignment operator
  CollTraceHandle(CollTraceHandle&&) = delete;
  CollTraceHandle& operator=(CollTraceHandle&&) = delete;

 private:
  CommsMaybeVoid checkTriggerStateValidity(
      CollTraceHandleTriggerState state) noexcept;

  // Record the last trigger state to ensure we don't trigger the same state
  // or trigger it in the wrong order. This is mostly being used to guard some
  // software error on the calling side. Including it in the state is going to
  // cause every thread to serialize on CollTraceHandleTrigger, which is not
  // what we want. So we use an atomic to record the last trigger state. We
  // might still hit some race condition, but it should not be a big deal.
  std::atomic<CollTraceHandleTriggerState> lastTriggerState_{
      CollTraceHandleTriggerState::NumTriggerStates};

  struct CollTraceHandleState {
    ICollTrace* collTrace_;
    // For event, it is intended to be a handle to a CollTraceEvent object
    // it should only be used to pass into CollTrace so that CollTrace can
    // trigger the right event. It should not be used to access the event
    // object directly.
    CollTraceEvent* event_;

    // This is used to ensure that the handle is not used after the collective
    // or colltrace is destroyed. CollTrace will be responsible for signaling
    // the invalidation of CollTrace or CollTraceEvent.
    bool referenceInvalidated_{false};
  };
  folly::Synchronized<CollTraceHandleState> state_;
};

} // namespace meta::comms::colltrace
