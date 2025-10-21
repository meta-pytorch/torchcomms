// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CollTraceHandle.h"

#include <folly/Unit.h>

#include "comms/utils/CommsMaybeChecks.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

std::string_view triggerStateToStr(CollTraceHandleTriggerState state) {
  switch (state) {
    case CollTraceHandleTriggerState::BeforeEnqueueKernel:
      return "BeforeEnqueueKernel";
    case CollTraceHandleTriggerState::AfterEnqueueKernel:
      return "AfterEnqueueKernel";
    case CollTraceHandleTriggerState::KernelStarted:
      return "KernelStarted";
    case CollTraceHandleTriggerState::KernelFinished:
      return "KernelFinished";
    default:
      return "UnknownState";
  }
}

CollTraceHandle::CollTraceHandle(ICollTrace* collTrace, CollTraceEvent* event)
    : state_(CollTraceHandleState{.collTrace_ = collTrace, .event_ = event}) {}

CommsMaybeVoid CollTraceHandle::checkTriggerStateValidity(
    CollTraceHandleTriggerState state) noexcept {
  // Check if we're triggering in the correct order. All of the checks below
  // are unlikely to be true, but we want to ensure that we're not triggering
  // states in the wrong order, which might create pretty hard to debug issues

  auto lastState = lastTriggerState_.load();

  if (lastState == CollTraceHandleTriggerState::NumTriggerStates) {
    if (state != CollTraceHandleTriggerState::BeforeEnqueueKernel)
        [[unlikely]] {
      return folly::makeUnexpected(CommsError(
          "First state must be BeforeEnqueueKernel", commInvalidArgument));
    } else {
      return folly::unit;
    }
  }

  // Simple validation to ensure states are triggered in the correct order
  if (lastState == state) [[unlikely]] {
    return folly::makeUnexpected(
        CommsError("Same state triggered multiple times", commInvalidArgument));
  }

  // Ensure proper ordering of states. It IS possible for KernelStarted to be
  // triggered before AfterEnqueueKernel, as GPE thread and Kernel might start
  // before the enqueue thread to call AfterEnqueueKernel. In rare cases, the
  // kernel might even finish before enqueue kernel is called
  if ((state == CollTraceHandleTriggerState::KernelStarted &&
       (lastState != CollTraceHandleTriggerState::BeforeEnqueueKernel &&
        lastState != CollTraceHandleTriggerState::AfterEnqueueKernel)) ||
      (state == CollTraceHandleTriggerState::KernelFinished &&
       (lastState != CollTraceHandleTriggerState::KernelStarted &&
        lastState != CollTraceHandleTriggerState::BeforeEnqueueKernel &&
        lastState != CollTraceHandleTriggerState::AfterEnqueueKernel)))
      [[unlikely]] {
    return folly::makeUnexpected(CommsError(
        fmt::format(
            "States triggered in incorrect order, last state: {}, triggering state: {}",
            triggerStateToStr(lastState),
            triggerStateToStr(state)),
        commInvalidArgument));
  }
  return folly::unit;
}

CommsMaybeVoid CollTraceHandle::trigger(
    CollTraceHandleTriggerState state) noexcept {
  auto stateReadLocked = state_.rlock();
  if (stateReadLocked->referenceInvalidated_) [[unlikely]] {
    return folly::makeUnexpected(
        CommsError("Handle is invalidated", commInvalidArgument));
  }

  if (stateReadLocked->collTrace_ == nullptr) [[unlikely]] {
    return folly::makeUnexpected(CommsError(
        "CollTrace is null, cannot trigger event state", commInternalError));
  }

  if (stateReadLocked->event_ == nullptr) [[unlikely]] {
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent is null, cannot trigger event state",
        commInternalError));
  }

  EXPECT_CHECK(checkTriggerStateValidity(state));

  auto res = stateReadLocked->collTrace_->triggerEventState(
      *stateReadLocked->event_, state);
  if (!res.hasError()) {
    lastTriggerState_.store(state);
  }
  return res;
}

CommsMaybeVoid CollTraceHandle::triggerPlugin(
    std::string pluginName,
    folly::dynamic params) noexcept {
  return folly::makeUnexpected(CommsError(
      "Currently trigger plugin from the new CollTrace is not supported",
      commInternalError));
}

CommsMaybe<std::shared_ptr<ICollRecord>>
CollTraceHandle::getCollRecord() noexcept {
  auto stateReadLocked = state_.rlock();
  if (stateReadLocked->referenceInvalidated_) [[unlikely]] {
    return folly::makeUnexpected(
        CommsError("Handle is invalidated", commInvalidArgument));
  }

  if (stateReadLocked->event_ == nullptr) {
    return nullptr;
  }

  return stateReadLocked->event_->collRecord;
}

CommsMaybeVoid CollTraceHandle::invalidate() noexcept {
  // Set the invalidated flag
  state_.wlock()->referenceInvalidated_ = true;

  return folly::unit;
}

} // namespace meta::comms::colltrace
