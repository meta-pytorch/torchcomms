// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <mutex>

#include <folly/Synchronized.h>

#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollWaitEvent.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

class GraphCancellationGate {
 public:
  using Cancel = std::function<CommsMaybeVoid(uint32_t)>;

  explicit GraphCancellationGate(Cancel cancel) : cancel_(std::move(cancel)) {}

  CommsMaybeVoid cancel(uint32_t collId) noexcept {
    std::lock_guard lock(mutex_);
    if (!cancel_) {
      return folly::unit;
    }
    return cancel_(collId);
  }

  void shutdown() noexcept {
    std::lock_guard lock(mutex_);
    cancel_ = nullptr;
  }

 private:
  std::mutex mutex_;
  Cancel cancel_;
};

// Handle for graph-captured collectives. Unlike CollTraceHandle, this does
// not interact with the serial queue. It delegates before/after kernel
// scheduling to the underlying ICollWaitEvent and is otherwise a no-op.
class GraphCollTraceHandle : public ICollTraceHandle {
 public:
  explicit GraphCollTraceHandle(
      ICollWaitEvent* waitEvent,
      std::shared_ptr<ICollRecord> record,
      std::shared_ptr<GraphCancellationGate> cancellationGate,
      uint32_t collId)
      : state_(
            State{
                .waitEvent = waitEvent,
                .record = std::move(record),
                .cancellationGate = std::move(cancellationGate),
                .collId = collId,
            }) {}

  ~GraphCollTraceHandle() override = default;

  CommsMaybeVoid trigger(CollTraceHandleTriggerState state) noexcept override {
    auto handleState = state_.rlock();
    if (handleState->waitEvent == nullptr) {
      return folly::Unit{};
    }
    switch (state) {
      case CollTraceHandleTriggerState::BeforeEnqueueKernel:
        return handleState->waitEvent->beforeCollKernelScheduled();
      case CollTraceHandleTriggerState::AfterEnqueueKernel:
        return handleState->waitEvent->afterCollKernelScheduled();
      case CollTraceHandleTriggerState::KernelStarted:
        return handleState->waitEvent->signalCollStart();
      case CollTraceHandleTriggerState::KernelFinished:
        return handleState->waitEvent->signalCollEnd();
      case CollTraceHandleTriggerState::NumTriggerStates:
        return folly::Unit{};
    }
    return folly::Unit{};
  }

  CommsMaybeVoid triggerPlugin(
      std::string /* pluginName */,
      folly::dynamic /* params */) noexcept override {
    return folly::Unit{};
  }

  CommsMaybe<std::shared_ptr<ICollRecord>> getCollRecord() noexcept override {
    return state_.rlock()->record;
  }

  CommsMaybeVoid cancel() noexcept override {
    std::shared_ptr<GraphCancellationGate> cancellationGate;
    uint32_t collId;
    {
      auto handleState = state_.wlock();
      cancellationGate = std::move(handleState->cancellationGate);
      collId = handleState->collId;
      handleState->waitEvent = nullptr;
      handleState->record = nullptr;
    }
    if (cancellationGate == nullptr) {
      return folly::unit;
    }
    return cancellationGate->cancel(collId);
  }

  CommsMaybeVoid invalidate() noexcept override {
    auto handleState = state_.wlock();
    handleState->waitEvent = nullptr;
    handleState->record = nullptr;
    handleState->cancellationGate = nullptr;
    return folly::Unit{};
  }

  ColltraceDeviceHandle getColltraceDeviceHandle() noexcept override {
    auto handleState = state_.rlock();
    auto* graphWaitEvent = graphWaitEvent_(handleState->waitEvent);
    if (graphWaitEvent == nullptr || !graphWaitEvent->hasRingBuffer()) {
      return {};
    }
    // emitStart/emitEnd are set by the arming code (armInKernelColltrace /
    // armBaselineInKernelColltrace) per the kernel's role; initialize them
    // explicitly here since ColltraceDeviceHandle has no default member
    // initializers (for __shared__ compatibility).
    return {
        .collId = graphWaitEvent->getCollId(),
        .emitStart = false,
        .emitEnd = false,
        .ring = graphWaitEvent->deviceHandle()};
  }

 private:
  struct State {
    ICollWaitEvent* waitEvent;
    std::shared_ptr<ICollRecord> record;
    std::shared_ptr<GraphCancellationGate> cancellationGate;
    uint32_t collId;
  };

  // waitEvent_ is always a GraphCudaWaitEvent for this handle (constructed by
  // CollTrace::recordGraphCollectiveImpl), or null after invalidate(). This is
  // on the submit hot path, so static_cast avoids RTTI; the debug-only assert
  // catches any future violation of that invariant.
  static GraphCudaWaitEvent* graphWaitEvent_(ICollWaitEvent* waitEvent) {
    assert(
        waitEvent == nullptr ||
        dynamic_cast<GraphCudaWaitEvent*>(waitEvent) != nullptr);
    return static_cast<GraphCudaWaitEvent*>(waitEvent);
  }

  folly::Synchronized<State> state_;
};

} // namespace meta::comms::colltrace
