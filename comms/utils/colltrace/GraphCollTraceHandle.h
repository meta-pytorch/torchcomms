// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cassert>

#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollWaitEvent.h"
#include "comms/utils/colltrace/GraphCudaWaitEvent.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

// Handle for graph-captured collectives. Unlike CollTraceHandle, this does
// not interact with the serial queue. It delegates before/after kernel
// scheduling to the underlying ICollWaitEvent and is otherwise a no-op.
class GraphCollTraceHandle : public ICollTraceHandle {
 public:
  explicit GraphCollTraceHandle(
      ICollWaitEvent* waitEvent,
      std::shared_ptr<ICollRecord> record)
      : waitEvent_(waitEvent), record_(std::move(record)) {}

  ~GraphCollTraceHandle() override = default;

  CommsMaybeVoid trigger(CollTraceHandleTriggerState state) noexcept override {
    if (waitEvent_ == nullptr) {
      return folly::Unit{};
    }
    switch (state) {
      case CollTraceHandleTriggerState::BeforeEnqueueKernel:
        return waitEvent_->beforeCollKernelScheduled();
      case CollTraceHandleTriggerState::AfterEnqueueKernel:
        return waitEvent_->afterCollKernelScheduled();
      case CollTraceHandleTriggerState::KernelStarted:
        return waitEvent_->signalCollStart();
      case CollTraceHandleTriggerState::KernelFinished:
        return waitEvent_->signalCollEnd();
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
    return record_;
  }

  CommsMaybeVoid invalidate() noexcept override {
    waitEvent_ = nullptr;
    record_ = nullptr;
    return folly::Unit{};
  }

  ColltraceDeviceHandle getColltraceDeviceHandle() noexcept override {
    auto* graphWaitEvent = graphWaitEvent_();
    if (graphWaitEvent == nullptr || !graphWaitEvent->hasRingBuffer()) {
      return {};
    }
    return {
        .collId = graphWaitEvent->getCollId(),
        .ring = graphWaitEvent->deviceHandle()};
  }

  void suppressInterKernelTimestamps() noexcept override {
    if (auto* graphWaitEvent = graphWaitEvent_()) {
      graphWaitEvent->setIntraKernelEmitActive();
    }
  }

 private:
  // waitEvent_ is always a GraphCudaWaitEvent for this handle (constructed by
  // CollTrace::recordGraphCollectiveImpl), or null after invalidate(). This is
  // on the submit hot path, so static_cast avoids RTTI; the debug-only assert
  // catches any future violation of that invariant.
  GraphCudaWaitEvent* graphWaitEvent_() const {
    assert(
        waitEvent_ == nullptr ||
        dynamic_cast<GraphCudaWaitEvent*>(waitEvent_) != nullptr);
    return static_cast<GraphCudaWaitEvent*>(waitEvent_);
  }

  ICollWaitEvent* waitEvent_;
  std::shared_ptr<ICollRecord> record_;
};

} // namespace meta::comms::colltrace
