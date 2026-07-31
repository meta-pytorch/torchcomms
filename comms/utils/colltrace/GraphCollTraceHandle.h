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
    // Killswitch: when NCCLX_COLLTRACE_DEVICE_WRITE is off, hand out no device
    // ring handle, so neither the ctran nor the baseline arming path arms the
    // kernel (both gate on valid()). This is the single gate for the in-kernel
    // device-write across all backends.
    if (!colltraceDeviceWriteEnabled()) {
      return {};
    }
    auto* graphWaitEvent = graphWaitEvent_();
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

  // TEMPORARY (removed at the in-kernel cutover): forward the "kernel
  // self-emits" signal to the wait event so it skips the host-launched
  // timestamp path.
  void markInKernelEmit() noexcept override {
    // Gate identically to getColltraceDeviceHandle(): only mark the wait event
    // as self-emitting when the killswitch is on and a ring is attached, so we
    // never skip the host-launched fallback for a kernel that isn't actually
    // armed to self-emit.
    if (!colltraceDeviceWriteEnabled()) {
      return;
    }
    auto* graphWaitEvent = graphWaitEvent_();
    if (graphWaitEvent == nullptr || !graphWaitEvent->hasRingBuffer()) {
      return;
    }
    graphWaitEvent->markInKernelEmit();
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
