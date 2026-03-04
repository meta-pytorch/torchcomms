// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/CollTraceLegacyHandle.h"

#include "comms/ctran/gpe/CtranChecksum.h"
#include "meta/colltrace/CollTraceFunc.h"

// It is confusing that we have two namespaces for colltrace...
// But this should be temporary until we fully migrate to the new colltrace
using namespace ncclx::colltrace;

namespace meta::comms::colltrace {

CollTraceLegacyHandle::CollTraceLegacyHandle(
    CtranComm* comm,
    std::unique_ptr<::CollTraceEvent> event,
    HandleType type)
    : comm_(comm), event_(std::move(event)), handleType_(type) {
  XLOG(DBG2) << "CollTraceLegacyHandle constructor called.";
  if (event_ != nullptr) {
    collRecord_ = std::make_shared<CollTraceColl>(event_->coll);
    cpuStartEvent_ = collTraceGetCpuStartWaitEvent(*event_);
    cpuEndEvent_ = collTraceGetCpuEndWaitEvent(*event_);
  }
}

CommsMaybeVoid CollTraceLegacyHandle::trigger(
    CollTraceHandleTriggerState state) noexcept {
  if (!isValid()) {
    XLOG(DBG2)
        << "The current CollTraceLegacyHandle is not in valid state, cannot trigger";
    return folly::unit;
  }

  XLOG(DBG2) << "Trigger function called with state: "
             << static_cast<int>(state);
  switch (state) {
    case CollTraceHandleTriggerState::BeforeEnqueueKernel: {
      XLOG(DBG2) << "State: BeforeEnqueueKernel";
      // For CPU events in Ctran, we don't need to do anything here.
      // For baseline, CPU events means tracking cuda graph, which still needs
      // to be handled.
      if (event_->eventType == ::CollTraceEvent::EventType::COMM_CPU &&
          handleType_ == HandleType::ctran) {
        return folly::unit;
      }
      collTraceRecordStartEvent(event_->coll.stream, event_.get());
      return folly::unit;
    }
    case CollTraceHandleTriggerState::AfterEnqueueKernel: {
      XLOG(DBG2) << "State: AfterEnqueueKernel";
      if (event_->eventType == ::CollTraceEvent::EventType::COMM_CPU &&
          handleType_ == HandleType::ctran) {
        collTraceCtranSubmitEvent(std::move(event_));
      } else {
        // Could not directly pass event_->coll.stream to the next function
        // because event_ is being moved in the same function.
        auto stream = event_->coll.stream;
        collTraceRecordEndEvent(comm_, stream, std::move(event_));
      }
      return folly::unit;
    }
    case CollTraceHandleTriggerState::KernelStarted: {
      XLOG(DBG2) << "State: KernelStarted, CPU start event: "
                 << reinterpret_cast<uintptr_t>(cpuStartEvent_);
      if (cpuStartEvent_ != nullptr) {
        cpuStartEvent_->setFinished();
      }
      return folly::unit;
    }
    case CollTraceHandleTriggerState::KernelFinished: {
      XLOG(DBG2) << "State: KernelFinished, CPU start event: "
                 << reinterpret_cast<uintptr_t>(cpuEndEvent_);
      if (cpuEndEvent_ != nullptr) {
        cpuEndEvent_->setFinished();
      }
      return folly::unit;
    }
    default: {
      XLOG(WARNING) << "Unexpected trigger state";
      // Handle unexpected state
      return folly::makeUnexpected(
          CommsError("Unexpected trigger state", commInvalidArgument));
    }
  }
}

CommsMaybeVoid CollTraceLegacyHandle::triggerPlugin(
    std::string pluginName,
    folly::dynamic params) noexcept {
  if (event_ == nullptr) {
    XLOG(DBG2) << "CollTraceEvent is null, cannot trigger plugin";
    return folly::unit;
  }

  XLOG(DBG2) << "TriggerPlugin function called with pluginName: " << pluginName;
  if (pluginName != "ctranChecksum") {
    return folly::makeUnexpected(
        CommsError("Unexpected plugin name", commInvalidArgument));
  }
  ChecksumItem* checksum =
      reinterpret_cast<ChecksumItem*>(params["checksumItem"].asInt());
  event_->ctranChecksumItem = checksum;
  return folly::unit;
}

CommsMaybeVoid CollTraceLegacyHandle::invalidate() noexcept {
  XLOG(DBG2) << "Invalidate function called.";
  // Dummy implementation for invalidate
  event_.reset();
  cpuStartEvent_ = nullptr;
  cpuEndEvent_ = nullptr;
  return folly::unit;
}

CommsMaybe<std::shared_ptr<ICollRecord>>
CollTraceLegacyHandle::getCollRecord() noexcept {
  XLOG(DBG2) << "GetCollRecord function called.";
  return collRecord_;
}

bool CollTraceLegacyHandle::isValid() const noexcept {
  // We define valid as either event_ is not null or both cpuStartEvent_ and
  // cpuEndEvent_ are not null. This is because we will move event_ into
  // Colltrace after the kernel is enqueued, so we need to check both.
  return event_ != nullptr ||
      (cpuStartEvent_ != nullptr && cpuEndEvent_ != nullptr);
}

} // namespace meta::comms::colltrace
