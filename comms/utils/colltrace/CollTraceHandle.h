// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>

#include <folly/Synchronized.h>
#include <folly/dynamic.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/ColltraceDeviceHandle.h"
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

class EagerCancellationGate {
 public:
  using Cancel = std::function<CommsMaybeVoid(CollTraceEvent&)>;

  explicit EagerCancellationGate(Cancel cancel) : cancel_(std::move(cancel)) {}

  // Takes the event by pointer, not by reference: the caller cannot safely
  // form the reference before entering the gate, because teardown can destroy
  // the owner's event storage first. The pointer is dereferenced only under
  // mutex_, after the callback has been confirmed live.
  CommsMaybeVoid cancel(CollTraceEvent* event) noexcept {
    std::lock_guard lock(mutex_);
    if (!cancel_ || event == nullptr) {
      return folly::unit;
    }
    return cancel_(*event);
  }

  void shutdown() noexcept {
    std::lock_guard lock(mutex_);
    cancel_ = nullptr;
  }

 private:
  std::mutex mutex_;
  Cancel cancel_;
};

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
  // Cancel is part of the enqueue state machine and must be called by the
  // thread that records and triggers this handle, before AfterEnqueueKernel.
  virtual CommsMaybeVoid cancel() noexcept = 0;
  virtual CommsMaybeVoid invalidate() noexcept = 0;

  // In-kernel colltrace: expose the graph ring device handle + collId so the
  // GPE can point the collective kernel at where to write its own start/end
  // timestamps, instead of the host-launched timestamp kernels. Returns a
  // non-capable handle for cases that don't support it (eager, dummy); only the
  // graph handle overrides this.
  virtual ColltraceDeviceHandle getColltraceDeviceHandle() noexcept {
    return {};
  }
};

// Cancels the handle on scope exit unless disarmed. A launch path that records
// a handle and then returns early -- driver lookup failure, kernel launch
// failure -- would otherwise strand the pending record, and the next
// collective would report an overlap and drop the stranded trace. Arm this at
// the record site and disarm it once AfterEnqueueKernel has succeeded.
class CollTraceEnqueueGuard {
 public:
  CollTraceEnqueueGuard() noexcept = default;
  explicit CollTraceEnqueueGuard(
      std::shared_ptr<ICollTraceHandle> handle) noexcept
      : handle_(std::move(handle)) {}

  CollTraceEnqueueGuard(CollTraceEnqueueGuard&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  CollTraceEnqueueGuard& operator=(CollTraceEnqueueGuard&& other) noexcept {
    if (this != &other) {
      cancelNow();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  CollTraceEnqueueGuard(const CollTraceEnqueueGuard&) = delete;
  CollTraceEnqueueGuard& operator=(const CollTraceEnqueueGuard&) = delete;

  ~CollTraceEnqueueGuard() {
    cancelNow();
  }

  void disarm() noexcept {
    handle_ = nullptr;
  }

 private:
  void cancelNow() noexcept {
    if (handle_ != nullptr) {
      handle_->cancel();
      handle_ = nullptr;
    }
  }

  std::shared_ptr<ICollTraceHandle> handle_;
};

// Handle to be returned to the user for triggering stages for the collective.
class CollTraceHandle : public ICollTraceHandle {
 public:
  CollTraceHandle(
      ICollTrace* collTrace,
      CollTraceEvent* event,
      std::shared_ptr<EagerCancellationGate> cancellationGate = nullptr);

  CommsMaybeVoid trigger(CollTraceHandleTriggerState state) noexcept override;

  CommsMaybeVoid triggerPlugin(
      std::string pluginName,
      folly::dynamic params) noexcept override;

  CommsMaybe<std::shared_ptr<ICollRecord>> getCollRecord() noexcept override;

  CommsMaybeVoid cancel() noexcept override;

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
    std::shared_ptr<EagerCancellationGate> cancellationGate_;

    // This is used to ensure that the handle is not used after the collective
    // or colltrace is destroyed. CollTrace will be responsible for signaling
    // the invalidation of CollTrace or CollTraceEvent.
    bool referenceInvalidated_{false};
  };
  folly::Synchronized<CollTraceHandleState> state_;
};

} // namespace meta::comms::colltrace
