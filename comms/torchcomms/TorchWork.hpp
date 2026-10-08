// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// IWYU pragma: no_include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <vector>

namespace at {
class Tensor;
} // namespace at
namespace c10::ivalue {
struct Future;
} // namespace c10::ivalue

namespace torch::comms {

/**
 * TorchWork - Base class representing asynchronous work.
 *
 * Thread Safety:
 * Partially thread-safe -- read this before assuming either extreme.
 *
 * Safe against concurrent access:
 *  - status() and isCompleted()
 *  - setStatus(), including two threads racing to a terminal status. The first
 *    terminal transition wins and is sticky; later ones are ignored entirely.
 *  - registerWorkStartHook() / registerWorkEndHook() against a concurrent
 *    transition. A hook is either queued and fired by the transition, or fired
 *    immediately on the registering thread if the transition already
 *    happened -- never fired twice. See the start-hook exception below.
 *
 * NOT thread-safe, still single-threaded by contract:
 *  - wait(), waitBlocking(), hostSynchronize() and backend accessors.
 *  - derived-backend state, including tensor references, events, and streams.
 *    Backends must synchronize that state independently.
 *
 * Start-hook exception, by design:
 *  - a start hook queued before a direct NOT_STARTED -> terminal transition is
 *    DROPPED, not fired. Start hooks fire only on setStatus(INPROGRESS): a work
 *    that never started must not report that it did. This is deliberate, and
 *    TorchWorkTest.StartHookNotFiredOnTerminalStatus pins it. A hook registered
 *    *after* such a transition still fires immediately, so registering late is
 *    safe.
 *
 * status() is race-free but does not publish or order access to other state.
 *
 * Hooks may query the work they are attached to. They run on the thread that
 * made the transition, or the registering thread on the late path. Start and
 * end hooks are failure-isolated observers: an exception is logged and does
 * not escape or suppress later lifecycle observers. Reusable wait-hook
 * exceptions still propagate to the caller of wait().
 *
 * Work objects should not be destroyed while wait() is in progress.
 */
class TorchWork : public c10::intrusive_ptr_target {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWork() = default;
  ~TorchWork() override = default;

  WorkStatus status() const {
    return status_.load(std::memory_order_relaxed);
  }
  bool isCompleted() const {
    return status() == WorkStatus::COMPLETED;
  }

  bool hasTerminalStatusProducer() const noexcept {
    return has_terminal_status_producer_.load(std::memory_order_relaxed);
  }

  void enableTerminalStatusProducer() noexcept {
    has_terminal_status_producer_.store(true, std::memory_order_relaxed);
  }

  // Pure virtual functions that derived classes must implement
  virtual void wait() = 0;

  // Returns the timeout for this work object.
  // Derived classes with timeout support should override this.
  // Returns max() by default for work types that don't support timeout.
  virtual std::chrono::milliseconds getTimeout() const {
    return std::chrono::milliseconds::max();
  }

  // Block the calling CPU thread until the device work behind this object has
  // completed (in addition to the stream-ordered wait()). Invoked by the c10d
  // WorkWrapper for synchronous barriers to mirror stock ProcessGroupNCCL,
  // whose barrier host-blocks the CPU thread. No-op by default; backends whose
  // wait() is already host-blocking (e.g. CPU/gloo) need not override it.
  virtual void hostSynchronize() {}

  // Fault Tolerance API

  /**
   * Block the CPU thread until the work is completed.
   * Unlike wait(), which blocks only the current CUDA stream, this method
   * blocks the CPU thread itself until the operation completes.
   *
   * @throws std::runtime_error if not implemented by the backend.
   */
  virtual void waitBlocking() {
    throw std::runtime_error(
        "[TorchWork]: waitBlocking not implemented for this work type");
  }

  // -- Work lifecycle hooks --
  //
  // These hooks allow external observers to track work object state
  // transitions without coupling to specific backend implementations.
  //
  // - Start hook:  fired when setStatus(INPROGRESS) is called
  // - End hook:    fired when setStatus(COMPLETED/ERROR/TIMEDOUT) is called
  // - Wait pre hook:  fired at the start of wait(), before the sync
  // - Wait post hook: fired at the end of wait(), after the sync
  //
  // Hooks registered before a transition run in order. Late start and end
  // hooks run immediately and may overlap callbacks already being dispatched.

  using WorkHook = std::function<void()>;

  void registerWorkStartHook(WorkHook hook) {
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      if (!start_hooks_fired_) {
        start_hooks_.push_back(std::move(hook));
        return;
      }
    }
    // Fire late hooks immediately so observers see a complete lifecycle.
    runLifecycleHook(hook);
  }

  void registerWorkEndHook(WorkHook hook) {
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      if (!end_hooks_fired_) {
        end_hooks_.push_back(std::move(hook));
        return;
      }
    }
    // Fire late hooks immediately because no later terminal transition exists.
    runLifecycleHook(hook);
  }

  // Wait hooks are reusable, so registration is synchronized with cleanup and
  // snapshotting rather than latched.
  void registerWorkWaitPreHook(WorkHook hook) {
    std::lock_guard<std::mutex> lock(hooks_mutex_);
    wait_pre_hooks_.push_back(std::move(hook));
  }

  void registerWorkWaitPostHook(WorkHook hook) {
    std::lock_guard<std::mutex> lock(hooks_mutex_);
    wait_post_hooks_.push_back(std::move(hook));
  }

  // Disable copy and move semantics
  TorchWork(const TorchWork&) = delete;
  TorchWork& operator=(const TorchWork&) = delete;
  TorchWork(TorchWork&&) = delete;
  TorchWork& operator=(TorchWork&&) = delete;

 protected:
  static bool isTerminal(WorkStatus status) {
    return status == WorkStatus::COMPLETED || status == WorkStatus::ERROR ||
        status == WorkStatus::TIMEDOUT;
  }

  // Serialize status and hook transitions so the first terminal state remains
  // authoritative and direct-to-terminal work cannot strand late start hooks.
  void setStatus(WorkStatus status) {
    std::vector<WorkHook> to_fire;
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      if (end_hooks_fired_) {
        return;
      }
      status_.store(status, std::memory_order_relaxed);
      if (isTerminal(status)) {
        end_hooks_fired_ = true;
        to_fire.swap(end_hooks_);
        // Direct-to-terminal work cannot emit a start event, so discard queued
        // start hooks while latching the phase for late registrants.
        start_hooks_fired_ = true;
        start_hooks_.clear();
      } else if (status == WorkStatus::INPROGRESS) {
        if (start_hooks_fired_) {
          return;
        }
        start_hooks_fired_ = true;
        to_fire.swap(start_hooks_);
      }
    }
    // Invoke callbacks outside the lock so reentrant hooks cannot deadlock.
    for (auto& hook : to_fire) {
      runLifecycleHook(hook);
    }
  }

  // Snapshot reusable wait hooks under the lock, then invoke them outside it
  // so callbacks may safely reenter the work.
  void runWaitPreHooks() {
    std::vector<WorkHook> hooks;
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      hooks = wait_pre_hooks_;
    }
    for (auto& hook : hooks) {
      hook();
    }
  }

  void runWaitPostHooks() {
    std::vector<WorkHook> hooks;
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      hooks = wait_post_hooks_;
    }
    for (auto& hook : hooks) {
      hook();
    }
  }

  friend class TorchComm;
  friend class WorkWrapper;

  virtual void markCompleted(
      c10::intrusive_ptr<c10::ivalue::Future> future_,
      std::vector<at::Tensor> outputTensors_);

  template <typename T, typename NullType>
  friend class c10::intrusive_ptr;

 private:
  static void runLifecycleHook(WorkHook& hook) noexcept;

  // break weak-ref cycle: hooks registered via postHook() may capture a
  // weak_intrusive_ptr back to this object. after the strong refcount
  // reaches 0, release_resources() clears the hooks, destroying the weak
  // pointers and allowing the weak refcount to reach 0 so the object is
  // deleted.
  void release_resources() override {
    std::lock_guard<std::mutex> lock(hooks_mutex_);
    start_hooks_.clear();
    end_hooks_.clear();
    wait_pre_hooks_.clear();
    wait_post_hooks_.clear();
  }

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  std::atomic<bool> has_terminal_status_producer_{false};

  // Guard each fired-flag check with its hook-vector mutation so registration
  // cannot be lost during a transition.
  std::mutex hooks_mutex_;
  bool start_hooks_fired_{false};
  bool end_hooks_fired_{false};

  std::vector<WorkHook> start_hooks_;
  std::vector<WorkHook> end_hooks_;
  std::vector<WorkHook> wait_pre_hooks_;
  std::vector<WorkHook> wait_post_hooks_;
};

class TorchWorkCompleted : public TorchWork {
 public:
  TorchWorkCompleted();
  ~TorchWorkCompleted() override = default;

  // Override virtual functions from TorchWork
  void wait() override;

  void waitBlocking() override;
};

class TorchWorkThread : public TorchWork {
 public:
  explicit TorchWorkThread(std::function<void()> fn);
  ~TorchWorkThread() override = default;

  // Override virtual functions from TorchWork
  void wait() override;

 private:
  std::future<void> future_;
};

} // namespace torch::comms
