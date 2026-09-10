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
 *  - anything a derived backend owns. This base synchronizes only its own
 *    status and hook state; backend members (tensor references, events,
 *    streams) remain single-threaded by contract. Several backends currently
 *    clear tensor references from both the watchdog and the waiting thread --
 *    that is a pre-existing race in those backends, not something this class
 *    makes safe.
 *
 * Start-hook exception, by design:
 *  - a start hook queued before a direct NOT_STARTED -> terminal transition is
 *    DROPPED, not fired. Start hooks fire only on setStatus(INPROGRESS): a work
 *    that never started must not report that it did. This is deliberate, and
 *    TorchWorkTest.StartHookNotFiredOnTerminalStatus pins it. A hook registered
 *    *after* such a transition still fires immediately, so registering late is
 *    safe.
 *
 * status() is a relaxed atomic load: it is coherent and race-free, and it
 * publishes nothing else. Do not use it to order access to other state.
 *
 * Hooks are invoked outside the internal lock, so a hook may query the work it
 * is attached to. Do not rely on which thread runs a hook: it is whichever
 * thread made the transition, or the registering thread on the late path.
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
  // Multiple hooks can be registered; they fire in registration order.
  // Registration is safe against a concurrent transition -- see the class
  // comment.

  using WorkHook = std::function<void()>;

  void registerWorkStartHook(WorkHook hook) {
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      if (!start_hooks_fired_) {
        start_hooks_.push_back(std::move(hook));
        return;
      }
    }
    // Already started: fire now, or the event would be lost entirely. This
    // handles backends (e.g. MCCL) whose ctor sets INPROGRESS before the
    // post-hook registers the start hook; without it the clog has no "S".
    hook();
  }

  void registerWorkEndHook(WorkHook hook) {
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      if (!end_hooks_fired_) {
        end_hooks_.push_back(std::move(hook));
        return;
      }
    }
    // Already terminal: fire now rather than enqueuing a hook that would never
    // run.
    hook();
  }

  // The wait hooks have no fired-flag: unlike start/end they can run more than
  // once, so registration only needs to be safe against release_resources()
  // and against the snapshot taken in runWaitPre/PostHooks().
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

  // The single choke point for status transitions and hook firing. Both
  // invariants have to be enforced together, which is why this is one
  // synchronized block rather than a pair of fire-once helpers:
  //  - the first terminal transition wins and is sticky, so a watchdog TIMEDOUT
  //    and a training-thread COMPLETED cannot overwrite each other and leave
  //    the surviving status dependent on scheduling;
  //  - a work that goes straight from NOT_STARTED to terminal still latches
  //    start hooks, so one registered afterwards fires instead of queueing
  //    forever.
  void setStatus(WorkStatus status) {
    std::vector<WorkHook> to_fire;
    {
      std::lock_guard<std::mutex> lock(hooks_mutex_);
      if (end_hooks_fired_) {
        // Terminal already latched: ignore, status included.
        return;
      }
      status_.store(status, std::memory_order_relaxed);
      if (isTerminal(status)) {
        end_hooks_fired_ = true;
        to_fire.swap(end_hooks_);
        // Start hooks can never fire after this point, so mark them fired and
        // release them. Hooks queued *before* a direct-to-terminal transition
        // are dropped, which preserves the pre-existing behavior.
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
    // Invoked outside the lock: a hook may legitimately query this work, and
    // holding a work-level lock across user code would deadlock.
    for (auto& hook : to_fire) {
      hook();
    }
  }

  // Backend wait() implementations should call these around the actual wait.
  // Snapshot under the lock, invoke outside it -- same rule as the start/end
  // hooks. These copy rather than swap: wait hooks are not one-shot.
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
  // break weak-ref cycle: hooks registered via postHook() may capture a
  // weak_intrusive_ptr back to this object. after the strong refcount
  // reaches 0, release_resources() clears the hooks, destroying the weak
  // pointers and allowing the weak refcount to reach 0 so the object is
  // deleted.
  //
  // Locked: this is a third writer to the hook vectors, alongside registration
  // and firing.
  void release_resources() override {
    std::lock_guard<std::mutex> lock(hooks_mutex_);
    start_hooks_.clear();
    end_hooks_.clear();
    wait_pre_hooks_.clear();
    wait_post_hooks_.clear();
  }

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};

  // Guards the hook vectors and their fired flags. The flags are plain bools
  // on purpose: the "already fired?" test and the append must be atomic with
  // respect to each other, so both only ever happen under this mutex. Making
  // them std::atomic would invite testing them outside the lock, which is
  // exactly the lost-hook bug -- a registrar that reads "not fired", is
  // preempted by the transition, and then appends to a vector nobody will read
  // again.
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
