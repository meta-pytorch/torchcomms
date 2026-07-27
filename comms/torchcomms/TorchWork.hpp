// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/core/ivalue.h> // @manual=//caffe2:ATen-core
#include <c10/util/Logging.h>
#include <c10/util/intrusive_ptr.h>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace torch::comms {

/**
 * Async work with concurrent status/hooks and single-caller waits.
 * Work objects must outlive any wait() in progress.
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

  // Opting in requires nonblocking status refresh that is safe alongside
  // concurrent backend watchdog polling.
  virtual bool supportsActivePolling() const {
    return false;
  }

  virtual WorkStatus pollStatus() {
    return status();
  }

  // Pure virtual functions that derived classes must implement
  virtual void wait() = 0;

  // Returns the timeout for this work object.
  // Derived classes with timeout support should override this.
  // Returns max() by default for work types that don't support timeout.
  virtual std::chrono::milliseconds getTimeout() const {
    return std::chrono::milliseconds::max();
  }

  // Host-blocks for operations, such as c10d barriers, whose contract requires
  // device completion before returning. Backends may override this no-op.
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

  // Observers can register concurrently with status transitions. Each start
  // or end hook runs exactly once, outside the hook mutex.

  using WorkHook = std::function<void()>;

  void registerWorkStartHook(WorkHook hook) {
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
      if (status() == WorkStatus::NOT_STARTED) {
        start_hooks_.push_back(std::move(hook));
        return;
      }
    }
    hook();
  }

  void registerWorkEndHook(WorkHook hook) {
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
      const auto s = status();
      if (s != WorkStatus::COMPLETED && s != WorkStatus::ERROR &&
          s != WorkStatus::TIMEDOUT) {
        end_hooks_.push_back(std::move(hook));
        return;
      }
    }
    hook();
  }

  void registerWorkWaitPreHook(WorkHook hook) {
    std::lock_guard<std::mutex> lock(hooksMutex_);
    wait_pre_hooks_.push_back(std::move(hook));
  }

  void registerWorkWaitPostHook(WorkHook hook) {
    std::lock_guard<std::mutex> lock(hooksMutex_);
    wait_post_hooks_.push_back(std::move(hook));
  }

  // Disable copy and move semantics
  TorchWork(const TorchWork&) = delete;
  TorchWork& operator=(const TorchWork&) = delete;
  TorchWork(TorchWork&&) = delete;
  TorchWork& operator=(TorchWork&&) = delete;

 protected:
  bool tryTransitionStatus(WorkStatus status) {
    auto current = status_.load(std::memory_order_relaxed);
    while (true) {
      if (isTerminal(current) || current == status ||
          (current == WorkStatus::INPROGRESS &&
           status == WorkStatus::NOT_STARTED)) {
        return false;
      }
      if (status_.compare_exchange_weak(
              current,
              status,
              std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  void dispatchStatusTransition(WorkStatus status) {
    if (status == WorkStatus::INPROGRESS) {
      runStartHooks();
    } else if (
        status == WorkStatus::COMPLETED || status == WorkStatus::ERROR ||
        status == WorkStatus::TIMEDOUT) {
      runEndHooks();
    }
  }

  void setStatus(WorkStatus status) {
    if (tryTransitionStatus(status)) {
      dispatchStatusTransition(status);
    }
  }

  // Backend wait() implementations should call these around the actual wait.
  void runWaitPreHooks() {
    std::vector<WorkHook> hooks;
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
      hooks = wait_pre_hooks_;
    }
    for (auto& hook : hooks) {
      hook();
    }
  }

  void runWaitPostHooks() {
    std::vector<WorkHook> hooks;
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
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
  static bool isTerminal(WorkStatus status) {
    return status == WorkStatus::COMPLETED || status == WorkStatus::TIMEDOUT ||
        status == WorkStatus::ERROR;
  }

  static void runHooks(std::vector<WorkHook>& hooks) {
    std::exception_ptr firstError;
    for (auto& hook : hooks) {
      try {
        hook();
      } catch (...) {
        if (!firstError) {
          firstError = std::current_exception();
          continue;
        }
        try {
          throw;
        } catch (const std::exception& error) {
          LOG(ERROR) << "[TC][TorchWork] Subsequent work hook failed: "
                     << error.what();
        } catch (...) {
          LOG(ERROR)
              << "[TC][TorchWork] Subsequent work hook failed with an unknown exception";
        }
      }
    }
    if (firstError) {
      std::rethrow_exception(firstError);
    }
  }

  void runStartHooks() {
    std::vector<WorkHook> hooks;
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
      hooks.swap(start_hooks_);
    }
    runHooks(hooks);
  }

  void runEndHooks() {
    std::vector<WorkHook> hooks;
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
      if (end_hooks_fired_) {
        return;
      }
      end_hooks_fired_ = true;
      hooks.swap(end_hooks_);
    }
    runHooks(hooks);
  }

  // Release captured weak pointers after the strong refcount reaches zero so
  // their destruction can release the remaining weak refcount.
  void release_resources() override {
    std::vector<WorkHook> startHooks;
    std::vector<WorkHook> endHooks;
    std::vector<WorkHook> waitPreHooks;
    std::vector<WorkHook> waitPostHooks;
    {
      std::lock_guard<std::mutex> lock(hooksMutex_);
      startHooks.swap(start_hooks_);
      endHooks.swap(end_hooks_);
      waitPreHooks.swap(wait_pre_hooks_);
      waitPostHooks.swap(wait_post_hooks_);
    }
  }

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  mutable std::mutex hooksMutex_;
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

  static c10::intrusive_ptr<TorchWorkThread> create(std::function<void()> fn);

  // Override virtual functions from TorchWork
  void wait() override;

 protected:
  TorchWorkThread() = default;
  std::future<void> future_;
};

} // namespace torch::comms
