// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/util/intrusive_ptr.h>
#include <chrono>
#include <functional>
#include <future>

namespace torch::comms {

/**
 * TorchWork - Base class representing asynchronous work.
 *
 * Thread Safety:
 * TorchWork is NOT thread-safe. All methods (status(), isCompleted(), wait())
 * must be called from a single thread. Concurrent calls from multiple threads
 * are not supported.
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

  // Disable copy and move semantics
  TorchWork(const TorchWork&) = delete;
  TorchWork& operator=(const TorchWork&) = delete;
  TorchWork(TorchWork&&) = delete;
  TorchWork& operator=(TorchWork&&) = delete;

 protected:
  void setStatus(WorkStatus status) {
    status_ = status;

    if (status == WorkStatus::COMPLETED || status == WorkStatus::ERROR ||
        status == WorkStatus::TIMEDOUT) {
      runCallback();
    }
  }

  friend class TorchComm;
  friend class WorkWrapper;

  void setCallback(std::function<void()> callback) {
    callback_ = std::move(callback);
  }

  void runCallback() {
    if (callback_) {
      std::call_once(callback_once_, [this]() {
        callback_();
        callback_ = nullptr;
      });
    }
  }

  // break weak-ref cycle: postHook() stores a lambda in callback_ that
  // captures a weak_intrusive_ptr back to this object. after the strong
  // refcount reaches 0, release_resources() clears the callback, destroying the
  // weak pointer and allowing the weak refcount to reach 0 so the object is
  // deleted.
  void release_resources() override {
    callback_ = nullptr;
  }

  template <typename T, typename NullType>
  friend class c10::intrusive_ptr;

 private:
  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};

  std::once_flag callback_once_;
  std::function<void()> callback_;
};

class TorchWorkCompleted : public TorchWork {
 public:
  TorchWorkCompleted();
  ~TorchWorkCompleted() override = default;

  // Override virtual functions from TorchWork
  void wait() override;
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
