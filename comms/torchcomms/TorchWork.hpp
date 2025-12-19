// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/util/intrusive_ptr.h>
#include <functional>
#include <future>

namespace torch {
namespace comms {

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
  virtual ~TorchWork() = default;

  WorkStatus status() const {
    return status_.load(std::memory_order_relaxed);
  }
  bool isCompleted() const {
    return status() == WorkStatus::COMPLETED;
  }

  // Pure virtual functions that derived classes must implement
  virtual void wait() = 0;

  // Disable copy and move semantics
  TorchWork(const TorchWork&) = delete;
  TorchWork& operator=(const TorchWork&) = delete;
  TorchWork(TorchWork&&) = delete;
  TorchWork& operator=(TorchWork&&) = delete;

 protected:
  void setStatus(WorkStatus status) {
    status_ = status;
  }

 private:
  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
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
  TorchWorkThread(std::function<void()> fn);
  ~TorchWorkThread() override = default;

  // Override virtual functions from TorchWork
  void wait() override;

 private:
  std::future<void> future_;
};

} // namespace comms
} // namespace torch
