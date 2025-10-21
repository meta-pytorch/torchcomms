// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <future>

namespace torch {
namespace comms {

class TorchWork {
 public:
  TorchWork() = default;
  virtual ~TorchWork() = default;

  // Pure virtual functions that derived classes must implement
  virtual bool isCompleted() = 0;
  virtual void wait() = 0;

  // Disable copy and move semantics
  TorchWork(const TorchWork&) = delete;
  TorchWork& operator=(const TorchWork&) = delete;
  TorchWork(TorchWork&&) = delete;
  TorchWork& operator=(TorchWork&&) = delete;
};

class TorchWorkCompleted : public TorchWork {
 public:
  TorchWorkCompleted();
  ~TorchWorkCompleted() override = default;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;
};

class TorchWorkThread : public TorchWork {
 public:
  TorchWorkThread(std::function<void()> fn);
  ~TorchWorkThread() override = default;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

 private:
  std::future<void> future_;
};

} // namespace comms
} // namespace torch
