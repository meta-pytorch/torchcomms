// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

TorchWorkCompleted::TorchWorkCompleted() {
  setStatus(WorkStatus::COMPLETED);
}

void TorchWorkCompleted::wait() {
  return;
}

TorchWorkThread::TorchWorkThread(std::function<void()> fn)
    : future_(std::async(std::launch::async, [this, fn = std::move(fn)]() {
        try {
          fn();
          setStatus(WorkStatus::COMPLETED);
        } catch (...) {
          setStatus(WorkStatus::ERROR);
          throw;
        }
      })) {}

void TorchWorkThread::wait() {
  if (!future_.valid()) {
    // already waited on
    return;
  }
  future_.get();
}

} // namespace comms
} // namespace torch
