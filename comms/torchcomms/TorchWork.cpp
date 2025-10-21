// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

TorchWorkCompleted::TorchWorkCompleted() {}

bool TorchWorkCompleted::isCompleted() {
  return true;
}

void TorchWorkCompleted::wait() {
  return;
}

TorchWorkThread::TorchWorkThread(std::function<void()> fn)
    : future_(std::async(std::launch::async, std::move(fn))) {}

bool TorchWorkThread::isCompleted() {
  return !future_.valid() ||
      future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

void TorchWorkThread::wait() {
  if (!future_.valid()) {
    // already waited on
    return;
  }
  future_.get();
}

} // namespace comms
} // namespace torch
