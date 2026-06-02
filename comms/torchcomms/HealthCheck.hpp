// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>

namespace torch::comms {

// Process-level singleton tracking whether any TorchComm communicator
// has experienced a watchdog timeout. Used by the debug server's health
// check endpoint to detect unhealthy ranks and trigger flight recorder dumps.
class TorchCommsHealthCheck {
 public:
  static TorchCommsHealthCheck* get() {
    // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
    static auto* instance = new TorchCommsHealthCheck();
    return instance;
  }

  void setTimedOut() {
    timed_out_.store(true, std::memory_order_release);
  }

  bool isTimedOut() const {
    return timed_out_.load(std::memory_order_acquire);
  }

 private:
  TorchCommsHealthCheck() = default;
  std::atomic<bool> timed_out_{false};
};

} // namespace torch::comms
