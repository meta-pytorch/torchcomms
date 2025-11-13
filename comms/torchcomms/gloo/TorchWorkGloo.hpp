// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

// Forward declaration
class TorchCommGloo;

class TorchWorkGloo : public TorchWork {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWorkGloo();
  ~TorchWorkGloo() override;

  // Delete copy and move operations
  TorchWorkGloo(const TorchWorkGloo&) = delete;
  TorchWorkGloo(TorchWorkGloo&&) = delete;
  TorchWorkGloo& operator=(const TorchWorkGloo&) = delete;
  TorchWorkGloo& operator=(TorchWorkGloo&&) = delete;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

 protected:
  friend class TorchCommGloo;
  friend class TorchWorkGlooQueue;
};

} // namespace comms
} // namespace torch
