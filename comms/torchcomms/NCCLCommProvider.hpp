// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

namespace torch::comms {

// Internal TorchComm extension point for backend implementations that can
// expose an underlying NCCL communicator to BackendWrapper.
class NCCLCommProvider {
 public:
  virtual ~NCCLCommProvider() = default;
  virtual void* getNCCLCommPtr() const = 0;
};

} // namespace torch::comms
