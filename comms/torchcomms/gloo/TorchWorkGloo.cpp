// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/gloo/TorchWorkGloo.hpp"

#include <thread>

#include "comms/torchcomms/gloo/TorchCommGloo.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

namespace torch::comms {

TorchWorkGloo::TorchWorkGloo() {
  setStatus(WorkStatus::COMPLETED);
}

TorchWorkGloo::~TorchWorkGloo() {
  TC_LOG(INFO, nullptr) << "TorchWorkGloo destroyed";
}

void TorchWorkGloo::wait() {
  return;
}

} // namespace torch::comms
