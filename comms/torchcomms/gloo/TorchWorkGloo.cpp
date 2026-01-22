// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/gloo/TorchWorkGloo.hpp"

#include <thread>

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/gloo/TorchCommGloo.hpp"

namespace torch {
namespace comms {

TorchWorkGloo::TorchWorkGloo() {
  setStatus(WorkStatus::COMPLETED);
}

TorchWorkGloo::~TorchWorkGloo() {
  TC_LOG(INFO, nullptr) << "TorchWorkGloo destroyed";
}

void TorchWorkGloo::wait() {
  return;
}

} // namespace comms
} // namespace torch
