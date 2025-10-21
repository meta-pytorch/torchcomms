// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/gloo/TorchWorkGloo.hpp"

#include <thread>

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/gloo/TorchCommGloo.hpp"

namespace torch {
namespace comms {

TorchWorkGloo::TorchWorkGloo() {}

TorchWorkGloo::~TorchWorkGloo() {
  TC_LOG(INFO) << "TorchWorkGloo destroyed";
}

bool TorchWorkGloo::isCompleted() {
  return true;
}

void TorchWorkGloo::wait() {
  return;
}

} // namespace comms
} // namespace torch
