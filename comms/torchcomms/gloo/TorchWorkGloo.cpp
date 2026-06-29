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
  runWaitPreHooks();
  runWaitPostHooks();
}

// --- P2P Send Work ---

TorchWorkGlooSend::TorchWorkGlooSend(
    at::Tensor tensor,
    std::unique_ptr<gloo::transport::UnboundBuffer> buf,
    std::chrono::milliseconds timeout)
    : tensor_(std::move(tensor)), buf_(std::move(buf)), timeout_(timeout) {
  setStatus(WorkStatus::INPROGRESS);
}

void TorchWorkGlooSend::wait() {
  if (waited_) {
    return;
  }
  waited_ = true;
  runWaitPreHooks();
  if (timeout_ <= std::chrono::milliseconds::zero()) {
    buf_->waitSend();
  } else {
    buf_->waitSend(timeout_);
  }
  setStatus(WorkStatus::COMPLETED);
  runWaitPostHooks();
}

// --- P2P Recv Work ---

TorchWorkGlooRecv::TorchWorkGlooRecv(
    at::Tensor originalTensor,
    at::Tensor cpuTensor,
    std::unique_ptr<gloo::transport::UnboundBuffer> buf,
    std::chrono::milliseconds timeout)
    : originalTensor_(std::move(originalTensor)),
      cpuTensor_(std::move(cpuTensor)),
      buf_(std::move(buf)),
      timeout_(timeout) {
  setStatus(WorkStatus::INPROGRESS);
}

void TorchWorkGlooRecv::wait() {
  if (waited_) {
    return;
  }
  waited_ = true;
  runWaitPreHooks();
  if (timeout_ <= std::chrono::milliseconds::zero()) {
    buf_->waitRecv();
  } else {
    buf_->waitRecv(timeout_);
  }
  if (cpuTensor_.device() != originalTensor_.device()) {
    originalTensor_.copy_(cpuTensor_);
  }
  setStatus(WorkStatus::COMPLETED);
  runWaitPostHooks();
}

} // namespace torch::comms
