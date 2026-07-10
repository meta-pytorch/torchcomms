// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <gloo/transport/unbound_buffer.h>
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/utils/TracingGuard.hpp"

namespace torch::comms {

// Forward declaration
class TorchCommGloo;

class TorchWorkGloo : public TorchWork {
 public:
  TorchWorkGloo();
  ~TorchWorkGloo() override;

  // Delete copy and move operations
  TorchWorkGloo(const TorchWorkGloo&) = delete;
  TorchWorkGloo(TorchWorkGloo&&) = delete;
  TorchWorkGloo& operator=(const TorchWorkGloo&) = delete;
  TorchWorkGloo& operator=(TorchWorkGloo&&) = delete;

  // Override virtual functions from TorchWork
  void wait() override;

 protected:
  friend class TorchCommGloo;
  friend class TorchWorkGlooQueue;
};

// P2P send work: buf->send() already called inline; wait() calls waitSend().
// Matches native ProcessGroupGloo::SendWork semantics.
class TorchWorkGlooSend : public TorchWork {
 public:
  TorchWorkGlooSend(
      at::Tensor tensor,
      std::unique_ptr<gloo::transport::UnboundBuffer> buf,
      std::chrono::milliseconds timeout);
  void wait() override;

 private:
  at::Tensor tensor_;
  std::unique_ptr<gloo::transport::UnboundBuffer> buf_;
  std::chrono::milliseconds timeout_;
  bool waited_{false};
};

// P2P recv work: buf->recv() already called inline; wait() calls waitRecv()
// then copies back to original device if needed.
class TorchWorkGlooRecv : public TorchWork {
 public:
  TorchWorkGlooRecv(
      at::Tensor originalTensor,
      at::Tensor cpuTensor,
      std::unique_ptr<gloo::transport::UnboundBuffer> buf,
      std::chrono::milliseconds timeout);
  void wait() override;

 private:
  at::Tensor originalTensor_;
  at::Tensor cpuTensor_;
  std::unique_ptr<gloo::transport::UnboundBuffer> buf_;
  std::chrono::milliseconds timeout_;
  bool waited_{false};
};

} // namespace torch::comms
