// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include "comms/torchcomms/TorchCommOptions.hpp"

namespace torch::comms {

// Forward declaration
class TorchComm;
class TorchWork;

class BatchSendRecv {
 public:
  explicit BatchSendRecv(TorchComm* parent);
  ~BatchSendRecv() = default;
  BatchSendRecv(const BatchSendRecv&) = default;
  BatchSendRecv& operator=(const BatchSendRecv&) = default;
  BatchSendRecv(BatchSendRecv&&) = default;
  BatchSendRecv& operator=(BatchSendRecv&&) = default;

  void send(const at::Tensor& tensor, int dst);
  void recv(at::Tensor& tensor, int src);
  c10::intrusive_ptr<TorchWork> issue(
      bool async_op,
      const BatchP2POptions& options = {});

  class P2POp {
   public:
    enum class OpType { SEND, RECV };
    P2POp(OpType type, const at::Tensor& tensor, int peer);
    ~P2POp() = default;
    P2POp(const P2POp&) = default;
    P2POp& operator=(const P2POp&) = default;
    P2POp(P2POp&&) = default;
    P2POp& operator=(P2POp&&) = default;

    OpType type;
    at::Tensor tensor;
    int peer;
  };

  std::vector<P2POp> ops;

 private:
  TorchComm* parent_;
};

} // namespace torch::comms
