// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/tcp_devmem/transport.h"

namespace ctran {

struct CtranTcpDmConfig {};

class CtranTcpDmRequest {
 public:
  explicit CtranTcpDmRequest() {}

  bool isComplete() {
    if (request_ == nullptr) {
      // Pending irecv where sender has not been connected yet or send/recv
      // control.
      return done_;
    }

    // Can't access request after testRequest for send requests.
    bool isRecv = request_->op == ::comms::tcp_devmem::RequestOp::Recv;

    int requestDone = 0;
    COMMCHECKTHROW(
        transport_->testRequest(request_, &requestDone, /*size*/ nullptr));
    if (requestDone) {
      done_ = true;

      if (isRecv) {
        // For receive, after testRequest singnals request completion, the
        // callers should let the plugin know when the bounce buffer is safe
        // to be reused. This is done by calling consumedRequest. With
        // CTRAN, the plugin itself drives the unpack so it's safe to reuse
        // the buffers immediately.

        COMMCHECKTHROW(transport_->consumedRequest(request_));
      }
    }

    return done_;
  }

  void complete() {
    done_ = true;
  }

  void track(
      std::shared_ptr<::comms::tcp_devmem::Transport> transport,
      ::comms::tcp_devmem::Request* request) {
    done_ = false;
    transport_ = transport;
    request_ = request;
  }

 private:
  ::comms::tcp_devmem::Request* request_{nullptr};
  std::shared_ptr<::comms::tcp_devmem::Transport> transport_{nullptr};
  bool done_{false};
};

} // namespace ctran
