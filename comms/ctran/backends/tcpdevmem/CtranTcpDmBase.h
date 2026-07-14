// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <utility>

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
      if (done_) {
        lifetime_.reset();
      }
      if (done_ && status_ != ::comms::tcp_devmem::Status::Ok) {
        COMMCHECKTHROW(status_);
      }
      return done_;
    }

    // Can't access request after testRequest for send requests.
    bool isRecv = request_->getOp() == ::comms::tcp_devmem::RequestOp::Recv;

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
      lifetime_.reset();
    }

    return done_;
  }

  void complete(
      ::comms::tcp_devmem::Status status = ::comms::tcp_devmem::Status::Ok) {
    status_ = status;
    done_ = true;
    if (request_ == nullptr) {
      lifetime_.reset();
    }
  }

  void markQueuedRecv(std::shared_ptr<void> lifetime = nullptr) {
    done_ = false;
    status_ = ::comms::tcp_devmem::Status::Ok;
    request_ = nullptr;
    lifetime_ = std::move(lifetime);
  }

  void track(
      ::comms::tcp_devmem::TransportInterface* transport,
      ::comms::tcp_devmem::RequestInterface* request,
      std::shared_ptr<void> lifetime = nullptr) {
    done_ = false;
    status_ = ::comms::tcp_devmem::Status::Ok;
    transport_ = transport;
    request_ = request;
    lifetime_ = std::move(lifetime);
  }

 private:
  ::comms::tcp_devmem::RequestInterface* request_{nullptr};
  ::comms::tcp_devmem::TransportInterface* transport_{nullptr};
  ::comms::tcp_devmem::Status status_{::comms::tcp_devmem::Status::Ok};
  std::shared_ptr<void> lifetime_;
  bool done_{false};
};

} // namespace ctran
