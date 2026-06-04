// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace comms::tcp_devmem {
struct RequestInterface;
class TransportInterface;
} // namespace comms::tcp_devmem

namespace ctran {

struct CtranTcpDmConfig {};

class CtranTcpDmRequest {
 public:
  explicit CtranTcpDmRequest() {}

  bool isComplete();

  void complete() {
    done_ = true;
  }

  void track(
      ::comms::tcp_devmem::TransportInterface* transport,
      ::comms::tcp_devmem::RequestInterface* request) {
    done_ = false;
    transport_ = transport;
    request_ = request;
  }

 private:
  ::comms::tcp_devmem::RequestInterface* request_{nullptr};
  ::comms::tcp_devmem::TransportInterface* transport_{nullptr};
  bool done_{false};
};

} // namespace ctran
