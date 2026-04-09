// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/controller/Controller.h"

namespace uniflow {

class Connection {
 public:
  explicit Connection(
      std::unique_ptr<controller::Conn> ctrl,
      std::unique_ptr<MultiTransport> transport)
      : ctrl_(std::move(ctrl)), transport_(std::move(transport)) {}

  ~Connection();

  /// Graceful shutdown: stop reader, shut down transport, close ctrl.
  void shutdown();

  Status sendCtrlMsg(std::span<const uint8_t> payload);
  Result<size_t> recvCtrlMsg(std::vector<uint8_t>& payload);

  std::future<Status> put(
      RegisteredSegment::Span src,
      RemoteRegisteredSegment::Span dst,
      RequestOptions options = {});

  std::future<Status> get(
      RemoteRegisteredSegment::Span src,
      RegisteredSegment::Span dst,
      RequestOptions options = {});

  std::future<Status> put(
      std::vector<TransferRequest> requests,
      RequestOptions options = {});

  std::future<Status> get(
      std::vector<TransferRequest> requests,
      RequestOptions options = {});

  // Zero copy send/recv operations
  std::future<Status> send(
      RegisteredSegment::Span src,
      RequestOptions options = {});

  std::future<Result<size_t>> recv(
      RegisteredSegment::Span dst,
      RequestOptions options = {});

  // Copy based send/recv operations
  std::future<Status> send(Segment::Span src, RequestOptions options = {});

  std::future<Result<size_t>> recv(
      Segment::Span dst,
      RequestOptions options = {});

 private:
  std::unique_ptr<controller::Conn> ctrl_;
  std::unique_ptr<MultiTransport> transport_;
};

} // namespace uniflow
