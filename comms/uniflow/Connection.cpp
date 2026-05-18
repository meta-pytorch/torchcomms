// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/Connection.h"

#include <cstring>

namespace uniflow {

Connection::~Connection() {
  shutdown();
}

void Connection::shutdown() {
  if (transport_) {
    transport_->shutdown();
  }
  if (ctrl_) {
    ctrl_->close();
  }
}

Status Connection::sendCtrlMsg(std::span<const uint8_t> payload) {
  size_t idx = 0;
  size_t len = payload.size();
  while (idx < len) {
    auto send = ctrl_->send(payload.subspan(idx, len - idx)).get();
    CHECK_RETURN(send);
    idx += send.value();
  }
  return Ok();
}

Result<size_t> Connection::recvCtrlMsg(std::vector<uint8_t>& payload) {
  return ctrl_->recv(payload).get();
}

std::future<Status> Connection::put(
    RegisteredSegment::Span src,
    RemoteRegisteredSegment::Span dst,
    const RequestOptions& options) {
  return transport_->put({{src, dst}}, options);
}

std::future<Status> Connection::get(
    RemoteRegisteredSegment::Span src,
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return transport_->get({{dst, src}}, options);
}

std::future<Status> Connection::put(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options) {
  return transport_->put(requests, options);
}

std::future<Status> Connection::get(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options) {
  return transport_->get(requests, options);
}

// Zero copy send/recv operations
std::future<Status> Connection::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  return transport_->send(src, options);
}

std::future<Result<size_t>> Connection::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return transport_->recv(dst, options);
}

// Copy based send/recv operations
std::future<Status> Connection::send(
    Segment::Span src,
    const RequestOptions& options) {
  return transport_->send(src, options);
}

std::future<Result<size_t>> Connection::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return transport_->recv(dst, options);
}

} // namespace uniflow
