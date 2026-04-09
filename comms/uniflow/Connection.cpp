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
    auto send = ctrl_->send(payload.subspan(idx, len - idx));
    CHECK_RETURN(send);
    idx += send.value();
  }
  return Ok();
}

Result<size_t> Connection::recvCtrlMsg(std::vector<uint8_t>& payload) {
  return ctrl_->recv(payload);
}

std::future<Status> Connection::put(
    RegisteredSegment::Span src,
    RemoteRegisteredSegment::Span dst,
    RequestOptions options) {
  return transport_->put({{src, dst}}, std::move(options));
}

std::future<Status> Connection::get(
    RemoteRegisteredSegment::Span src,
    RegisteredSegment::Span dst,
    RequestOptions options) {
  return transport_->get({{dst, src}}, std::move(options));
}

std::future<Status> Connection::put(
    std::vector<TransferRequest> requests,
    RequestOptions options) {
  return transport_->put(requests, std::move(options));
}

std::future<Status> Connection::get(
    std::vector<TransferRequest> requests,
    RequestOptions options) {
  return transport_->get(requests, std::move(options));
}

// Zero copy send/recv operations
std::future<Status> Connection::send(
    RegisteredSegment::Span src,
    RequestOptions options) {
  return transport_->send(src, std::move(options));
}

std::future<Result<size_t>> Connection::recv(
    RegisteredSegment::Span dst,
    RequestOptions options) {
  return transport_->recv(dst, std::move(options));
}

// Copy based send/recv operations
std::future<Status> Connection::send(
    Segment::Span src,
    RequestOptions options) {
  return transport_->send(src, std::move(options));
}

std::future<Result<size_t>> Connection::recv(
    Segment::Span dst,
    RequestOptions options) {
  return transport_->recv(dst, std::move(options));
}

} // namespace uniflow
