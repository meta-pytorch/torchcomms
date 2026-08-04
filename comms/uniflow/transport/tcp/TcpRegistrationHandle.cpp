// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/tcp/TcpRegistrationHandle.h"

#include <cstring>

namespace uniflow {

TcpRegistrationHandle::TcpRegistrationHandle(
    uint64_t segId,
    uint64_t base,
    uint64_t len,
    std::function<void()> onDestroy)
    : segId_(segId), base_(base), len_(len), onDestroy_(std::move(onDestroy)) {}

TcpRegistrationHandle::~TcpRegistrationHandle() {
  if (onDestroy_) {
    onDestroy_();
  }
}

std::vector<uint8_t> TcpRegistrationHandle::serialize() const {
  Header header{
      .segId = segId_,
      .base = base_,
      .len = len_,
  };
  std::vector<uint8_t> data(sizeof(Header));
  std::memcpy(data.data(), &header, sizeof(header));
  return data;
}

TcpRemoteRegistrationHandle::TcpRemoteRegistrationHandle(
    uint64_t segId,
    uint64_t base,
    uint64_t len)
    : segId_(segId), base_(base), len_(len) {}

Result<std::unique_ptr<TcpRemoteRegistrationHandle>>
TcpRemoteRegistrationHandle::deserialize(
    size_t segmentLength,
    std::span<const uint8_t> payload) {
  if (payload.size() < sizeof(TcpRegistrationHandle::Header)) {
    return Err(
        ErrCode::InvalidArgument,
        "tcp registration payload is smaller than its header");
  }

  TcpRegistrationHandle::Header header;
  std::memcpy(&header, payload.data(), sizeof(header));

  if (payload.size() != sizeof(header)) {
    return Err(
        ErrCode::InvalidArgument, "tcp registration payload size mismatch");
  }
  if (header.len != segmentLength) {
    return Err(
        ErrCode::InvalidArgument,
        "tcp registration length does not match segment length");
  }
  return std::make_unique<TcpRemoteRegistrationHandle>(
      header.segId, header.base, header.len);
}

} // namespace uniflow
