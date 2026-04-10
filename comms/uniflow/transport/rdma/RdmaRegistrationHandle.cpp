// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/RdmaRegistrationHandle.h"

#include <cassert>
#include <cstring>

namespace uniflow {

// ---------------------------------------------------------------------------
// RdmaRegistrationHandle
// ---------------------------------------------------------------------------

RdmaRegistrationHandle::RdmaRegistrationHandle(
    std::vector<ibv_mr*> mrs,
    std::shared_ptr<IbvApi> ibvApi,
    uint64_t domainId)
    : mrs_(std::move(mrs)), ibvApi_(std::move(ibvApi)), domainId_(domainId) {}

RdmaRegistrationHandle::~RdmaRegistrationHandle() {
  for (auto* mr : mrs_) {
    if (mr) {
      ibvApi_->deregMr(mr);
    }
  }
}

std::vector<uint8_t> RdmaRegistrationHandle::serialize() const {
  assert(!mrs_.empty() && "Cannot serialize with no MRs");
  size_t totalSize = kPayloadHeaderSize + mrs_.size() * sizeof(uint32_t);
  std::vector<uint8_t> buf(totalSize);

  Header header{
      .domainId = domainId_,
      .numMrs = static_cast<uint8_t>(mrs_.size()),
  };
  std::memcpy(buf.data(), &header, sizeof(header));

  // Append per-NIC rkeys.
  size_t offset = kPayloadHeaderSize;
  for (const auto* mr : mrs_) {
    uint32_t rkey = mr->rkey;
    std::memcpy(buf.data() + offset, &rkey, sizeof(rkey));
    offset += sizeof(rkey);
  }

  return buf;
}

// ---------------------------------------------------------------------------
// RdmaRemoteRegistrationHandle
// ---------------------------------------------------------------------------

RdmaRemoteRegistrationHandle::RdmaRemoteRegistrationHandle(
    std::vector<uint32_t> rkeys,
    uint64_t domainId)
    : rkeys_(std::move(rkeys)), domainId_(domainId) {}

} // namespace uniflow
