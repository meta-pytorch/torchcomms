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
    uint64_t domainId,
    uint64_t registrationBase,
    std::shared_ptr<DeviceAdapter> deviceAdapter,
    int deviceId,
    int hostBufferNumaNode)
    : mrs_(std::move(mrs)),
      ibvApi_(std::move(ibvApi)),
      domainId_(domainId),
      hostBufferNumaNode_(hostBufferNumaNode),
      registrationBase_(registrationBase),
      deviceAdapter_(std::move(deviceAdapter)),
      deviceId_(deviceId) {
  // toWireAddr() dereferences deviceAdapter_ whenever deviceId_ >= 0, so a
  // device-backed handle must be given a non-null adapter.
  assert(
      (deviceId_ < 0 || deviceAdapter_ != nullptr) &&
      "device-backed RdmaRegistrationHandle requires a non-null DeviceAdapter");
}

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
      .registrationBase = registrationBase_,
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
uint64_t RdmaRegistrationHandle::toWireAddr(const void* ptr) const {
  auto devAddr = (deviceId_ >= 0) ? deviceAdapter_->resolveDevicePointer(ptr)
                                  : reinterpret_cast<uint64_t>(ptr);
  return devAddr - registrationBase_;
}

// ---------------------------------------------------------------------------
// RdmaRemoteRegistrationHandle
// ---------------------------------------------------------------------------

RdmaRemoteRegistrationHandle::RdmaRemoteRegistrationHandle(
    std::vector<uint32_t> rkeys,
    uint64_t domainId,
    uint64_t registrationBase,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : rkeys_(std::move(rkeys)),
      domainId_(domainId),
      registrationBase_(registrationBase),
      deviceAdapter_(std::move(deviceAdapter)) {
  // toWireAddr() dereferences deviceAdapter_ whenever registrationBase_ != 0,
  // so a translated handle must be given a non-null adapter.
  assert(
      (registrationBase_ == 0 || deviceAdapter_ != nullptr) &&
      "translated RdmaRemoteRegistrationHandle requires a non-null DeviceAdapter");
}

uint64_t RdmaRemoteRegistrationHandle::toWireAddr(const void* ptr) const {
  /// Remote handles don't carry information on whether the target is a device
  /// ptr or not So we use the fact that only targets that have a non-zero
  /// registrationBase require pointer translation
  auto devAddr = registrationBase_ ? deviceAdapter_->resolveDevicePointer(ptr)
                                   : reinterpret_cast<uint64_t>(ptr);
  return devAddr - registrationBase_;
}

} // namespace uniflow
