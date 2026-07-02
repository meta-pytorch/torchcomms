// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

#include <algorithm>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

P2pRegistrationHandle::P2pRegistrationHandle(
    const CudaApi::IpcMemHandle& ipcHandle,
    int32_t ownerPid,
    uint64_t base,
    uint64_t offset,
    uint64_t size)
    : payload_{ownerPid, base, offset, size, ipcHandle} {}

std::vector<uint8_t> P2pRegistrationHandle::serialize() const {
  std::vector<uint8_t> buf(kSerializedSize);
  const auto* bytes = reinterpret_cast<const uint8_t*>(&payload_);
  std::copy_n(bytes, kSerializedSize, buf.begin());
  return buf;
}

Result<P2pRegistrationHandle::Payload> P2pRegistrationHandle::deserialize(
    std::span<const uint8_t> bytes) {
  if (bytes.size() != kSerializedSize) {
    return Err(
        ErrCode::InvalidArgument,
        "P2pRegistrationHandle: serialized payload has wrong size");
  }
  Payload payload{};
  std::copy_n(
      bytes.begin(), kSerializedSize, reinterpret_cast<uint8_t*>(&payload));
  return payload;
}

P2pRemoteRegistrationHandle::P2pRemoteRegistrationHandle(
    void* mappedBase,
    uint64_t offset,
    size_t size,
    bool ownedByIpc,
    std::shared_ptr<CudaApi> cudaApi)
    : mappedBase_(mappedBase),
      offset_(offset),
      size_(size),
      ownedByIpc_(ownedByIpc),
      cudaApi_(std::move(cudaApi)) {
  if (!cudaApi_) {
    cudaApi_ = std::make_shared<CudaApi>();
  }
}

void P2pRemoteRegistrationHandle::closeMapping() noexcept {
  if (ownedByIpc_ && mappedBase_ != nullptr) {
    auto st = cudaApi_->ipcCloseMemHandle(mappedBase_);
    if (st.hasError()) {
      // Best-effort cleanup: log and continue. Throwing here would escape a
      // noexcept context (destructor / move-assignment) and call
      // std::terminate.
      UNIFLOW_LOG_ERROR(
          "P2P: ipcCloseMemHandle failed: {}", st.error().message());
    }
  }
  mappedBase_ = nullptr;
  ownedByIpc_ = false;
}

P2pRemoteRegistrationHandle::~P2pRemoteRegistrationHandle() {
  closeMapping();
}

P2pRemoteRegistrationHandle::P2pRemoteRegistrationHandle(
    P2pRemoteRegistrationHandle&& other) noexcept
    : mappedBase_(other.mappedBase_),
      offset_(other.offset_),
      size_(other.size_),
      ownedByIpc_(other.ownedByIpc_),
      cudaApi_(std::move(other.cudaApi_)) {
  // Null out the source so its destructor does not also close the mapping.
  other.mappedBase_ = nullptr;
  other.ownedByIpc_ = false;
}

P2pRemoteRegistrationHandle& P2pRemoteRegistrationHandle::operator=(
    P2pRemoteRegistrationHandle&& other) noexcept {
  if (this != &other) {
    // Release any mapping we currently own before taking over the source's.
    closeMapping();
    mappedBase_ = other.mappedBase_;
    offset_ = other.offset_;
    size_ = other.size_;
    ownedByIpc_ = other.ownedByIpc_;
    cudaApi_ = std::move(other.cudaApi_);
    other.mappedBase_ = nullptr;
    other.ownedByIpc_ = false;
  }
  return *this;
}

void* P2pRemoteRegistrationHandle::mappedPtr() const noexcept {
  if (mappedBase_ == nullptr) {
    // Moved-from or invalid handle: avoid UB pointer arithmetic.
    return nullptr;
  }
  return static_cast<uint8_t*>(mappedBase_) + offset_;
}

} // namespace uniflow
