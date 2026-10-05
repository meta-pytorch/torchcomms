// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {
namespace {

using VmmChunk = P2pRegistrationHandle::VmmChunk;
using VmmPayload = P2pRegistrationHandle::VmmPayload;
using IpcPayload = P2pRegistrationHandle::IpcPayload;

// Fixed-size VMM fields between the mode byte and the chunk array.
struct __attribute__((packed)) VmmHeader {
  uint8_t bootId[16];
  uint64_t pidNamespace;
  int32_t ownerPid;
  int32_t exporterDevice;
  uint64_t offset;
  uint64_t size;
  uint32_t numChunks;
};

constexpr size_t kModeSize = sizeof(P2pSharingMode);
constexpr size_t kVmmFixedSize = kModeSize + sizeof(VmmHeader);

// IPC payloads are recognized by their length, so no VMM payload may share it.
static_assert(
    (P2pRegistrationHandle::kIpcSerializedSize - kVmmFixedSize) %
        sizeof(VmmChunk) !=
    0);

template <typename T>
void appendBytes(std::vector<uint8_t>& buf, const T& value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  buf.insert(buf.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T readBytes(std::span<const uint8_t> bytes, size_t at) {
  T value{};
  std::copy_n(bytes.data() + at, sizeof(T), reinterpret_cast<uint8_t*>(&value));
  return value;
}

Err invalidPayload(std::string_view what) {
  return Err(
      ErrCode::InvalidArgument, "P2pRegistrationHandle: " + std::string(what));
}

Result<P2pRegistrationHandle::Payload> parseVmm(
    std::span<const uint8_t> bytes) {
  if (bytes.size() < kVmmFixedSize) {
    return invalidPayload("VMM payload is truncated");
  }
  const auto header = readBytes<VmmHeader>(bytes, kModeSize);
  if (header.numChunks == 0 || header.numChunks > kP2pMaxVmmChunks) {
    return invalidPayload(
        "VMM chunk count " + std::to_string(header.numChunks) +
        " outside [1, " + std::to_string(kP2pMaxVmmChunks) + "]");
  }
  if (bytes.size() !=
      kVmmFixedSize +
          static_cast<size_t>(header.numChunks) * sizeof(VmmChunk)) {
    return invalidPayload("VMM payload has wrong size");
  }
  if (header.ownerPid <= 0 || header.exporterDevice < 0) {
    return invalidPayload("VMM payload has an invalid owner pid or device");
  }
  if (header.size == 0) {
    return invalidPayload("VMM segment is empty");
  }

  VmmPayload payload{
      .scope = {.pidNamespace = header.pidNamespace},
      .ownerPid = header.ownerPid,
      .exporterDevice = header.exporterDevice,
      .offset = header.offset,
      .size = header.size,
  };
  std::copy_n(
      header.bootId, sizeof(header.bootId), payload.scope.bootId.begin());
  payload.chunks.reserve(header.numChunks);
  uint64_t total = 0;
  for (uint32_t i = 0; i < header.numChunks; ++i) {
    const auto chunk =
        readBytes<VmmChunk>(bytes, kVmmFixedSize + i * sizeof(VmmChunk));
    if (chunk.fd < 0 || chunk.size == 0) {
      return invalidPayload("VMM chunk has an invalid fd or size");
    }
    if (chunk.size > std::numeric_limits<uint64_t>::max() - total) {
      return invalidPayload("VMM chunk sizes overflow");
    }
    total += chunk.size;
    payload.chunks.push_back(chunk);
  }
  if (payload.offset > total || payload.size > total - payload.offset) {
    return invalidPayload("VMM segment extends past its chunks");
  }
  return payload;
}

} // namespace

P2pRegistrationHandle::P2pRegistrationHandle(
    const CudaApi::IpcMemHandle& ipcHandle,
    int32_t ownerPid,
    uint64_t base,
    uint64_t offset,
    uint64_t size)
    : payload_{IpcPayload{ownerPid, base, offset, size, ipcHandle}} {}

P2pRegistrationHandle::P2pRegistrationHandle(VmmPayload payload)
    : payload_{std::move(payload)} {}

P2pRegistrationHandle::~P2pRegistrationHandle() {
  const auto* vmm = std::get_if<VmmPayload>(&payload_);
  if (vmm == nullptr) {
    return;
  }
  for (const auto& chunk : vmm->chunks) {
    const int fd = chunk.fd;
    if (fd >= 0 && ::close(fd) != 0) {
      UNIFLOW_LOG_ERROR(
          "P2P: closing exported VMM fd {} failed: {}",
          fd,
          std::system_category().message(errno));
    }
  }
}

P2pSharingMode P2pRegistrationHandle::sharingMode() const noexcept {
  return std::holds_alternative<VmmPayload>(payload_) ? P2pSharingMode::PosixFd
                                                      : P2pSharingMode::Ipc;
}

std::vector<uint8_t> P2pRegistrationHandle::serialize() const {
  std::vector<uint8_t> buf;
  if (const auto* ipc = std::get_if<IpcPayload>(&payload_)) {
    appendBytes(buf, *ipc);
    return buf;
  }
  const auto& vmm = std::get<VmmPayload>(payload_);
  buf.reserve(kVmmFixedSize + vmm.chunks.size() * sizeof(VmmChunk));
  appendBytes(buf, P2pSharingMode::PosixFd);
  VmmHeader header{
      .pidNamespace = vmm.scope.pidNamespace,
      .ownerPid = vmm.ownerPid,
      .exporterDevice = vmm.exporterDevice,
      .offset = vmm.offset,
      .size = vmm.size,
      .numChunks = static_cast<uint32_t>(vmm.chunks.size())};
  std::copy_n(vmm.scope.bootId.begin(), sizeof(header.bootId), header.bootId);
  appendBytes(buf, header);
  for (const auto& chunk : vmm.chunks) {
    appendBytes(buf, chunk);
  }
  return buf;
}

Result<P2pRegistrationHandle::Payload> P2pRegistrationHandle::deserialize(
    std::span<const uint8_t> bytes) {
  if (bytes.size() == kIpcSerializedSize) {
    return readBytes<IpcPayload>(bytes, 0);
  }
  if (bytes.empty() ||
      static_cast<P2pSharingMode>(bytes[0]) != P2pSharingMode::PosixFd) {
    return invalidPayload(
        "unrecognized payload of " + std::to_string(bytes.size()) + " bytes");
  }
  return parseVmm(bytes);
}

P2pIpcMapping::P2pIpcMapping(void* base, std::shared_ptr<CudaApi> cudaApi)
    : base_(base), cudaApi_(std::move(cudaApi)) {}

P2pIpcMapping::~P2pIpcMapping() {
  auto st = cudaApi_->ipcCloseMemHandle(base_);
  if (st.hasError()) {
    UNIFLOW_LOG_ERROR(
        "P2P: ipcCloseMemHandle failed: {}", st.error().message());
  }
}

P2pRemoteRegistrationHandle::P2pRemoteRegistrationHandle(
    void* mappedBase,
    uint64_t offset,
    size_t size,
    std::unique_ptr<P2pMapping> mapping)
    : mappedBase_(mappedBase),
      offset_(offset),
      size_(size),
      mapping_(std::move(mapping)) {}

void* UNIFLOW_NULLABLE P2pRemoteRegistrationHandle::mappedPtr() const noexcept {
  if (mappedBase_ == nullptr) {
    return nullptr;
  }
  return static_cast<uint8_t*>(mappedBase_) + offset_;
}

} // namespace uniflow
