// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/Segment.h"
#include <cstring>

namespace {

constexpr uint8_t kSegmentVersion = 0;
constexpr size_t kSegmentHeaderSize = 24;

} // namespace

namespace uniflow {

Result<std::vector<uint8_t>> RegisteredSegment::exportId() const {
  size_t totalSize = kSegmentHeaderSize;
  std::vector<std::vector<uint8_t>> handlesData;
  for (auto& h : handles_) {
    auto data = h->serialize();
    if (data.empty()) {
      return Err(
          ErrCode::InvalidArgument, "handle serialize() returned empty data");
    }
    // For each handle, we store the transport type and the handle data size at
    // the beginning
    totalSize += data.size() + sizeof(uint8_t) + sizeof(uint32_t);
    handlesData.emplace_back(std::move(data));
  }
  if (handlesData.empty()) {
    return Err(ErrCode::InvalidArgument, "segment has no registration handles");
  }

  std::vector<uint8_t> id(totalSize);
  size_t pos = 0;

  // version
  id[pos++] = kSegmentVersion;

  // memory type
  id[pos++] = static_cast<uint8_t>(memType_);

  // handles number
  id[pos++] = static_cast<uint8_t>(handles_.size());

  // reserve
  id[pos++] = 0;

  // device id
  auto deviceId = static_cast<int32_t>(deviceId_);
  std::memcpy(id.data() + pos, &deviceId, sizeof(deviceId));
  pos += sizeof(deviceId);

  // addr
  auto addr = reinterpret_cast<uint64_t>(buf_);
  std::memcpy(id.data() + pos, &addr, sizeof(addr));
  pos += sizeof(addr);

  // length
  auto len = static_cast<uint64_t>(len_);
  std::memcpy(id.data() + pos, &len, sizeof(len));
  pos += sizeof(len);

  // handles data
  for (size_t i = 0; i < handlesData.size(); ++i) {
    // transport type
    id[pos++] = static_cast<uint8_t>(handles_[i]->transportType());

    // handle data size
    auto size = static_cast<uint32_t>(handlesData[i].size());
    std::memcpy(id.data() + pos, &size, sizeof(size));
    pos += sizeof(size);

    // handle data
    std::memcpy(id.data() + pos, handlesData[i].data(), handlesData[i].size());
    pos += handlesData[i].size();
  }

  return id;
}

Result<RemoteRegisteredSegment> RemoteRegisteredSegment::from(
    std::span<const uint8_t> exportId,
    const std::function<
        remoteHandleT(TransportType, size_t, std::span<const uint8_t>)>&
        getHandle) {
  if (exportId.size() < kSegmentHeaderSize) {
    return Err(
        ErrCode::InvalidArgument,
        "exportId is too small, less than header size");
  }

  size_t pos = 0;

  // version
  if (exportId[pos++] != kSegmentVersion) {
    return Err(
        ErrCode::InvalidArgument,
        "invalid segment version, expected " + std::to_string(kSegmentVersion) +
            " get " + std::to_string(exportId[pos - 1]));
  }

  // memory type
  auto memType = static_cast<MemoryType>(exportId[pos++]);

  // handles number
  uint8_t numHandles = exportId[pos++];

  // reserve
  pos++;

  // device id
  int32_t deviceId = 0;
  std::memcpy(&deviceId, exportId.data() + pos, sizeof(deviceId));
  pos += sizeof(deviceId);

  // addr
  uint64_t addr = 0;
  std::memcpy(&addr, exportId.data() + pos, sizeof(addr));
  pos += sizeof(addr);

  // length
  uint64_t len = 0;
  std::memcpy(&len, exportId.data() + pos, sizeof(len));
  pos += sizeof(len);

  RemoteRegisteredSegment segment(
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(static_cast<uintptr_t>(addr)),
      static_cast<size_t>(len),
      memType,
      static_cast<int>(deviceId));

  // handles data
  for (uint8_t i = 0; i < numHandles; ++i) {
    const size_t headerSize = sizeof(uint8_t) + sizeof(uint32_t);
    if (pos + headerSize > exportId.size()) {
      return Err(
          ErrCode::InvalidArgument,
          "exportId truncated in handle header at index " + std::to_string(i) +
              ": need " + std::to_string(headerSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(exportId.size() - pos) + " remaining");
    }

    // transport type
    auto transportType = static_cast<TransportType>(exportId[pos++]);

    // handle data size
    uint32_t handleSize = 0;
    std::memcpy(&handleSize, exportId.data() + pos, sizeof(handleSize));
    pos += sizeof(handleSize);

    if (pos + handleSize > exportId.size()) {
      return Err(
          ErrCode::InvalidArgument,
          "exportId truncated in handle data at index " + std::to_string(i) +
              ": need " + std::to_string(handleSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(exportId.size() - pos) + " remaining");
    }

    // handle data
    std::span<const uint8_t> handleData(
        exportId.data() + pos, static_cast<size_t>(handleSize));
    pos += handleSize;

    auto handleResult = getHandle(transportType, len, handleData);
    CHECK_RETURN(handleResult);
    segment.handles_.emplace_back(std::move(handleResult).value());
  }

  return segment;
}

} // namespace uniflow
