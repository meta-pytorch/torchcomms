// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

#include <cstring>

namespace uniflow {

namespace {

bool isCpu(int deviceId) {
  return deviceId == -1;
}

} // namespace

// ============================================================================
// MultiTransport Implementation
// ============================================================================

std::vector<std::string> MultiTransportFactory::selectNics() {
  auto& topo = Topology::get();
  return isCpu(deviceId_) ? topo.selectCpuNics(nicFilter_)
                          : topo.selectGpuNics(deviceId_, nicFilter_);
}

Status MultiTransportFactory::supported(TransportType type) {
  switch (type) {
    case TransportType::NVLink:
      return NVLinkTransportFactory::supported();
    case TransportType::RDMA:
      return RdmaTransportFactory::supported();
    case TransportType::TCP:
      return Err(ErrCode::NotImplemented, "tcp transport is not implemented");
    case TransportType::Mock:
      return Ok();
    default:
      return Err(ErrCode::InvalidArgument, "unknown transport type");
  }
}

Status MultiTransport::validateRequests(
    const std::vector<TransferRequest>& requests) {
  if (requests.empty()) {
    return Err(ErrCode::InvalidArgument, "empty request list");
  }

  auto localMemType = requests.front().local.memType();
  auto remoteMemType = requests.front().remote.memType();
  auto localMemDeviceId = requests.front().local.deviceId();
  auto localHandles = requests.front().local.handles_;
  auto remoteHandles = requests.front().remote.handles_;

  for (size_t i = 1; i < requests.size(); ++i) {
    auto& local = requests[i].local;
    auto& remote = requests[i].remote;
    if (local.memType() != localMemType || remote.memType() != remoteMemType ||
        local.deviceId() != localMemDeviceId) {
      return Err(
          ErrCode::InvalidArgument,
          "all requests must have the same memory type and device id");
    }

    if (localHandles.size() != local.handles_.size() ||
        remoteHandles.size() != remote.handles_.size()) {
      return Err(
          ErrCode::InvalidArgument,
          "all requests must have the same number of handles");
    }

    for (size_t j = 0; j < localHandles.size(); ++j) {
      if (localHandles[j]->transportType() !=
          local.handles_[j]->transportType()) {
        return Err(
            ErrCode::InvalidArgument,
            "all requests must have the same handles type");
      }
    }

    for (size_t j = 0; j < remoteHandles.size(); ++j) {
      if (remoteHandles[j]->transportType() !=
          remote.handles_[j]->transportType()) {
        return Err(
            ErrCode::InvalidArgument,
            "all requests must have the same handles type");
      }
    }
  }
  return Ok();
}

Transport* MultiTransport::findTransport(TransportType type) const {
  for (auto& t : transports_) {
    if (t->transportType() == type) {
      return t.get();
    }
  }
  return nullptr;
}

MultiTransportFactory::MultiTransportFactory(int deviceId, NicFilter nicFilter)
    : deviceId_(deviceId),
      nicFilter_(std::move(nicFilter)),
      eventBaseThread_(std::make_shared<ScopedEventBaseThread>()) {
  auto& topo = Topology::get();
  CHECK_THROW_EXCEPTION(
      deviceId_ >= -1 && deviceId_ < static_cast<int>(topo.gpuCount()),
      std::runtime_error);
  // cuda device
  if (deviceId_ >= 0) {
    auto nvlink = std::make_shared<NVLinkTransportFactory>(
        deviceId, eventBaseThread_->getEventBase());
    factories_.emplace_back(std::move(nvlink));
  }

  auto nics = selectNics();
  if (!nics.empty()) {
    RdmaTransportConfig config;
    config.numQps = static_cast<uint32_t>(nics.size());
    auto rdma = std::make_shared<RdmaTransportFactory>(
        std::move(nics), eventBaseThread_->getEventBase(), config);
    factories_.emplace_back(std::move(rdma));
  }
}

void MultiTransport::addTransport(std::unique_ptr<Transport> transport) {
  transports_.emplace_back(std::move(transport));
}

Result<TransportInfo> MultiTransport::bind() {
  const auto numTransport = static_cast<uint8_t>(transports_.size());
  std::vector<std::vector<uint8_t>> infoData;
  infoData.reserve(numTransport);

  size_t totalSize = sizeof(uint8_t);
  totalSize += sizeof(uint32_t) * numTransport;
  for (auto& t : transports_) {
    infoData.emplace_back(t->bind());
    totalSize += infoData.back().size();
  }

  std::vector<uint8_t> info(totalSize);
  size_t pos = 0;

  info[pos++] = numTransport;
  for (const auto& data : infoData) {
    auto size = static_cast<uint32_t>(data.size());
    std::memcpy(info.data() + pos, &size, sizeof(size));
    pos += sizeof(size);

    std::memcpy(info.data() + pos, data.data(), size);
    pos += size;
  }

  return info;
}

Status MultiTransport::connect(std::span<const uint8_t> info) {
  if (info.empty()) {
    return Err(ErrCode::ConnectionFailed, "empty transport info");
  }

  size_t pos = 0;
  uint8_t num = info[pos++];
  if (num != transports_.size()) {
    return Err(
        ErrCode::ConnectionFailed,
        "transport count mismatch: local=" +
            std::to_string(transports_.size()) +
            ", peer=" + std::to_string(num));
  }

  size_t headerSize = sizeof(uint32_t);
  for (auto& t : transports_) {
    if (pos + headerSize > info.size()) {
      return Err(
          ErrCode::ConnectionFailed,
          "peer topology info truncated at transport " + t->name() + ": need " +
              std::to_string(headerSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(info.size() - pos) + " remaining");
    }

    uint32_t size = 0;
    std::memcpy(&size, info.data() + pos, sizeof(size));
    pos += sizeof(size);

    if (pos + size > info.size()) {
      return Err(
          ErrCode::ConnectionFailed,
          "peer topology truncated at transport " + t->name() + ": need " +
              std::to_string(size) + " bytes at pos " + std::to_string(pos) +
              ", but only " + std::to_string(info.size() - pos) + " remaining");
    }

    std::span<const uint8_t> infoData(
        info.data() + pos, static_cast<size_t>(size));
    pos += size;

    CHECK_EXPR(t->connect(infoData));
  }
  return Ok();
}

Result<Transport*> MultiTransport::selectTransport(
    const std::vector<TransferRequest>& requests) {
  CHECK_EXPR(validateRequests(requests));

  auto localMemType = requests.front().local.memType();
  auto remoteMemType = requests.front().remote.memType();
  auto localDeviceId = requests.front().local.deviceId();
  auto localHandles = requests.front().local.handles_;
  auto remoteHandles = requests.front().remote.handles_;

  // Helper to check if handles contain a given transport type.
  auto hasHandleType = [](auto& handles, TransportType type) {
    for (auto& h : handles) {
      if (h->transportType() == type) {
        return true;
      }
    }
    return false;
  };

  // VRAM→VRAM on matching device: prefer NVLink if both sides have handles.
  if (localMemType == MemoryType::VRAM && remoteMemType == MemoryType::VRAM &&
      localDeviceId == deviceId_) {
    if (hasHandleType(localHandles, TransportType::NVLink) &&
        hasHandleType(remoteHandles, TransportType::NVLink)) {
      if (auto* t = findTransport(TransportType::NVLink)) {
        return t;
      }
    }
  }

  // Fallback priority: RDMA → TCP
  static constexpr TransportType kFallback[] = {
      TransportType::RDMA, TransportType::TCP};
  for (auto type : kFallback) {
    if (auto* t = findTransport(type)) {
      return t;
    }
  }

  return Err(
      ErrCode::NotConnected,
      "no transport available for the given memory type and device id");
}

std::future<Status> MultiTransport::doTransfer(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options,
    TransferOp op) {
  auto transport = selectTransport(requests);
  if (transport.hasError()) {
    return make_ready_future<Status>(std::move(transport).error());
  }

  auto& t = transport.value();
  ++transferCounts_[t->transportType()];
  return (t->*op)(requests, options);
}

std::future<Status> MultiTransport::put(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options) {
  return doTransfer(requests, options, &Transport::put);
}

std::future<Status> MultiTransport::get(
    const std::vector<TransferRequest>& requests,
    const RequestOptions& options) {
  return doTransfer(requests, options, &Transport::get);
}

// send/recv left as NotImplemented — Phase 6 only uses put/get
std::future<Status> MultiTransport::send(
    RegisteredSegment::Span src,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

std::future<Status> MultiTransport::send(
    Segment::Span src,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

std::future<Result<size_t>> MultiTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return make_ready_future<Result<size_t>>(
      Result<size_t>(ErrCode::NotImplemented));
}

std::future<Result<size_t>> MultiTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return make_ready_future<Result<size_t>>(
      Result<size_t>(ErrCode::NotImplemented));
}

Result<RegisteredSegment> MultiTransportFactory::registerSegment(
    Segment& segment) {
  RegisteredSegment regSeg(segment);
  for (auto& f : factories_) {
    auto handle = f->registerSegment(segment);
    if (handle) {
      regSeg.handles_.emplace_back(std::move(handle).value());
    } else {
      UNIFLOW_LOG_WARN(
          "Segment {} cannot be registered on transport {}: {}",
          segment.data(),
          f->transportType(),
          handle.error().message());
    }
  }
  if (regSeg.handles_.empty()) {
    return Err(
        ErrCode::MemoryRegistrationError,
        "no transport backend could register this segment");
  }
  return regSeg;
}

Result<RemoteRegisteredSegment> MultiTransportFactory::importSegment(
    std::span<const uint8_t> exportId) {
  return RemoteRegisteredSegment::from(
      exportId,
      [this](
          TransportType transportType,
          size_t segmentLength,
          std::span<const uint8_t> payload)
          -> RemoteRegisteredSegment::remoteHandleT {
        for (auto& f : factories_) {
          if (f->transportType() == transportType) {
            return f->importSegment(segmentLength, payload);
          }
        }
        return Err(
            ErrCode::InvalidArgument,
            "importSegment: Invalid transport type " +
                std::to_string(transportType));
      });
}

Result<std::vector<MultiTransportFactory::TopologyEntry>>
MultiTransportFactory::parse(std::span<const uint8_t> peerTopology) {
  if (peerTopology.empty()) {
    return Err(ErrCode::TopologyDisconnect, "empty peer topology");
  }

  size_t pos = 0;
  uint8_t num = peerTopology[pos++];
  std::vector<TopologyEntry> entries(num);

  size_t topoHeaderSize = sizeof(uint8_t) + sizeof(uint32_t);
  for (size_t i = 0; i < entries.size(); ++i) {
    if (pos + topoHeaderSize > peerTopology.size()) {
      return Err(
          ErrCode::TopologyDisconnect,
          "peer topology truncated at transport " + std::to_string(i) +
              ": need " + std::to_string(topoHeaderSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(peerTopology.size() - pos) + " remaining");
    }
    auto transportType = static_cast<TransportType>(peerTopology[pos++]);

    uint32_t topoSize = 0;
    std::memcpy(&topoSize, peerTopology.data() + pos, sizeof(topoSize));
    pos += sizeof(topoSize);

    if (pos + topoSize > peerTopology.size()) {
      return Err(
          ErrCode::TopologyDisconnect,
          "peer topology truncated at transport " + std::to_string(i) +
              ": need " + std::to_string(topoSize) + " bytes at pos " +
              std::to_string(pos) + ", but only " +
              std::to_string(peerTopology.size() - pos) + " remaining");
    }

    entries[i] = {transportType, peerTopology.subspan(pos, topoSize)};
    pos += topoSize;
  }
  return entries;
}

Result<std::unique_ptr<MultiTransport>> MultiTransportFactory::createTransport(
    std::span<const uint8_t> peerTopology) {
  auto parsed = MultiTransportFactory::parse(peerTopology);
  CHECK_RETURN(parsed);
  auto& entries = parsed.value();
  if (entries.size() != factories_.size()) {
    return Err(
        ErrCode::InvalidArgument,
        "transport count mismatch: local=" + std::to_string(factories_.size()) +
            ", peer=" + std::to_string(entries.size()));
  }

  auto mt = std::make_unique<MultiTransport>(deviceId_, eventBaseThread_);
  for (size_t i = 0; i < factories_.size(); ++i) {
    if (entries[i].type != factories_[i]->transportType()) {
      return Err(
          ErrCode::TopologyDisconnect,
          "transport type mismatch at " + std::to_string(i) +
              ": local=" + std::to_string(factories_[i]->transportType()) +
              ", peer=" + std::to_string(entries[i].type));
    }

    auto transport = factories_[i]->createTransport(entries[i].data);
    if (transport) {
      mt->addTransport(std::move(transport).value());
    } else {
      UNIFLOW_LOG_WARN(
          "Transport {} cannot be created: {}",
          factories_[i]->transportType(),
          transport.error().message());
    }
  }
  if (mt->transports_.empty()) {
    return Err(ErrCode::TopologyDisconnect, "no transport can be connected");
  }
  return mt;
}

std::vector<uint8_t> MultiTransportFactory::getTopology() {
  uint8_t numTransport = factories_.size();
  std::vector<std::vector<uint8_t>> topoData;
  topoData.reserve(numTransport);

  // total number of transports
  size_t totalSize = sizeof(uint8_t);
  // header for each transport
  totalSize += (sizeof(uint8_t) + sizeof(uint32_t)) * numTransport;
  for (auto& f : factories_) {
    auto topo = f->getTopology();
    CHECK_THROW_EXCEPTION(!topo.empty(), std::runtime_error);
    auto size = topo.size();
    topoData.emplace_back(std::move(topo));
    CHECK_THROW_EXCEPTION(size > 0, std::runtime_error);
    totalSize += size;
  }

  size_t pos = 0;
  std::vector<uint8_t> topology(totalSize);
  topology[pos++] = numTransport;
  for (size_t i = 0; i < topoData.size(); ++i) {
    // topo transport type
    topology[pos++] = factories_[i]->transportType();

    // topo data size
    auto size = static_cast<uint32_t>(topoData[i].size());
    std::memcpy(topology.data() + pos, &size, sizeof(size));
    pos += sizeof(size);

    // topo data
    std::memcpy(topology.data() + pos, topoData[i].data(), size);
    pos += size;
  }

  return topology;
}

void MultiTransport::shutdown() {
  for (auto& tt : transports_) {
    tt->shutdown();
  }
}

} // namespace uniflow
