// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/logging/Logger.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace uniflow {

Status MultiTransport::validateRequests(
    const std::vector<TransferRequest>& requests) {
  if (requests.empty()) {
    return Err(ErrCode::InvalidArgument, "empty request list");
  }

  auto localMemType = requests.front().local.memType();
  auto remoteMemType = requests.front().remote.memType();
  auto localMemDeviceId = requests.front().local.deviceId();

  for (size_t i = 1; i < requests.size(); ++i) {
    const auto& local = requests[i].local;
    const auto& remote = requests[i].remote;
    if (local.memType() != localMemType || remote.memType() != remoteMemType ||
        local.deviceId() != localMemDeviceId) {
      return Err(
          ErrCode::InvalidArgument,
          "all requests must have the same memory type and device id");
    }
  }

  // Handle counts and transport types are intentionally not validated here.
  // A segment's handles describe the transports that were usable when that
  // specific segment was registered or imported; they are capabilities, not a
  // batch schema. For example, a GPU cache segment may expose both NVLink and
  // RDMA, while a peer process that cannot import the NVLink handle still has a
  // valid RDMA handle. selectTransport() enforces the real transfer invariant:
  // one common transport must be available on every local and remote segment in
  // the batch.
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

MultiTransportFactory::MultiTransportFactory(
    int deviceId,
    MultiTransportFactoryOptions options,
    TransportProviderList providers)
    : deviceId_(deviceId), options_(std::move(options)) {
  if (deviceId_ < -1) {
    throw std::invalid_argument("deviceId must be -1 or non-negative");
  }

  std::array<bool, NumTransportType> seen{};
  for (const auto& provider : providers) {
    if (!provider) {
      throw std::invalid_argument("transport provider must not be null");
    }
    const auto type = provider->transportType();
    const auto id = static_cast<size_t>(type);
    if (id >= seen.size()) {
      throw std::invalid_argument(
          "invalid transport provider type: " + std::to_string(id));
    }
    if (seen[id]) {
      throw std::invalid_argument(
          "duplicate transport provider: " + std::string(toStringView(type)));
    }
    seen[id] = true;
  }

  if (providers.empty()) {
    return;
  }

  std::sort(
      providers.begin(), providers.end(), [](const auto& lhs, const auto& rhs) {
        return lhs->transportType() < rhs->transportType();
      });

  eventBaseThread_ = std::make_shared<ScopedEventBaseThread>();
  const TransportProviderContext context{
      .deviceId = deviceId_,
      .eventBase = eventBaseThread_->getEventBase(),
      .options = &options_,
  };
  factories_.reserve(providers.size());
  for (const auto& provider : providers) {
    auto result = provider->createFactory(context);
    if (!result) {
      throw std::runtime_error(
          "Failed to create " +
          std::string(toStringView(provider->transportType())) +
          " transport factory: " + result.error().toString());
    }
    auto factory = std::move(result).value();
    if (!factory) {
      continue;
    }
    if (factory->transportType() != provider->transportType()) {
      throw std::invalid_argument(
          "transport factory type does not match provider: expected " +
          std::string(toStringView(provider->transportType())) + ", got " +
          std::string(toStringView(factory->transportType())));
    }
    factories_.emplace_back(std::move(factory));
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
    auto data = t->bind();
    /*
     * A successful bind always yields a non-empty serialized TransportInfo
     * (header + QP/NIC info). An empty result means the underlying transport
     * failed to acquire its resources (CQ/QP/MR) and set itself to Error.
     * Surface that as a real error instead of packing an empty sub-info that
     * the peer would fail to deserialize during connect().
     */
    if (data.empty()) {
      return Err(
          ErrCode::ConnectionFailed,
          "MultiTransport::bind: a transport failed to bind (empty info)");
    }
    infoData.emplace_back(std::move(data));
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

  auto hasHandleType = [](const auto& handles, TransportType type) {
    for (const auto& h : handles) {
      if (h->transportType() == type) {
        return true;
      }
    }
    return false;
  };

  // A transport is eligible only if every request in the batch has a matching
  // handle on both local and remote sides. This preserves the invariant that a
  // single transport implementation executes the entire batch. Example use
  // case: distributed KV-cache transfer can mix peers where NVLink is available
  // with peers where RDMA is the only common capability; the batch should use
  // NVLink only when all segments support it, otherwise fall back to RDMA.
  auto allHaveTransport = [&](TransportType type) {
    for (const auto& req : requests) {
      if (!hasHandleType(req.local.registrationHandles(), type) ||
          !hasHandleType(req.remote.registrationHandles(), type)) {
        return false;
      }
    }
    return true;
  };

  // preferredTransport: global force -- if set and available on all requests,
  // it wins regardless of topology (both intra- and inter-node).
  if (preferredTransport_.has_value() &&
      allHaveTransport(*preferredTransport_)) {
    if (auto* t = findTransport(*preferredTransport_)) {
      return t;
    }
  }

  // Intra-node = VRAM<->VRAM on this device (reachable over the on-host
  // interconnect tier); otherwise inter-node. One transport per batch.
  const bool intraNode = localMemType == MemoryType::VRAM &&
      remoteMemType == MemoryType::VRAM && localDeviceId == deviceId_;

  // Per-case ordered preference, evaluated allocation-free with no transport
  // checked twice: the case override first (if set), then the case defaults
  // skipping whichever tier equals the override.
  //   intra-node: [intraNodeTransport?] NVLink -> RDMA -> TCP
  //   inter-node: [interNodeTransport?] RDMA -> TCP
  const std::optional<TransportType>& caseOverride =
      intraNode ? intraNodeTransport_ : interNodeTransport_;
  auto tryTier = [&](TransportType type) -> Transport* {
    return allHaveTransport(type) ? findTransport(type) : nullptr;
  };

  if (caseOverride.has_value()) {
    if (auto* t = tryTier(*caseOverride)) {
      return t;
    }
  }
  if (intraNode && caseOverride != TransportType::NVLink) {
    if (auto* t = tryTier(TransportType::NVLink)) {
      return t;
    }
  }
  if (caseOverride != TransportType::RDMA) {
    if (auto* t = tryTier(TransportType::RDMA)) {
      return t;
    }
  }
  if (caseOverride != TransportType::TCP) {
    if (auto* t = tryTier(TransportType::TCP)) {
      return t;
    }
  }

  return Err(
      ErrCode::NotConnected,
      "no common transport available across all requests");
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

std::future<Status> MultiTransport::recv(
    RegisteredSegment::Span dst,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
}

std::future<Status> MultiTransport::recv(
    Segment::Span dst,
    const RequestOptions& options) {
  return make_ready_future<Status>(ErrCode::NotImplemented);
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
          toStringView(f->transportType()),
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
  if (factories_.empty()) {
    return Err(ErrCode::TopologyDisconnect, "no transport can be connected");
  }
  auto& entries = parsed.value();
  if (entries.size() != factories_.size()) {
    UNIFLOW_LOG_WARN(
        "transport count mismatch: local={}, peer={}",
        factories_.size(),
        entries.size());
  }

  auto mt = std::make_unique<MultiTransport>(
      deviceId_,
      eventBaseThread_,
      options_.intraNodeTransport,
      options_.preferredTransport,
      options_.interNodeTransport);
  for (size_t i = 0, j = 0; i < entries.size() && j < factories_.size();) {
    if (entries[i].type < factories_[j]->transportType()) {
      ++i;
      continue;
    }
    if (entries[i].type > factories_[j]->transportType()) {
      ++j;
      continue;
    }

    auto transport = factories_[j]->createTransport(entries[i].data);
    if (transport) {
      mt->addTransport(std::move(transport).value());
    } else {
      UNIFLOW_LOG_WARN(
          "Transport {} cannot be created: {}",
          toStringView(factories_[j]->transportType()),
          transport.error().message());
    }
    ++i;
    ++j;
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
