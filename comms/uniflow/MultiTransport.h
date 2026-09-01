// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/Topology.h"
#include "comms/uniflow/transport/Transport.h"

#include <array>
#include <cstddef>
#include <optional>

namespace uniflow {

// Defined in transport/tcp/TcpTransport.h, which this header cannot include:
// TCP is AMD-only, so the type is not available on every platform.
struct TcpTransportConfig;

enum class CpuNicSelectionPolicy {
  kAll,
  kNumaLocalBounded,
};

struct MultiTransportFactoryOptions {
  NicFilter nicFilter;
  std::string netdevPrefix{"beth"};
  int16_t gidIndex{-1};
  uint8_t trafficClass{0};
  CpuNicSelectionPolicy cpuNicSelectionPolicy{
      CpuNicSelectionPolicy::kNumaLocalBounded};
  size_t maxCpuNics{2};
  // Runtime intra-node transport selection. Intra-node VRAM<->VRAM defaults to
  // the on-host interconnect tier (NVLink on NVIDIA, P2P/XGMI on AMD). Set this
  // to flip that choice -- e.g. TransportType::RDMA to route intra-node traffic
  // over RDMA instead. Both tiers stay registered, so this is a selection
  // override, not a kill-switch. Caller-owned (no transport-internal
  // config-system dependency).
  std::optional<TransportType> intraNodeTransport;
  // preferredTransport: global force, checked first in selectTransport --
  // applies to both intra- and inter-node. interNodeTransport: inter-node
  // override (default RDMA). Each falls through to the remaining tiers if the
  // chosen transport is unavailable on all requests.
  std::optional<TransportType> preferredTransport;
  std::optional<TransportType> interNodeTransport;
  // The TCP data transport auto-registers whenever a routable bind address is
  // available (tcpBindHost if set, else resolveTcpBindHost: netdevPrefix match,
  // else first global address). It is skipped when only loopback resolves,
  // since advertising loopback in the connect handshake breaks cross-host
  // connections for all transports. Set enableTcp to force-register even on
  // loopback (e.g. same-host testing).
  bool enableTcp{false};
  std::string tcpBindHost{};
  // Overrides for the TCP data transport: socket options, lane count
  // (numSockets), and the devices lanes bind to. Null keeps the
  // TcpTransportConfig defaults, so this is an override seam rather than
  // required config, and it is the only way to reach lane striping from outside
  // the transport.
  //
  // Held by pointer, and the type only forward-declared, because TCP is
  // AMD-only (see the ovr_config//gpu:amd deps in BUCK): a by-value member
  // would drag TcpTransport.h into every consumer of this header and fail to
  // compile where the transport is not built. The struct layout stays
  // platform-independent this way, so callers need no #ifdef to leave it unset.
  std::shared_ptr<const TcpTransportConfig> tcpTransportConfig;
};

class MultiTransport {
 public:
  explicit MultiTransport(
      int deviceId,
      std::shared_ptr<ScopedEventBaseThread> evbThread = nullptr,
      std::optional<TransportType> intraNodeTransport = std::nullopt,
      std::optional<TransportType> preferredTransport = std::nullopt,
      std::optional<TransportType> interNodeTransport = std::nullopt)
      : deviceId_(deviceId),
        preferredTransport_(preferredTransport),
        intraNodeTransport_(intraNodeTransport),
        interNodeTransport_(interNodeTransport),
        evbThread_(std::move(evbThread)) {
    if (!evbThread_) {
      evbThread_ = std::make_shared<ScopedEventBaseThread>();
    }
  }
  ~MultiTransport() = default;

  void addTransport(std::unique_ptr<Transport> transport);

  Result<TransportInfo> bind();

  Status connect(std::span<const uint8_t> info);

  // Batch transfer operations
  std::future<Status> put(
      const std::vector<TransferRequest>& requests,
      const RequestOptions& options = {});

  std::future<Status> get(
      const std::vector<TransferRequest>& requests,
      const RequestOptions& options = {});

  // Zero copy send/recv operations
  std::future<Status> send(
      RegisteredSegment::Span src,
      const RequestOptions& options = {});

  std::future<Status> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {});

  // Copy based send/recv operations
  std::future<Status> send(
      Segment::Span src,
      const RequestOptions& options = {});

  std::future<Status> recv(
      Segment::Span dst,
      const RequestOptions& options = {});

  /// Number of transfer operations dispatched to a given transport type.
  uint64_t transferCount(TransportType type) const {
    return transferCounts_[type];
  }

  void shutdown();

  friend class MultiTransportFactory;

 private:
  using TransferOp = std::future<Status> (
      Transport::*)(std::span<const TransferRequest>, const RequestOptions&);

  Result<Transport*> selectTransport(
      const std::vector<TransferRequest>& requests);

  Status validateRequests(const std::vector<TransferRequest>& requests);

  std::future<Status> doTransfer(
      const std::vector<TransferRequest>& requests,
      const RequestOptions& options,
      TransferOp op);

  Transport* findTransport(TransportType type) const;

  const int deviceId_;
  std::optional<TransportType> preferredTransport_;
  std::optional<TransportType> intraNodeTransport_;
  std::optional<TransportType> interNodeTransport_;
  // Prevents destruction of the shared EventBase while transports are live.
  // Transports hold raw EventBase* borrowed from the ScopedEventBaseThread
  // owned by MultiTransportFactory; this shared_ptr ensures the thread (and
  // its EventBase) outlives the transports.
  std::shared_ptr<ScopedEventBaseThread> evbThread_;
  std::vector<std::unique_ptr<Transport>> transports_;
  std::array<uint64_t, NumTransportType> transferCounts_{};
};

class MultiTransportFactory {
 public:
  explicit MultiTransportFactory(
      int deviceId,
      MultiTransportFactoryOptions options = {});

  Result<RegisteredSegment> registerSegment(Segment& segment);

  Result<RemoteRegisteredSegment> importSegment(
      std::span<const uint8_t> exportId);

  Result<std::unique_ptr<MultiTransport>> createTransport(
      std::span<const uint8_t> peerTopology);

  std::vector<uint8_t> getTopology();

  static Status supported(TransportType type);

  friend class MultiTransportFactoryTest;

 private:
  struct TopologyEntry {
    TransportType type{};
    std::span<const uint8_t> data;
  };
  static Result<std::vector<TopologyEntry>> parse(
      std::span<const uint8_t> peerTopology);

  explicit MultiTransportFactory(
      std::vector<std::shared_ptr<TransportFactory>> factories)
      : factories_(std::move(factories)) {}

  std::vector<std::string> selectNics();
  std::vector<std::string> selectCpuNics();

  int deviceId_{-1};
  MultiTransportFactoryOptions options_;
  std::shared_ptr<ScopedEventBaseThread> eventBaseThread_;
  std::vector<std::shared_ptr<TransportFactory>> factories_;
};

} // namespace uniflow
