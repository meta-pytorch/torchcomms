// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/uniflow/MultiTransportOptions.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/Topology.h"
#include "comms/uniflow/transport/Transport.h"
#include "comms/uniflow/transport/TransportProvider.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace uniflow {

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

  /// Uses only the supplied providers, without default backend discovery.
  /// This overload applies only intraNodeTransport, preferredTransport, and
  /// interNodeTransport from options; providers own backend configuration.
  /// deviceId is -1 for CPU or non-negative; providers validate device
  /// availability without relying on the default GPU topology.
  /// Invalid providers throw invalid_argument; creation errors throw
  /// runtime_error.
  MultiTransportFactory(
      int deviceId,
      MultiTransportFactoryOptions options,
      TransportProviderList providers);

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

  int deviceId_{-1};
  MultiTransportFactoryOptions options_;
  std::shared_ptr<ScopedEventBaseThread> eventBaseThread_;
  std::vector<std::shared_ptr<TransportFactory>> factories_;
};

} // namespace uniflow
