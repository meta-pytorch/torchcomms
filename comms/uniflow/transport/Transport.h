// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <span>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/transport/TransportType.h"

namespace uniflow {

struct TransferRequest {
  RegisteredSegment::Span local;
  RemoteRegisteredSegment::Span remote;
};

/// Options passed to data-transfer operations (put/get/send/recv).
struct RequestOptions {
  /// Optional GPU stream handle (e.g., cudaStream_t) for backends that
  /// support GPU-async transfers. Non-GPU backends ignore it.
  std::optional<void*> stream = std::nullopt;
  /// Optional per-request timeout. RDMA ignores it unless
  /// RdmaTransportConfig::requestTimeout is set. When it is set, every check
  /// below runs on the caller thread, before dispatch:
  /// - nullopt or 0ms inherits requestTimeout; any other value that differs
  ///   from requestTimeout returns InvalidArgument.
  /// - Copy send()/recv() return NotImplemented.
  ///
  /// The timeout starts at the transport's first processing pass after its
  /// EventBase admits the operation, so EventBase delays can lengthen it but
  /// never shorten it.
  ///
  /// When a timeout fires, that operation gets Timeout, and every other
  /// in-flight or later operation on the transport gets Aborted. A Timeout or
  /// Aborted result leaves the operation's outcome indeterminate (the remote
  /// range of a PUT may be partially written), but local buffers are reusable
  /// as soon as the result is visible, because the transport destroys its QPs
  /// first. The peer's operations on the paired QPs fail. The transport cannot
  /// be reused: recover with shutdown() and a new transport.
  ///
  /// Only Timeout and Aborted carry this guarantee. Any other error, such as
  /// DriverError from a failed work request, can be returned while the
  /// operation's work requests on other QPs are still running, so keep its
  /// buffers until shutdown() returns.
  std::optional<std::chrono::milliseconds> timeout = std::nullopt;
};

using TransportInfo = std::vector<uint8_t>;

enum class TransportState : uint8_t {
  Disconnected,
  Initialized,
  Connected,
  Error,
};

class Transport {
 public:
  virtual ~Transport() = default;

  virtual const std::string& name() const noexcept = 0;

  virtual TransportType transportType() const noexcept = 0;

  virtual TransportState state() const noexcept = 0;

  virtual TransportInfo bind() = 0;

  virtual Status connect(std::span<const uint8_t> remoteInfo) = 0;

  // Batch transfer operations
  virtual std::future<Status> put(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) = 0;

  virtual std::future<Status> get(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) = 0;

  // Zero copy send/recv operations
  virtual std::future<Status> send(
      RegisteredSegment::Span src,
      const RequestOptions& options = {}) = 0;

  virtual std::future<Status> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {}) = 0;

  // Copy based send/recv operations
  virtual std::future<Status> send(
      Segment::Span src,
      const RequestOptions& options = {}) = 0;

  virtual std::future<Status> recv(
      Segment::Span dst,
      const RequestOptions& options = {}) = 0;

  virtual void shutdown() = 0;

 protected:
  Transport() = default;
};

class TransportFactory {
 public:
  explicit TransportFactory(TransportType transportType)
      : transportType_(transportType) {}

  virtual ~TransportFactory() = default;

  TransportType transportType() const noexcept {
    return transportType_;
  }

  virtual Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& segment) = 0;

  virtual Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t segmentLength,
      std::span<const uint8_t> payload) = 0;

  virtual Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> peerTopology) = 0;

  virtual std::vector<uint8_t> getTopology() = 0;

  virtual Status canConnect(std::span<const uint8_t> peerTopology) = 0;

 protected:
  const TransportType transportType_;
};

} // namespace uniflow
