// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/Segment.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/Transport.h"
#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

namespace uniflow {

/// Intra-node GPU P2P transport (XGMI on AMD; also works on NVIDIA P2P).
///
/// Shares the NVLink interconnect tier (`TransportType::NVLink`). Memory is
/// shared via HIP IPC handles (P2pRegistrationHandle); transfers are
/// device-to-device copies over the GPU interconnect. Neutral by construction:
/// all GPU contact goes through the CudaApi seam, so this is a plain,
/// non-hipified library.
///
/// Threading: a transport instance is single-threaded-owned. The lifecycle
/// methods (bind/connect/shutdown) and state(), peerDeviceId_ access are not
/// synchronized and must all be driven by one owner thread; only the GPU copy
/// work submitted by transfer() runs off-thread (on the EventBase and the
/// driver completion callback), and it never touches the lifecycle state. This
/// matches NVLinkTransport's ownership model.
class P2pTransport : public Transport {
 public:
  P2pTransport(
      int deviceId,
      EventBase* evb,
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  const std::string& name() const noexcept override {
    return deviceName_;
  }

  TransportType transportType() const noexcept override {
    return TransportType::NVLink;
  }

  TransportState state() const noexcept override {
    return state_;
  }

  TransportInfo bind() override;
  Status connect(std::span<const uint8_t> remoteInfo) override;

  std::future<Status> put(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  std::future<Status> get(
      std::span<const TransferRequest> requests,
      const RequestOptions& options = {}) override;

  std::future<Status> send(
      RegisteredSegment::Span src,
      const RequestOptions& options = {}) override;

  std::future<Status> send(
      Segment::Span src,
      const RequestOptions& options = {}) override;

  std::future<Status> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {}) override;

  std::future<Status> recv(
      Segment::Span dst,
      const RequestOptions& options = {}) override;

  void shutdown() override;

 private:
  struct CopyOp {
    void* dst{nullptr};
    const void* src{nullptr};
    size_t size{};
  };

  std::future<Status> transfer(std::vector<CopyOp> ops, void* stream);

  Result<const P2pRemoteRegistrationHandle*> findRemoteHandle(
      const RemoteRegisteredSegment::Span& span) const;

  int deviceId_{-1};
  int peerDeviceId_{-1};
  std::string deviceName_;
  TransportState state_{TransportState::Disconnected};
  EventBase* evb_{nullptr};
  std::shared_ptr<CudaApi> cudaApi_;
};

/// Factory for the intra-node P2P transport. Registration exports a HIP IPC
/// handle; import opens it (cross-process) or reuses the exporter pointer
/// (same-process).
class P2pTransportFactory : public TransportFactory {
 public:
  /// Available when at least one GPU is present.
  static Status supported(std::shared_ptr<CudaApi> cudaApi = nullptr);

  P2pTransportFactory(
      int deviceId,
      EventBase* evb,
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  ~P2pTransportFactory() override = default;

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& segment) override;

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t segmentLength,
      std::span<const uint8_t> payload) override;

  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> peerTopology) override;

  std::vector<uint8_t> getTopology() override;

 private:
  Status canConnect(std::span<const uint8_t> peerTopology) override;

  int deviceId_{-1};
  EventBase* evb_{nullptr};
  std::shared_ptr<CudaApi> cudaApi_;
};

} // namespace uniflow
