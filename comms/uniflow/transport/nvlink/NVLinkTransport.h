// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/nvml/NvmlApi.h"
#include "comms/uniflow/executor/EventBase.h"
#include "comms/uniflow/transport/Transport.h"
#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"

namespace uniflow {

// ---------------------------------------------------------------------------
// NVLinkTransport
// ---------------------------------------------------------------------------

class NVLinkTransport : public Transport {
 public:
  /// Construct a transport for a local GPU to communicate with a peer GPU.
  NVLinkTransport(
      int deviceId,
      EventBase* evb,
      std::shared_ptr<CudaApi> cuda_api = nullptr,
      std::shared_ptr<CudaDriverApi> cuda_driver_api = nullptr);

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

  std::future<Result<size_t>> recv(
      RegisteredSegment::Span dst,
      const RequestOptions& options = {}) override;

  std::future<Result<size_t>> recv(
      Segment::Span dst,
      const RequestOptions& options = {}) override;

  void shutdown() override;

 private:
  struct CopyOp {
    void* dst{nullptr};
    const void* src{nullptr};
    size_t size{};
  };

  std::future<Status> transfer(std::vector<CopyOp> ops, cudaStream_t stream);

  Result<const NVLinkRemoteRegistrationHandle*> findRemoteHandle(
      const RemoteRegisteredSegment::Span& span) const;

  int deviceId_{-1};
  int peerDeviceId_{-1};
  std::string deviceName_;
  TransportState state_{TransportState::Disconnected};
  EventBase* evb_{nullptr};
  std::shared_ptr<CudaApi> cuda_api_{nullptr};
  std::shared_ptr<CudaDriverApi> cuda_driver_api_{nullptr};
};

// ---------------------------------------------------------------------------
// NVLinkTransportFactory
// ---------------------------------------------------------------------------

class NVLinkTransportFactory : public TransportFactory {
 public:
  /// Construct a factory for the given CUDA device.
  /// Queries NVML for GPU fabric info to populate local NVLink topology.
  /// Falls back to POSIX FD-based IPC for single-node sharing when
  /// fabric handles are not supported by the device (e.g. H100).
  explicit NVLinkTransportFactory(
      int deviceId,
      EventBase* evb,
      std::shared_ptr<NvmlApi> nvmlApi = nullptr,
      std::shared_ptr<CudaApi> cuda_api = nullptr,
      std::shared_ptr<CudaDriverApi> cuda_driver_api = nullptr);

  ~NVLinkTransportFactory() override = default;

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& segment) override;

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t segmentLength,
      std::span<const uint8_t> payload) override;

  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> peerTopology) override;

  std::vector<uint8_t> getTopology() override;

  /// Return the CUDA handle type that allocations must be created with
  /// (via CUmemAllocationProp::requestedHandleTypes) for registerSegment
  /// to successfully export them.
  CUmemAllocationHandleType handleType() const noexcept {
    return handleType_;
  }

 private:
  Status canConnect(std::span<const uint8_t> peerTopology) override;

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegmentFabric(
      std::span<const uint8_t> payload);

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegmentFd(
      std::span<const uint8_t> payload);

  Result<std::unique_ptr<RemoteRegistrationHandle>> mapImportedAllocation(
      CUmemGenericAllocationHandle allocHandle,
      size_t segmentLength,
      CUmemAllocationHandleType handleType);

  int deviceId_{-1};
  EventBase* evb_{nullptr};
  CUmemAllocationHandleType handleType_{CU_MEM_HANDLE_TYPE_NONE};
  std::shared_ptr<NvmlApi> nvmlApi_{nullptr};
  std::shared_ptr<CudaApi> cuda_api_{nullptr};
  std::shared_ptr<CudaDriverApi> cuda_driver_api_{nullptr};
};

} // namespace uniflow
