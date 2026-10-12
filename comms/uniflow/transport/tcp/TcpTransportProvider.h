// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/transport/TransportProvider.h"
#include "comms/uniflow/transport/tcp/TcpTransport.h"

namespace uniflow {

/// Supplies TCP data factories without replacing caller settings.
class TcpTransportProvider : public TransportProvider {
 public:
  explicit TcpTransportProvider(
      TcpTransportConfig config = {},
      std::string host = "127.0.0.1",
      std::shared_ptr<CudaApi> cudaApi = nullptr);

  TransportType transportType() const noexcept override;
  Status supported() const override;
  Result<std::shared_ptr<TransportFactory>> createFactory(
      const TransportProviderContext& context) const override;

 private:
  TcpTransportConfig config_;
  std::string host_;
  std::shared_ptr<CudaApi> cudaApi_;
};

} // namespace uniflow
