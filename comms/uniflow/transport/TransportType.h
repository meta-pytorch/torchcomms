// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string_view>

namespace uniflow {

enum TransportType : uint8_t {
  NVLink = 0, // NVLink for intra-node or MNNVL
  RDMA, // InfiniBand or RoCE RDMA
  TCP, // TCP/IP fallback
  Mock, // Mock transport for testing
  NumTransportType,
};

constexpr std::string_view toStringView(TransportType t) noexcept {
  switch (t) {
    case TransportType::NVLink:
      return "NVLink";
    case TransportType::RDMA:
      return "RDMA";
    case TransportType::TCP:
      return "TCP";
    case TransportType::Mock:
      return "Mock";
    case TransportType::NumTransportType:
      break;
  }
  return "Unknown";
}

} // namespace uniflow
