// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

namespace uniflow {

enum TransportType : uint8_t {
  NVLink = 0, // NVLink for intra-node or MNNVL
  RDMA, // InfiniBand or RoCE RDMA
  TCP, // TCP/IP fallback
  Mock, // Mock transport for testing
  NumTransportType,
};

} // namespace uniflow
