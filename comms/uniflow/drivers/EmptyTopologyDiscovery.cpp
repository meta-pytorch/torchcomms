// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/TopologyDiscovery.h"

namespace uniflow {
namespace {

class EmptyTopologyDiscovery final : public TopologyDiscoveryBackend {
 public:
  Status discover(Topology& topology) override {
    topology.clear();
    topology.setStatus(Ok());
    return Ok();
  }
};

} // namespace

std::unique_ptr<TopologyDiscoveryBackend> createDefaultDiscoveryBackend() {
  return std::make_unique<EmptyTopologyDiscovery>();
}

} // namespace uniflow
