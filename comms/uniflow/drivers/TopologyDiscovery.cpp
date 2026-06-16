// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/TopologyDiscovery.h"

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

// `createDefaultDiscoveryBackend()` is provided by the platform-specific
// library selected at build time via Buck `select()` in
// `comms/uniflow/drivers:topology-discovery` (CUDA by default, or the
// accelerator backend selected by the build's GPU config).

Topology& sharedTopology() {
  static Topology topo;
  static Status init = [] {
    Status st = createDefaultDiscoveryBackend()->discover(topo);
    if (st.hasError()) {
      UNIFLOW_LOG_ERROR("Topology discovery failed: {}", st.error().toString());
    }
    return st;
  }();
  (void)init;
  return topo;
}

} // namespace uniflow
