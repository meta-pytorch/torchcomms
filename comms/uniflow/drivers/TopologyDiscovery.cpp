// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/TopologyDiscovery.h"

namespace uniflow {

// `discoverTopology()` is provided by the platform-specific library
// selected at build time via Buck `select()` in
// `comms/uniflow/drivers:topology-discovery` (CUDA by default, MTIA when
// `ovr_config//gpu:mtia` is active).

Topology& sharedTopology() {
  static Topology topo;
  static Status init = discoverTopology(topo);
  (void)init;
  return topo;
}

} // namespace uniflow
