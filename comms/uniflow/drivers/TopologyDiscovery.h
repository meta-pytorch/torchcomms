// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/transport/Topology.h"

namespace uniflow {

/// Backend-neutral topology discovery strategy.
///
/// Each accelerator backend (e.g. CUDA) implements this interface and
/// owns its own backend-specific dependencies (driver APIs) internally or via
/// a backend-specific constructor. `Topology` stays a pure graph/data model and
/// this interface stays stable as new backends are added — no backend's driver
/// concepts leak into the common discovery surface.
class TopologyDiscoveryBackend {
 public:
  virtual ~TopologyDiscoveryBackend() = default;

  /// Fully populate @p topology by probing this backend on the local host.
  ///
  /// DESTRUCTIVE: calls `topology.clear()` before probing, so any nodes,
  /// links, paths, and status previously set on @p topology are discarded.
  /// Safe to call repeatedly on the same `Topology` instance to re-discover.
  /// Tolerates partial discovery — populates whatever is available and
  /// returns Ok.
  virtual Status discover(Topology& topology) = 0;
};

/// Create the host's default discovery backend, instantiated with real driver
/// APIs. The concrete backend is chosen at build time via Buck `select()` in
/// `comms/uniflow/drivers:topology-discovery` (CUDA by default, or the
/// accelerator backend selected by the build's GPU config).
std::unique_ptr<TopologyDiscoveryBackend> createDefaultDiscoveryBackend();

/// Process-wide singleton Topology, populated lazily on first call via
/// `createDefaultDiscoveryBackend()->discover(topo)`. If discovery fails the
/// error is logged; the (unavailable) topology is still returned.
///
/// This singleton always uses real driver instances and cannot be overridden.
/// To inject mock drivers in tests, do NOT call `sharedTopology()` — instead
/// construct a local `Topology` and run the concrete backend directly with
/// mock APIs. For CUDA:
///
/// ```cpp
/// #include "comms/uniflow/drivers/cuda/CudaTopologyDiscovery.h"
///
/// auto cuda  = std::make_shared<MockCudaApi>();
/// auto nvml  = std::make_shared<MockNvmlApi>();
/// auto ibv   = std::make_shared<MockIbvApi>();
/// auto sysfs = std::make_shared<MockSysfsApi>();
/// // ... set up mock expectations ...
///
/// Topology topo;
/// ASSERT_TRUE(CudaTopologyDiscovery(cuda, nvml, ibv, sysfs).discover(topo));
/// // `topo` is now a hermetic, mock-driven topology for the test.
/// ```
///
/// Other backends expose their own backend-specific entry points for tests.
Topology& sharedTopology();

} // namespace uniflow
