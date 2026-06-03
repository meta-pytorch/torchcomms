// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/drivers/nvml/NvmlApi.h"
#include "comms/uniflow/drivers/sysfs/SysfsApi.h"
#include "comms/uniflow/transport/Topology.h"

namespace uniflow {

/// Fully populate @p topology by probing the accelerator backend on this host.
///
/// DESTRUCTIVE: calls `topology.clear()` before probing, so any nodes,
/// links, paths, and status previously set on @p topology are discarded.
/// Safe to call repeatedly on the same `Topology` instance to re-discover.
///
/// Each driver argument is optional; when null, a default-constructed real
/// instance is used. The function tolerates partial discovery — if a given
/// backend (e.g. CUDA) is not present it still populates NUMA nodes, NICs,
/// and any other backends that are available, and returns Ok.
///
/// Backend selection is done at build-time.
Status discoverTopology(
    Topology& topology,
    std::shared_ptr<CudaApi> cudaApi = nullptr,
    std::shared_ptr<NvmlApi> nvmlApi = nullptr,
    std::shared_ptr<IbvApi> ibvApi = nullptr,
    std::shared_ptr<SysfsApi> sysfsApi = nullptr);

/// Process-wide singleton Topology, populated lazily on first call via
/// `discoverTopology(topo)` with default-constructed driver instances.
///
/// This singleton always uses real driver instances and cannot be
/// overridden. To inject mock drivers in tests, do NOT call
/// `sharedTopology()` — instead construct a local `Topology` and invoke
/// the backend-specific discovery directly. For CUDA:
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
/// ASSERT_TRUE(discoverCudaTopology(topo, cuda, nvml, ibv, sysfs));
/// // `topo` is now a hermetic, mock-driven topology for the test.
/// ```
Topology& sharedTopology();

} // namespace uniflow
