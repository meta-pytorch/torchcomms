// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/drivers/nvml/NvmlApi.h"
#include "comms/uniflow/drivers/sysfs/SysfsApi.h"
#include "comms/uniflow/transport/Topology.h"

namespace uniflow {

/// CUDA/NVIDIA topology discovery backend.
///
/// Probes GPUs (CUDA/NVML), NICs (ibverbs), and PCIe/NUMA topology (sysfs).
/// Each driver dependency is injectable for testing; any argument left null is
/// resolved to a real driver instance. Unit tests construct this directly with
/// mock APIs (see `sharedTopology()` docs in `TopologyDiscovery.h`).
class CudaTopologyDiscovery final : public TopologyDiscoveryBackend {
 public:
  explicit CudaTopologyDiscovery(
      std::shared_ptr<CudaApi> cudaApi = nullptr,
      std::shared_ptr<NvmlApi> nvmlApi = nullptr,
      std::shared_ptr<IbvApi> ibvApi = nullptr,
      std::shared_ptr<SysfsApi> sysfsApi = nullptr);

  Status discover(Topology& topology) override;

 private:
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<NvmlApi> nvmlApi_;
  std::shared_ptr<IbvApi> ibvApi_;
  std::shared_ptr<SysfsApi> sysfsApi_;
};

} // namespace uniflow
