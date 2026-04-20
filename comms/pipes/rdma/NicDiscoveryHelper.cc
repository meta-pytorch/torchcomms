// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/rdma/NicDiscoveryHelper.h"

#include "comms/pipes/rdma/NicDiscovery.h"

namespace comms::pipes {

std::string NicDiscoveryHelper::getCudaPciBusId(int cudaDevice) {
  return GpuNicDiscovery::getCudaPciBusId(cudaDevice);
}

std::string NicDiscoveryHelper::discoverBestNic(
    int cudaDevice,
    const std::string& ibHca) {
  auto discovery = GpuNicDiscovery(cudaDevice, ibHca);
  return discovery.getCandidates()[0].name;
}

} // namespace comms::pipes
