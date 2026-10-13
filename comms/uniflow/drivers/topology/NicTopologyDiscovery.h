// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "comms/uniflow/drivers/TopologyDiscovery.h" // @manual=//comms/uniflow/drivers:topology-discovery-interface
#include "comms/uniflow/drivers/topology/PciTopology.h"

namespace uniflow {
class IbvApi;
class SysfsApi;

namespace topology {

struct NicDiscovery {
  std::string name;
  TopoNode::NicData data;
  std::string sysfsPath;
  std::vector<std::string> ancestorChain;
  PcieLinkInfo linkInfo;
};

struct NicDiscoveryData {
  std::vector<NicDiscovery> nics;
  int numaCount{1};
};

/// NIC/NUMA probing only, with partial-discovery behavior shared by GPU
/// backends.
NicDiscoveryData discoverNicHardware(IbvApi* ibvApi, SysfsApi& sysfs);
void addNicNodes(
    Topology& topology,
    std::span<const NicDiscovery> nics,
    int numaCount);
void appendNicPciDevices(
    std::vector<PciDevice>& devices,
    const Topology& topology,
    std::span<const NicDiscovery> nics);

} // namespace topology

class NicTopologyDiscovery final : public TopologyDiscoveryBackend {
 public:
  explicit NicTopologyDiscovery(
      std::shared_ptr<IbvApi> ibvApi = nullptr,
      std::shared_ptr<SysfsApi> sysfsApi = nullptr);

  Status discover(Topology& topology) override;

 private:
  std::shared_ptr<IbvApi> ibvApi_;
  std::shared_ptr<SysfsApi> sysfsApi_;
};

/// CPU/NIC graph only. This entry point never selects or initializes a GPU.
Topology& sharedNicTopology();

} // namespace uniflow
