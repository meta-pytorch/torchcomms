// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "comms/uniflow/transport/Topology.h"

namespace uniflow {
class SysfsApi;

namespace topology {

struct PciDevice {
  int nodeId;
  int numaNode;
  std::vector<std::string> ancestorChain;
  uint32_t portSpeedMBps{0};
};

std::string_view extractBdf(std::string_view sysfsPath);
std::vector<std::string> buildAncestorChain(
    SysfsApi& sysfs,
    const std::string& sysfsPath);
int readNumaNode(SysfsApi& sysfs, std::string_view pciDevicePath);
PcieLinkInfo readPcieLinkInfo(
    SysfsApi& sysfs,
    const std::string& sysfsDevicePath);
int discoverNumaNodeCount(SysfsApi& sysfs);

void addPciCpuEdges(
    Topology& topology,
    SysfsApi& sysfs,
    std::span<const PciDevice> devices,
    int numaCount);
void addPciPeerEdges(
    Topology& topology,
    SysfsApi& sysfs,
    std::span<const PciDevice> devices);
void addInterCpuEdges(Topology& topology, int numaCount);

} // namespace topology
} // namespace uniflow
