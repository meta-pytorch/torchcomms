// Copyright (c) Meta Platforms, Inc. and affiliates.

/// Prints a topology matrix similar to `nvidia-smi topo -m` using our
/// Topology component. Run with: buck run
/// fbcode//comms/uniflow/transport/tests/integration:topology_dump
/// @fbcode//mode/opt

#include "comms/uniflow/transport/Topology.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace uniflow;

int main() {
  auto& topo = Topology::get();
  if (!topo.available()) {
    fprintf(
        stderr,
        "discover() failed: %s\n",
        topo.available().error().toString().c_str());
    return 1;
  }

  // Collect device labels and node IDs.
  struct Device {
    std::string label;
    int nodeId;
  };
  std::vector<Device> devices;

  devices.reserve(topo.gpuCount() + topo.nicCount());
  for (size_t i = 0; i < topo.gpuCount(); ++i) {
    devices.push_back({
        .label = "GPU" + std::to_string(i),
        .nodeId = topo.getGpuNode(i).id,
    });
  }
  for (size_t i = 0; i < topo.nicCount(); ++i) {
    devices.push_back({
        .label = topo.getNicNode(i).name,
        .nodeId = topo.getNicNode(i).id,
    });
  }

  // Determine column width.
  size_t colWidth = 6;
  for (const auto& dev : devices) {
    colWidth = std::max(colWidth, dev.label.size() + 2);
  }

  // Print header.
  printf("%*s", static_cast<int>(colWidth), "");
  for (const auto& dev : devices) {
    printf("%-*s", static_cast<int>(colWidth), dev.label.c_str());
  }
  printf("\n");

  // Print matrix rows.
  for (const auto& row : devices) {
    printf("%-*s", static_cast<int>(colWidth), row.label.c_str());
    for (const auto& col : devices) {
      if (row.nodeId == col.nodeId) {
        printf("%-*s", static_cast<int>(colWidth), "X");
        continue;
      }
      const auto& path = topo.getPath(row.nodeId, col.nodeId);
      printf("%-*s", static_cast<int>(colWidth), pathTypeToString(path.type));
    }
    printf("\n");
  }

  // Print legend.
  printf("\nLegend:\n");
  printf("  X    = Self\n");
  printf("  NVL  = NVLink\n");
  printf("  C2C  = Chip-to-chip (e.g. Grace Hopper)\n");
  printf("  PIX  = Connection traversing at most a single PCIe bridge\n");
  printf("  PXB  = Connection traversing multiple PCIe bridges\n");
  printf("  PXN  = PCIe + NVLink proxy through peer GPU\n");
  printf("  PHB  = Same PCIe host bridge (same root complex)\n");
  printf("  SYS  = Cross NUMA\n");
  printf("  DIS  = Disconnected\n");

  // Print NIC legend.
  if (topo.nicCount() > 0) {
    printf("\nNIC Legend:\n");
    for (size_t i = 0; i < topo.nicCount(); ++i) {
      const auto& nicNode = topo.getNicNode(static_cast<int>(i));
      const auto& nicData = std::get<TopoNode::NicData>(nicNode.data);
      printf(
          "  %s: %s (port: %d) numa %d\n",
          nicNode.name.c_str(),
          nicData.bdf.c_str(),
          nicData.port,
          nicData.numaNode);
    }
  }

  // Print GPU info.
  printf("\nGPU Info:\n");
  for (size_t i = 0; i < topo.gpuCount(); ++i) {
    const auto& gpuData = std::get<TopoNode::GpuData>(topo.getGpuNode(i).data);
    printf(
        "  GPU%zu: %s (SM %d) numa %d\n",
        i,
        gpuData.bdf.c_str(),
        gpuData.sm,
        gpuData.numaNode);
  }

  return 0;
}
