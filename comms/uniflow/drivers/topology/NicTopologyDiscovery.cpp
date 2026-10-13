// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/topology/NicTopologyDiscovery.h"

#include <algorithm>
#include <memory>
#include <utility>

#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
#include "comms/uniflow/drivers/ibverbs/IbvApi.h" // @manual=//comms/uniflow/drivers/ibverbs:uniflow-rdma-deps
#endif
#include "comms/uniflow/drivers/sysfs/SysfsApi.h"

namespace uniflow {

namespace topology {
namespace {

#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
/// Read the backing netdev name for an IB device by listing
/// /sys/class/infiniband/<dev>/device/net/. Vendor-neutral (works for mlx5,
/// bnxt_re, etc.). Returns the first entry, or empty if none is found.
std::string readNetdevName(SysfsApi& sysfs, const std::string& ibdevPath) {
  auto entries = sysfs.listDir(ibdevPath + "/device/net");
  return entries.empty() ? std::string{} : entries.front();
}

/// IB width multipliers indexed by bit position in ibv_port_attr.active_width.
/// Bit 0 = 1x, bit 1 = 4x, bit 2 = 8x, bit 3 = 12x, bit 4 = 2x.
// clang-format off
constexpr int kIbvWidths[] = {1, 4, 8, 12, 2};
// clang-format on

/// IB per-lane speed (Mbps) indexed by bit position in active_speed.
/// Bit 0 = SDR, 1 = DDR, 2 = QDR, 3 = QDR, 4 = FDR, 5 = EDR,
/// 6 = HDR, 7 = NDR.
// clang-format off
constexpr int kIbvSpeeds[] = {
    2500,   // SDR
    5000,   // DDR
    10000,  // QDR
    10000,  // QDR
    14000,  // FDR
    25000,  // EDR
    50000,  // HDR
    100000, // NDR
};
// clang-format on

/// Return the index of the lowest set bit, or -1 if none.
int firstBitSet(uint32_t v) {
  for (int i = 0; v != 0; ++i, v >>= 1) {
    if (v & 1) {
      return i;
    }
  }
  return -1;
}

/// Compute RDMA port speed in Mbps from ibv_port_attr bitmask fields.
uint32_t ibPortSpeedMbps(uint32_t activeSpeed, uint32_t activeWidth) {
  int si = firstBitSet(activeSpeed);
  int wi = firstBitSet(activeWidth);
  if (si < 0 || wi < 0) {
    return 0;
  }
  int speed =
      (si < static_cast<int>(std::size(kIbvSpeeds))) ? kIbvSpeeds[si] : 0;
  int width =
      (wi < static_cast<int>(std::size(kIbvWidths))) ? kIbvWidths[wi] : 0;
  return static_cast<uint32_t>(speed * width);
}
#endif

#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
/// Check whether an ibdev sysfs path belongs to a virtual (software) RDMA
/// device such as RXE (Soft-RoCE).  These live under
/// /sys/devices/virtual/ and have no PCI backing.
bool isVirtualRdmaDevice(SysfsApi& sysfs, const std::string& ibdevPath) {
  auto resolved = sysfs.resolvePath(ibdevPath);
  if (!resolved) {
    return false;
  }
  return resolved.value().find("/devices/virtual/") != std::string::npos;
}

constexpr size_t kDefaultRxeSpeed = 10000; // 10 Gbps

/// Discover NICs via IbvApi: enumerate IB devices and active ports.
/// Returns empty vector if ibverbs is not available.
Result<std::vector<NicDiscovery>> discoverNics(
    IbvApi& ibvApi,
    SysfsApi& sysfs) {
  std::vector<NicDiscovery> nics;
  CHECK_EXPR(ibvApi.init());

  int numDevices = 0;
  auto devListResult = ibvApi.getDeviceList(&numDevices);
  CHECK_RETURN(devListResult);

  auto devListDeleter = [&ibvApi](ibv_device** ptr) {
    if (ptr) {
      ibvApi.freeDeviceList(ptr);
    }
  };
  std::unique_ptr<ibv_device*[], decltype(devListDeleter)> devListGuard(
      devListResult.value(), devListDeleter);
  ibv_device** devList = devListGuard.get();
  if (devList == nullptr) {
    return nics;
  }

  for (int i = 0; i < numDevices; ++i) {
    ibv_device* dev = devList[i];
    if (dev == nullptr) {
      continue;
    }

    auto name = ibvApi.getDeviceName(dev);
    if (!name) {
      continue;
    }
    std::string devName = name.value();

    // Resolve PCI device path from ibdev sysfs path.
    std::string ibdevPath = dev->ibdev_path;

    bool isVirtual = isVirtualRdmaDevice(sysfs, ibdevPath);
    std::string pciDevicePath;

    if (!isVirtual) {
      auto resolved = sysfs.resolvePath(ibdevPath + "/device");
      if (!resolved) {
        continue;
      }
      pciDevicePath = std::move(resolved).value();
    }

    // Open device and query ports.
    auto ctxResult = ibvApi.openDevice(dev);
    if (!ctxResult) {
      continue;
    }
    ibv_context* ctx = ctxResult.value();

    // Collect active ports and RDMA port speed for this device.
    int activePort = -1;
    uint32_t portSpeedMbps = 0;
    ibv_device_attr devAttr{};
    if (ibvApi.queryDevice(ctx, &devAttr)) {
      for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
        ibv_port_attr portAttr{};
        if (ibvApi.queryPort(ctx, port, &portAttr) &&
            portAttr.state == IBV_PORT_ACTIVE) {
          auto speed =
              ibPortSpeedMbps(portAttr.active_speed, portAttr.active_width);
          if (activePort == -1 || speed > portSpeedMbps) {
            activePort = port;
            portSpeedMbps =
                ibPortSpeedMbps(portAttr.active_speed, portAttr.active_width);
          }
        }
      }
    }

    if (activePort != -1) {
      if (isVirtual) {
        if (portSpeedMbps == 0) {
          portSpeedMbps = kDefaultRxeSpeed;
        }
        // Virtual devices have no PCI topology. Use NUMA 0, empty ancestor
        // chain, and a default port speed if the ibverbs query returned 0.
        nics.push_back({
            .name = std::move(devName),
            .data =
                TopoNode::NicData{
                    .bdf = "virtual",
                    .numaNode = 0,
                    .port = activePort,
                    .portSpeedMbps = portSpeedMbps,
                    .netdevName = readNetdevName(sysfs, ibdevPath),
                },
            .sysfsPath = ibdevPath,
            .ancestorChain = {},
            .linkInfo = {},
        });
      } else {
        nics.push_back({
            .name = std::move(devName),
            .data =
                TopoNode::NicData{
                    .bdf = std::string(extractBdf(pciDevicePath)),
                    .numaNode = readNumaNode(sysfs, pciDevicePath),
                    .port = activePort,
                    .portSpeedMbps = portSpeedMbps,
                    .netdevName = readNetdevName(sysfs, ibdevPath),
                },
            .sysfsPath = pciDevicePath,
            .ancestorChain = buildAncestorChain(sysfs, pciDevicePath),
            .linkInfo = readPcieLinkInfo(sysfs, pciDevicePath),
        });
      }
    }

    ibvApi.closeDevice(ctx);
  }
  return nics;
}
#endif

} // namespace

NicDiscoveryData discoverNicHardware(
    [[maybe_unused]] IbvApi* ibvApi,
    SysfsApi& sysfs) {
  NicDiscoveryData data;
#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
  if (ibvApi != nullptr) {
    auto result = discoverNics(*ibvApi, sysfs);
    if (result) {
      data.nics = std::move(result).value();
    }
  }
#endif
  data.numaCount = discoverNumaNodeCount(sysfs);
  if (data.numaCount == 0) {
    data.numaCount = 1;
  }
  return data;
}

void addNicNodes(
    Topology& topo,
    std::span<const NicDiscovery> nics,
    int numaCount) {
  for (int i = 0; i < numaCount; ++i) {
    int nodeId = topo.addNode({
        .type = NodeType::CPU,
        .name = "cpu:" + std::to_string(i),
        .data = TopoNode::CpuData{.numaId = i},
    });
    topo.registerCpuNode(i, nodeId);
  }

  for (int i = 0; i < static_cast<int>(nics.size()); ++i) {
    int nodeId = topo.addNode({
        .type = NodeType::NIC,
        .name = nics[i].name,
        .data = nics[i].data,
    });
    topo.registerNicNode(i, nodeId);
  }
}

void appendNicPciDevices(
    std::vector<PciDevice>& devices,
    const Topology& topo,
    std::span<const NicDiscovery> nics) {
  for (int i = 0; i < static_cast<int>(nics.size()); ++i) {
    // Convert RDMA port speed from Mbps to MB/s for bandwidth comparison.
    uint32_t portMBps = nics[i].data.portSpeedMbps / 8;
    devices.push_back({
        .nodeId = topo.getNicNode(i).id,
        .numaNode = nics[i].data.numaNode,
        .ancestorChain = nics[i].ancestorChain,
        .portSpeedMBps = portMBps,
    });
  }
}

} // namespace topology

NicTopologyDiscovery::NicTopologyDiscovery(
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<SysfsApi> sysfsApi)
#if !defined(UNIFLOW_CMAKE_BUILD) || defined(UNIFLOW_ENABLE_RDMA_TRANSPORT)
    : ibvApi_(ibvApi ? std::move(ibvApi) : std::make_shared<IbvApi>()),
#else
    : ibvApi_(std::move(ibvApi)),
#endif
      sysfsApi_(sysfsApi ? std::move(sysfsApi) : std::make_shared<SysfsApi>()) {
}

Status NicTopologyDiscovery::discover(Topology& topo) {
  topo.clear();
  auto data = topology::discoverNicHardware(ibvApi_.get(), *sysfsApi_);
  topology::addNicNodes(topo, data.nics, data.numaCount);
  std::vector<topology::PciDevice> devices;
  topology::appendNicPciDevices(devices, topo, data.nics);
  topology::addPciCpuEdges(topo, *sysfsApi_, devices, data.numaCount);
  topology::addPciPeerEdges(topo, *sysfsApi_, devices);
  topology::addInterCpuEdges(topo, data.numaCount);
  topo.recomputePaths();
  topo.setStatus(Ok());
  return Ok();
}

Topology& sharedNicTopology() {
  static Topology topology = [] {
    Topology value;
    NicTopologyDiscovery().discover(value);
    return value;
  }();
  return topology;
}

} // namespace uniflow
