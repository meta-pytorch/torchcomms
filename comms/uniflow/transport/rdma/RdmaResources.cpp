// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaResources.h"

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

namespace {

constexpr uint8_t kDefaultRoceV2GidIndex = 3;

/// True if @p gid is an IPv4-mapped IPv6 address (::ffff:a.b.c.d).
bool isIpv4MappedGid(const ibv_gid& gid) {
  for (int i = 0; i < 10; ++i) {
    if (gid.raw[i] != 0) {
      return false;
    }
  }
  return gid.raw[10] == 0xff && gid.raw[11] == 0xff;
}

/// True if @p gid is a link-local address that cannot be routed across NICs or
/// hosts: IPv6 link-local (fe80::/10) or IPv4-mapped IPv4 link-local
/// (169.254.0.0/16). Such GIDs are unusable for cross-NIC/cross-host RoCE.
bool isLinkLocalGid(const ibv_gid& gid) {
  if (gid.raw[0] == 0xfe && (gid.raw[1] & 0xc0) == 0x80) {
    return true;
  }
  return isIpv4MappedGid(gid) && gid.raw[12] == 169 && gid.raw[13] == 254;
}

} // namespace

// ---------------------------------------------------------------------------
// NicResources
// ---------------------------------------------------------------------------

NicResources::NicResources(
    ibv_device* device,
    std::shared_ptr<IbvApi> api,
    int numaNode,
    int configuredGidIndex,
    std::optional<uint8_t> port)
    : numaNode(numaNode), ibvApi(std::move(api)) {
  try {
    if (!ibvApi) {
      ibvApi = std::make_shared<IbvApi>();
    }

    auto ctxResult = ibvApi->openDevice(device);
    if (ctxResult.hasError()) {
      throw std::runtime_error(
          "NicResources: failed to open device" + ctxResult.error().message());
    }
    ctx = ctxResult.value();

    portNum = port.value_or(0);
    if (portNum == 0) {
      portNum = findActivePort();
      if (portNum == 0) {
        throw std::runtime_error("NicResources: no active port found");
      }
    }

    auto pdResult = ibvApi->allocPd(ctx);
    if (pdResult.hasError()) {
      throw std::runtime_error("NicResources: failed to allocate PD");
    }
    pd = pdResult.value();

    auto dmaBufResult = ibvApi->isDmaBufSupported(pd);
    if (dmaBufResult.hasError()) {
      throw std::runtime_error("NicResources: failed to probe DMA-BUF support");
    }
    dmaBufSupported = dmaBufResult.value();

    ibv_port_attr portAttr{};
    auto portStatus = ibvApi->queryPort(ctx, portNum, &portAttr);
    if (portStatus.hasError()) {
      throw std::runtime_error("NicResources: failed to query port");
    }
    lid = portAttr.lid;
    mtu = portAttr.active_mtu;
    linkLayer = portAttr.link_layer;

    gidIndex = resolveGidIndex(configuredGidIndex, portAttr);

    auto gidStatus = ibvApi->queryGid(ctx, portNum, gidIndex, &gid);
    if (gidStatus.hasError()) {
      throw std::runtime_error("NicResources: failed to query GID");
    }
  } catch (...) {
    cleanup();
    throw;
  }
}

NicResources::NicResources(NicResources&& other) noexcept
    : ctx(other.ctx),
      pd(other.pd),
      lid(other.lid),
      gid(other.gid),
      gidIndex(other.gidIndex),
      mtu(other.mtu),
      linkLayer(other.linkLayer),
      portNum(other.portNum),
      dmaBufSupported(other.dmaBufSupported),
      numaNode(other.numaNode),
      ibvApi(std::move(other.ibvApi)) {
  other.ctx = nullptr;
  other.pd = nullptr;
}

void NicResources::cleanup() {
  if (ibvApi) {
    if (pd) {
      ibvApi->deallocPd(pd);
      pd = nullptr;
    }
    if (ctx) {
      ibvApi->closeDevice(ctx);
      ctx = nullptr;
    }
  }
}

NicResources::~NicResources() {
  cleanup();
}

uint8_t NicResources::findActivePort() const {
  ibv_device_attr devAttr{};
  auto status = ibvApi->queryDevice(ctx, &devAttr);
  if (status.hasError()) {
    return 0;
  }

  for (uint8_t p = 1; p <= devAttr.phys_port_cnt; ++p) {
    ibv_port_attr portAttr{};
    auto portStatus = ibvApi->queryPort(ctx, p, &portAttr);
    if (portStatus.hasError()) {
      continue;
    }
    if (portAttr.state == IBV_PORT_ACTIVE) {
      return p;
    }
  }
  return 0;
}

uint8_t NicResources::resolveGidIndex(
    int configuredGidIndex,
    const ibv_port_attr& portAttr) const {
  if (configuredGidIndex >= 0) {
    if (configuredGidIndex >= portAttr.gid_tbl_len) {
      throw std::runtime_error(
          "NicResources: configured GID index " +
          std::to_string(configuredGidIndex) +
          " out of range (gid_tbl_len=" + std::to_string(portAttr.gid_tbl_len) +
          "); check configured gidIndex");
    }
    return static_cast<uint8_t>(configuredGidIndex);
  }

  // The GID index only matters for RoCE (Ethernet) addressing; IB ports use
  // LID-based address vectors. Auto-selection also requires ibv_query_gid_ex,
  // which older rdma-core may not export.
  if (linkLayer != IBV_LINK_LAYER_ETHERNET ||
      !ibvApi->isQueryGidExSupported()) {
    if (portAttr.gid_tbl_len > 0 &&
        kDefaultRoceV2GidIndex >= portAttr.gid_tbl_len) {
      throw std::runtime_error(
          "NicResources: default GID index " +
          std::to_string(kDefaultRoceV2GidIndex) +
          " out of range (gid_tbl_len=" + std::to_string(portAttr.gid_tbl_len) +
          "); set gidIndex explicitly");
    }
    return kDefaultRoceV2GidIndex;
  }
  int firstGlobalRoceV2 = -1;
  for (int i = 0; i < portAttr.gid_tbl_len; ++i) {
    ibv_gid_entry entry{};
    auto status =
        ibvApi->queryGidEx(ctx, portNum, static_cast<uint32_t>(i), &entry, 0);
    if (status.hasError() || entry.gid_type != IBV_GID_TYPE_ROCE_V2) {
      continue;
    }
    // Link-local GIDs are not routable across NICs/hosts; skip them so that
    // cross-NIC and cross-host RoCE connections use a globally routable GID.
    if (isLinkLocalGid(entry.gid)) {
      continue;
    }
    // Prefer an IPv4-mapped RoCEv2 entry; otherwise remember the first global.
    if (isIpv4MappedGid(entry.gid)) {
      UNIFLOW_LOG_INFO(
          "Selected IPv4-mapped RoCEv2 GID index {} on port {}", i, portNum);
      return static_cast<uint8_t>(i);
    }
    if (firstGlobalRoceV2 < 0) {
      firstGlobalRoceV2 = i;
    }
  }

  if (firstGlobalRoceV2 >= 0) {
    if (firstGlobalRoceV2 > 255) {
      UNIFLOW_LOG_WARN(
          "GID index {} exceeds uint8_t range on port {}; using default",
          firstGlobalRoceV2,
          portNum);
    } else {
      UNIFLOW_LOG_INFO(
          "Selected RoCEv2 GID index {} on port {}",
          firstGlobalRoceV2,
          portNum);
      return static_cast<uint8_t>(firstGlobalRoceV2);
    }
  }

  if (kDefaultRoceV2GidIndex < portAttr.gid_tbl_len) {
    UNIFLOW_LOG_WARN(
        "No global RoCEv2 GID found on port {}; falling back to GID index {}",
        portNum,
        kDefaultRoceV2GidIndex);
    return kDefaultRoceV2GidIndex;
  }

  throw std::runtime_error(
      "NicResources: no global RoCEv2 GID found on port " +
      std::to_string(portNum) + " and default GID index " +
      std::to_string(kDefaultRoceV2GidIndex) + " out of range (gid_tbl_len=" +
      std::to_string(portAttr.gid_tbl_len) + ")");
}

} // namespace uniflow
