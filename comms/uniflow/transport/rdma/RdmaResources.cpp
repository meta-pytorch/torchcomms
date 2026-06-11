// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/rdma/RdmaResources.h"

namespace uniflow {

// ---------------------------------------------------------------------------
// NicResources
// ---------------------------------------------------------------------------

NicResources::NicResources(
    ibv_device* device,
    std::shared_ptr<IbvApi> api,
    int numaNode,
    uint8_t gidIndex,
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

} // namespace uniflow
