// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvDevice.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

namespace {

class RoceHca {
 public:
  RoceHca(std::string hcaStr, int defaultPort) {
    std::string s = std::move(hcaStr);
    std::string delim = ":";

    std::vector<std::string> hcaStrPair;
    folly::split(':', s, hcaStrPair);
    if (hcaStrPair.size() == 1) {
      this->name = s;
      this->port = defaultPort;
    } else if (hcaStrPair.size() == 2) {
      this->name = hcaStrPair.at(0);
      this->port = std::stoi(hcaStrPair.at(1));
    }
  }
  std::string name;
  int port{-1};
};

bool mlx5dvDmaBufDataDirectLinkCapable(
    ibv_device* device,
    ibv_context* context) {
  if (ibvSymbols.mlx5dv_internal_is_supported == nullptr ||
      ibvSymbols.mlx5dv_internal_reg_dmabuf_mr == nullptr ||
      ibvSymbols.mlx5dv_internal_get_data_direct_sysfs_path == nullptr) {
    return false;
  }

  if (!ibvSymbols.mlx5dv_internal_is_supported(device)) {
    return false;
  }
  int dev_fail = 0;
  ibv_pd* pd = nullptr;
  pd = ibvSymbols.ibv_internal_alloc_pd(context);
  if (!pd) {
    XLOG(ERR) << "ibv_alloc_pd failed: " << folly::errnoStr(errno);
    return false;
  }

  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)ibvSymbols.ibv_internal_reg_dmabuf_mr(
      pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
  // supported (EBADF otherwise)
  (void)ibvSymbols.mlx5dv_internal_reg_dmabuf_mr(
      pd,
      0ULL /*offset*/,
      0ULL /*len*/,
      0ULL /*iova*/,
      -1 /*fd*/,
      0 /*flags*/,
      0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
  // supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  if (ibvSymbols.ibv_internal_dealloc_pd(pd) != 0) {
    XLOGF(
        WARN,
        "ibv_dealloc_pd failed: {} DMA-BUF support status: {}",
        folly::errnoStr(errno),
        dev_fail);
    return false;
  }
  if (dev_fail) {
    XLOGF(
        INFO,
        "MLX5DV Kernel DMA-BUF is not supported on device {}",
        device->name);
    return false;
  }

  char dataDirectDevicePath[PATH_MAX];
  snprintf(dataDirectDevicePath, PATH_MAX, "/sys");
  return ibvSymbols.mlx5dv_internal_get_data_direct_sysfs_path(
             context, dataDirectDevicePath + 4, PATH_MAX - 4) == 0;
}

} // namespace

/*** IbvDevice ***/

// hcaList format examples:
// - Without port: "mlx5_0,mlx5_1,mlx5_2"
// - With port: "mlx5_0:1,mlx5_1:0,mlx5_2:1"
// - Prefix match: "mlx5"
// hcaPrefix: use "=" for exact match, "^" for exclude match, "" for prefix
// match. See guidelines:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca
folly::Expected<std::vector<IbvDevice>, Error> IbvDevice::ibvGetDeviceList(
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix,
    int defaultPort,
    int ibDataDirect) {
  // Get device list
  ibv_device** devs{nullptr};
  int numDevs;
  devs = ibvSymbols.ibv_internal_get_device_list(&numDevs);
  if (!devs) {
    return folly::makeUnexpected(Error(errno));
  }
  auto devices = ibvFilterDeviceList(
      numDevs, devs, hcaList, hcaPrefix, defaultPort, ibDataDirect);
  // Free device list
  ibvSymbols.ibv_internal_free_device_list(devs);
  return devices;
}

std::vector<IbvDevice> IbvDevice::ibvFilterDeviceList(
    int numDevs,
    ibv_device** devs,
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix,
    int defaultPort,
    int ibDataDirect) {
  std::vector<IbvDevice> devices;
  bool dataDirect = ibDataDirect == 1;

  if (hcaList.empty()) {
    devices.reserve(numDevs);
    for (int i = 0; i < numDevs; i++) {
      devices.emplace_back(devs[i], defaultPort, dataDirect);
    }
    return devices;
  }

  // Convert the provided list of HCA strings into a vector of RoceHca
  // objects, which enables efficient device filter operation
  std::vector<RoceHca> hcas;
  // Avoid copy triggered by resize
  hcas.reserve(hcaList.size());
  for (const auto& hca : hcaList) {
    // Copy value to each vector element so it can be freed automatically
    hcas.emplace_back(hca, defaultPort);
  }

  // Filter devices
  if (hcaPrefix == "=") {
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (hca.name == devs[i]->name) {
          devices.emplace_back(devs[i], hca.port, dataDirect);
          break;
        }
      }
    }
    return devices;
  } else if (hcaPrefix == "^") {
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (hca.name != devs[i]->name) {
          devices.emplace_back(devs[i], defaultPort, dataDirect);
          break;
        }
      }
    }
    return devices;
  } else {
    // Prefix match
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (strncmp(devs[i]->name, hca.name.c_str(), hca.name.length()) == 0) {
          devices.emplace_back(devs[i], hca.port, dataDirect);
          break;
        }
      }
    }
    return devices;
  }
}

IbvDevice::IbvDevice(ibv_device* ibvDevice, int port, bool dataDirect)
    : device_(ibvDevice), deviceId_(nextDeviceId_.fetch_add(1)) {
  port_ = port;
  context_ = ibvSymbols.ibv_internal_open_device(device_);
  if (!context_) {
    XLOGF(ERR, "Failed to open device {}", device_->name);
    throw std::runtime_error(
        fmt::format("Failed to open device {}", device_->name));
  }
  if (dataDirect && (mlx5dvDmaBufDataDirectLinkCapable(device_, context_))) {
    dataDirect_ = true;
    XLOGF(
        INFO,
        "NET/IB: Data Direct DMA Interface is detected for device: {} dataDirect: {}",
        device_->name,
        dataDirect_);
  }
}

IbvDevice::~IbvDevice() {
  if (context_) {
    int rc = ibvSymbols.ibv_internal_close_device(context_);
    if (rc != 0) {
      XLOGF(
          WARN,
          "Failed to close device rc: {}, {}. "
          "This is a post-failure warning likely due to an uncleaned RDMA resource on the failure path.",
          rc,
          strerror(errno));
    }
  }
}

IbvDevice::IbvDevice(IbvDevice&& other) noexcept {
  device_ = other.device_;
  context_ = other.context_;
  port_ = other.port_;
  dataDirect_ = other.dataDirect_;
  deviceId_ = other.deviceId_;

  other.device_ = nullptr;
  other.context_ = nullptr;
  other.deviceId_ = -1;
}

IbvDevice& IbvDevice::operator=(IbvDevice&& other) noexcept {
  device_ = other.device_;
  context_ = other.context_;
  port_ = other.port_;
  dataDirect_ = other.dataDirect_;
  deviceId_ = other.deviceId_;

  other.device_ = nullptr;
  other.context_ = nullptr;
  other.deviceId_ = -1;
  return *this;
}

ibv_device* IbvDevice::device() const {
  return device_;
}

ibv_context* IbvDevice::context() const {
  return context_;
}

int IbvDevice::port() const {
  return port_;
}

int32_t IbvDevice::getDeviceId() const {
  return deviceId_;
}

folly::Expected<IbvPd, Error> IbvDevice::allocPd() {
  ibv_pd* pd;
  pd = ibvSymbols.ibv_internal_alloc_pd(context_);
  if (!pd) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvPd(pd, deviceId_, dataDirect_);
}

folly::Expected<IbvPd, Error> IbvDevice::allocParentDomain(
    ibv_parent_domain_init_attr* attr) {
  ibv_pd* pd;

  if (ibvSymbols.ibv_internal_alloc_parent_domain == nullptr) {
    return folly::makeUnexpected(Error(ENOSYS));
  }

  pd = ibvSymbols.ibv_internal_alloc_parent_domain(context_, attr);

  if (!pd) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvPd(pd, deviceId_, dataDirect_);
}

folly::Expected<ibv_device_attr, Error> IbvDevice::queryDevice() const {
  ibv_device_attr deviceAttr{};
  int rc = ibvSymbols.ibv_internal_query_device(context_, &deviceAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return deviceAttr;
}

folly::Expected<ibv_port_attr, Error> IbvDevice::queryPort(
    uint8_t portNum) const {
  ibv_port_attr portAttr{};
  int rc = ibvSymbols.ibv_internal_query_port(context_, portNum, &portAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return portAttr;
}

folly::Expected<ibv_gid, Error> IbvDevice::queryGid(
    uint8_t portNum,
    int gidIndex) const {
  ibv_gid gid{};
  int rc = ibvSymbols.ibv_internal_query_gid(context_, portNum, gidIndex, &gid);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return gid;
}

folly::Expected<IbvCq, Error> IbvDevice::createCq(
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) const {
  ibv_cq* cq;
  cq = ibvSymbols.ibv_internal_create_cq(
      context_, cqe, cq_context, channel, comp_vector);
  if (!cq) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvCq(cq, deviceId_);
}

folly::Expected<IbvVirtualCq, Error> IbvDevice::createVirtualCq(
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) {
  auto maybeCq = createCq(cqe, cq_context, channel, comp_vector);
  if (maybeCq.hasError()) {
    return folly::makeUnexpected(maybeCq.error());
  }
  return IbvVirtualCq(std::move(*maybeCq), cqe);
}

folly::Expected<IbvCq, Error> IbvDevice::createCq(
    ibv_cq_init_attr_ex* attr) const {
  ibv_cq_ex* cqEx;
  cqEx = ibvSymbols.ibv_internal_create_cq_ex(context_, attr);
  if (!cqEx) {
    return folly::makeUnexpected(Error(errno));
  }
  ibv_cq* cq = ibv_cq_ex_to_cq(cqEx);
  return IbvCq(cq, deviceId_);
}

folly::Expected<ibv_comp_channel*, Error> IbvDevice::createCompChannel() const {
  ibv_comp_channel* channel;
  channel = ibvSymbols.ibv_internal_create_comp_channel(context_);
  if (!channel) {
    return folly::makeUnexpected(Error(errno));
  }
  return channel;
}

folly::Expected<folly::Unit, Error> IbvDevice::destroyCompChannel(
    ibv_comp_channel* channel) const {
  int rc = ibvSymbols.ibv_internal_destroy_comp_channel(channel);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<bool, Error> IbvDevice::isPortActive(
    uint8_t portNum,
    std::unordered_set<int> linkLayers) const {
  auto maybePortAttr = queryPort(portNum);
  if (maybePortAttr.hasError()) {
    return folly::makeUnexpected(maybePortAttr.error());
  }

  auto portAttr = maybePortAttr.value();

  // Check if port is active
  if (portAttr.state != IBV_PORT_ACTIVE) {
    return false;
  }

  // Check if link layer matches (if specified)
  if (!linkLayers.empty() &&
      linkLayers.find(portAttr.link_layer) == linkLayers.end()) {
    return false;
  }

  return true;
}

folly::Expected<uint8_t, Error> IbvDevice::findActivePort(
    std::unordered_set<int> const& linkLayers) const {
  // If specific port requested, check if it is active
  if (port_ != kIbAnyPort) {
    auto maybeActive = isPortActive(port_, linkLayers);
    if (maybeActive.hasError()) {
      return folly::makeUnexpected(maybeActive.error());
    }

    if (!maybeActive.value()) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Port {} is not active on device {}", port_, device_->name)));
    }
    return port_;
  }

  // No specific port requested, find any active port
  auto maybeDeviceAttr = queryDevice();
  if (maybeDeviceAttr.hasError()) {
    return folly::makeUnexpected(maybeDeviceAttr.error());
  }

  for (uint8_t port = 1; port <= maybeDeviceAttr->phys_port_cnt; port++) {
    auto maybeActive = isPortActive(port, linkLayers);
    if (maybeActive.hasError()) {
      continue; // Skip ports we can't query
    }

    if (maybeActive.value()) {
      return port;
    }
  }

  return folly::makeUnexpected(Error(
      ENODEV, fmt::format("No active port found on device {}", device_->name)));
}

} // namespace ibverbx
