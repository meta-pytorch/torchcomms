// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvDevice.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/utils/CtranLogger.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>

#include <fmt/core.h>

namespace ibverbx {

extern IbvSymbols ibvSymbols;

namespace {

class RoceHca {
 public:
  RoceHca(std::string hcaStr, int defaultPort) {
    std::string s = std::move(hcaStr);

    std::vector<std::string> hcaStrPair;
    // Stands in for folly::split(':', s, hcaStrPair), which keeps empty tokens:
    // "mlx5_0:" yields two, and "" yields one.
    for (size_t pos = 0;;) {
      const size_t delim = s.find(':', pos);
      if (delim == std::string::npos) {
        hcaStrPair.push_back(s.substr(pos));
        break;
      }
      hcaStrPair.push_back(s.substr(pos, delim - pos));
      pos = delim + 1;
    }
    if (hcaStrPair.size() == 1) {
      this->name = hcaStrPair.at(0);
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
    CTRAN_LOG(ERR, "ibv_alloc_pd failed: {}", errnoStr(errno));
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
    CTRAN_LOG(
        WARN,
        "ibv_dealloc_pd failed: {} DMA-BUF support status: {}",
        errnoStr(errno),
        dev_fail);
    return false;
  }
  if (dev_fail) {
    CTRAN_LOG(
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
Expected<std::vector<IbvDevice>> IbvDevice::ibvGetDeviceList(
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix,
    int defaultPort,
    int ibDataDirect) {
  // Get device list
  ibv_device** devs{nullptr};
  int numDevs;
  devs = ibvSymbols.ibv_internal_get_device_list(&numDevs);
  if (!devs) {
    return makeUnexpected(Error(errno));
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
    CTRAN_LOG(ERR, "Failed to open device {}", device_->name);
    throw std::runtime_error(
        fmt::format("Failed to open device {}", device_->name));
  }
  if (dataDirect && (mlx5dvDmaBufDataDirectLinkCapable(device_, context_))) {
    dataDirect_ = true;
    CTRAN_LOG(
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
      CTRAN_LOG(
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

Expected<IbvPd> IbvDevice::allocPd() {
  ibv_pd* pd;
  pd = ibvSymbols.ibv_internal_alloc_pd(context_);
  if (!pd) {
    return makeUnexpected(Error(errno));
  }
  return IbvPd(pd, deviceId_, dataDirect_);
}

Expected<IbvPd> IbvDevice::allocParentDomain(
    ibv_parent_domain_init_attr* attr) {
  ibv_pd* pd;

  if (ibvSymbols.ibv_internal_alloc_parent_domain == nullptr) {
    return makeUnexpected(Error(ENOSYS));
  }

  pd = ibvSymbols.ibv_internal_alloc_parent_domain(context_, attr);

  if (!pd) {
    return makeUnexpected(Error(errno));
  }
  return IbvPd(pd, deviceId_, dataDirect_);
}

Expected<ibv_device_attr> IbvDevice::queryDevice() const {
  ibv_device_attr deviceAttr{};
  int rc = ibvSymbols.ibv_internal_query_device(context_, &deviceAttr);
  if (rc != 0) {
    return makeUnexpected(Error(rc));
  }
  return deviceAttr;
}

Expected<ibv_port_attr> IbvDevice::queryPort(uint8_t portNum) const {
  ibv_port_attr portAttr{};
  int rc = ibvSymbols.ibv_internal_query_port(context_, portNum, &portAttr);
  if (rc != 0) {
    return makeUnexpected(Error(rc));
  }
  return portAttr;
}

Expected<ibv_gid> IbvDevice::queryGid(uint8_t portNum, int gidIndex) const {
  ibv_gid gid{};
  int rc = ibvSymbols.ibv_internal_query_gid(context_, portNum, gidIndex, &gid);
  if (rc != 0) {
    return makeUnexpected(Error(rc));
  }
  return gid;
}

Expected<int> detail::selectRoceGidIndex(
    int gidTableLength,
    int requestedIndex,
    const std::function<Expected<ibv_gid>(int)>& queryGid,
    const std::function<Expected<std::string>(int)>& queryGidType) {
  if (requestedIndex < -1 || requestedIndex >= gidTableLength) {
    return makeUnexpected(Error(
        EINVAL,
        fmt::format(
            "Invalid RoCE GID index {} (GID table length {})",
            requestedIndex,
            gidTableLength)));
  }
  if (requestedIndex >= 0) {
    auto gid = queryGid(requestedIndex);
    if (gid.hasError()) {
      return makeUnexpected(gid.error());
    }
    return requestedIndex;
  }

  int v2Index = -1;
  int untypedIndex = -1;
  for (int index = 0; index < gidTableLength; ++index) {
    auto gid = queryGid(index);
    if (gid.hasError() ||
        std::all_of(
            gid->raw,
            gid->raw + sizeof(gid->raw),
            [](uint8_t byte) { return byte == 0; }) ||
        (gid->raw[0] == 0xfe && (gid->raw[1] & 0xc0) == 0x80)) {
      continue;
    }

    auto type = queryGidType(index);
    if (type.hasError()) {
      if (type.error().errNum == EINVAL) {
        // Container sysfs may expose a GID but not its type at this index.
        continue;
      }
      return makeUnexpected(type.error());
    }
    const bool ipv4Mapped =
        std::all_of(
            gid->raw, gid->raw + 10, [](uint8_t byte) { return byte == 0; }) &&
        gid->raw[10] == 0xff && gid->raw[11] == 0xff;
    if (type->find("RoCE v2") != std::string::npos) {
      if (ipv4Mapped) {
        return index;
      }
      if (v2Index == -1) {
        v2Index = index;
      }
    } else if (type->empty() && ipv4Mapped) {
      // The kernel lists the RoCE v1 entry before the v2 entry for each IP.
      // If type files are unavailable, retain the last matching entry to
      // avoid choosing the earlier, non-routable v1 GID.
      untypedIndex = index;
    }
  }
  if (v2Index >= 0) {
    return v2Index;
  }
  if (untypedIndex >= 0) {
    CTRAN_LOG(
        WARN,
        "IBVERBX: GID types unavailable; selecting index {} from configured IPv4 GIDs. Specify an explicit index if this is not RoCE v2.",
        untypedIndex);
    return untypedIndex;
  }
  return makeUnexpected(
      Error(ENOENT, "No usable RoCE v2 GID; specify an explicit GID index"));
}

Expected<int> IbvDevice::resolveRoceGidIndex(
    uint8_t portNum,
    const ibv_port_attr& portAttr,
    int requestedIndex) const {
  if (portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
    return makeUnexpected(Error(EINVAL, "RoCE GID requires an Ethernet port"));
  }
  auto result = detail::selectRoceGidIndex(
      portAttr.gid_tbl_len,
      requestedIndex,
      [this, portNum](int index) { return queryGid(portNum, index); },
      [this, portNum](int index) -> Expected<std::string> {
        const std::string path = fmt::format(
            "/sys/class/infiniband/{}/ports/{}/gid_attrs/types/{}",
            device_->name,
            portNum,
            index);
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
          const int openError = errno;
          if (openError == ENOENT) {
            // Some containers expose GIDs but do not mount their type files.
            return std::string{};
          }
          return makeUnexpected(Error(
              openError != 0 ? openError : EIO,
              fmt::format("Cannot read GID type from {}", path)));
        }
        std::array<char, 32> type{};
        const ssize_t bytesRead = ::read(fd, type.data(), type.size());
        const int readError = errno;
        ::close(fd);
        if (bytesRead < 0) {
          return makeUnexpected(Error(
              readError, fmt::format("Cannot read GID type from {}", path)));
        }
        return std::string(type.data(), bytesRead);
      });
  if (result.hasError()) {
    return makeUnexpected(Error(
        result.error().errNum,
        fmt::format(
            "Cannot resolve RoCE GID index {} on {} port {}: {}",
            requestedIndex,
            device_->name,
            portNum,
            result.error().errStr)));
  }
  return result;
}

Expected<IbvCq> IbvDevice::createCq(
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) const {
  ibv_cq* cq;
  cq = ibvSymbols.ibv_internal_create_cq(
      context_, cqe, cq_context, channel, comp_vector);
  if (!cq) {
    return makeUnexpected(Error(errno));
  }
  return IbvCq(cq, deviceId_);
}

Expected<IbvVirtualCq> IbvDevice::createVirtualCq(
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) {
  auto maybeCq = createCq(cqe, cq_context, channel, comp_vector);
  if (maybeCq.hasError()) {
    return makeUnexpected(maybeCq.error());
  }
  return IbvVirtualCq(std::move(*maybeCq), cqe);
}

Expected<IbvCq> IbvDevice::createCq(ibv_cq_init_attr_ex* attr) const {
  ibv_cq_ex* cqEx;
  cqEx = ibvSymbols.ibv_internal_create_cq_ex(context_, attr);
  if (!cqEx) {
    return makeUnexpected(Error(errno));
  }
  ibv_cq* cq = ibv_cq_ex_to_cq(cqEx);
  return IbvCq(cq, deviceId_);
}

Expected<ibv_comp_channel*> IbvDevice::createCompChannel() const {
  ibv_comp_channel* channel;
  channel = ibvSymbols.ibv_internal_create_comp_channel(context_);
  if (!channel) {
    return makeUnexpected(Error(errno));
  }
  return channel;
}

Status IbvDevice::destroyCompChannel(ibv_comp_channel* channel) const {
  int rc = ibvSymbols.ibv_internal_destroy_comp_channel(channel);
  if (rc != 0) {
    return makeUnexpected(Error(rc));
  }
  return ok();
}

Expected<bool> IbvDevice::isPortActive(
    uint8_t portNum,
    std::unordered_set<int> linkLayers) const {
  auto maybePortAttr = queryPort(portNum);
  if (maybePortAttr.hasError()) {
    return makeUnexpected(maybePortAttr.error());
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

Expected<uint8_t> IbvDevice::findActivePort(
    std::unordered_set<int> const& linkLayers) const {
  // If specific port requested, check if it is active
  if (port_ != kIbAnyPort) {
    auto maybeActive = isPortActive(port_, linkLayers);
    if (maybeActive.hasError()) {
      return makeUnexpected(maybeActive.error());
    }

    if (!maybeActive.value()) {
      return makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Port {} is not active on device {}", port_, device_->name)));
    }
    return port_;
  }

  // No specific port requested, find any active port
  auto maybeDeviceAttr = queryDevice();
  if (maybeDeviceAttr.hasError()) {
    return makeUnexpected(maybeDeviceAttr.error());
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

  return makeUnexpected(Error(
      ENODEV, fmt::format("No active port found on device {}", device_->name)));
}

} // namespace ibverbx
