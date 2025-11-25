// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

#ifdef IBVERBX_BUILD_RDMA_CORE
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#endif

#include <dlfcn.h>
#include <folly/ScopeGuard.h>
#include <folly/Singleton.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <folly/synchronization/CallOnce.h>
#include <stdexcept>
#include "comms/utils/cvars/nccl_cvars.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

namespace {

folly::once_flag initIbvSymbolOnce;

folly::Singleton<Coordinator> coordinatorSingleton{};

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

folly::Expected<folly::Unit, Error> ibvInit() {
  static std::atomic<int> errNum{1};
  folly::call_once(initIbvSymbolOnce, [&]() {
    errNum = buildIbvSymbols(ibvSymbols, NCCL_IBVERBS_PATH);
  });
  if (errNum != 0) {
    return folly::makeUnexpected(Error(errNum));
  }
  return folly::unit;
}

folly::Expected<folly::Unit, Error>
ibvGetCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context) {
  int rc = ibvSymbols.ibv_internal_get_cq_event(channel, cq, cq_context);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

void ibvAckCqEvents(ibv_cq* cq, unsigned int nevents) {
  ibvSymbols.ibv_internal_ack_cq_events(cq, nevents);
}

/*** IbvMr ***/

IbvMr::IbvMr(ibv_mr* mr) : mr_(mr) {}

IbvMr::IbvMr(IbvMr&& other) noexcept {
  mr_ = other.mr_;
  other.mr_ = nullptr;
}

IbvMr& IbvMr::operator=(IbvMr&& other) noexcept {
  mr_ = other.mr_;
  other.mr_ = nullptr;
  return *this;
}

IbvMr::~IbvMr() {
  if (mr_) {
    int rc = ibvSymbols.ibv_internal_dereg_mr(mr_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to deregister mr rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_mr* IbvMr::mr() const {
  return mr_;
}

/*** IbvPd ***/

IbvPd::IbvPd(ibv_pd* pd, bool dataDirect) : pd_(pd), dataDirect_(dataDirect) {}

IbvPd::IbvPd(IbvPd&& other) noexcept {
  pd_ = other.pd_;
  dataDirect_ = other.dataDirect_;
  other.pd_ = nullptr;
}

IbvPd& IbvPd::operator=(IbvPd&& other) noexcept {
  pd_ = other.pd_;
  dataDirect_ = other.dataDirect_;
  other.pd_ = nullptr;
  return *this;
}

IbvPd::~IbvPd() {
  if (pd_) {
    int rc = ibvSymbols.ibv_internal_dealloc_pd(pd_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to deallocate pd rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_pd* IbvPd::pd() const {
  return pd_;
}

bool IbvPd::useDataDirect() const {
  return dataDirect_;
}

folly::Expected<IbvMr, Error>
IbvPd::regMr(void* addr, size_t length, ibv_access_flags access) const {
  ibv_mr* mr;
  mr = ibvSymbols.ibv_internal_reg_mr(pd_, addr, length, access);
  if (!mr) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvMr(mr);
}

folly::Expected<IbvMr, Error> IbvPd::regDmabufMr(
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    ibv_access_flags access) const {
  ibv_mr* mr;
  if (dataDirect_) {
    mr = ibvSymbols.mlx5dv_internal_reg_dmabuf_mr(
        pd_,
        offset,
        length,
        iova,
        fd,
        access,
        MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  } else {
    mr = ibvSymbols.ibv_internal_reg_dmabuf_mr(
        pd_, offset, length, iova, fd, access);
  }
  if (!mr) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvMr(mr);
}

folly::Expected<IbvQp, Error> IbvPd::createQp(
    ibv_qp_init_attr* initAttr) const {
  ibv_qp* qp;
  qp = ibvSymbols.ibv_internal_create_qp(pd_, initAttr);
  if (!qp) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvQp(qp);
}

folly::Expected<IbvVirtualQp, Error> IbvPd::createVirtualQp(
    int totalQps,
    ibv_qp_init_attr* initAttr,
    IbvVirtualCq* sendCq,
    IbvVirtualCq* recvCq,
    int maxMsgCntPerQp,
    int maxMsgSize,
    LoadBalancingScheme loadBalancingScheme) const {
  std::vector<IbvQp> qps;
  qps.reserve(totalQps);

  if (sendCq == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "Empty sendCq being provided to createVirtualQp"));
  }

  if (recvCq == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "Empty recvCq being provided to createVirtualQp"));
  }

  // Overwrite the CQs in the initAttr to point to the virtual CQ
  initAttr->send_cq = sendCq->getPhysicalCqRef().cq();
  initAttr->recv_cq = recvCq->getPhysicalCqRef().cq();

  // First create all the data QPs
  for (int i = 0; i < totalQps; i++) {
    auto maybeQp = createQp(initAttr);
    if (maybeQp.hasError()) {
      return folly::makeUnexpected(maybeQp.error());
    }
    qps.emplace_back(std::move(*maybeQp));
  }

  // Create notify QP
  auto maybeNotifyQp = createQp(initAttr);
  if (maybeNotifyQp.hasError()) {
    return folly::makeUnexpected(maybeNotifyQp.error());
  }

  // Create the IbvVirtualQp instance, with coordinator registartion happens
  // within IbvVirtualQp constructor
  return IbvVirtualQp(
      std::move(qps),
      std::move(*maybeNotifyQp),
      sendCq,
      recvCq,
      maxMsgCntPerQp,
      maxMsgSize,
      loadBalancingScheme);
}

/*** IbvCq ***/

IbvCq::IbvCq(ibv_cq* cq) : cq_(cq) {}

IbvCq::~IbvCq() {
  if (cq_) {
    int rc = ibvSymbols.ibv_internal_destroy_cq(cq_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy cq rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvCq::IbvCq(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  other.cq_ = nullptr;
}

IbvCq& IbvCq::operator=(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  other.cq_ = nullptr;
  return *this;
}

ibv_cq* IbvCq::cq() const {
  return cq_;
}

folly::Expected<folly::Unit, Error> IbvCq::reqNotifyCq(
    int solicited_only) const {
  int rc = cq_->context->ops.req_notify_cq(cq_, solicited_only);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

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
    int defaultPort) {
  // Get device list
  ibv_device** devs{nullptr};
  int numDevs;
  devs = ibvSymbols.ibv_internal_get_device_list(&numDevs);
  if (!devs) {
    return folly::makeUnexpected(Error(errno));
  }
  auto devices =
      ibvFilterDeviceList(numDevs, devs, hcaList, hcaPrefix, defaultPort);
  // Free device list
  ibvSymbols.ibv_internal_free_device_list(devs);
  return devices;
}

std::vector<IbvDevice> IbvDevice::ibvFilterDeviceList(
    int numDevs,
    ibv_device** devs,
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix,
    int defaultPort) {
  std::vector<IbvDevice> devices;

  if (hcaList.empty()) {
    devices.reserve(numDevs);
    for (int i = 0; i < numDevs; i++) {
      devices.emplace_back(devs[i], defaultPort);
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
          devices.emplace_back(devs[i], hca.port);
          break;
        }
      }
    }
    return devices;
  } else if (hcaPrefix == "^") {
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (hca.name != devs[i]->name) {
          devices.emplace_back(devs[i], defaultPort);
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
          devices.emplace_back(devs[i], hca.port);
          break;
        }
      }
    }
    return devices;
  }
}

IbvDevice::IbvDevice(ibv_device* ibvDevice, int port) : device_(ibvDevice) {
  port_ = port;
  context_ = ibvSymbols.ibv_internal_open_device(device_);
  if (!context_) {
    XLOGF(ERR, "Failed to open device {}", device_->name);
    throw std::runtime_error(
        fmt::format("Failed to open device {}", device_->name));
  }
  if ((mlx5dvDmaBufDataDirectLinkCapable(device_, context_))) {
    // Now check whether Data Direct has been disabled by the user
    dataDirect_ = NCCL_IB_DATA_DIRECT == 1;
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
      XLOGF(ERR, "Failed to close device rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvDevice::IbvDevice(IbvDevice&& other) noexcept {
  device_ = other.device_;
  context_ = other.context_;
  port_ = other.port_;
  dataDirect_ = other.dataDirect_;

  other.device_ = nullptr;
  other.context_ = nullptr;
}

IbvDevice& IbvDevice::operator=(IbvDevice&& other) noexcept {
  device_ = other.device_;
  context_ = other.context_;
  port_ = other.port_;
  dataDirect_ = other.dataDirect_;

  other.device_ = nullptr;
  other.context_ = nullptr;
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

folly::Expected<IbvPd, Error> IbvDevice::allocPd() {
  ibv_pd* pd;
  pd = ibvSymbols.ibv_internal_alloc_pd(context_);
  if (!pd) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvPd(pd, dataDirect_);
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
  return IbvPd(pd, dataDirect_);
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
  return IbvCq(cq);
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
  return IbvCq(cq);
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

/*** IbvQp ***/
IbvQp::IbvQp(ibv_qp* qp) : qp_(qp) {}

IbvQp::~IbvQp() {
  if (qp_) {
    int rc = ibvSymbols.ibv_internal_destroy_qp(qp_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy qp rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvQp::IbvQp(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  other.qp_ = nullptr;
}

IbvQp& IbvQp::operator=(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  other.qp_ = nullptr;
  return *this;
}

ibv_qp* IbvQp::qp() const {
  return qp_;
}

folly::Expected<folly::Unit, Error> IbvQp::modifyQp(
    ibv_qp_attr* attr,
    int attrMask) {
  int rc = ibvSymbols.ibv_internal_modify_qp(qp_, attr, attrMask);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<std::pair<ibv_qp_attr, ibv_qp_init_attr>, Error> IbvQp::queryQp(
    int attrMask) const {
  ibv_qp_attr qpAttr{};
  ibv_qp_init_attr initAttr{};
  int rc = ibvSymbols.ibv_internal_query_qp(qp_, &qpAttr, attrMask, &initAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return std::make_pair(qpAttr, initAttr);
}

void IbvQp::enquePhysicalSendWrStatus(int physicalWrId, int virtualWrId) {
  physicalSendWrStatus_.emplace_back(physicalWrId, virtualWrId);
}

void IbvQp::dequePhysicalSendWrStatus() {
  physicalSendWrStatus_.pop_front();
}

void IbvQp::dequePhysicalRecvWrStatus() {
  physicalRecvWrStatus_.pop_front();
}

void IbvQp::enquePhysicalRecvWrStatus(int physicalWrId, int virtualWrId) {
  physicalRecvWrStatus_.emplace_back(physicalWrId, virtualWrId);
}

bool IbvQp::isSendQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalSendWrStatus_.size() < maxMsgCntPerQp;
}

bool IbvQp::isRecvQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalRecvWrStatus_.size() < maxMsgCntPerQp;
}

/*** IbvVirtualQp ***/

IbvVirtualQp::IbvVirtualQp(
    std::vector<IbvQp>&& qps,
    IbvQp&& notifyQp,
    IbvVirtualCq* sendCq,
    IbvVirtualCq* recvCq,
    int maxMsgCntPerQp,
    int maxMsgSize,
    LoadBalancingScheme loadBalancingScheme)
    : physicalQps_(std::move(qps)),
      maxMsgCntPerQp_(maxMsgCntPerQp),
      maxMsgSize_(maxMsgSize),
      loadBalancingScheme_(loadBalancingScheme),
      notifyQp_(std::move(notifyQp)) {
  virtualQpNum_ =
      nextVirtualQpNum_.fetch_add(1); // Assign unique virtual QP number

  for (int i = 0; i < physicalQps_.size(); i++) {
    qpNumToIdx_[physicalQps_.at(i).qp()->qp_num] = i;
  }

  // Register the virtual QP and all its mappings with the coordinator
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualQp construction!";

  // Use the consolidated registration API
  coordinator->registerVirtualQpWithVirtualCqMappings(
      this, sendCq->getVirtualCqNum(), recvCq->getVirtualCqNum());
}

size_t IbvVirtualQp::getTotalQps() const {
  return physicalQps_.size();
}

const std::vector<IbvQp>& IbvVirtualQp::getQpsRef() const {
  return physicalQps_;
}

std::vector<IbvQp>& IbvVirtualQp::getQpsRef() {
  return physicalQps_;
}

const IbvQp& IbvVirtualQp::getNotifyQpRef() const {
  return notifyQp_;
}

uint32_t IbvVirtualQp::getVirtualQpNum() const {
  return virtualQpNum_;
}

IbvVirtualQp::IbvVirtualQp(IbvVirtualQp&& other) noexcept
    : pendingSendVirtualWrQue_(std::move(other.pendingSendVirtualWrQue_)),
      pendingRecvVirtualWrQue_(std::move(other.pendingRecvVirtualWrQue_)),
      virtualQpNum_(std::move(other.virtualQpNum_)),
      physicalQps_(std::move(other.physicalQps_)),
      qpNumToIdx_(std::move(other.qpNumToIdx_)),
      nextSendPhysicalQpIdx_(std::move(other.nextSendPhysicalQpIdx_)),
      nextRecvPhysicalQpIdx_(std::move(other.nextRecvPhysicalQpIdx_)),
      maxMsgCntPerQp_(std::move(other.maxMsgCntPerQp_)),
      maxMsgSize_(std::move(other.maxMsgSize_)),
      nextPhysicalWrId_(std::move(other.nextPhysicalWrId_)),
      loadBalancingScheme_(std::move(other.loadBalancingScheme_)),
      pendingSendNotifyVirtualWrQue_(
          std::move(other.pendingSendNotifyVirtualWrQue_)),
      notifyQp_(std::move(other.notifyQp_)),
      dqplbSeqTracker(std::move(other.dqplbSeqTracker)),
      dqplbReceiverInitialized_(std::move(other.dqplbReceiverInitialized_)) {
  // Update coordinator pointer mapping for this virtual QP after move
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualQp move construction!";
  coordinator->updateVirtualQpPointer(virtualQpNum_, this);
}

IbvVirtualQp& IbvVirtualQp::operator=(IbvVirtualQp&& other) noexcept {
  if (this != &other) {
    physicalQps_ = std::move(other.physicalQps_);
    notifyQp_ = std::move(other.notifyQp_);
    nextSendPhysicalQpIdx_ = std::move(other.nextSendPhysicalQpIdx_);
    nextRecvPhysicalQpIdx_ = std::move(other.nextRecvPhysicalQpIdx_);
    qpNumToIdx_ = std::move(other.qpNumToIdx_);
    maxMsgCntPerQp_ = std::move(other.maxMsgCntPerQp_);
    maxMsgSize_ = std::move(other.maxMsgSize_);
    loadBalancingScheme_ = std::move(other.loadBalancingScheme_);
    pendingSendVirtualWrQue_ = std::move(other.pendingSendVirtualWrQue_);
    pendingRecvVirtualWrQue_ = std::move(other.pendingRecvVirtualWrQue_);
    virtualQpNum_ = std::move(other.virtualQpNum_);
    nextPhysicalWrId_ = std::move(other.nextPhysicalWrId_);
    pendingSendNotifyVirtualWrQue_ =
        std::move(other.pendingSendNotifyVirtualWrQue_);
    dqplbSeqTracker = std::move(other.dqplbSeqTracker);
    dqplbReceiverInitialized_ = std::move(other.dqplbReceiverInitialized_);

    // Update coordinator pointer mapping for this virtual QP after move
    auto coordinator = Coordinator::getCoordinator();
    CHECK(coordinator)
        << "Coordinator should not be nullptr during IbvVirtualQp move construction!";
    coordinator->updateVirtualQpPointer(virtualQpNum_, this);
  }
  return *this;
}

IbvVirtualQp::~IbvVirtualQp() {
  // Always call unregister - the coordinator will check if the pointer matches
  // and do nothing if the object was moved
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualQp destruction!";
  coordinator->unregisterVirtualQp(virtualQpNum_, this);
}

folly::Expected<folly::Unit, Error> IbvVirtualQp::modifyVirtualQp(
    ibv_qp_attr* attr,
    int attrMask,
    const IbvVirtualQpBusinessCard& businessCard) {
  // If businessCard is not empty, use it to modify QPs with specific
  // dest_qp_num values
  if (!businessCard.qpNums_.empty()) {
    // Make sure the businessCard has the same number of QPs as physicalQps_
    if (businessCard.qpNums_.size() != physicalQps_.size()) {
      return folly::makeUnexpected(Error(
          EINVAL, "BusinessCard QP count doesn't match physical QP count"));
    }

    // Modify each QP with its corresponding dest_qp_num from the businessCard
    for (auto i = 0; i < physicalQps_.size(); i++) {
      attr->dest_qp_num = businessCard.qpNums_.at(i);
      auto maybeModifyQp = physicalQps_.at(i).modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    attr->dest_qp_num = businessCard.notifyQpNum_;
    auto maybeModifyQp = notifyQp_.modifyQp(attr, attrMask);
    if (maybeModifyQp.hasError()) {
      return folly::makeUnexpected(maybeModifyQp.error());
    }
  } else {
    // If no businessCard provided, modify all QPs with the same attributes
    for (auto& qp : physicalQps_) {
      auto maybeModifyQp = qp.modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    auto maybeModifyQp = notifyQp_.modifyQp(attr, attrMask);
    if (maybeModifyQp.hasError()) {
      return folly::makeUnexpected(maybeModifyQp.error());
    }
  }
  return folly::unit;
}

IbvVirtualQpBusinessCard IbvVirtualQp::getVirtualQpBusinessCard() const {
  std::vector<uint32_t> qpNums;
  qpNums.reserve(physicalQps_.size());
  for (auto& qp : physicalQps_) {
    qpNums.push_back(qp.qp()->qp_num);
  }
  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQp_.qp()->qp_num);
}

LoadBalancingScheme IbvVirtualQp::getLoadBalancingScheme() const {
  return loadBalancingScheme_;
}

/*** IbvVirtualQpBusinessCard ***/

IbvVirtualQpBusinessCard::IbvVirtualQpBusinessCard(
    std::vector<uint32_t> qpNums,
    uint32_t notifyQpNum)
    : qpNums_(std::move(qpNums)), notifyQpNum_(notifyQpNum) {}

folly::dynamic IbvVirtualQpBusinessCard::toDynamic() const {
  folly::dynamic obj = folly::dynamic::object;
  folly::dynamic qpNumsArray = folly::dynamic::array;

  // Use fixed-width string formatting to ensure consistent size
  // All uint32_t values will be formatted as 10-digit zero-padded strings
  for (const auto& qpNum : qpNums_) {
    std::string paddedQpNum = fmt::format("{:010d}", qpNum);
    qpNumsArray.push_back(paddedQpNum);
  }

  obj["qpNums"] = std::move(qpNumsArray);
  obj["notifyQpNum"] = fmt::format("{:010d}", notifyQpNum_);
  return obj;
}

folly::Expected<IbvVirtualQpBusinessCard, Error>
IbvVirtualQpBusinessCard::fromDynamic(const folly::dynamic& obj) {
  std::vector<uint32_t> qpNums;

  if (obj.count("qpNums") > 0 && obj["qpNums"].isArray()) {
    const auto& qpNumsArray = obj["qpNums"];
    qpNums.reserve(qpNumsArray.size());

    for (const auto& qpNum : qpNumsArray) {
      CHECK(qpNum.isString()) << "qp num is not string!";
      try {
        uint32_t qpNumValue =
            static_cast<uint32_t>(std::stoul(qpNum.asString()));
        qpNums.push_back(qpNumValue);
      } catch (const std::exception& e) {
        return folly::makeUnexpected(Error(
            EINVAL,
            fmt::format(
                "Invalid QP number string format: {}. Exception: {}",
                qpNum.asString(),
                e.what())));
      }
    }
  } else {
    return folly::makeUnexpected(
        Error(EINVAL, "Invalid qpNums array received from remote side"));
  }

  uint32_t notifyQpNum = 0; // Default value for backwards compatibility
  if (obj.count("notifyQpNum") > 0 && obj["notifyQpNum"].isString()) {
    try {
      notifyQpNum =
          static_cast<uint32_t>(std::stoul(obj["notifyQpNum"].asString()));
    } catch (const std::exception& e) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Invalid notifyQpNum string format: {}. Exception: {}",
              obj["notifyQpNum"].asString(),
              e.what())));
    }
  }

  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}

std::string IbvVirtualQpBusinessCard::serialize() const {
  return folly::toJson(toDynamic());
}

folly::Expected<IbvVirtualQpBusinessCard, Error>
IbvVirtualQpBusinessCard::deserialize(const std::string& jsonStr) {
  try {
    folly::dynamic obj = folly::parseJson(jsonStr);
    return fromDynamic(obj);
  } catch (const std::exception& e) {
    return folly::makeUnexpected(Error(
        EINVAL,
        fmt::format(
            "Failed to parse JSON in IbvVirtualQpBusinessCard Deserialize. Exception: {}",
            e.what())));
  }
}

/*** IbvVirtualCq ***/

IbvVirtualCq::IbvVirtualCq(IbvCq&& physicalCq, int maxCqe)
    : physicalCq_(std::move(physicalCq)), maxCqe_(maxCqe) {
  virtualCqNum_ =
      nextVirtualCqNum_.fetch_add(1); // Assign unique virtual CQ number

  // Register the virtual CQ with Coordinator
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualCq construction!";
  coordinator->registerVirtualCq(virtualCqNum_, this);
}

IbvVirtualCq::IbvVirtualCq(IbvVirtualCq&& other) noexcept {
  physicalCq_ = std::move(other.physicalCq_);
  pendingSendVirtualWcQue_ = std::move(other.pendingSendVirtualWcQue_);
  pendingRecvVirtualWcQue_ = std::move(other.pendingRecvVirtualWcQue_);
  maxCqe_ = other.maxCqe_;
  virtualWrIdToVirtualWc_ = std::move(other.virtualWrIdToVirtualWc_);
  virtualCqNum_ = other.virtualCqNum_;

  // Update coordinator pointer mapping for this virtual CQ after move
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualCq move construction!";
  coordinator->updateVirtualCqPointer(virtualCqNum_, this);
}

IbvVirtualCq& IbvVirtualCq::operator=(IbvVirtualCq&& other) noexcept {
  if (this != &other) {
    physicalCq_ = std::move(other.physicalCq_);
    pendingSendVirtualWcQue_ = std::move(other.pendingSendVirtualWcQue_);
    pendingRecvVirtualWcQue_ = std::move(other.pendingRecvVirtualWcQue_);
    maxCqe_ = other.maxCqe_;
    virtualWrIdToVirtualWc_ = std::move(other.virtualWrIdToVirtualWc_);
    virtualCqNum_ = other.virtualCqNum_;

    // Update coordinator pointer mapping for this virtual CQ after move
    auto coordinator = Coordinator::getCoordinator();
    CHECK(coordinator)
        << "Coordinator should not be nullptr during IbvVirtualCq move construction!";
    coordinator->updateVirtualCqPointer(virtualCqNum_, this);
  }
  return *this;
}

IbvCq& IbvVirtualCq::getPhysicalCqRef() {
  return physicalCq_;
}

uint32_t IbvVirtualCq::getVirtualCqNum() const {
  return virtualCqNum_;
}

void IbvVirtualCq::enqueSendCq(VirtualWc virtualWc) {
  pendingSendVirtualWcQue_.push_back(std::move(virtualWc));
}

void IbvVirtualCq::enqueRecvCq(VirtualWc virtualWc) {
  pendingRecvVirtualWcQue_.push_back(std::move(virtualWc));
}

IbvVirtualCq::~IbvVirtualCq() {
  // Always call unregister - the coordinator will check if the pointer matches
  // and do nothing if the object was moved
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualCq destruction!";
  coordinator->unregisterVirtualCq(virtualCqNum_, this);
}

/*** Coordinator ***/

std::shared_ptr<Coordinator> Coordinator::getCoordinator() {
  return coordinatorSingleton.try_get();
}

// Register APIs for mapping management
void Coordinator::registerVirtualQp(
    uint32_t virtualQpNum,
    IbvVirtualQp* virtualQp) {
  virtualQpNumToVirtualQp_[virtualQpNum] = virtualQp;
}

void Coordinator::registerVirtualCq(
    uint32_t virtualCqNum,
    IbvVirtualCq* virtualCq) {
  virtualCqNumToVirtualCq_[virtualCqNum] = virtualCq;
}

void Coordinator::registerPhysicalQpToVirtualQp(
    int physicalQpNum,
    uint32_t virtualQpNum) {
  physicalQpNumToVirtualQpNum_[physicalQpNum] = virtualQpNum;
}

void Coordinator::registerVirtualQpToVirtualSendCq(
    uint32_t virtualQpNum,
    uint32_t virtualSendCqNum) {
  virtualQpNumToVirtualSendCqNum_[virtualQpNum] = virtualSendCqNum;
}

void Coordinator::registerVirtualQpToVirtualRecvCq(
    uint32_t virtualQpNum,
    uint32_t virtualRecvCqNum) {
  virtualQpNumToVirtualRecvCqNum_[virtualQpNum] = virtualRecvCqNum;
}

void Coordinator::registerVirtualQpWithVirtualCqMappings(
    IbvVirtualQp* virtualQp,
    uint32_t virtualSendCqNum,
    uint32_t virtualRecvCqNum) {
  // Extract virtual QP number from the virtual QP object
  uint32_t virtualQpNum = virtualQp->getVirtualQpNum();

  // Register the virtual QP
  registerVirtualQp(virtualQpNum, virtualQp);

  // Register all physical QP to virtual QP mappings
  for (const auto& qp : virtualQp->getQpsRef()) {
    registerPhysicalQpToVirtualQp(qp.qp()->qp_num, virtualQpNum);
  }
  // Register notify QP
  registerPhysicalQpToVirtualQp(
      virtualQp->getNotifyQpRef().qp()->qp_num, virtualQpNum);

  // Register virtual QP to virtual CQ relationships
  registerVirtualQpToVirtualSendCq(virtualQpNum, virtualSendCqNum);
  registerVirtualQpToVirtualRecvCq(virtualQpNum, virtualRecvCqNum);
}

// Access APIs for testing and internal use
const std::unordered_map<uint32_t, IbvVirtualQp*>&
Coordinator::getVirtualQpMap() const {
  return virtualQpNumToVirtualQp_;
}

const std::unordered_map<uint32_t, IbvVirtualCq*>&
Coordinator::getVirtualCqMap() const {
  return virtualCqNumToVirtualCq_;
}

const std::unordered_map<int, uint32_t>&
Coordinator::getPhysicalQpToVirtualQpMap() const {
  return physicalQpNumToVirtualQpNum_;
}

const std::unordered_map<uint32_t, uint32_t>&
Coordinator::getVirtualQpToVirtualSendCqMap() const {
  return virtualQpNumToVirtualSendCqNum_;
}

const std::unordered_map<uint32_t, uint32_t>&
Coordinator::getVirtualQpToVirtualRecvCqMap() const {
  return virtualQpNumToVirtualRecvCqNum_;
}

// Update API for move operations - only need to update pointer maps
void Coordinator::updateVirtualQpPointer(
    uint32_t virtualQpNum,
    IbvVirtualQp* newPtr) {
  virtualQpNumToVirtualQp_[virtualQpNum] = newPtr;
}

void Coordinator::updateVirtualCqPointer(
    uint32_t virtualCqNum,
    IbvVirtualCq* newPtr) {
  virtualCqNumToVirtualCq_[virtualCqNum] = newPtr;
}

void Coordinator::unregisterVirtualQp(
    uint32_t virtualQpNum,
    IbvVirtualQp* ptr) {
  // Only unregister if the pointer in the map matches the object being
  // destroyed. This handles the case where the object was moved and the map was
  // already updated with the new pointer.
  auto it = virtualQpNumToVirtualQp_.find(virtualQpNum);
  if (it == virtualQpNumToVirtualQp_.end() || it->second != ptr) {
    // Object was moved, map already updated, nothing to do
    return;
  }

  // Remove entries from all maps related to this virtual QP
  virtualQpNumToVirtualQp_.erase(virtualQpNum);
  virtualQpNumToVirtualSendCqNum_.erase(virtualQpNum);
  virtualQpNumToVirtualRecvCqNum_.erase(virtualQpNum);

  // Remove all physical QP to virtual QP mappings that point to this virtual QP
  for (auto it = physicalQpNumToVirtualQpNum_.begin();
       it != physicalQpNumToVirtualQpNum_.end();) {
    if (it->second == virtualQpNum) {
      it = physicalQpNumToVirtualQpNum_.erase(it);
    } else {
      ++it;
    }
  }
}

void Coordinator::unregisterVirtualCq(
    uint32_t virtualCqNum,
    IbvVirtualCq* ptr) {
  // Only unregister if the pointer in the map matches the object being
  // destroyed. This handles the case where the object was moved and the map was
  // already updated with the new pointer.
  auto it = virtualCqNumToVirtualCq_.find(virtualCqNum);
  if (it == virtualCqNumToVirtualCq_.end() || it->second != ptr) {
    // Object was moved, map already updated, nothing to do
    return;
  }

  // Remove the virtual CQ from the pointer map
  virtualCqNumToVirtualCq_.erase(virtualCqNum);

  // Remove all virtual QP to virtual send CQ mappings that point to this
  // virtual CQ
  for (auto it = virtualQpNumToVirtualSendCqNum_.begin();
       it != virtualQpNumToVirtualSendCqNum_.end();) {
    if (it->second == virtualCqNum) {
      it = virtualQpNumToVirtualSendCqNum_.erase(it);
    } else {
      ++it;
    }
  }

  // Remove all virtual QP to virtual recv CQ mappings that point to this
  // virtual CQ
  for (auto it = virtualQpNumToVirtualRecvCqNum_.begin();
       it != virtualQpNumToVirtualRecvCqNum_.end();) {
    if (it->second == virtualCqNum) {
      it = virtualQpNumToVirtualRecvCqNum_.erase(it);
    } else {
      ++it;
    }
  }
}

/*** RoceHCA ***/

RoceHca::RoceHca(std::string hcaStr, int defaultPort) {
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

folly::Expected<folly::Unit, Error> Mlx5dv::initObj(
    mlx5dv_obj* obj,
    uint64_t obj_type) {
  int rc = ibvSymbols.mlx5dv_internal_init_obj(obj, obj_type);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

} // namespace ibverbx
