// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvPd.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvPd ***/

IbvPd::IbvPd(ibv_pd* pd, int32_t deviceId, bool dataDirect)
    : pd_(pd), deviceId_(deviceId), dataDirect_(dataDirect) {}

IbvPd::IbvPd(IbvPd&& other) noexcept {
  pd_ = other.pd_;
  dataDirect_ = other.dataDirect_;
  deviceId_ = other.deviceId_;
  other.pd_ = nullptr;
  other.deviceId_ = -1;
}

IbvPd& IbvPd::operator=(IbvPd&& other) noexcept {
  pd_ = other.pd_;
  dataDirect_ = other.dataDirect_;
  deviceId_ = other.deviceId_;
  other.pd_ = nullptr;
  other.deviceId_ = -1;
  return *this;
}

IbvPd::~IbvPd() {
  if (pd_) {
    int rc = ibvSymbols.ibv_internal_dealloc_pd(pd_);
    if (rc != 0) {
      XLOGF(
          WARN,
          "Failed to deallocate pd rc: {}, {}. "
          "This is a post-failure warning likely due to an uncleaned RDMA resource on the failure path.",
          rc,
          strerror(errno));
    }
  }
}

ibv_pd* IbvPd::pd() const {
  return pd_;
}

bool IbvPd::useDataDirect() const {
  return dataDirect_;
}

int32_t IbvPd::getDeviceId() const {
  return deviceId_;
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
  return IbvQp(qp, deviceId_);
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
  initAttr->send_cq = sendCq->getPhysicalCqsRef().at(0).cq();
  initAttr->recv_cq = recvCq->getPhysicalCqsRef().at(0).cq();

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

} // namespace ibverbx
