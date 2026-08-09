// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvPd.h"
#include "comms/ctran/ibverbx/IbvVirtualCq.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/utils/CtranLogger.h"

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
      CTRAN_LOG(
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

std::string IbvPd::getDeviceName() const {
  if (pd_ && pd_->context && pd_->context->device) {
    return std::string(pd_->context->device->name);
  }
  return "unknown";
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
    if (ibvSymbols.mlx5dv_internal_reg_dmabuf_mr == nullptr) {
      return folly::makeUnexpected(Error(ENOSYS));
    }
    mr = ibvSymbols.mlx5dv_internal_reg_dmabuf_mr(
        pd_,
        offset,
        length,
        iova,
        fd,
        access,
        MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  } else {
    if (ibvSymbols.ibv_internal_reg_dmabuf_mr == nullptr) {
      return folly::makeUnexpected(Error(ENOSYS));
    }
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
    IbvVirtualCq* virtualCq,
    int maxMsgCntPerQp,
    int maxMsgSize,
    LoadBalancingScheme loadBalancingScheme) const {
  std::vector<IbvQp> qps;
  qps.reserve(totalQps);

  if (virtualCq == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "Empty virtualCq being provided to createVirtualQp"));
  }

  // Overwrite the CQs in the initAttr to point to the virtual CQ
  initAttr->send_cq = virtualCq->getPhysicalCqsRef().at(0).cq();
  initAttr->recv_cq = virtualCq->getPhysicalCqsRef().at(0).cq();

  // First create all the data QPs
  for (int i = 0; i < totalQps; i++) {
    auto maybeQp = createQp(initAttr);
    if (maybeQp.hasError()) {
      return folly::makeUnexpected(maybeQp.error());
    }
    qps.emplace_back(std::move(*maybeQp));
  }

  // Create notify QP when using multiple data QPs, to support all
  // load balancing schemes that may require it (e.g., SPRAY)
  std::optional<IbvQp> notifyQp;
  if (totalQps > 1) {
    auto maybeNotifyQp = createQp(initAttr);
    if (maybeNotifyQp.hasError()) {
      return folly::makeUnexpected(maybeNotifyQp.error());
    }
    notifyQp = std::move(*maybeNotifyQp);
  }

  // Create the IbvVirtualQp instance, with registration happening
  // within IbvVirtualQp constructor
  return IbvVirtualQp(
      std::move(qps),
      virtualCq,
      maxMsgCntPerQp,
      maxMsgSize,
      loadBalancingScheme,
      std::move(notifyQp));
}

folly::Expected<IbvSrq, Error> IbvPd::createSrq(
    ibv_srq_init_attr* srqInitAttr) const {
  if (ibvSymbols.ibv_internal_create_srq == nullptr) {
    return folly::makeUnexpected(Error(ENOSYS));
  }

  ibv_srq* srq;
  srq = ibvSymbols.ibv_internal_create_srq(pd_, srqInitAttr);
  if (!srq) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvSrq(srq);
}

folly::Expected<IbvAh, Error> IbvPd::createAh(ibv_ah_attr* ahAttr) const {
  ibv_ah* ah;
  ah = ibvSymbols.ibv_internal_create_ah(pd_, ahAttr);
  if (!ah) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvAh(ah);
}

folly::Expected<IbvQp, Error> IbvPd::createDcQp(
    ibv_qp_init_attr_ex* initAttrEx,
    mlx5dv_qp_init_attr* mlx5InitAttr) const {
  if (!ibvSymbols.mlx5dv_internal_create_qp) {
    return folly::makeUnexpected(
        Error(ENOTSUP, "mlx5dv_create_qp not available"));
  }

  // Set the PD in the extended attributes
  initAttrEx->pd = pd_;

  ibv_qp* qp;
  qp = ibvSymbols.mlx5dv_internal_create_qp(
      pd_->context, initAttrEx, mlx5InitAttr);
  if (!qp) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvQp(qp, deviceId_);
}

folly::Expected<IbvQp, Error> IbvPd::createExtRcQpMlx5(
    ibv_qp_init_attr_ex* initAttrEx,
    mlx5dv_qp_init_attr* mlx5InitAttr) const {
  if (initAttrEx == nullptr || mlx5InitAttr == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "createExtRcQpMlx5: null init attrs"));
  }
  if (initAttrEx->qp_type != IBV_QPT_RC) {
    return folly::makeUnexpected(
        Error(EINVAL, "createExtRcQpMlx5: only IBV_QPT_RC supported"));
  }
  if ((mlx5InitAttr->comp_mask & MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS) ==
      0) {
    return folly::makeUnexpected(Error(
        EINVAL,
        "createExtRcQpMlx5: MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS bit not "
        "set in comp_mask — caller must advertise create_flags via comp_mask"));
  }
  if (mlx5InitAttr->create_flags == 0) {
    return folly::makeUnexpected(Error(
        EINVAL,
        "createExtRcQpMlx5: create_flags is 0 — use plain createRcQp instead "
        "of the mlx5dv path when no mlx5-specific flags are needed"));
  }
  if (!ibvSymbols.mlx5dv_internal_create_qp) {
    return folly::makeUnexpected(
        Error(ENOTSUP, "mlx5dv_create_qp not available"));
  }

  // mlx5dv_create_qp only honors initAttrEx->pd when IBV_QP_INIT_ATTR_PD is
  // set in comp_mask. Set both together to avoid asymmetry that would let a
  // caller who forgot the bit silently get a QP with an unexpected PD.
  initAttrEx->pd = pd_;
  initAttrEx->comp_mask |= IBV_QP_INIT_ATTR_PD;

  ibv_qp* qp = ibvSymbols.mlx5dv_internal_create_qp(
      pd_->context, initAttrEx, mlx5InitAttr);
  if (!qp) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvQp(qp, deviceId_);
}

} // namespace ibverbx
