// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/ibverbx/IbvQp.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/Mlx5dv.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvQp ***/
IbvQp::IbvQp(ibv_qp* qp, int32_t deviceId) : qp_(qp), deviceId_(deviceId) {}

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
  physicalSendQueStatus_ = std::move(other.physicalSendQueStatus_);
  physicalRecvQueStatus_ = std::move(other.physicalRecvQueStatus_);
  deviceId_ = other.deviceId_;
  other.qp_ = nullptr;
  other.deviceId_ = -1;
}

IbvQp& IbvQp::operator=(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendQueStatus_ = std::move(other.physicalSendQueStatus_);
  physicalRecvQueStatus_ = std::move(other.physicalRecvQueStatus_);
  deviceId_ = other.deviceId_;
  other.qp_ = nullptr;
  other.deviceId_ = -1;
  return *this;
}

ibv_qp* IbvQp::qp() const {
  return qp_;
}

int32_t IbvQp::getDeviceId() const {
  return deviceId_;
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

void IbvQp::logInfo() const {
  auto maybeQpAttrPair = queryQp(IBV_QP_STATE | IBV_QP_CAP | IBV_QP_PKEY_INDEX);
  if (maybeQpAttrPair.hasValue()) {
    auto [qpAttr, initAttr] = maybeQpAttrPair.value();
    XLOGF(
        INFO,
        "QP details: qp_num={}, qp_ptr={}, context={}, pd={}, state info: state={}, max_send_wr={}, max_recv_wr={}, max_send_sge={}, max_recv_sge={}, pkey_index={}",
        qp_->qp_num,
        fmt::ptr(qp_),
        fmt::ptr(qp_->context),
        fmt::ptr(qp_->pd),
        static_cast<int>(qpAttr.qp_state),
        qpAttr.cap.max_send_wr,
        qpAttr.cap.max_recv_wr,
        qpAttr.cap.max_send_sge,
        qpAttr.cap.max_recv_sge,
        qpAttr.pkey_index);
  } else {
    XLOGF(
        ERR,
        "QP details: qp_num={}, qp_ptr={}, context={}, pd={}, Failed to query local QP state: errno={} ({})",
        qp_->qp_num,
        fmt::ptr(qp_),
        fmt::ptr(qp_->context),
        fmt::ptr(qp_->pd),
        maybeQpAttrPair.error().errNum,
        maybeQpAttrPair.error().errStr);
  }
}

void IbvQp::enquePhysicalSendWrStatus(int physicalWrId, int virtualWrId) {
  physicalSendQueStatus_.emplace_back(physicalWrId, virtualWrId);
}

void IbvQp::dequePhysicalSendWrStatus() {
  physicalSendQueStatus_.pop_front();
}

void IbvQp::dequePhysicalRecvWrStatus() {
  physicalRecvQueStatus_.pop_front();
}

void IbvQp::enquePhysicalRecvWrStatus(int physicalWrId, int virtualWrId) {
  physicalRecvQueStatus_.emplace_back(physicalWrId, virtualWrId);
}

bool IbvQp::isSendQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalSendQueStatus_.size() < maxMsgCntPerQp;
}

bool IbvQp::isRecvQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalRecvQueStatus_.size() < maxMsgCntPerQp;
}

folly::Expected<struct device_qp, Error> IbvQp::getDeviceQp(
    device_cq* cq) const noexcept {
#if defined(__HIP_PLATFORM_AMD__)
  throw std::runtime_error("getDeviceQp() is not supported on AMD GPUs");
#else
  struct device_qp deviceQp;
  deviceQp.cq = cq;

  // create mlx5dv_qp

  ibverbx::mlx5dv_obj obj = {};
  ibverbx::mlx5dv_qp mlx5_qp{};
  obj.qp.in = qp();
  obj.qp.out = &mlx5_qp;
  {
    auto ret = Mlx5dv::initObj(&obj, ibverbx::MLX5DV_OBJ_QP);
    if (ret.hasError()) {
      return folly::makeUnexpected(ret.error());
    }
  }

  // Set QP number
  deviceQp.qp_num = qp()->qp_num;

  // Map QP work queue buffer
  {
    auto ret = cudaHostGetDevicePointer(&deviceQp.wq_buf, mlx5_qp.sq.buf, 0);
    if (ret != cudaSuccess) {
      return folly::makeUnexpected(Error(
          static_cast<int>(ret),
          fmt::format(
              "Mapping QP WQE to GPU buffer failed: {}",
              cudaGetErrorString(ret))));
    }
    XLOGF(
        INFO,
        "Mapped QP work queue host={} device={} wqe_cnt={} stride={}",
        fmt::ptr(mlx5_qp.sq.buf),
        fmt::ptr(deviceQp.wq_buf),
        mlx5_qp.sq.wqe_cnt,
        mlx5_qp.sq.stride);
  }

  deviceQp.nwqes = mlx5_qp.sq.wqe_cnt;

  // Map QP doorbell record
  // This region contains two 32-bit (4-byte) doorbells:
  // RQ doorbell is at offset 0, SQ doorbell is at offset +sizeof(__be32)
  void* gpu_qp_dbrec;
  {
    auto ret =
        cudaHostGetDevicePointer((void**)&gpu_qp_dbrec, mlx5_qp.dbrec, 0);
    if (ret != cudaSuccess) {
      return folly::makeUnexpected(Error(
          static_cast<int>(ret),
          fmt::format(
              "Mapping QP DBREC to GPU buffer failed: {}",
              cudaGetErrorString(ret))));
    }
    XLOGF(
        INFO,
        "Mapped QP doorbell host={} device={}",
        fmt::ptr(mlx5_qp.dbrec),
        fmt::ptr(gpu_qp_dbrec));
  }

  // TODO - add rq support if needed:
  // deviceQp.rq_dbrec = (__be32*)gpu_qp_dbrec;
  deviceQp.sq_dbrec = (__be32*)((uintptr_t)gpu_qp_dbrec + sizeof(__be32));

  // Map QP BlueFlame register
  if (mlx5_qp.bf.size <= 0) {
    return folly::makeUnexpected(Error(
        1,
        fmt::format(
            "BlueFlame register not allocated by QP creation: size={}. ",
            mlx5_qp.bf.size)));
  }
  {
    auto ret = cudaHostRegister(
        mlx5_qp.bf.reg,
        mlx5_qp.bf.size * 2,
        cudaHostRegisterPortable | cudaHostRegisterMapped |
            cudaHostRegisterIoMemory);
    if (ret != cudaSuccess) {
      return folly::makeUnexpected(Error(
          static_cast<int>(ret),
          fmt::format(
              "Registering BlueFlame register failed: {}",
              cudaGetErrorString(ret))));
    }
  }
  {
    auto ret =
        cudaHostGetDevicePointer((void**)&deviceQp.bf_reg, mlx5_qp.bf.reg, 0);
    if (ret != cudaSuccess) {
      return folly::makeUnexpected(Error(
          static_cast<int>(ret),
          fmt::format(
              "Mapping BlueFlame register to GPU buffer failed: {}",
              cudaGetErrorString(ret))));
    }
    XLOGF(
        INFO,
        "Mapped BlueFlame host={} device={} size={}",
        fmt::ptr(mlx5_qp.bf.reg),
        fmt::ptr(deviceQp.bf_reg),
        mlx5_qp.bf.size);
  }

  if (mlx5_qp.sq.stride != WQE_STRIDE) {
    return folly::makeUnexpected(Error(
        1,
        fmt::format(
            "QP WQE stride does not match: expected={}, actual={}",
            WQE_STRIDE,
            mlx5_qp.sq.stride)));
  }

  return deviceQp;
#endif
}

} // namespace ibverbx
