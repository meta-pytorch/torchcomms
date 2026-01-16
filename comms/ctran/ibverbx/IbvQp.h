// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include <folly/logging/xlog.h>
#include <deque>
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/device/structs.h"

namespace ibverbx {

// Ibv Queue Pair
class IbvQp {
 public:
  ~IbvQp();

  // disable copy constructor
  IbvQp(const IbvQp&) = delete;
  IbvQp& operator=(const IbvQp&) = delete;

  // move constructor
  IbvQp(IbvQp&& other) noexcept;
  IbvQp& operator=(IbvQp&& other) noexcept;

  ibv_qp* qp() const;
  int32_t getDeviceId() const;

  // create device qp, map qp buffer to GPU
  folly::Expected<struct device_qp, Error> getDeviceQp(
      device_cq* cq) const noexcept;

  folly::Expected<folly::Unit, Error> modifyQp(ibv_qp_attr* attr, int attrMask);
  folly::Expected<std::pair<ibv_qp_attr, ibv_qp_init_attr>, Error> queryQp(
      int attrMask) const;

  // Log QP details for debugging
  void logInfo() const;

  inline uint32_t getQpNum() const;
  inline folly::Expected<folly::Unit, Error> postRecv(
      ibv_recv_wr* recvWr,
      ibv_recv_wr* recvWrBad);
  inline folly::Expected<folly::Unit, Error> postSend(
      ibv_send_wr* sendWr,
      ibv_send_wr* sendWrBad);

  void enquePhysicalSendWrStatus(int physicalWrId, int virtualWrId);
  void enquePhysicalRecvWrStatus(int physicalWrId, int virtualWrId);
  void dequePhysicalSendWrStatus();
  void dequePhysicalRecvWrStatus();
  bool isSendQueueAvailable(int maxMsgCntPerQp) const;
  bool isRecvQueueAvailable(int maxMsgCntPerQp) const;

 private:
  friend class IbvPd;
  friend class IbvVirtualQp;
  friend class IbvVirtualCq;

  struct PhysicalSendWrStatus {
    PhysicalSendWrStatus(uint64_t physicalWrId, uint64_t virtualWrId)
        : physicalWrId(physicalWrId), virtualWrId(virtualWrId) {}
    uint64_t physicalWrId{0};
    uint64_t virtualWrId{0};
  };
  struct PhysicalRecvWrStatus {
    PhysicalRecvWrStatus(uint64_t physicalWrId, uint64_t virtualWrId)
        : physicalWrId(physicalWrId), virtualWrId(virtualWrId) {}
    uint64_t physicalWrId{0};
    uint64_t virtualWrId{0};
  };
  explicit IbvQp(ibv_qp* qp, int32_t deviceId);

  ibv_qp* qp_{nullptr};
  std::deque<PhysicalSendWrStatus> physicalSendWrStatus_;
  std::deque<PhysicalRecvWrStatus> physicalRecvWrStatus_;
  int32_t deviceId_{-1}; // The IbvDevice's DeviceId that corresponds to this
                         // Queue Pair (QP)
};

// IbvQp inline functions
inline uint32_t IbvQp::getQpNum() const {
  XCHECK_NE(qp_, nullptr);
  return qp_->qp_num;
}

inline folly::Expected<folly::Unit, Error> IbvQp::postRecv(
    ibv_recv_wr* recvWr,
    ibv_recv_wr* recvWrBad) {
  int rc = qp_->context->ops.post_recv(qp_, recvWr, &recvWrBad);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

inline folly::Expected<folly::Unit, Error> IbvQp::postSend(
    ibv_send_wr* sendWr,
    ibv_send_wr* sendWrBad) {
  int rc = qp_->context->ops.post_send(qp_, sendWr, &sendWrBad);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

} // namespace ibverbx
