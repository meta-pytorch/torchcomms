// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvMr.h"
#include "comms/ctran/ibverbx/IbvQp.h"
#include "comms/ctran/ibverbx/IbvVirtualQp.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

class IbvVirtualCq;

// IbvPd: Protection Domain
class IbvPd {
 public:
  ~IbvPd();

  // disable copy constructor
  IbvPd(const IbvPd&) = delete;
  IbvPd& operator=(const IbvPd&) = delete;

  // move constructor
  IbvPd(IbvPd&& other) noexcept;
  IbvPd& operator=(IbvPd&& other) noexcept;

  ibv_pd* pd() const;
  bool useDataDirect() const;
  int32_t getDeviceId() const;
  std::string getDeviceName() const;

  folly::Expected<IbvMr, Error>
  regMr(void* addr, size_t length, ibv_access_flags access) const;

  folly::Expected<IbvMr, Error> regDmabufMr(
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      ibv_access_flags access) const;

  folly::Expected<IbvQp, Error> createQp(ibv_qp_init_attr* initAttr) const;

  // The send_cq and recv_cq fields in initAttr are ignored.
  // Instead, initAttr.send_cq and initAttr.recv_cq will be set to the physical
  // CQs contained within sendCq and recvCq, respectively.
  folly::Expected<IbvVirtualQp, Error> createVirtualQp(
      int totalQps,
      ibv_qp_init_attr* initAttr,
      IbvVirtualCq* sendCq,
      IbvVirtualCq* recvCq,
      int maxMsgCntPerQp = kIbMaxMsgCntPerQp,
      int maxMsgSize = kIbMaxMsgSizeByte,
      LoadBalancingScheme loadBalancingScheme =
          LoadBalancingScheme::SPRAY) const;

 private:
  friend class IbvDevice;

  IbvPd(ibv_pd* pd, int32_t deviceId, bool dataDirect = false);

  ibv_pd* pd_{nullptr};
  int32_t deviceId_{-1}; // The IbvDevice's DeviceId that corresponds to this
                         // Protection Domain (PD)
  bool dataDirect_{false}; // Relevant only to mlx5
};

} // namespace ibverbx
