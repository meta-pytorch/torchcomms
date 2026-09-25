// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_IMPL_H_
#define CTRAN_IB_IMPL_H_

#include <mutex>
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranLogUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ScubaLogger.h"

#define CTRAN_IB_PER_OBJ_LOCK_GUARD(mutex_, code) \
  if (NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE) {          \
    code;                                         \
  } else {                                        \
    std::lock_guard<std::mutex> lock(mutex_);     \
    code;                                         \
  }
#define CQE_ERROR_CHECK(wc, peerRank, qpnType)                                                        \
  do {                                                                                                \
    if (wc.status != ibverbx::IBV_WC_SUCCESS) {                                                       \
      /* NOTE: wc.opcode may be arbitrary if the poll_cq returned an error.                           \
       * DO NOT print it to avoid confusion. */                                                       \
      auto errMsg = fmt::format(                                                                      \
          "CTRAN-IB: wrap_ibv_poll_cq failed, commHash {:x} peer {} {} qpn {}, with status={}, '{}'", \
          commHash,                                                                                   \
          peerRank,                                                                                   \
          qpnType,                                                                                    \
          wc.qp_num,                                                                                  \
          wc.status,                                                                                  \
          ibv_wc_status_str(wc.status));                                                              \
      CTRAN_ERR(commRemoteError, "{}", errMsg);                                                       \
      return commRemoteError;                                                                         \
    }                                                                                                 \
  } while (0)

namespace ctran::ib {
inline commResult_t validateConfiguredDeviceCount(
    const int configuredDeviceCount,
    const size_t rkeyCapacity) {
  if (configuredDeviceCount <= 0) {
    CTRAN_ERR(
        commInvalidArgument,
        "CTRAN-IB: invalid configured IB device count {}",
        configuredDeviceCount);
    return commInvalidArgument;
  }
  if (static_cast<size_t>(configuredDeviceCount) > rkeyCapacity) {
    CTRAN_ERR(
        commInvalidArgument,
        "CTRAN-IB: configured IB device count {} exceeds rkey capacity {}",
        configuredDeviceCount,
        rkeyCapacity);
    return commInvalidArgument;
  }
  return commSuccess;
}

inline commResult_t getRemoteKeysImpl(
    void* ibRegElem,
    const int configuredDeviceCount,
    std::array<uint32_t, CTRAN_MAX_IB_DEVICES_PER_RANK>& rkeys) {
  if (ibRegElem == nullptr) {
    CTRAN_ERR(
        commInvalidArgument,
        "CTRAN-IB: getRemoteKeys called with a null IB registration handle");
    return commInvalidArgument;
  }

  FB_COMMCHECK(
      validateConfiguredDeviceCount(configuredDeviceCount, rkeys.size()));
  const auto deviceCount = static_cast<size_t>(configuredDeviceCount);

  const auto* mrs =
      reinterpret_cast<const std::vector<ibverbx::IbvMr>*>(ibRegElem);
  if (mrs->size() < deviceCount) {
    CTRAN_ERR(
        commInvalidArgument,
        "CTRAN-IB: registration carries {} memory regions but {} devices are "
        "configured",
        mrs->size(),
        deviceCount);
    return commInvalidArgument;
  }
  // Validate every MR before mutating the output.
  for (size_t device = 0; device < deviceCount; device++) {
    if ((*mrs)[device].mr() == nullptr) {
      CTRAN_ERR(
          commInvalidArgument,
          "CTRAN-IB: registration memory region {} is null",
          device);
      return commInvalidArgument;
    }
  }
  for (size_t device = 0; device < deviceCount; device++) {
    rkeys[device] = (*mrs)[device].mr()->rkey;
  }
  return commSuccess;
}
} // namespace ctran::ib
#endif
