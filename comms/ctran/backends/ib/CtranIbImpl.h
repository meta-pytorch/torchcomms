// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_IB_IMPL_H_
#define CTRAN_IB_IMPL_H_

#include <mutex>
#include "comms/ctran/backends/CtranCtrl.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
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
      ProcessGlobalErrorsUtil::setNic(                                                                \
          devices[device].devName, wc.qp_num, errMsg);                                                \
      CLOGF(ERR, "{}", errMsg);                                                                       \
      return commRemoteError;                                                                         \
    }                                                                                                 \
  } while (0)

namespace ctran::ib {
inline void getRemoteKeysImpl(
    void* ibRegElem,
    std::array<uint32_t, CTRAN_MAX_IB_DEVICES_PER_RANK>& rkeys) {
  auto mrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);
  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
    rkeys.at(device) = (*mrs)[device].mr()->rkey;
  }
}
} // namespace ctran::ib
#endif
