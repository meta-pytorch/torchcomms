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
#include "comms/utils/cvars/nccl_cvars.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

namespace {

folly::once_flag initIbvSymbolOnce;

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

} // namespace ibverbx
