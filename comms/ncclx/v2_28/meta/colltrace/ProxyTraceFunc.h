// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"
#include "proxy.h"

#include "comms/ctran/CtranComm.h"
#include "meta/colltrace/ProxyTrace.h"

namespace ncclx::colltrace {
void proxyTraceInfoCopy(ncclProxyOp& proxyOp, CtranComm* comm);

// TODO: remove this function after refactoring done
// !!! DO NOT USE THIS FUNCTION !!!
inline void proxyTraceInfoCopy(ncclProxyOp& proxyOp, ncclComm* comm) {
  proxyTraceInfoCopy(proxyOp, comm->ctranComm_.get());
}

void proxyTraceAddBasicInfo(
    ncclProxyOp& proxyOp,
    int nChannels,
    ncclFunc_t coll);

ncclResult_t proxyTraceInit(struct ncclProxyState* state, CtranComm* comm);

// TODO: remove this once ctran refactoring is done
// !!! DO NOT USE THIS FUNCTION !!!
inline ncclResult_t proxyTraceInit(
    struct ncclProxyState* state,
    ncclComm* comm) {
  return proxyTraceInit(state, comm->ctranComm_.get());
}

ncclResult_t proxyTraceDestroy(struct ncclProxyState* state);
} // namespace ncclx::colltrace
