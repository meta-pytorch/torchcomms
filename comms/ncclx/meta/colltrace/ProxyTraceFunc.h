// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comm.h"
#include "proxy.h"

#include "meta/colltrace/ProxyTrace.h"

namespace ncclx::colltrace {
void proxyTraceInfoCopy(ncclProxyOp& proxyOp, ncclComm* comm);

void proxyTraceAddBasicInfo(
    ncclProxyOp& proxyOp,
    int nChannels,
    ncclFunc_t coll);

ncclResult_t proxyTraceInit(struct ncclProxyState* state, ncclComm* comm);

ncclResult_t proxyTraceDestroy(struct ncclProxyState* state);
} // namespace ncclx::colltrace
