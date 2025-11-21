// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/ProxyTraceFunc.h"

#include "comms/utils/colltrace/NetworkPerfMonitor.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/ProxyTrace.h"

namespace ncclx::colltrace {
void proxyTraceInfoCopy(ncclProxyOp& proxyOp, CtranComm* comm) {
  proxyOp.traceArgs.collInfo.commHash = comm->statex_->commHash();
  proxyOp.traceArgs.collInfo.opCount = *comm->opCount_;
  proxyOp.traceArgs.rank = comm->statex_->rank();

  proxyOp.traceArgs.remoteRank = proxyOp.root;
}

void proxyTraceAddBasicInfo(
    ncclProxyOp& proxyOp,
    int nChannels,
    ncclFunc_t coll) {
  proxyOp.traceArgs.collInfo.nChannels = nChannels;
  proxyOp.traceArgs.collInfo.coll = coll;
}

ncclResult_t proxyTraceInit(struct ncclProxyState* state, CtranComm* comm) {
  if (NCCL_PROXYTRACE.empty()) {
    return ncclSuccess;
  }
  auto networkPerfMonitorPtr =
      ncclx::colltrace::NetworkPerfMonitor::getInstance();
  if (networkPerfMonitorPtr != nullptr && comm != nullptr) {
    networkPerfMonitorPtr->storeCommInfo(
        comm->logMetaData_, comm->statex_->cudaDev(), comm->statex_->busId());
  }
  try {
    state->trace = std::make_unique<ProxyTrace>();
  } catch (const std::exception& e) {
    WARN(
        "PROXYTRACE: failed to initialize ProxyTrace, comm %p commDesc %s: %s",
        comm,
        comm->config_.commDesc.c_str(),
        e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t proxyTraceDestroy(struct ncclProxyState* state) {
  if (state->trace) {
    state->trace.reset();
  }
  return ncclSuccess;
}
} // namespace ncclx::colltrace
