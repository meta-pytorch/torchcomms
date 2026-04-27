// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCLX_IB_NET_COMM_CONFIG_H_
#define NCCLX_IB_NET_COMM_CONFIG_H_

#include "nccl.h" // @manual

namespace ncclx {

struct NcclxIbNetCommConfig {
  int trafficClass{-1}; // -1 == NCCL_NET_TRAFFIC_CLASS_UNDEF
  int ibSplitDataOnQps{NCCL_CONFIG_UNDEF_INT};
  int ibQpsPerConnection{NCCL_CONFIG_UNDEF_INT};
};

inline int ibResolveQpsPerConnection(
    const NcclxIbNetCommConfig* ctx,
    int envDefault) {
  return (ctx && ctx->ibQpsPerConnection != NCCL_CONFIG_UNDEF_INT)
      ? ctx->ibQpsPerConnection
      : envDefault;
}

inline int ibResolveSplitDataOnQps(
    const NcclxIbNetCommConfig* ctx,
    int envDefault) {
  return (ctx && ctx->ibSplitDataOnQps != NCCL_CONFIG_UNDEF_INT)
      ? ctx->ibSplitDataOnQps
      : envDefault;
}

// Per-comm IB config overrides applied after ncclIbSendCommInit.
// Uses a template to avoid including the IB transport common.h
// from meta/ — the template is instantiated in connect.cc where
// ncclIbSendComm/ncclIbRecvComm is fully defined.
template <typename IbComm>
void ncclxIbCommInit(IbComm* comm, void* ctx) {
  auto* ncclxCtx = static_cast<NcclxIbNetCommConfig*>(ctx);
  if (ncclxCtx)
    comm->base.splitDataOnQps =
        ibResolveSplitDataOnQps(ncclxCtx, comm->base.splitDataOnQps);
}

} // namespace ncclx

#endif // NCCLX_IB_NET_COMM_CONFIG_H_
