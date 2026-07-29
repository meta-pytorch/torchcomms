// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/ibverbx/Ibverbx.h"

namespace ibverbx {

/**
 * Common utility functions for InfiniBand Queue Pair (QP) management.
 *
 * These functions provide a reusable interface for QP creation and state
 * transitions that can be used across different components of the Ctran IB
 * backend. They were extracted from CtranIbVcImpl to enable reuse in other QP
 * initialization scenarios.
 */

struct RemoteQpInfo {
  enum ibv_mtu mtu;
  uint32_t qpn;
  uint8_t port;
  int linkLayer;
  union {
    struct {
      uint64_t spn;
      uint64_t iid;
    } eth;
    struct {
      uint16_t lid;
    } ib;
  } u;
};

// createRcQp - Creates a new Reliable Connection (RC) QP
folly::Expected<IbvQp, Error>
createRcQp(const IbvPd* ibvPd, ibv_cq* cq, int maxSendWr, int maxRecvWr);

// createRcQpWithOooDp - Creates an RC QP with the option to enable mlx5
// out-of-order data placement (OOO_DP). When oooDp=true, goes through
// mlx5dv_create_qp with MLX5DV_QP_CREATE_OOO_DP so packet-sprayed data
// (adaptive routing) may arrive out of order on the RQ. When oooDp=false,
// bypasses mlx5dv entirely and delegates to the standard createRcQp() path
// — so deployments without libmlx5 loaded still create QPs normally.
// Caller must ensure the underlying device supports OOO DP when oooDp=true
// (mlx5 provider, adaptive routing enabled, ooo_recv_wrs_caps.max_rc >=
// maxRecvWr) — this helper does not gate.
folly::Expected<IbvQp, Error> createRcQpWithOooDp(
    const IbvPd* ibvPd,
    ibv_cq* cq,
    int maxSendWr,
    int maxRecvWr,
    bool oooDp);

// initQp - Transitions QP to INIT state with port and access
// configuration
folly::Expected<folly::Unit, Error>
initQp(IbvQp& ibvQp, int port, int qp_access_flags);

// rtrQp - Transitions QP to Ready To Receive (RTR) state with remote
// endpoint info
folly::Expected<folly::Unit, Error> rtrQp(
    const RemoteQpInfo& remoteQpInfo,
    IbvQp& ibvQp,
    uint8_t trafficClass,
    uint8_t gid_index,
    uint8_t ib_sl,
    uint32_t psn = 0,
    uint8_t maxRdAtomic = 1);

// rtsQp - Transitions QP to Ready To Send (RTS) state for active
// communication
folly::Expected<folly::Unit, Error> rtsQp(
    IbvQp& ibvQp,
    uint8_t timeout,
    uint8_t retryCnt,
    uint32_t psn = 0,
    uint8_t maxRdAtomic = 1);

} // namespace ibverbx
