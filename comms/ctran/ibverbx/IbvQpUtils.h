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

// initQp - Transitions QP to INIT state with port and access
// configuration
folly::Expected<folly::Unit, Error>
initQp(IbvQp& ibvQp, int port, int qp_access_flags);

// rtrQp - Transitions QP to Ready To Receive (RTR) state with remote
// endpoint info
folly::Expected<folly::Unit, Error>
rtrQp(const RemoteQpInfo& remoteQpInfo, IbvQp& ibvQp, uint8_t trafficClass);

// rtsQp - Transitions QP to Ready To Send (RTS) state for active
// communication
folly::Expected<folly::Unit, Error> rtsQp(IbvQp& ibvQp);

} // namespace ibverbx
