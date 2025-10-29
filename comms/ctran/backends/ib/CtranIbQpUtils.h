// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/ibverbx/Ibverbx.h"

namespace ctran::ib {

/**
 * Common utility functions for InfiniBand Queue Pair (QP) management.
 *
 * These functions provide a reusable interface for QP creation and state
 * transitions that can be used across different components of the Ctran IB
 * backend. They were extracted from CtranIbVcImpl to enable reuse in other QP
 * initialization scenarios.
 */

constexpr int MAX_CONTROL_MSGS{128};
constexpr int MAX_SEND_WR{256};

struct CtranIbRemoteQpInfo {
  enum ibverbx::ibv_mtu mtu;
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

// ctranIbQpCreate - Creates a new Reliable Connection (RC) QP
folly::Expected<ibverbx::IbvQp, ibverbx::Error> ctranIbQpCreate(
    const ibverbx::IbvPd* ibvPd,
    ibverbx::ibv_cq* cq);

// ctranIbQpInit - Transitions QP to INIT state with port and access
// configuration
folly::Expected<folly::Unit, ibverbx::Error>
ctranIbQpInit(ibverbx::IbvQp& ibvQp, int port, int qp_access_flags);

// ctranIbQpRTR - Transitions QP to Ready To Receive (RTR) state with remote
// endpoint info
folly::Expected<folly::Unit, ibverbx::Error> ctranIbQpRTR(
    const CtranIbRemoteQpInfo& remoteQpInfo,
    ibverbx::IbvQp& ibvQp,
    uint8_t trafficClass);

// ctranIbQpRTS - Transitions QP to Ready To Send (RTS) state for active
// communication
folly::Expected<folly::Unit, ibverbx::Error> ctranIbQpRTS(
    ibverbx::IbvQp& ibvQp);

} // namespace ctran::ib
