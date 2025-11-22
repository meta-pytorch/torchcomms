// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/gpe/CtranGpe.h"

namespace ctran::sendrecv {

// Launch send/recv operations using copy engine for NVL backend
commResult_t launchSendRecvCopyEngine(
    std::vector<OpElem*>& nvlOps,
    std::vector<OpElem*>& sendNvlOps,
    CtranComm* comm);

} // namespace ctran::sendrecv
