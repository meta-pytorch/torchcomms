// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvDevice.h" // IWYU pragma: keep
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/Mlx5dv.h"

namespace ibverbx {

// Forward declarations
class IbvVirtualQp;

/*** ibverbx APIs ***/

folly::Expected<folly::Unit, Error> ibvInit();

// Get a completion event from the completion channel
folly::Expected<folly::Unit, Error>
ibvGetCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context);

// Acknowledge completion events
void ibvAckCqEvents(ibv_cq* cq, unsigned int nevents);

} // namespace ibverbx
