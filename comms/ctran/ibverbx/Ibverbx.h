// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvDevice.h" // IWYU pragma: keep
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// Forward declarations
class IbvVirtualQp;
class Coordinator;

/*** ibverbx APIs ***/

folly::Expected<folly::Unit, Error> ibvInit();

// Get a completion event from the completion channel
folly::Expected<folly::Unit, Error>
ibvGetCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context);

// Acknowledge completion events
void ibvAckCqEvents(ibv_cq* cq, unsigned int nevents);

class Mlx5dv {
 public:
  static folly::Expected<folly::Unit, Error> initObj(
      mlx5dv_obj* obj,
      uint64_t obj_type);
};

} // namespace ibverbx
