// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Expected.h>
#include <folly/dynamic.h>
#include <folly/json.h>

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/IbvDevice.h" // IWYU pragma: keep
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/Mlx5dv.h"

namespace ibverbx {

// Forward declarations
class IbvVirtualQp;

/*** ibverbx APIs ***/

// dlopen the libibverbs named by IBVERBX_IBVERBS_SO, or libibverbs.so.1 when
// unset. A plain environment variable rather than a cvar: it is read here, at
// the point of use, so it works the same in every ibverbx consumer -- including
// ctranx, prims and uniflow-light, none of which populate ncclx cvars before
// this runs. Point it at //comms/ctran/ibverbx/ib_injection to inject verbs
// failures or completion skew.
folly::Expected<folly::Unit, Error> ibvInit();

// Get a completion event from the completion channel
folly::Expected<folly::Unit, Error>
ibvGetCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context);

// Acknowledge completion events
void ibvAckCqEvents(ibv_cq* cq, unsigned int nevents);

} // namespace ibverbx
