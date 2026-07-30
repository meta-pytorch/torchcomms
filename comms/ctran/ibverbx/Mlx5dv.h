// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/Mlx5core.h"

namespace ibverbx {

class Mlx5dv {
 public:
  static folly::Expected<folly::Unit, Error> initObj(
      mlx5dv_obj* obj,
      uint64_t obj_type);

  // Query mlx5-specific device attributes via mlx5dv_query_device.
  // Caller sets the requested caps via compMask (enum
  // mlx5dv_context_comp_mask). Returns EOPNOTSUPP if libmlx5 is unavailable
  // (dynamic-loading path failed to resolve the symbol); other errors from
  // mlx5dv_query_device pass through.
  static folly::Expected<mlx5dv_context, Error> queryDevice(
      ibv_context* ctx,
      uint64_t compMask);
};

} // namespace ibverbx
