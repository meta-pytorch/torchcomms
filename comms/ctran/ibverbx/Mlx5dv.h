// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>

#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

class Mlx5dv {
 public:
  static folly::Expected<folly::Unit, Error> initObj(
      mlx5dv_obj* obj,
      uint64_t obj_type);
};

} // namespace ibverbx
