// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/Mlx5dv.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

folly::Expected<folly::Unit, Error> Mlx5dv::initObj(
    mlx5dv_obj* obj,
    uint64_t obj_type) {
  int rc = ibvSymbols.mlx5dv_internal_init_obj(obj, obj_type);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

} // namespace ibverbx
