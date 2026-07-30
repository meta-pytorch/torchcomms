// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/Mlx5dv.h"

#include <fmt/core.h>

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

folly::Expected<mlx5dv_context, Error> Mlx5dv::queryDevice(
    ibv_context* ctx,
    uint64_t compMask) {
  if (ctx == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "mlx5dv_query_device: null ibv_context"));
  }
  if (ibvSymbols.mlx5dv_internal_query_device == nullptr) {
    return folly::makeUnexpected(
        Error(EOPNOTSUPP, "mlx5dv_query_device symbol unavailable"));
  }
  mlx5dv_context dvCtx{};
  dvCtx.comp_mask = compMask;
  // mlx5dv_query_device returns errno-style values on failure (positive
  // errno directly, per rdma-core convention for this symbol) — pass rc
  // through unchanged rather than reading errno separately.
  const int rc = ibvSymbols.mlx5dv_internal_query_device(ctx, &dvCtx);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc, "mlx5dv_query_device failed"));
  }
  // Driver silently omits caps if firmware/kernel is too old — a partial
  // response looks successful but leaves requested fields unpopulated.
  if ((dvCtx.comp_mask & compMask) != compMask) {
    return folly::makeUnexpected(Error(
        EOPNOTSUPP,
        fmt::format(
            "mlx5dv_query_device: driver did not populate requested caps "
            "(requested=0x{:x} got=0x{:x})",
            compMask,
            dvCtx.comp_mask)));
  }
  return dvCtx;
}

} // namespace ibverbx
