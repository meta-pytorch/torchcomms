// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// [META] Temporary shim for NCCL 2.30 → DOCA 2.x compatibility.
//
// NCCL 2.30 calls `doca_verbs_qp_attr_set_max_rd_atomic` and
// `doca_verbs_qp_attr_set_max_dest_rd_atomic` from gin_host_gdaki.cc, but
// those setters were added by NVIDIA in DOCA 3.x. Meta's vendored
// third-party/nvidia-doca/gpunetio/ source bundle predates them and the
// 2.x `struct doca_verbs_qp_attr` has no fields to store the values.
//
// This shim provides stub definitions so v2_30 NCCLX links cleanly. The
// stubs are no-ops — they validate the pointer and return DOCA_SUCCESS,
// but the underlying `doca_verbs_qp_modify` does not honor the values
// (the corresponding DOCA_VERBS_QP_ATTR_MAX_(QP|DEST)_RD_ATOMIC attr_mask
// bits are also defined as 0 in v2_30's local doca_verbs.h to keep the
// 2.x validity check from rejecting them).
//
// Net runtime effect: NCCLX 2.30 silently uses the MLX5 driver default
// for `max_rd_atomic` / `max_dest_rd_atomic` (typically 1). This is a
// known performance trade-off, accepted because NCCLX 2.30 is not yet in
// active use; the long-term fix is for `comms/ctran/prims` to migrate off the
// shared 2.x doca-gpunetio shim layer so v2_30+ can consume a real DOCA
// 3.x prebuilt that supplies these symbols natively.
//
// DELETE THIS SHIM once `comms/ctran/prims` migrates off `doca_gpunetio_dl`
// AND v2_30+ NCCLX can link against a 3.x doca verbs library.

#include <stddef.h>
#include <stdint.h>

#include "host/doca_verbs.h"

extern "C" {

doca_error_t doca_verbs_qp_attr_set_max_rd_atomic(
    struct doca_verbs_qp_attr* verbs_qp_attr,
    uint8_t /* max_rd_atomic */) {
  if (verbs_qp_attr == nullptr) {
    return DOCA_ERROR_INVALID_VALUE;
  }
  return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_attr_set_max_dest_rd_atomic(
    struct doca_verbs_qp_attr* verbs_qp_attr,
    uint8_t /* max_dest_rd_atomic */) {
  if (verbs_qp_attr == nullptr) {
    return DOCA_ERROR_INVALID_VALUE;
  }
  return DOCA_SUCCESS;
}

} // extern "C"
