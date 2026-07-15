// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// IONIC Direct Verbs (ionic_dv) runtime loader for pipes-gda
// =============================================================================
//
// Function-pointer typedefs and the runtime function table for the AMD Pensando
// "ionic" direct-verbs API (`ionic_dv_*`), loaded at runtime from libionic.so
// via dlopen/dlsym. Mirrors the BnxtReDv.h pattern. The `ionic_dv_*` structs
// (ionic_dv_ctx / queue / cq / qp, ionic_cq_init_attr_ex) are defined in
// IonicHsi.h, which both this header and the device backend share.
//
// These DV calls extract the raw GDA handles (SQ/CQ rings, doorbell page) from
// a QP/CQ that was created with the standard ibv_create_{qp,cq}_ex on a
// parent domain whose allocator returns GPU-uncached memory for the rings.
// =============================================================================

#pragma once

#include <infiniband/verbs.h>
#include <stdbool.h>
#include <stdint.h>

#include "nic/ionic/IonicHsi.h" // @manual  (ionic_dv_* structs + ionic_cq_init_attr_ex)

#ifdef __cplusplus
extern "C" {
#endif

// Function-pointer types for dlsym resolution. Signatures match ionic_dv.h.
typedef int (*ionic_dv_get_ctx_fn)(struct ionic_dv_ctx*, struct ibv_context*);
typedef uint8_t (*ionic_dv_qp_get_udma_idx_fn)(struct ibv_qp*);
typedef int (*ionic_dv_get_cq_fn)(struct ionic_dv_cq*, struct ibv_cq*, uint8_t);
typedef int (*ionic_dv_get_qp_fn)(struct ionic_dv_qp*, struct ibv_qp*);
typedef int (*ionic_dv_pd_set_sqcmb_fn)(struct ibv_pd*, bool, bool, bool);
typedef int (*ionic_dv_pd_set_rqcmb_fn)(struct ibv_pd*, bool, bool, bool);
typedef int (*ionic_dv_pd_set_udma_mask_fn)(struct ibv_pd*, uint8_t);
typedef struct ibv_cq_ex* (*ionic_dv_create_cq_ex_fn)(
    struct ibv_context*,
    struct ibv_cq_init_attr_ex*,
    struct ionic_cq_init_attr_ex*);

// Runtime-loaded function table. `create_cq_ex` may be null if libionic lacks
// the symbol, but the ionic backend only implements the compressed CCQE
// completion queue, so CQ setup aborts (no ibv_create_cq_ex fallback) if
// absent.
struct ionic_dv_funcs {
  void* dl_handle;
  ionic_dv_get_ctx_fn get_ctx;
  ionic_dv_qp_get_udma_idx_fn qp_get_udma_idx;
  ionic_dv_get_cq_fn get_cq;
  ionic_dv_get_qp_fn get_qp;
  ionic_dv_pd_set_sqcmb_fn pd_set_sqcmb;
  ionic_dv_pd_set_rqcmb_fn pd_set_rqcmb;
  ionic_dv_pd_set_udma_mask_fn pd_set_udma_mask;
  ionic_dv_create_cq_ex_fn create_cq_ex; // optional
};

#ifdef __cplusplus
}
#endif
