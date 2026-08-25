// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PrimsAmdGdaHost - host-side prims_amd_gda APIs for AMD/HIP builds
// =============================================================================
//
// AMD-native host APIs that mirror the DOCA host surface
// (`doca_gpu_*`, `doca_verbs_*`, `doca_gpu_verbs_*`) used by
// `comms/prims/MultipeerIbgdaTransport.{h,cc}`. Implementations call
// HSA + libibverbs directly.
//
// Call-site translation `doca_* -> prims_amd_gda::prims_amd_gda_*` lives in
// `comms/prims/transport/amd/DocaCompat.h` so cross-platform call sites stay
// unchanged on the consumer side.
//
// Companion to the device-side `prims_amd_gda_*` APIs in `PrimsAmdGdaOps.h` and
// `PrimsAmdGdaDev.h`. Functions live in the `prims_amd_gda::` namespace;
// structs stay at global scope with the `prims_amd_gda_*` prefix to match the
// existing convention in `PrimsAmdGdaDev.h`.
// =============================================================================

#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>

// `<infiniband/mlx5dv.h>` is intentionally NOT included here — it's an
// implementation detail of the mlx5 backend (`PrimsAmdGdaHost.cc`). The BNXT
// backend (`PrimsAmdGdaBnxtHost.cc`) compiles the same public header against
// `bnxt_re_dv` instead. Each .cc pulls in its own vendor header.

// ===========================================================================
// Error codes
// ===========================================================================
//
// Reduced to a simple int enum; only PRIMS_AMD_GDA_SUCCESS is checked by
// `MultipeerIbgdaTransport.cc` (everything else triggers the same error
// path).

using prims_amd_gda_error_t = int;

constexpr prims_amd_gda_error_t PRIMS_AMD_GDA_SUCCESS = 0;
constexpr prims_amd_gda_error_t PRIMS_AMD_GDA_ERROR_INVALID_VALUE = 1;
constexpr prims_amd_gda_error_t PRIMS_AMD_GDA_ERROR_NO_MEMORY = 2;
constexpr prims_amd_gda_error_t PRIMS_AMD_GDA_ERROR_NOT_FOUND = 3;
constexpr prims_amd_gda_error_t PRIMS_AMD_GDA_ERROR_INITIALIZATION = 4;
constexpr prims_amd_gda_error_t PRIMS_AMD_GDA_ERROR_DRIVER = 5;

// ===========================================================================
// prims_amd_gda_gpu - GPU context handle
// ===========================================================================
//
// Wraps HSA agent + HIP device for the GPU we're targeting.

struct prims_amd_gda_gpu;

// ===========================================================================
// prims_amd_gda_verbs_* - libibverbs wrappers
// ===========================================================================

struct prims_amd_gda_verbs_qp_attr;
struct prims_amd_gda_verbs_ah_attr;
struct prims_amd_gda_verbs_gid {
  uint8_t raw[16];
};

enum prims_amd_gda_verbs_addr_type {
  PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IPv4 = 0,
  PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IPv6 = 1,
  PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IB = 2,
  PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IB_NO_GRH = 3,
};

enum prims_amd_gda_verbs_mtu_size {
  PRIMS_AMD_GDA_VERBS_MTU_SIZE_256_BYTES = IBV_MTU_256,
  PRIMS_AMD_GDA_VERBS_MTU_SIZE_512_BYTES = IBV_MTU_512,
  PRIMS_AMD_GDA_VERBS_MTU_SIZE_1K_BYTES = IBV_MTU_1024,
  PRIMS_AMD_GDA_VERBS_MTU_SIZE_2K_BYTES = IBV_MTU_2048,
  PRIMS_AMD_GDA_VERBS_MTU_SIZE_4K_BYTES = IBV_MTU_4096,
};
using prims_amd_gda_mtu = prims_amd_gda_verbs_mtu_size;
constexpr prims_amd_gda_mtu PRIMS_AMD_GDA_MTU_SIZE_256_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_256_BYTES;
constexpr prims_amd_gda_mtu PRIMS_AMD_GDA_MTU_SIZE_512_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_512_BYTES;
constexpr prims_amd_gda_mtu PRIMS_AMD_GDA_MTU_SIZE_1024_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_1K_BYTES;
constexpr prims_amd_gda_mtu PRIMS_AMD_GDA_MTU_SIZE_2048_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_2K_BYTES;
constexpr prims_amd_gda_mtu PRIMS_AMD_GDA_MTU_SIZE_4096_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_4K_BYTES;

enum prims_amd_gda_verbs_qp_state {
  PRIMS_AMD_GDA_VERBS_QP_STATE_RST = IBV_QPS_RESET,
  PRIMS_AMD_GDA_VERBS_QP_STATE_INIT = IBV_QPS_INIT,
  PRIMS_AMD_GDA_VERBS_QP_STATE_RTR = IBV_QPS_RTR,
  PRIMS_AMD_GDA_VERBS_QP_STATE_RTS = IBV_QPS_RTS,
};

enum prims_amd_gda_verbs_qp_atomic_mode {
  PRIMS_AMD_GDA_VERBS_QP_ATOMIC_MODE_NONE = 0,
  PRIMS_AMD_GDA_VERBS_QP_ATOMIC_MODE_IB_SPEC = 1,
};

// QP attribute mask flags. `prims_amd_gda_verbs_qp_modify` ORs the caller's
// mask (translated to IBV_QP_* space) with the IBV mask accumulated by the
// individual setters; passing a flag here ensures the corresponding IBV
// attribute is applied even if no setter was called (e.g. zero-init
// `pkey_index`, mirroring NVIDIA DOCA's rst2init semantics).
enum prims_amd_gda_verbs_qp_attr_mask {
  PRIMS_AMD_GDA_VERBS_QP_ATTR_NEXT_STATE = 1 << 0,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_PKEY_INDEX = 1 << 1,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_PORT_NUM = 1 << 2,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE = 1 << 3,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_READ = 1 << 4,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_ATOMIC = 1 << 5,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_PATH_MTU = 1 << 6,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_DEST_QP_NUM = 1 << 7,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_RQ_PSN = 1 << 8,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_AH_ATTR = 1 << 9,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_MIN_RNR_TIMER = 1 << 10,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_SQ_PSN = 1 << 11,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_ACK_TIMEOUT = 1 << 12,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_RETRY_CNT = 1 << 13,
  PRIMS_AMD_GDA_VERBS_QP_ATTR_RNR_RETRY = 1 << 14,
};

// ===========================================================================
// prims_amd_gda_gpu_verbs_* - GPU-side QP / CQ creation
// ===========================================================================
//
// Manual orchestration via `ibv_create_qp` + `mlx5dv_init_obj` +
// `hsa_amd_memory_lock`.

struct prims_amd_gda_gpu_verbs_qp_init_attr_hl {
  prims_amd_gda_gpu* gpu_dev{nullptr};
  ibv_pd* ibpd{nullptr};
  uint32_t sq_nwqe{0};
  int nic_handler{0};
  int mreg_type{0};
};

constexpr int PRIMS_AMD_GDA_GPUNETIO_VERBS_NIC_HANDLER_AUTO = 0;
constexpr int PRIMS_AMD_GDA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT = 0;

// Forward-declared by the existing prims_amd_gda QP type in
// `amd/prims_amd_gda/PrimsAmdGdaDev.h`.
struct prims_amd_gda_gpu_dev_verbs_qp;

// Opaque GPU-verbs context handle. Refers back to the host-side QP handle
// via a tagged self-pointer.
struct prims_amd_gda_gpu_verbs_qp;

// Single-QP host handle that owns both host-side libibverbs objects and
// the device-mapped GPU-side QP descriptor.
struct prims_amd_gda_gpu_verbs_qp_hl {
  ibv_qp* qp{nullptr};
  ibv_cq* cq{nullptr};
  prims_amd_gda_gpu_dev_verbs_qp* gpu_qp{nullptr};
  // Tagged self-pointer; recovered by `prims_amd_gda_gpu_verbs_get_qp_dev`.
  prims_amd_gda_gpu_verbs_qp* qp_gverbs{nullptr};
  // Auxiliary AMD-only resources (UAR, registered host buffers, etc.)
  // are tracked in an opaque per-QP control block.
  void* amd_internal{nullptr};
};

// QP "group" = primary + companion QP (used for compound put+signal+counter).
struct prims_amd_gda_gpu_verbs_qp_group_hl {
  prims_amd_gda_gpu_verbs_qp_hl qp_main;
  prims_amd_gda_gpu_verbs_qp_hl qp_companion;
};

// ===========================================================================
// Function declarations
// ===========================================================================

namespace prims_amd_gda {

// --- prims_amd_gda_gpu lifecycle ---
prims_amd_gda_error_t prims_amd_gda_gpu_create(
    const char* gpu_pci_bus_id,
    prims_amd_gda_gpu** out_gpu);
prims_amd_gda_error_t prims_amd_gda_gpu_destroy(prims_amd_gda_gpu* gpu);
prims_amd_gda_error_t prims_amd_gda_gpu_mem_alloc(
    prims_amd_gda_gpu* gpu,
    std::size_t size,
    std::size_t alignment,
    int mem_type,
    int access_type,
    void** out_ptr,
    void** out_gpu_ptr);

// --- prims_amd_gda_verbs ibv wrappers ---
//
// Declared (not inlined) so each backend chooses how to dispatch:
//   - PrimsAmdGdaHost.cc (mlx5):      direct calls into fbcode static
//   libibverbs
//   - PrimsAmdGdaBnxtHost.cc (BNXT):  routed through dlopen'd system libibverbs
//                                  (PABI 34, supports kernel uverbs ABI 8)
// The fbcode static libibverbs (PABI 59) cannot register the system BNXT
// provider, so BNXT must use the system library throughout.
prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_get_device_list(
    int* num_devices,
    ibv_device*** out_list);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_free_device_list(
    ibv_device** list);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_get_device_name(
    ibv_device* dev,
    const char** out_name);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_open_device(
    ibv_device* dev,
    ibv_context** out_ctx);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_close_device(
    ibv_context* ctx);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_alloc_pd(
    ibv_context* ctx,
    ibv_pd** out_pd);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_dealloc_pd(ibv_pd* pd);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_query_device(
    ibv_context* ctx,
    ibv_device_attr* attr);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_query_port(
    ibv_context* ctx,
    uint8_t port,
    ibv_port_attr* attr);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_query_gid(
    ibv_context* ctx,
    uint8_t port,
    int index,
    union ibv_gid* gid);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_reg_mr(
    ibv_pd* pd,
    void* addr,
    std::size_t length,
    int access,
    ibv_mr** out_mr);

prims_amd_gda_error_t prims_amd_gda_verbs_wrapper_ibv_dereg_mr(ibv_mr* mr);

// --- prims_amd_gda_verbs QP attribute setters ---
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_create(
    prims_amd_gda_verbs_qp_attr** out_attr);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_destroy(
    prims_amd_gda_verbs_qp_attr* attr);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_next_state(
    prims_amd_gda_verbs_qp_attr* attr,
    prims_amd_gda_verbs_qp_state state);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_path_mtu(
    prims_amd_gda_verbs_qp_attr* attr,
    prims_amd_gda_mtu mtu);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_port_num(
    prims_amd_gda_verbs_qp_attr* attr,
    uint8_t port);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_dest_qp_num(
    prims_amd_gda_verbs_qp_attr* attr,
    uint32_t dest_qp_num);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_rq_psn(
    prims_amd_gda_verbs_qp_attr* attr,
    uint32_t psn);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_sq_psn(
    prims_amd_gda_verbs_qp_attr* attr,
    uint32_t psn);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_min_rnr_timer(
    prims_amd_gda_verbs_qp_attr* attr,
    uint8_t v);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_ack_timeout(
    prims_amd_gda_verbs_qp_attr* attr,
    uint8_t v);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_retry_cnt(
    prims_amd_gda_verbs_qp_attr* attr,
    uint8_t v);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_rnr_retry(
    prims_amd_gda_verbs_qp_attr* attr,
    uint8_t v);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_allow_remote_read(
    prims_amd_gda_verbs_qp_attr* attr,
    bool allow);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_allow_remote_write(
    prims_amd_gda_verbs_qp_attr* attr,
    bool allow);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_allow_remote_atomic(
    prims_amd_gda_verbs_qp_attr* attr,
    int atomic_mode);
prims_amd_gda_error_t prims_amd_gda_verbs_qp_attr_set_ah_attr(
    prims_amd_gda_verbs_qp_attr* attr,
    prims_amd_gda_verbs_ah_attr* ah_attr);

// --- prims_amd_gda_verbs address handle attribute setters ---
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_create(
    ibv_context* ctx,
    prims_amd_gda_verbs_ah_attr** out_attr);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_destroy(
    prims_amd_gda_verbs_ah_attr* attr);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_addr_type(
    prims_amd_gda_verbs_ah_attr* attr,
    prims_amd_gda_verbs_addr_type t);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_dlid(
    prims_amd_gda_verbs_ah_attr* attr,
    uint16_t dlid);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_gid(
    prims_amd_gda_verbs_ah_attr* attr,
    const prims_amd_gda_verbs_gid& gid);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_sgid_index(
    prims_amd_gda_verbs_ah_attr* attr,
    int idx);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_hop_limit(
    prims_amd_gda_verbs_ah_attr* attr,
    uint8_t hop);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_sl(
    prims_amd_gda_verbs_ah_attr* attr,
    uint8_t sl);
prims_amd_gda_error_t prims_amd_gda_verbs_ah_attr_set_traffic_class(
    prims_amd_gda_verbs_ah_attr* attr,
    uint8_t tc);

// --- prims_amd_gda_verbs QP modify / query ---
prims_amd_gda_error_t prims_amd_gda_verbs_qp_modify(
    ibv_qp* qp,
    prims_amd_gda_verbs_qp_attr* attr,
    int attr_mask = 0);
inline uint32_t prims_amd_gda_verbs_qp_get_qpn(ibv_qp* qp) {
  return qp->qp_num;
}

inline std::size_t prims_amd_gda_verbs_mtu_size_in_bytes(prims_amd_gda_mtu m) {
  switch (m) {
    case PRIMS_AMD_GDA_MTU_SIZE_256_BYTES:
      return 256;
    case PRIMS_AMD_GDA_MTU_SIZE_512_BYTES:
      return 512;
    case PRIMS_AMD_GDA_MTU_SIZE_1024_BYTES:
      return 1024;
    case PRIMS_AMD_GDA_MTU_SIZE_2048_BYTES:
      return 2048;
    case PRIMS_AMD_GDA_MTU_SIZE_4096_BYTES:
      return 4096;
  }
  return 4096;
}

// --- prims_amd_gda_gpu_verbs QP creation/destruction ---
prims_amd_gda_error_t prims_amd_gda_gpu_verbs_create_qp_hl(
    const prims_amd_gda_gpu_verbs_qp_init_attr_hl* attr,
    prims_amd_gda_gpu_verbs_qp_hl** out_qp);
prims_amd_gda_error_t prims_amd_gda_gpu_verbs_destroy_qp_hl(
    prims_amd_gda_gpu_verbs_qp_hl* qp);
prims_amd_gda_error_t prims_amd_gda_gpu_verbs_create_qp_group_hl(
    const prims_amd_gda_gpu_verbs_qp_init_attr_hl* attr,
    prims_amd_gda_gpu_verbs_qp_group_hl** out_grp);
prims_amd_gda_error_t prims_amd_gda_gpu_verbs_destroy_qp_group_hl(
    prims_amd_gda_gpu_verbs_qp_group_hl* g);

// Get the device-side QP handle for a `prims_amd_gda_gpu_verbs_qp`.
prims_amd_gda_error_t prims_amd_gda_gpu_verbs_get_qp_dev(
    prims_amd_gda_gpu_verbs_qp* qp_gverbs,
    prims_amd_gda_gpu_dev_verbs_qp** out_dev_qp);

} // namespace prims_amd_gda

#endif // __HIP_PLATFORM_AMD__
