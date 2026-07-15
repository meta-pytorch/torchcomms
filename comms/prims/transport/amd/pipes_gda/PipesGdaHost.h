// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PipesGdaHost - host-side pipes_gda APIs for AMD/HIP builds
// =============================================================================
//
// AMD-native host APIs that mirror the DOCA host surface
// (`doca_gpu_*`, `doca_verbs_*`, `doca_gpu_verbs_*`) used by
// `comms/prims/MultipeerIbgdaTransport.{h,cc}`. Implementations call
// HSA + libibverbs directly.
//
// Call-site translation `doca_* -> pipes_gda::pipes_gda_*` lives in
// `comms/prims/transport/amd/DocaCompat.h` so cross-platform call sites stay
// unchanged on the consumer side.
//
// Companion to the device-side `pipes_gda_*` APIs in `PipesGdaOps.h` and
// `PipesGdaDev.h`. Functions live in the `pipes_gda::` namespace; structs
// stay at global scope with the `pipes_gda_*` prefix to match the existing
// convention in `PipesGdaDev.h`.
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
// implementation detail of the mlx5 backend (`PipesGdaHost.cc`). The BNXT
// backend (`PipesGdaBnxtHost.cc`) compiles the same public header against
// `bnxt_re_dv` instead. Each .cc pulls in its own vendor header.

// ===========================================================================
// Error codes
// ===========================================================================
//
// Reduced to a simple int enum; only PIPES_GDA_SUCCESS is checked by
// `MultipeerIbgdaTransport.cc` (everything else triggers the same error
// path).

using pipes_gda_error_t = int;

constexpr pipes_gda_error_t PIPES_GDA_SUCCESS = 0;
constexpr pipes_gda_error_t PIPES_GDA_ERROR_INVALID_VALUE = 1;
constexpr pipes_gda_error_t PIPES_GDA_ERROR_NO_MEMORY = 2;
constexpr pipes_gda_error_t PIPES_GDA_ERROR_NOT_FOUND = 3;
constexpr pipes_gda_error_t PIPES_GDA_ERROR_INITIALIZATION = 4;
constexpr pipes_gda_error_t PIPES_GDA_ERROR_DRIVER = 5;

// ===========================================================================
// pipes_gda_gpu - GPU context handle
// ===========================================================================
//
// Wraps HSA agent + HIP device for the GPU we're targeting.

struct pipes_gda_gpu;

// ===========================================================================
// pipes_gda_verbs_* - libibverbs wrappers
// ===========================================================================

struct pipes_gda_verbs_qp_attr;
struct pipes_gda_verbs_ah_attr;
struct pipes_gda_verbs_gid {
  uint8_t raw[16];
};

enum pipes_gda_verbs_addr_type {
  PIPES_GDA_VERBS_ADDR_TYPE_IPv4 = 0,
  PIPES_GDA_VERBS_ADDR_TYPE_IPv6 = 1,
  PIPES_GDA_VERBS_ADDR_TYPE_IB = 2,
  PIPES_GDA_VERBS_ADDR_TYPE_IB_NO_GRH = 3,
};

enum pipes_gda_verbs_mtu_size {
  PIPES_GDA_VERBS_MTU_SIZE_256_BYTES = IBV_MTU_256,
  PIPES_GDA_VERBS_MTU_SIZE_512_BYTES = IBV_MTU_512,
  PIPES_GDA_VERBS_MTU_SIZE_1K_BYTES = IBV_MTU_1024,
  PIPES_GDA_VERBS_MTU_SIZE_2K_BYTES = IBV_MTU_2048,
  PIPES_GDA_VERBS_MTU_SIZE_4K_BYTES = IBV_MTU_4096,
};
using pipes_gda_mtu = pipes_gda_verbs_mtu_size;
constexpr pipes_gda_mtu PIPES_GDA_MTU_SIZE_256_BYTES =
    PIPES_GDA_VERBS_MTU_SIZE_256_BYTES;
constexpr pipes_gda_mtu PIPES_GDA_MTU_SIZE_512_BYTES =
    PIPES_GDA_VERBS_MTU_SIZE_512_BYTES;
constexpr pipes_gda_mtu PIPES_GDA_MTU_SIZE_1024_BYTES =
    PIPES_GDA_VERBS_MTU_SIZE_1K_BYTES;
constexpr pipes_gda_mtu PIPES_GDA_MTU_SIZE_2048_BYTES =
    PIPES_GDA_VERBS_MTU_SIZE_2K_BYTES;
constexpr pipes_gda_mtu PIPES_GDA_MTU_SIZE_4096_BYTES =
    PIPES_GDA_VERBS_MTU_SIZE_4K_BYTES;

enum pipes_gda_verbs_qp_state {
  PIPES_GDA_VERBS_QP_STATE_RST = IBV_QPS_RESET,
  PIPES_GDA_VERBS_QP_STATE_INIT = IBV_QPS_INIT,
  PIPES_GDA_VERBS_QP_STATE_RTR = IBV_QPS_RTR,
  PIPES_GDA_VERBS_QP_STATE_RTS = IBV_QPS_RTS,
};

enum pipes_gda_verbs_qp_atomic_mode {
  PIPES_GDA_VERBS_QP_ATOMIC_MODE_NONE = 0,
  PIPES_GDA_VERBS_QP_ATOMIC_MODE_IB_SPEC = 1,
};

// QP attribute mask flags. `pipes_gda_verbs_qp_modify` ORs the caller's
// mask (translated to IBV_QP_* space) with the IBV mask accumulated by the
// individual setters; passing a flag here ensures the corresponding IBV
// attribute is applied even if no setter was called (e.g. zero-init
// `pkey_index`, mirroring NVIDIA DOCA's rst2init semantics).
enum pipes_gda_verbs_qp_attr_mask {
  PIPES_GDA_VERBS_QP_ATTR_NEXT_STATE = 1 << 0,
  PIPES_GDA_VERBS_QP_ATTR_PKEY_INDEX = 1 << 1,
  PIPES_GDA_VERBS_QP_ATTR_PORT_NUM = 1 << 2,
  PIPES_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE = 1 << 3,
  PIPES_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_READ = 1 << 4,
  PIPES_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_ATOMIC = 1 << 5,
  PIPES_GDA_VERBS_QP_ATTR_PATH_MTU = 1 << 6,
  PIPES_GDA_VERBS_QP_ATTR_DEST_QP_NUM = 1 << 7,
  PIPES_GDA_VERBS_QP_ATTR_RQ_PSN = 1 << 8,
  PIPES_GDA_VERBS_QP_ATTR_AH_ATTR = 1 << 9,
  PIPES_GDA_VERBS_QP_ATTR_MIN_RNR_TIMER = 1 << 10,
  PIPES_GDA_VERBS_QP_ATTR_SQ_PSN = 1 << 11,
  PIPES_GDA_VERBS_QP_ATTR_ACK_TIMEOUT = 1 << 12,
  PIPES_GDA_VERBS_QP_ATTR_RETRY_CNT = 1 << 13,
  PIPES_GDA_VERBS_QP_ATTR_RNR_RETRY = 1 << 14,
};

// ===========================================================================
// pipes_gda_gpu_verbs_* - GPU-side QP / CQ creation
// ===========================================================================
//
// Manual orchestration via `ibv_create_qp` + `mlx5dv_init_obj` +
// `hsa_amd_memory_lock`.

struct pipes_gda_gpu_verbs_qp_init_attr_hl {
  pipes_gda_gpu* gpu_dev{nullptr};
  ibv_pd* ibpd{nullptr};
  uint32_t sq_nwqe{0};
  int nic_handler{0};
  int mreg_type{0};
};

constexpr int PIPES_GDA_GPUNETIO_VERBS_NIC_HANDLER_AUTO = 0;
constexpr int PIPES_GDA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT = 0;

// Forward-declared by the existing pipes_gda QP type in
// `amd/pipes_gda/PipesGdaDev.h`.
struct pipes_gda_gpu_dev_verbs_qp;

// Opaque GPU-verbs context handle. Refers back to the host-side QP handle
// via a tagged self-pointer.
struct pipes_gda_gpu_verbs_qp;

// Single-QP host handle that owns both host-side libibverbs objects and
// the device-mapped GPU-side QP descriptor.
struct pipes_gda_gpu_verbs_qp_hl {
  ibv_qp* qp{nullptr};
  ibv_cq* cq{nullptr};
  pipes_gda_gpu_dev_verbs_qp* gpu_qp{nullptr};
  // Tagged self-pointer; recovered by `pipes_gda_gpu_verbs_get_qp_dev`.
  pipes_gda_gpu_verbs_qp* qp_gverbs{nullptr};
  // Auxiliary AMD-only resources (UAR, registered host buffers, etc.)
  // are tracked in an opaque per-QP control block.
  void* amd_internal{nullptr};
};

// QP "group" = primary + companion QP (used for compound put+signal+counter).
struct pipes_gda_gpu_verbs_qp_group_hl {
  pipes_gda_gpu_verbs_qp_hl qp_main;
  pipes_gda_gpu_verbs_qp_hl qp_companion;
};

// ===========================================================================
// Function declarations
// ===========================================================================

namespace pipes_gda {

// --- pipes_gda_gpu lifecycle ---
pipes_gda_error_t pipes_gda_gpu_create(
    const char* gpu_pci_bus_id,
    pipes_gda_gpu** out_gpu);
pipes_gda_error_t pipes_gda_gpu_destroy(pipes_gda_gpu* gpu);
pipes_gda_error_t pipes_gda_gpu_mem_alloc(
    pipes_gda_gpu* gpu,
    std::size_t size,
    std::size_t alignment,
    int mem_type,
    int access_type,
    void** out_ptr,
    void** out_gpu_ptr);

// --- pipes_gda_verbs ibv wrappers ---
//
// Declared (not inlined) so each backend chooses how to dispatch:
//   - PipesGdaHost.cc (mlx5):      direct calls into fbcode static libibverbs
//   - PipesGdaBnxtHost.cc (BNXT):  routed through dlopen'd system libibverbs
//                                  (PABI 34, supports kernel uverbs ABI 8)
// The fbcode static libibverbs (PABI 59) cannot register the system BNXT
// provider, so BNXT must use the system library throughout.
pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_get_device_list(
    int* num_devices,
    ibv_device*** out_list);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_free_device_list(
    ibv_device** list);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_get_device_name(
    ibv_device* dev,
    const char** out_name);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_open_device(
    ibv_device* dev,
    ibv_context** out_ctx);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_close_device(ibv_context* ctx);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_alloc_pd(
    ibv_context* ctx,
    ibv_pd** out_pd);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_dealloc_pd(ibv_pd* pd);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_query_device(
    ibv_context* ctx,
    ibv_device_attr* attr);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_query_port(
    ibv_context* ctx,
    uint8_t port,
    ibv_port_attr* attr);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_query_gid(
    ibv_context* ctx,
    uint8_t port,
    int index,
    union ibv_gid* gid);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_reg_mr(
    ibv_pd* pd,
    void* addr,
    std::size_t length,
    int access,
    ibv_mr** out_mr);

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_dereg_mr(ibv_mr* mr);

// --- pipes_gda_verbs QP attribute setters ---
pipes_gda_error_t pipes_gda_verbs_qp_attr_create(
    pipes_gda_verbs_qp_attr** out_attr);
pipes_gda_error_t pipes_gda_verbs_qp_attr_destroy(
    pipes_gda_verbs_qp_attr* attr);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_next_state(
    pipes_gda_verbs_qp_attr* attr,
    pipes_gda_verbs_qp_state state);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_path_mtu(
    pipes_gda_verbs_qp_attr* attr,
    pipes_gda_mtu mtu);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_port_num(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t port);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_dest_qp_num(
    pipes_gda_verbs_qp_attr* attr,
    uint32_t dest_qp_num);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_rq_psn(
    pipes_gda_verbs_qp_attr* attr,
    uint32_t psn);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_sq_psn(
    pipes_gda_verbs_qp_attr* attr,
    uint32_t psn);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_min_rnr_timer(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_ack_timeout(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_retry_cnt(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_rnr_retry(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_allow_remote_read(
    pipes_gda_verbs_qp_attr* attr,
    bool allow);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_allow_remote_write(
    pipes_gda_verbs_qp_attr* attr,
    bool allow);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_allow_remote_atomic(
    pipes_gda_verbs_qp_attr* attr,
    int atomic_mode);
pipes_gda_error_t pipes_gda_verbs_qp_attr_set_ah_attr(
    pipes_gda_verbs_qp_attr* attr,
    pipes_gda_verbs_ah_attr* ah_attr);

// --- pipes_gda_verbs address handle attribute setters ---
pipes_gda_error_t pipes_gda_verbs_ah_attr_create(
    ibv_context* ctx,
    pipes_gda_verbs_ah_attr** out_attr);
pipes_gda_error_t pipes_gda_verbs_ah_attr_destroy(
    pipes_gda_verbs_ah_attr* attr);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_addr_type(
    pipes_gda_verbs_ah_attr* attr,
    pipes_gda_verbs_addr_type t);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_dlid(
    pipes_gda_verbs_ah_attr* attr,
    uint16_t dlid);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_gid(
    pipes_gda_verbs_ah_attr* attr,
    const pipes_gda_verbs_gid& gid);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_sgid_index(
    pipes_gda_verbs_ah_attr* attr,
    int idx);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_hop_limit(
    pipes_gda_verbs_ah_attr* attr,
    uint8_t hop);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_sl(
    pipes_gda_verbs_ah_attr* attr,
    uint8_t sl);
pipes_gda_error_t pipes_gda_verbs_ah_attr_set_traffic_class(
    pipes_gda_verbs_ah_attr* attr,
    uint8_t tc);

// --- pipes_gda_verbs QP modify / query ---
pipes_gda_error_t pipes_gda_verbs_qp_modify(
    ibv_qp* qp,
    pipes_gda_verbs_qp_attr* attr,
    int attr_mask = 0);
inline uint32_t pipes_gda_verbs_qp_get_qpn(ibv_qp* qp) {
  return qp->qp_num;
}

inline std::size_t pipes_gda_verbs_mtu_size_in_bytes(pipes_gda_mtu m) {
  switch (m) {
    case PIPES_GDA_MTU_SIZE_256_BYTES:
      return 256;
    case PIPES_GDA_MTU_SIZE_512_BYTES:
      return 512;
    case PIPES_GDA_MTU_SIZE_1024_BYTES:
      return 1024;
    case PIPES_GDA_MTU_SIZE_2048_BYTES:
      return 2048;
    case PIPES_GDA_MTU_SIZE_4096_BYTES:
      return 4096;
  }
  return 4096;
}

// --- pipes_gda_gpu_verbs QP creation/destruction ---
pipes_gda_error_t pipes_gda_gpu_verbs_create_qp_hl(
    const pipes_gda_gpu_verbs_qp_init_attr_hl* attr,
    pipes_gda_gpu_verbs_qp_hl** out_qp);
pipes_gda_error_t pipes_gda_gpu_verbs_destroy_qp_hl(
    pipes_gda_gpu_verbs_qp_hl* qp);
pipes_gda_error_t pipes_gda_gpu_verbs_create_qp_group_hl(
    const pipes_gda_gpu_verbs_qp_init_attr_hl* attr,
    pipes_gda_gpu_verbs_qp_group_hl** out_grp);
pipes_gda_error_t pipes_gda_gpu_verbs_destroy_qp_group_hl(
    pipes_gda_gpu_verbs_qp_group_hl* g);

// Get the device-side QP handle for a `pipes_gda_gpu_verbs_qp`.
pipes_gda_error_t pipes_gda_gpu_verbs_get_qp_dev(
    pipes_gda_gpu_verbs_qp* qp_gverbs,
    pipes_gda_gpu_dev_verbs_qp** out_dev_qp);

} // namespace pipes_gda

#endif // __HIP_PLATFORM_AMD__
