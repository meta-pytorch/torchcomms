// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// DocaCompat - DOCA-shaped compat shim for AMD/HIP builds
// =============================================================================
//
// Provides `doca_*` device-side type aliases, constants, and function
// wrappers that forward to the DOCA-aligned `prims_amd_gda_*` overloads in
// `amd/prims_amd_gda/PrimsAmdGdaOps.h`. Included by
// `comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh` when
// `__HIP_PLATFORM_AMD__` is defined, instead of the NVIDIA DOCA headers
// (`<device/doca_gpunetio_dev_verbs_*.cuh>` + `DocaVerbsUtils.cuh`).
//
// Each `doca_X` shim is a **pure name-prefix forward** to
// `prims_amd_gda::prims_amd_gda_X` with identical arguments and template
// parameters. The underlying `prims_amd_gda_*` DOCA-aligned overload (added
// in Phase 1) handles the parameter shape — accepts `opcode`/`lkey_id`
// and DOCA template selectors, supplies the NicBackend internally via a
// stack-local `ActiveNicBackend nic{}`. So this shim has zero logic;
// it's just a name table.
//
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

// AMD verbs/nic headers are exported by
// //comms/prims/transport/amd:prims_amd_gda_device with header_namespace = "",
// so they are included without the `comms/prims/transport/amd/` prefix.
#include "nic/NicSelector.h" // @manual
#include "prims_amd_gda/PrimsAmdGdaDev.h" // @manual
#include "prims_amd_gda/PrimsAmdGdaOps.h" // @manual
#include "prims_amd_gda/PrimsAmdGdaShared.h" // @manual

// =============================================================================
// Type aliases
// =============================================================================
//
// prims_amd_gda struct types are declared at global scope in
// `amd/prims_amd_gda/PrimsAmdGdaDev.h` (legacy C-compatible structs), not
// inside the `prims_amd_gda` namespace. Re-export with the `doca_*` names so
// call sites in `P2pIbgdaTransportDevice.cuh` resolve identically on both
// platforms.

using doca_gpu_dev_verbs_qp = ::prims_amd_gda_gpu_dev_verbs_qp;
using doca_gpu_dev_verbs_addr = ::prims_amd_gda_gpu_dev_verbs_addr;
using doca_gpu_dev_verbs_wqe = ::prims_amd_gda_gpu_dev_verbs_wqe;
using doca_gpu_dev_verbs_cq = ::prims_amd_gda_gpu_dev_verbs_cq;
using doca_gpu_dev_verbs_ticket_t = uint64_t;
using doca_gpu_dev_verbs_wqe_ctrl_flags = uint8_t;

// =============================================================================
// Constant aliases
// =============================================================================
//
// DOCA enum constants used by `P2pIbgdaTransportDevice.cuh`. Compile-time
// selectors that NVIDIA uses become inert numeric placeholders here (the
// DOCA-aligned `prims_amd_gda_*` overloads ignore the template values).
// Opcode constants and WQE-control flags forward to their `PRIMS_AMD_GDA_*`
// equivalents (same bit values).

constexpr int DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU = 0;
constexpr int DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO = 0;
constexpr int DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU = 0;
constexpr int DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD = 0;
constexpr int DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD = 0;
constexpr int DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD = 0;

constexpr uint8_t DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE =
    PRIMS_AMD_GDA_IB_MLX5_OPCODE_RDMA_WRITE;
constexpr uint8_t DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA =
    PRIMS_AMD_GDA_IB_MLX5_OPCODE_ATOMIC_FA;
constexpr uint8_t DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE =
    PRIMS_AMD_GDA_IB_MLX5_WQE_CTRL_CQ_UPDATE;
constexpr uint8_t DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FENCE =
    PRIMS_AMD_GDA_IB_MLX5_WQE_CTRL_FENCE;

// =============================================================================
// Function shims — pure name-prefix forwarders
// =============================================================================

template <int MODE = 0>
__device__ __forceinline__ uint64_t doca_gpu_dev_verbs_reserve_wq_slots(
    doca_gpu_dev_verbs_qp* qp,
    uint32_t numSlots) {
  return prims_amd_gda::prims_amd_gda_gpu_dev_verbs_reserve_wq_slots<MODE>(
      qp, numSlots);
}

__device__ __forceinline__ doca_gpu_dev_verbs_wqe*
doca_gpu_dev_verbs_get_wqe_ptr(doca_gpu_dev_verbs_qp* qp, uint64_t wqeIdx) {
  return prims_amd_gda::prims_amd_gda_gpu_dev_verbs_get_wqe_ptr(qp, wqeIdx);
}

__device__ __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_nop(
    doca_gpu_dev_verbs_qp* qp,
    doca_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t ctrlFlags) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_wqe_prepare_nop(
      qp, wqe, wqeIdx, ctrlFlags);
}

__device__ __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_write(
    doca_gpu_dev_verbs_qp* qp,
    doca_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t opcode,
    uint8_t ctrlFlags,
    uint32_t lkey_id,
    uint64_t remoteAddr,
    uint32_t remoteKey,
    uint64_t localAddr,
    uint32_t localKey,
    uint32_t size) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_wqe_prepare_write(
      qp,
      wqe,
      wqeIdx,
      opcode,
      ctrlFlags,
      lkey_id,
      remoteAddr,
      remoteKey,
      localAddr,
      localKey,
      size);
}

__device__ __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_atomic(
    doca_gpu_dev_verbs_qp* qp,
    doca_gpu_dev_verbs_wqe* wqe,
    uint64_t wqeIdx,
    uint8_t opcode,
    uint8_t ctrlFlags,
    uint64_t remoteAddr,
    uint32_t remoteKey,
    uint64_t localAddr,
    uint32_t localKey,
    uint32_t size,
    uint64_t addVal,
    uint64_t compareVal) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_wqe_prepare_atomic(
      qp,
      wqe,
      wqeIdx,
      opcode,
      ctrlFlags,
      remoteAddr,
      remoteKey,
      localAddr,
      localKey,
      size,
      addVal,
      compareVal);
}

template <int MODE = 0>
__device__ __forceinline__ void doca_gpu_dev_verbs_mark_wqes_ready(
    doca_gpu_dev_verbs_qp* qp,
    uint64_t firstIdx,
    uint64_t lastIdx) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_mark_wqes_ready<MODE>(
      qp, firstIdx, lastIdx);
}

template <int MODE = 0, int SCOPE = 0, int HANDLER = 0>
__device__ __forceinline__ void doca_gpu_dev_verbs_submit(
    doca_gpu_dev_verbs_qp* qp,
    uint64_t nextWqeIdx) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_submit<MODE, SCOPE, HANDLER>(
      qp, nextWqeIdx);
}

template <int MODE = 0, int HANDLER = 0>
__device__ __forceinline__ void doca_gpu_dev_verbs_wait(
    doca_gpu_dev_verbs_qp* qp,
    uint64_t ticket) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_wait<MODE, HANDLER>(qp, ticket);
}

template <int MODE = 0, int HANDLER = 0, int EXEC = 0>
__device__ __forceinline__ void doca_gpu_dev_verbs_put(
    doca_gpu_dev_verbs_qp* qp,
    doca_gpu_dev_verbs_addr raddr,
    doca_gpu_dev_verbs_addr laddr,
    std::size_t size,
    uint64_t* out_ticket) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_put<MODE, HANDLER, EXEC>(
      qp, raddr, laddr, size, out_ticket);
}

template <int OP = 0, int MODE = 0, int HANDLER = 0>
__device__ __forceinline__ void doca_gpu_dev_verbs_signal_counter(
    doca_gpu_dev_verbs_qp* mainQp,
    doca_gpu_dev_verbs_addr sigRemoteAddr,
    doca_gpu_dev_verbs_addr sigSinkAddr,
    uint64_t sigVal,
    doca_gpu_dev_verbs_qp* companionQp,
    doca_gpu_dev_verbs_addr counterRemoteAddr,
    doca_gpu_dev_verbs_addr counterSinkAddr,
    uint64_t counterVal) {
  prims_amd_gda::prims_amd_gda_gpu_dev_verbs_signal_counter<OP, MODE, HANDLER>(
      mainQp,
      sigRemoteAddr,
      sigSinkAddr,
      sigVal,
      companionQp,
      counterRemoteAddr,
      counterSinkAddr,
      counterVal);
}

__device__ __forceinline__ doca_gpu_dev_verbs_cq*
doca_gpu_dev_verbs_qp_get_cq_sq(doca_gpu_dev_verbs_qp* qp) {
  return prims_amd_gda::prims_amd_gda_gpu_dev_verbs_qp_get_cq_sq(qp);
}

template <int MODE = 0>
__device__ __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_at(
    doca_gpu_dev_verbs_cq* cq,
    uint64_t consIndex) {
  return prims_amd_gda::prims_amd_gda_gpu_dev_verbs_poll_one_cq_at<MODE>(
      cq, consIndex);
}

// =============================================================================
// doca_fence — wrapper from `comms/prims/platform/DocaVerbsUtils.cuh`
// =============================================================================
//
// On NVIDIA, this lives in `comms::prims::doca_fence` (defined in
// `DocaVerbsUtils.cuh`). On AMD, the `.cuh` skips that include and uses
// the version below. Forwards to `prims_amd_gda::prims_amd_gda_fence` (the
// no-nic overload added in Phase 1).

namespace comms::prims {

template <int MODE = 0, int HANDLER = 0>
__device__ __forceinline__ void doca_fence(doca_gpu_dev_verbs_qp* qp) {
  prims_amd_gda::prims_amd_gda_fence<MODE, HANDLER>(qp);
}

} // namespace comms::prims

// =============================================================================
// Host-side doca_* -> prims_amd_gda::prims_amd_gda_* translation
// =============================================================================
//
// `MultipeerIbgdaTransport.{h,cc}` and `rdma/NicDiscovery.h` call
// host-side `doca_*` APIs. On AMD we translate them to the
// `prims_amd_gda::prims_amd_gda_*` host APIs implemented in
// `prims_amd_gda/PrimsAmdGdaHost.{h,cc}` (which back the calls with HSA + raw
// mlx5dv + libibverbs underneath).
//
// All translation is pure name-prefix forwarding (zero-cost inlines)
// plus type/constant aliases — implementations live in `prims_amd_gda/`.
//
// This block is parseable in both device and host passes because
// `HipDeviceCompat.h` provides host-pass stubs for the HIP device
// intrinsics it uses (so the device chain pulled in by the existing
// device-side translations above doesn't break the host pass).

#include "prims_amd_gda/PrimsAmdGdaDmaBuf.h" // @manual
#include "prims_amd_gda/PrimsAmdGdaHost.h" // @manual

// --- Type aliases ---
using doca_error_t = prims_amd_gda_error_t;
using doca_gpu = prims_amd_gda_gpu;
using doca_verbs_qp_attr = prims_amd_gda_verbs_qp_attr;
using doca_verbs_ah_attr = prims_amd_gda_verbs_ah_attr;
using doca_verbs_gid = prims_amd_gda_verbs_gid;
using doca_verbs_addr_type = prims_amd_gda_verbs_addr_type;
using doca_verbs_mtu_size = prims_amd_gda_verbs_mtu_size;
using doca_mtu = prims_amd_gda_mtu;
using doca_verbs_qp_state = prims_amd_gda_verbs_qp_state;
using doca_verbs_qp_atomic_mode = prims_amd_gda_verbs_qp_atomic_mode;
using doca_verbs_qp_attr_mask = prims_amd_gda_verbs_qp_attr_mask;
using doca_gpu_verbs_qp_init_attr_hl = prims_amd_gda_gpu_verbs_qp_init_attr_hl;
using doca_gpu_verbs_qp = prims_amd_gda_gpu_verbs_qp;
using doca_gpu_verbs_qp_hl = prims_amd_gda_gpu_verbs_qp_hl;
using doca_gpu_verbs_qp_group_hl = prims_amd_gda_gpu_verbs_qp_group_hl;

// --- Constant aliases ---
constexpr doca_error_t DOCA_SUCCESS = PRIMS_AMD_GDA_SUCCESS;
constexpr doca_error_t DOCA_ERROR_INVALID_VALUE =
    PRIMS_AMD_GDA_ERROR_INVALID_VALUE;
constexpr doca_error_t DOCA_ERROR_NO_MEMORY = PRIMS_AMD_GDA_ERROR_NO_MEMORY;
constexpr doca_error_t DOCA_ERROR_NOT_FOUND = PRIMS_AMD_GDA_ERROR_NOT_FOUND;
constexpr doca_error_t DOCA_ERROR_INITIALIZATION =
    PRIMS_AMD_GDA_ERROR_INITIALIZATION;
constexpr doca_error_t DOCA_ERROR_DRIVER = PRIMS_AMD_GDA_ERROR_DRIVER;

constexpr doca_verbs_addr_type DOCA_VERBS_ADDR_TYPE_IPv4 =
    PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IPv4;
constexpr doca_verbs_addr_type DOCA_VERBS_ADDR_TYPE_IPv6 =
    PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IPv6;
constexpr doca_verbs_addr_type DOCA_VERBS_ADDR_TYPE_IB =
    PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IB;
constexpr doca_verbs_addr_type DOCA_VERBS_ADDR_TYPE_IB_NO_GRH =
    PRIMS_AMD_GDA_VERBS_ADDR_TYPE_IB_NO_GRH;

constexpr doca_verbs_mtu_size DOCA_VERBS_MTU_SIZE_256_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_256_BYTES;
constexpr doca_verbs_mtu_size DOCA_VERBS_MTU_SIZE_512_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_512_BYTES;
constexpr doca_verbs_mtu_size DOCA_VERBS_MTU_SIZE_1K_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_1K_BYTES;
constexpr doca_verbs_mtu_size DOCA_VERBS_MTU_SIZE_2K_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_2K_BYTES;
constexpr doca_verbs_mtu_size DOCA_VERBS_MTU_SIZE_4K_BYTES =
    PRIMS_AMD_GDA_VERBS_MTU_SIZE_4K_BYTES;
constexpr doca_mtu DOCA_MTU_SIZE_256_BYTES = PRIMS_AMD_GDA_MTU_SIZE_256_BYTES;
constexpr doca_mtu DOCA_MTU_SIZE_512_BYTES = PRIMS_AMD_GDA_MTU_SIZE_512_BYTES;
constexpr doca_mtu DOCA_MTU_SIZE_1024_BYTES = PRIMS_AMD_GDA_MTU_SIZE_1024_BYTES;
constexpr doca_mtu DOCA_MTU_SIZE_2048_BYTES = PRIMS_AMD_GDA_MTU_SIZE_2048_BYTES;
constexpr doca_mtu DOCA_MTU_SIZE_4096_BYTES = PRIMS_AMD_GDA_MTU_SIZE_4096_BYTES;

constexpr doca_verbs_qp_state DOCA_VERBS_QP_STATE_RST =
    PRIMS_AMD_GDA_VERBS_QP_STATE_RST;
constexpr doca_verbs_qp_state DOCA_VERBS_QP_STATE_INIT =
    PRIMS_AMD_GDA_VERBS_QP_STATE_INIT;
constexpr doca_verbs_qp_state DOCA_VERBS_QP_STATE_RTR =
    PRIMS_AMD_GDA_VERBS_QP_STATE_RTR;
constexpr doca_verbs_qp_state DOCA_VERBS_QP_STATE_RTS =
    PRIMS_AMD_GDA_VERBS_QP_STATE_RTS;

constexpr int DOCA_VERBS_QP_ATOMIC_MODE_NONE =
    PRIMS_AMD_GDA_VERBS_QP_ATOMIC_MODE_NONE;
constexpr int DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC =
    PRIMS_AMD_GDA_VERBS_QP_ATOMIC_MODE_IB_SPEC;

constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_NEXT_STATE =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_NEXT_STATE;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_PKEY_INDEX =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_PKEY_INDEX;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_PORT_NUM =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_PORT_NUM;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_READ;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_ATOMIC =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_ATOMIC;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_PATH_MTU =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_PATH_MTU;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_DEST_QP_NUM =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_DEST_QP_NUM;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_RQ_PSN =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_RQ_PSN;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_AH_ATTR =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_AH_ATTR;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_MIN_RNR_TIMER;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_SQ_PSN =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_SQ_PSN;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_ACK_TIMEOUT =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_ACK_TIMEOUT;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_RETRY_CNT =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_RETRY_CNT;
constexpr doca_verbs_qp_attr_mask DOCA_VERBS_QP_ATTR_RNR_RETRY =
    PRIMS_AMD_GDA_VERBS_QP_ATTR_RNR_RETRY;

constexpr int DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT =
    PRIMS_AMD_GDA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

// --- Function forwarders ---
// Note: `DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO` is already defined above
// for device-side use; the value is identical for the host APIs.

inline doca_error_t doca_gpu_create(
    const char* gpu_pci_bus_id,
    doca_gpu** out_gpu) {
  return prims_amd_gda::prims_amd_gda_gpu_create(gpu_pci_bus_id, out_gpu);
}
inline doca_error_t doca_gpu_destroy(doca_gpu* gpu) {
  return prims_amd_gda::prims_amd_gda_gpu_destroy(gpu);
}
inline doca_error_t doca_gpu_mem_alloc(
    doca_gpu* gpu,
    std::size_t size,
    std::size_t alignment,
    int mem_type,
    int access_type,
    void** out_ptr,
    void** out_gpu_ptr) {
  return prims_amd_gda::prims_amd_gda_gpu_mem_alloc(
      gpu, size, alignment, mem_type, access_type, out_ptr, out_gpu_ptr);
}

inline doca_error_t doca_verbs_wrapper_ibv_get_device_list(
    int* num_devices,
    ibv_device*** out_list) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_get_device_list(
      num_devices, out_list);
}
inline doca_error_t doca_verbs_wrapper_ibv_free_device_list(ibv_device** list) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_free_device_list(list);
}
inline doca_error_t doca_verbs_wrapper_ibv_get_device_name(
    ibv_device* dev,
    const char** out_name) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_get_device_name(
      dev, out_name);
}
inline doca_error_t doca_verbs_wrapper_ibv_open_device(
    ibv_device* dev,
    ibv_context** out_ctx) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_open_device(
      dev, out_ctx);
}
inline doca_error_t doca_verbs_wrapper_ibv_close_device(ibv_context* ctx) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_close_device(ctx);
}
inline doca_error_t doca_verbs_wrapper_ibv_alloc_pd(
    ibv_context* ctx,
    ibv_pd** out_pd) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_alloc_pd(ctx, out_pd);
}
inline doca_error_t doca_verbs_wrapper_ibv_dealloc_pd(ibv_pd* pd) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_dealloc_pd(pd);
}
inline doca_error_t doca_verbs_wrapper_ibv_query_device(
    ibv_context* ctx,
    ibv_device_attr* attr) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_query_device(ctx, attr);
}
inline doca_error_t doca_verbs_wrapper_ibv_query_port(
    ibv_context* ctx,
    uint8_t port,
    ibv_port_attr* attr) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_query_port(
      ctx, port, attr);
}
inline doca_error_t doca_verbs_wrapper_ibv_query_gid(
    ibv_context* ctx,
    uint8_t port,
    int index,
    union ibv_gid* gid) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_query_gid(
      ctx, port, index, gid);
}
inline doca_error_t doca_verbs_wrapper_ibv_reg_mr(
    ibv_pd* pd,
    void* addr,
    std::size_t length,
    int access,
    ibv_mr** out_mr) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_reg_mr(
      pd, addr, length, access, out_mr);
}
inline doca_error_t doca_verbs_wrapper_ibv_dereg_mr(ibv_mr* mr) {
  return prims_amd_gda::prims_amd_gda_verbs_wrapper_ibv_dereg_mr(mr);
}

inline doca_error_t doca_verbs_qp_attr_create(doca_verbs_qp_attr** out_attr) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_create(out_attr);
}
inline doca_error_t doca_verbs_qp_attr_destroy(doca_verbs_qp_attr* attr) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_destroy(attr);
}
inline doca_error_t doca_verbs_qp_attr_set_next_state(
    doca_verbs_qp_attr* attr,
    doca_verbs_qp_state state) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_next_state(attr, state);
}
inline doca_error_t doca_verbs_qp_attr_set_path_mtu(
    doca_verbs_qp_attr* attr,
    doca_mtu mtu) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_path_mtu(attr, mtu);
}
inline doca_error_t doca_verbs_qp_attr_set_port_num(
    doca_verbs_qp_attr* attr,
    uint8_t port) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_port_num(attr, port);
}
inline doca_error_t doca_verbs_qp_attr_set_dest_qp_num(
    doca_verbs_qp_attr* attr,
    uint32_t dest_qp_num) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_dest_qp_num(
      attr, dest_qp_num);
}
inline doca_error_t doca_verbs_qp_attr_set_rq_psn(
    doca_verbs_qp_attr* attr,
    uint32_t psn) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_rq_psn(attr, psn);
}
inline doca_error_t doca_verbs_qp_attr_set_sq_psn(
    doca_verbs_qp_attr* attr,
    uint32_t psn) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_sq_psn(attr, psn);
}
inline doca_error_t doca_verbs_qp_attr_set_min_rnr_timer(
    doca_verbs_qp_attr* attr,
    uint8_t v) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_min_rnr_timer(attr, v);
}
inline doca_error_t doca_verbs_qp_attr_set_ack_timeout(
    doca_verbs_qp_attr* attr,
    uint8_t v) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_ack_timeout(attr, v);
}
inline doca_error_t doca_verbs_qp_attr_set_retry_cnt(
    doca_verbs_qp_attr* attr,
    uint8_t v) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_retry_cnt(attr, v);
}
inline doca_error_t doca_verbs_qp_attr_set_rnr_retry(
    doca_verbs_qp_attr* attr,
    uint8_t v) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_rnr_retry(attr, v);
}
inline doca_error_t doca_verbs_qp_attr_set_allow_remote_read(
    doca_verbs_qp_attr* attr,
    bool allow) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_allow_remote_read(
      attr, allow);
}
inline doca_error_t doca_verbs_qp_attr_set_allow_remote_write(
    doca_verbs_qp_attr* attr,
    bool allow) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_allow_remote_write(
      attr, allow);
}
inline doca_error_t doca_verbs_qp_attr_set_allow_remote_atomic(
    doca_verbs_qp_attr* attr,
    int atomic_mode) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_allow_remote_atomic(
      attr, atomic_mode);
}
inline doca_error_t doca_verbs_qp_attr_set_ah_attr(
    doca_verbs_qp_attr* attr,
    doca_verbs_ah_attr* ah_attr) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_attr_set_ah_attr(attr, ah_attr);
}

inline doca_error_t doca_verbs_ah_attr_create(
    ibv_context* ctx,
    doca_verbs_ah_attr** out_attr) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_create(ctx, out_attr);
}
inline doca_error_t doca_verbs_ah_attr_destroy(doca_verbs_ah_attr* attr) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_destroy(attr);
}
inline doca_error_t doca_verbs_ah_attr_set_addr_type(
    doca_verbs_ah_attr* attr,
    doca_verbs_addr_type t) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_addr_type(attr, t);
}
inline doca_error_t doca_verbs_ah_attr_set_dlid(
    doca_verbs_ah_attr* attr,
    uint16_t dlid) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_dlid(attr, dlid);
}
inline doca_error_t doca_verbs_ah_attr_set_gid(
    doca_verbs_ah_attr* attr,
    const doca_verbs_gid& gid) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_gid(attr, gid);
}
inline doca_error_t doca_verbs_ah_attr_set_sgid_index(
    doca_verbs_ah_attr* attr,
    int idx) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_sgid_index(attr, idx);
}
inline doca_error_t doca_verbs_ah_attr_set_hop_limit(
    doca_verbs_ah_attr* attr,
    uint8_t hop) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_hop_limit(attr, hop);
}
inline doca_error_t doca_verbs_ah_attr_set_sl(
    doca_verbs_ah_attr* attr,
    uint8_t sl) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_sl(attr, sl);
}
inline doca_error_t doca_verbs_ah_attr_set_traffic_class(
    doca_verbs_ah_attr* attr,
    uint8_t tc) {
  return prims_amd_gda::prims_amd_gda_verbs_ah_attr_set_traffic_class(attr, tc);
}

inline doca_error_t
doca_verbs_qp_modify(ibv_qp* qp, doca_verbs_qp_attr* attr, int attr_mask = 0) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_modify(qp, attr, attr_mask);
}
inline uint32_t doca_verbs_qp_get_qpn(ibv_qp* qp) {
  return prims_amd_gda::prims_amd_gda_verbs_qp_get_qpn(qp);
}
inline std::size_t doca_verbs_mtu_size_in_bytes(doca_mtu m) {
  return prims_amd_gda::prims_amd_gda_verbs_mtu_size_in_bytes(m);
}

inline doca_error_t doca_gpu_verbs_create_qp_hl(
    const doca_gpu_verbs_qp_init_attr_hl* attr,
    doca_gpu_verbs_qp_hl** out_qp) {
  return prims_amd_gda::prims_amd_gda_gpu_verbs_create_qp_hl(attr, out_qp);
}
inline doca_error_t doca_gpu_verbs_destroy_qp_hl(doca_gpu_verbs_qp_hl* qp) {
  return prims_amd_gda::prims_amd_gda_gpu_verbs_destroy_qp_hl(qp);
}
inline doca_error_t doca_gpu_verbs_create_qp_group_hl(
    const doca_gpu_verbs_qp_init_attr_hl* attr,
    doca_gpu_verbs_qp_group_hl** out_grp) {
  return prims_amd_gda::prims_amd_gda_gpu_verbs_create_qp_group_hl(
      attr, out_grp);
}
inline doca_error_t doca_gpu_verbs_destroy_qp_group_hl(
    doca_gpu_verbs_qp_group_hl* g) {
  return prims_amd_gda::prims_amd_gda_gpu_verbs_destroy_qp_group_hl(g);
}
inline doca_error_t doca_gpu_verbs_get_qp_dev(
    doca_gpu_verbs_qp* qp_gverbs,
    prims_amd_gda_gpu_dev_verbs_qp** out_dev_qp) {
  return prims_amd_gda::prims_amd_gda_gpu_verbs_get_qp_dev(
      qp_gverbs, out_dev_qp);
}
