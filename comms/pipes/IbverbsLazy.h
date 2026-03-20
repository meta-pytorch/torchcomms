// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// Ibverbs type definitions and dlopen'd function wrappers for pipes.
//
// When pipes uses doca_gpunetio_dl (dlopen variant), neither libibverbs nor
// libmlx5 are linked at build time. DOCA internally handles its own ibverbs
// calls through its wrapper layer. However, pipes also calls a few ibverbs
// functions directly (e.g., ibv_reg_mr_iova2, ibv_reg_dmabuf_mr) that are
// NOT covered by DOCA's wrapper. This header provides:
//
//   1. Struct/enum definitions for ibverbs types that pipes accesses by value
//      (ibv_mr, ibv_port_attr, ibv_mtu, etc.). These mirror the ABI-stable
//      layouts from <infiniband/verbs.h>.
//
//   2. dlopen-based wrappers for ibverbs functions not covered by DOCA's
//      doca_verbs_wrapper_ibv_* API.
//
// Include order: include this header AFTER <doca_gpunetio_host.h> so that
// any forward declarations from DOCA headers are completed here.

#include <cstddef>
#include <cstdint>

// Forward declarations — always safe to repeat even after full definitions.
// Placed outside the INFINIBAND_VERBS_H guard so consumers can include this
// header purely for opaque-pointer declarations without pulling in verbs.h.
struct ibv_context;
struct ibv_pd;
struct ibv_device;

// If infiniband/verbs.h is already included (e.g., via doca_verbs_ibv_wrapper.h
// in conda builds where rdma-core headers are installed), skip our type
// definitions and only provide the dlopen wrappers below.
#ifndef INFINIBAND_VERBS_H

// ============================================================================
// ibv_mtu — MTU enum (matches libibverbs ABI)
// ============================================================================
enum ibv_mtu {
  IBV_MTU_256 = 1,
  IBV_MTU_512 = 2,
  IBV_MTU_1024 = 3,
  IBV_MTU_2048 = 4,
  IBV_MTU_4096 = 5,
};

// ============================================================================
// ibv_port_state — Port state enum
// ============================================================================
enum ibv_port_state {
  IBV_PORT_NOP = 0,
  IBV_PORT_DOWN = 1,
  IBV_PORT_INIT = 2,
  IBV_PORT_ARMED = 3,
  IBV_PORT_ACTIVE = 4,
  IBV_PORT_ACTIVE_DEFER = 5,
};

// ============================================================================
// ibv_port_attr — Port attributes (matches libibverbs ABI layout)
// ============================================================================
struct ibv_port_attr {
  enum ibv_port_state state;
  enum ibv_mtu max_mtu;
  enum ibv_mtu active_mtu;
  int gid_tbl_len;
  uint32_t port_cap_flags;
  uint32_t max_msg_sz;
  uint32_t bad_pkey_cntr;
  uint32_t qkey_viol_cntr;
  uint16_t pkey_tbl_len;
  uint16_t lid;
  uint16_t sm_lid;
  uint8_t lmc;
  uint8_t max_vl_num;
  uint8_t sm_sl;
  uint8_t subnet_timeout;
  uint8_t init_type_reply;
  uint8_t active_width;
  uint8_t active_speed;
  uint8_t phys_state;
  uint8_t link_layer;
  uint8_t flags;
  uint16_t port_cap_flags2;
  uint32_t active_speed_ex;
};

// Link-layer constants
constexpr uint8_t IBV_LINK_LAYER_INFINIBAND = 1;
constexpr uint8_t IBV_LINK_LAYER_ETHERNET = 2;

// ============================================================================
// ibv_mr — Memory region (matches libibverbs ABI layout)
// DOCA's ibv wrapper only forward-declares ibv_mr as an opaque type;
// we provide the full definition so pipes can access lkey/rkey fields.
// ============================================================================
struct ibv_mr {
  struct ibv_context* context;
  struct ibv_pd* pd;
  void* addr;
  size_t length;
  uint32_t handle;
  uint32_t lkey;
  uint32_t rkey;
};

#endif // INFINIBAND_VERBS_H

namespace comms::pipes {

// ============================================================================
// dlopen'd ibverbs functions not covered by DOCA's wrapper API
// ============================================================================

/**
 * ibv_reg_mr_iova2 — Register MR with explicit IOVA (zero-based MR support).
 * Wraps the libibverbs ibv_reg_mr_iova2() function loaded via dlopen.
 *
 * @return ibv_mr* on success, nullptr on failure.
 */
struct ibv_mr* lazy_ibv_reg_mr_iova2(
    struct ibv_pd* pd,
    void* addr,
    size_t length,
    uint64_t iova,
    unsigned int access);

/**
 * ibv_reg_dmabuf_mr — Register DMA-BUF memory region.
 * Wraps the libibverbs ibv_reg_dmabuf_mr() function loaded via dlopen.
 *
 * @return ibv_mr* on success, nullptr on failure.
 */
struct ibv_mr* lazy_ibv_reg_dmabuf_mr(
    struct ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access);

// ============================================================================
// dlopen'd mlx5dv functions (from libmlx5)
// ============================================================================

/**
 * mlx5dv_is_supported — Check if device supports mlx5 DV interface.
 * Wraps the libmlx5 mlx5dv_is_supported() function loaded via dlopen.
 *
 * @return non-zero if supported, 0 if not or libmlx5 unavailable.
 */
int lazy_mlx5dv_is_supported(struct ibv_device* device);

/**
 * mlx5dv_get_data_direct_sysfs_path — Get Data Direct sysfs path.
 * Wraps the libmlx5 mlx5dv_get_data_direct_sysfs_path() loaded via dlopen.
 *
 * @return 0 on success, non-zero on failure or libmlx5 unavailable.
 */
int lazy_mlx5dv_get_data_direct_sysfs_path(
    struct ibv_context* ctx,
    char* buf,
    size_t buf_len);

} // namespace comms::pipes
