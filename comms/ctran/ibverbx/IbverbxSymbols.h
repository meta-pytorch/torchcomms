// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/ibverbx/Ibvcore.h"

#ifdef IBVERBX_BUILD_RDMA_CORE
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#endif

namespace ibverbx {

struct IbvSymbols {
  int (*ibv_internal_fork_init)(void) = nullptr;
  struct ibv_device** (*ibv_internal_get_device_list)(int* num_devices) =
      nullptr;
  void (*ibv_internal_free_device_list)(struct ibv_device** list) = nullptr;
  const char* (*ibv_internal_get_device_name)(struct ibv_device* device) =
      nullptr;
  struct ibv_context* (*ibv_internal_open_device)(struct ibv_device* device) =
      nullptr;
  int (*ibv_internal_close_device)(struct ibv_context* context) = nullptr;
  int (*ibv_internal_get_async_event)(
      struct ibv_context* context,
      struct ibv_async_event* event) = nullptr;
  void (*ibv_internal_ack_async_event)(struct ibv_async_event* event) = nullptr;
  int (*ibv_internal_query_device)(
      struct ibv_context* context,
      struct ibv_device_attr* device_attr) = nullptr;
  int (*ibv_internal_query_port)(
      struct ibv_context* context,
      uint8_t port_num,
      struct ibv_port_attr* port_attr) = nullptr;
  int (*ibv_internal_query_gid)(
      struct ibv_context* context,
      uint8_t port_num,
      int index,
      union ibv_gid* gid) = nullptr;
  int (*ibv_internal_query_qp)(
      struct ibv_qp* qp,
      struct ibv_qp_attr* attr,
      int attr_mask,
      struct ibv_qp_init_attr* init_attr) = nullptr;
  struct ibv_pd* (*ibv_internal_alloc_pd)(struct ibv_context* context) =
      nullptr;
  struct ibv_pd* (*ibv_internal_alloc_parent_domain)(
      struct ibv_context* context,
      struct ibv_parent_domain_init_attr* attr) = nullptr;
  int (*ibv_internal_dealloc_pd)(struct ibv_pd* pd) = nullptr;
  struct ibv_mr* (*ibv_internal_reg_mr)(
      struct ibv_pd* pd,
      void* addr,
      size_t length,
      int access) = nullptr;
  struct ibv_mr* (*ibv_internal_reg_mr_iova2)(
      struct ibv_pd* pd,
      void* addr,
      size_t length,
      uint64_t iova,
      unsigned int access) = nullptr;
  struct ibv_mr* (*ibv_internal_reg_dmabuf_mr)(
      struct ibv_pd* pd,
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      int access) = nullptr;
  int (*ibv_internal_dereg_mr)(struct ibv_mr* mr) = nullptr;
  struct ibv_cq* (*ibv_internal_create_cq)(
      struct ibv_context* context,
      int cqe,
      void* cq_context,
      struct ibv_comp_channel* channel,
      int comp_vector) = nullptr;
  struct ibv_cq_ex* (*ibv_internal_create_cq_ex)(
      struct ibv_context* context,
      struct ibv_cq_init_attr_ex* attr) = nullptr;
  int (*ibv_internal_destroy_cq)(struct ibv_cq* cq) = nullptr;
  struct ibv_comp_channel* (*ibv_internal_create_comp_channel)(
      struct ibv_context* context) = nullptr;
  int (*ibv_internal_destroy_comp_channel)(struct ibv_comp_channel* channel) =
      nullptr;
  int (*ibv_internal_get_cq_event)(
      struct ibv_comp_channel* channel,
      struct ibv_cq** cq,
      void** cq_context) = nullptr;
  void (*ibv_internal_ack_cq_events)(struct ibv_cq* cq, unsigned int nevents) =
      nullptr;
  struct ibv_qp* (*ibv_internal_create_qp)(
      struct ibv_pd* pd,
      struct ibv_qp_init_attr* qp_init_attr) = nullptr;
  int (*ibv_internal_modify_qp)(
      struct ibv_qp* qp,
      struct ibv_qp_attr* attr,
      int attr_mask) = nullptr;
  int (*ibv_internal_destroy_qp)(struct ibv_qp* qp) = nullptr;
  const char* (*ibv_internal_event_type_str)(enum ibv_event_type event) =
      nullptr;
  int (*ibv_internal_query_ece)(struct ibv_qp* qp, struct ibv_ece* ece) =
      nullptr;
  int (*ibv_internal_set_ece)(struct ibv_qp* qp, struct ibv_ece* ece) = nullptr;
  enum ibv_fork_status (*ibv_internal_is_fork_initialized)() = nullptr;

  /* mlx5dv functions */
  int (*mlx5dv_internal_init_obj)(struct mlx5dv_obj* obj, uint64_t obj_type) =
      nullptr;
  bool (*mlx5dv_internal_is_supported)(struct ibv_device* device) = nullptr;
  int (*mlx5dv_internal_get_data_direct_sysfs_path)(
      struct ibv_context* context,
      char* buf,
      size_t buf_len) = nullptr;
  /* DMA-BUF support */
  struct ibv_mr* (*mlx5dv_internal_reg_dmabuf_mr)(
      struct ibv_pd* pd,
      uint64_t offset,
      size_t length,
      uint64_t iova,
      int fd,
      int access,
      int mlx5_access) = nullptr;

  /* DC (Dynamically Connected) transport support */
  struct ibv_qp* (*mlx5dv_internal_create_qp)(
      struct ibv_context* context,
      struct ibv_qp_init_attr_ex* qp_attr,
      struct mlx5dv_qp_init_attr* mlx5_qp_attr) = nullptr;
  struct mlx5dv_qp_ex* (*mlx5dv_internal_qp_ex_from_ibv_qp_ex)(
      struct ibv_qp_ex* qp) = nullptr;

  /* SRQ support */
  struct ibv_srq* (*ibv_internal_create_srq)(
      struct ibv_pd* pd,
      struct ibv_srq_init_attr* srq_init_attr) = nullptr;
  struct ibv_srq* (*ibv_internal_create_srq_ex)(
      struct ibv_context* context,
      struct ibv_srq_init_attr_ex* srq_init_attr_ex) = nullptr;
  int (*ibv_internal_modify_srq)(
      struct ibv_srq* srq,
      struct ibv_srq_attr* srq_attr,
      int srq_attr_mask) = nullptr;
  int (*ibv_internal_query_srq)(
      struct ibv_srq* srq,
      struct ibv_srq_attr* srq_attr) = nullptr;
  int (*ibv_internal_destroy_srq)(struct ibv_srq* srq) = nullptr;
  int (*ibv_internal_post_srq_recv)(
      struct ibv_srq* srq,
      struct ibv_recv_wr* recv_wr,
      struct ibv_recv_wr** bad_recv_wr) = nullptr;

  /* Address Handle (AH) support */
  struct ibv_ah* (*ibv_internal_create_ah)(
      struct ibv_pd* pd,
      struct ibv_ah_attr* attr) = nullptr;
  int (*ibv_internal_destroy_ah)(struct ibv_ah* ah) = nullptr;

  /* Extended QP support */
  struct ibv_qp* (*ibv_internal_create_qp_ex)(
      struct ibv_context* context,
      struct ibv_qp_init_attr_ex* qp_init_attr_ex) = nullptr;
  struct ibv_qp_ex* (*ibv_internal_qp_to_qp_ex)(struct ibv_qp* qp) = nullptr;
  void (*ibv_internal_wr_start)(struct ibv_qp_ex* qp) = nullptr;
  void (*ibv_internal_wr_rdma_write)(
      struct ibv_qp_ex* qp,
      uint32_t rkey,
      uint64_t remote_addr) = nullptr;
  void (*ibv_internal_wr_rdma_write_imm)(
      struct ibv_qp_ex* qp,
      uint32_t rkey,
      uint64_t remote_addr,
      __be32 imm_data) = nullptr;
  void (*ibv_internal_wr_set_sge_list)(
      struct ibv_qp_ex* qp,
      size_t num_sge,
      const struct ibv_sge* sg_list) = nullptr;
  int (*ibv_internal_wr_complete)(struct ibv_qp_ex* qp) = nullptr;
  void (*mlx5dv_internal_wr_set_dc_addr)(
      struct mlx5dv_qp_ex* mqp,
      struct ibv_ah* ah,
      uint32_t remote_dctn,
      uint64_t remote_dc_key) = nullptr;
};

extern IbvSymbols ibvSymbols;

int buildIbvSymbols(IbvSymbols& ibvSymbols, const std::string& ibv_path = "");

} // namespace ibverbx
