// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbverbxSymbols.h"

#include <dlfcn.h>
#include <folly/ScopeGuard.h>
#include <folly/logging/xlog.h>
#include <folly/synchronization/CallOnce.h>

namespace ibverbx {

IbvSymbols ibvSymbols;

#define IBVERBS_VERSION "IBVERBS_1.1"

#define MLX5DV_VERSION "MLX5_1.8"

#ifdef IBVERBX_BUILD_RDMA_CORE
// Wrapper functions to handle type conversions between custom and real types
struct ibv_device** linked_get_device_list(int* num_devices) {
  return reinterpret_cast<struct ibv_device**>(
      ibv_get_device_list(num_devices));
}

void linked_free_device_list(struct ibv_device** list) {
  ibv_free_device_list(reinterpret_cast<::ibv_device**>(list));
}

const char* linked_get_device_name(struct ibv_device* device) {
  return ibv_get_device_name(reinterpret_cast<::ibv_device*>(device));
}

struct ibv_context* linked_open_device(struct ibv_device* device) {
  return reinterpret_cast<struct ibv_context*>(
      ibv_open_device(reinterpret_cast<::ibv_device*>(device)));
}

int linked_close_device(struct ibv_context* context) {
  return ibv_close_device(reinterpret_cast<::ibv_context*>(context));
}

int linked_query_device(
    struct ibv_context* context,
    struct ibv_device_attr* device_attr) {
  return ibv_query_device(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_device_attr*>(device_attr));
}

int linked_query_port(
    struct ibv_context* context,
    uint8_t port_num,
    struct ibv_port_attr* port_attr) {
  return ibv_query_port(
      reinterpret_cast<::ibv_context*>(context),
      port_num,
      reinterpret_cast<::ibv_port_attr*>(port_attr));
}

int linked_query_gid(
    struct ibv_context* context,
    uint8_t port_num,
    int index,
    union ibv_gid* gid) {
  return ibv_query_gid(
      reinterpret_cast<::ibv_context*>(context),
      port_num,
      index,
      reinterpret_cast<::ibv_gid*>(gid));
}

struct ibv_pd* linked_alloc_pd(struct ibv_context* context) {
  return reinterpret_cast<struct ibv_pd*>(
      ibv_alloc_pd(reinterpret_cast<::ibv_context*>(context)));
}

struct ibv_pd* linked_alloc_parent_domain(
    struct ibv_context* context,
    struct ibv_parent_domain_init_attr* attr) {
  return reinterpret_cast<struct ibv_pd*>(ibv_alloc_parent_domain(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_parent_domain_init_attr*>(attr)));
}

int linked_dealloc_pd(struct ibv_pd* pd) {
  return ibv_dealloc_pd(reinterpret_cast<::ibv_pd*>(pd));
}

struct ibv_mr*
linked_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  return reinterpret_cast<struct ibv_mr*>(
      ibv_reg_mr(reinterpret_cast<::ibv_pd*>(pd), addr, length, access));
}

int linked_dereg_mr(struct ibv_mr* mr) {
  return ibv_dereg_mr(reinterpret_cast<::ibv_mr*>(mr));
}

struct ibv_cq* linked_create_cq(
    struct ibv_context* context,
    int cqe,
    void* cq_context,
    struct ibv_comp_channel* channel,
    int comp_vector) {
  return reinterpret_cast<struct ibv_cq*>(ibv_create_cq(
      reinterpret_cast<::ibv_context*>(context),
      cqe,
      cq_context,
      reinterpret_cast<::ibv_comp_channel*>(channel),
      comp_vector));
}

struct ibv_cq_ex* linked_create_cq_ex(
    struct ibv_context* context,
    struct ibv_cq_init_attr_ex* attr) {
  return reinterpret_cast<struct ibv_cq_ex*>(ibv_create_cq_ex(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_cq_init_attr_ex*>(attr)));
}

int linked_destroy_cq(struct ibv_cq* cq) {
  return ibv_destroy_cq(reinterpret_cast<::ibv_cq*>(cq));
}

struct ibv_qp* linked_create_qp(
    struct ibv_pd* pd,
    struct ibv_qp_init_attr* qp_init_attr) {
  return reinterpret_cast<struct ibv_qp*>(ibv_create_qp(
      reinterpret_cast<::ibv_pd*>(pd),
      reinterpret_cast<::ibv_qp_init_attr*>(qp_init_attr)));
}

int linked_modify_qp(
    struct ibv_qp* qp,
    struct ibv_qp_attr* attr,
    int attr_mask) {
  return ibv_modify_qp(
      reinterpret_cast<::ibv_qp*>(qp),
      reinterpret_cast<::ibv_qp_attr*>(attr),
      attr_mask);
}

int linked_destroy_qp(struct ibv_qp* qp) {
  return ibv_destroy_qp(reinterpret_cast<::ibv_qp*>(qp));
}

const char* linked_event_type_str(enum ibv_event_type event) {
  return ibv_event_type_str(static_cast<::ibv_event_type>(event));
}

int linked_get_async_event(
    struct ibv_context* context,
    struct ibv_async_event* event) {
  return ibv_get_async_event(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_async_event*>(event));
}

void linked_ack_async_event(struct ibv_async_event* event) {
  ibv_ack_async_event(reinterpret_cast<::ibv_async_event*>(event));
}

int linked_query_qp(
    struct ibv_qp* qp,
    struct ibv_qp_attr* attr,
    int attr_mask,
    struct ibv_qp_init_attr* init_attr) {
  return ibv_query_qp(
      reinterpret_cast<::ibv_qp*>(qp),
      reinterpret_cast<::ibv_qp_attr*>(attr),
      attr_mask,
      reinterpret_cast<::ibv_qp_init_attr*>(init_attr));
}

struct ibv_mr* linked_reg_mr_iova2(
    struct ibv_pd* pd,
    void* addr,
    size_t length,
    uint64_t iova,
    unsigned int access) {
  return reinterpret_cast<struct ibv_mr*>(ibv_reg_mr_iova2(
      reinterpret_cast<::ibv_pd*>(pd), addr, length, iova, access));
}

struct ibv_mr* linked_reg_dmabuf_mr(
    struct ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access) {
  return reinterpret_cast<struct ibv_mr*>(ibv_reg_dmabuf_mr(
      reinterpret_cast<::ibv_pd*>(pd), offset, length, iova, fd, access));
}

int linked_query_ece(struct ibv_qp* qp, struct ibv_ece* ece) {
  return ibv_query_ece(
      reinterpret_cast<::ibv_qp*>(qp), reinterpret_cast<::ibv_ece*>(ece));
}

int linked_set_ece(struct ibv_qp* qp, struct ibv_ece* ece) {
  return ibv_set_ece(
      reinterpret_cast<::ibv_qp*>(qp), reinterpret_cast<::ibv_ece*>(ece));
}

enum ibv_fork_status linked_is_fork_initialized() {
  return static_cast<enum ibv_fork_status>(ibv_is_fork_initialized());
}

struct ibv_comp_channel* linked_create_comp_channel(
    struct ibv_context* context) {
  return reinterpret_cast<struct ibv_comp_channel*>(
      ibv_create_comp_channel(reinterpret_cast<::ibv_context*>(context)));
}

int linked_destroy_comp_channel(struct ibv_comp_channel* channel) {
  return ibv_destroy_comp_channel(
      reinterpret_cast<::ibv_comp_channel*>(channel));
}

int linked_req_notify_cq(struct ibv_cq* cq, int solicited_only) {
  return ibv_req_notify_cq(reinterpret_cast<::ibv_cq*>(cq), solicited_only);
}

int linked_get_cq_event(
    struct ibv_comp_channel* channel,
    struct ibv_cq** cq,
    void** cq_context) {
  return ibv_get_cq_event(
      reinterpret_cast<::ibv_comp_channel*>(channel),
      reinterpret_cast<::ibv_cq**>(cq),
      cq_context);
}

void linked_ack_cq_events(struct ibv_cq* cq, unsigned int nevents) {
  ibv_ack_cq_events(reinterpret_cast<::ibv_cq*>(cq), nevents);
}

bool linked_mlx5dv_is_supported(struct ibv_device* device) {
  return mlx5dv_is_supported(reinterpret_cast<::ibv_device*>(device));
}

int linked_mlx5dv_init_obj(mlx5dv_obj* obj, uint64_t obj_type) {
  return mlx5dv_init_obj(reinterpret_cast<::mlx5dv_obj*>(obj), obj_type);
}

int linked_mlx5dv_get_data_direct_sysfs_path(
    struct ibv_context* context,
    char* buf,
    size_t buf_len) {
  return mlx5dv_get_data_direct_sysfs_path(
      reinterpret_cast<::ibv_context*>(context), buf, buf_len);
}

struct ibv_mr* linked_mlx5dv_reg_dmabuf_mr(
    struct ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access,
    int mlx5_access) {
  return reinterpret_cast<struct ibv_mr*>(mlx5dv_reg_dmabuf_mr(
      reinterpret_cast<::ibv_pd*>(pd),
      offset,
      length,
      iova,
      fd,
      access,
      mlx5_access));
}

struct ibv_qp* linked_mlx5dv_create_qp(
    struct ibv_context* context,
    struct ibv_qp_init_attr_ex* qp_attr,
    struct mlx5dv_qp_init_attr* mlx5_qp_attr) {
  return reinterpret_cast<struct ibv_qp*>(mlx5dv_create_qp(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_qp_init_attr_ex*>(qp_attr),
      reinterpret_cast<::mlx5dv_qp_init_attr*>(mlx5_qp_attr)));
}

struct mlx5dv_qp_ex* linked_mlx5dv_qp_ex_from_ibv_qp_ex(struct ibv_qp_ex* qp) {
  return reinterpret_cast<struct mlx5dv_qp_ex*>(
      mlx5dv_qp_ex_from_ibv_qp_ex(reinterpret_cast<::ibv_qp_ex*>(qp)));
}

struct ibv_srq* linked_create_srq(
    struct ibv_pd* pd,
    struct ibv_srq_init_attr* srq_init_attr) {
  return reinterpret_cast<struct ibv_srq*>(ibv_create_srq(
      reinterpret_cast<::ibv_pd*>(pd),
      reinterpret_cast<::ibv_srq_init_attr*>(srq_init_attr)));
}

struct ibv_srq* linked_create_srq_ex(
    struct ibv_context* context,
    struct ibv_srq_init_attr_ex* srq_init_attr_ex) {
  return reinterpret_cast<struct ibv_srq*>(ibv_create_srq_ex(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_srq_init_attr_ex*>(srq_init_attr_ex)));
}

int linked_modify_srq(
    struct ibv_srq* srq,
    struct ibv_srq_attr* srq_attr,
    int srq_attr_mask) {
  return ibv_modify_srq(
      reinterpret_cast<::ibv_srq*>(srq),
      reinterpret_cast<::ibv_srq_attr*>(srq_attr),
      srq_attr_mask);
}

int linked_query_srq(struct ibv_srq* srq, struct ibv_srq_attr* srq_attr) {
  return ibv_query_srq(
      reinterpret_cast<::ibv_srq*>(srq),
      reinterpret_cast<::ibv_srq_attr*>(srq_attr));
}

int linked_destroy_srq(struct ibv_srq* srq) {
  return ibv_destroy_srq(reinterpret_cast<::ibv_srq*>(srq));
}

int linked_post_srq_recv(
    struct ibv_srq* srq,
    struct ibv_recv_wr* recv_wr,
    struct ibv_recv_wr** bad_recv_wr) {
  return ibv_post_srq_recv(
      reinterpret_cast<::ibv_srq*>(srq),
      reinterpret_cast<::ibv_recv_wr*>(recv_wr),
      reinterpret_cast<::ibv_recv_wr**>(bad_recv_wr));
}

struct ibv_ah* linked_create_ah(struct ibv_pd* pd, struct ibv_ah_attr* attr) {
  return reinterpret_cast<struct ibv_ah*>(ibv_create_ah(
      reinterpret_cast<::ibv_pd*>(pd), reinterpret_cast<::ibv_ah_attr*>(attr)));
}

int linked_destroy_ah(struct ibv_ah* ah) {
  return ibv_destroy_ah(reinterpret_cast<::ibv_ah*>(ah));
}

struct ibv_qp* linked_create_qp_ex(
    struct ibv_context* context,
    struct ibv_qp_init_attr_ex* qp_init_attr_ex) {
  return reinterpret_cast<struct ibv_qp*>(ibv_create_qp_ex(
      reinterpret_cast<::ibv_context*>(context),
      reinterpret_cast<::ibv_qp_init_attr_ex*>(qp_init_attr_ex)));
}

struct ibv_qp_ex* linked_qp_to_qp_ex(struct ibv_qp* qp) {
  return reinterpret_cast<struct ibv_qp_ex*>(
      ibv_qp_to_qp_ex(reinterpret_cast<::ibv_qp*>(qp)));
}

void linked_wr_start(struct ibv_qp_ex* qp) {
  ibv_wr_start(reinterpret_cast<::ibv_qp_ex*>(qp));
}

void linked_wr_rdma_write(
    struct ibv_qp_ex* qp,
    uint32_t rkey,
    uint64_t remote_addr) {
  ibv_wr_rdma_write(reinterpret_cast<::ibv_qp_ex*>(qp), rkey, remote_addr);
}

void linked_wr_rdma_write_imm(
    struct ibv_qp_ex* qp,
    uint32_t rkey,
    uint64_t remote_addr,
    __be32 imm_data) {
  ibv_wr_rdma_write_imm(
      reinterpret_cast<::ibv_qp_ex*>(qp), rkey, remote_addr, imm_data);
}

void linked_wr_set_sge_list(
    struct ibv_qp_ex* qp,
    size_t num_sge,
    const struct ibv_sge* sg_list) {
  ibv_wr_set_sge_list(
      reinterpret_cast<::ibv_qp_ex*>(qp),
      num_sge,
      reinterpret_cast<const ::ibv_sge*>(sg_list));
}

int linked_wr_complete(struct ibv_qp_ex* qp) {
  return ibv_wr_complete(reinterpret_cast<::ibv_qp_ex*>(qp));
}

void linked_mlx5dv_wr_set_dc_addr(
    struct mlx5dv_qp_ex* mqp,
    struct ibv_ah* ah,
    uint32_t remote_dctn,
    uint64_t remote_dc_key) {
  mlx5dv_wr_set_dc_addr(
      reinterpret_cast<::mlx5dv_qp_ex*>(mqp),
      reinterpret_cast<::ibv_ah*>(ah),
      remote_dctn,
      remote_dc_key);
}
#endif

// Wrapper functions for extended QP operations
struct ibv_qp_ex* wrapper_qp_to_qp_ex(struct ibv_qp* qp) {
  return reinterpret_cast<struct ibv_qp_ex*>(qp);
}

void wrapper_wr_start(struct ibv_qp_ex* qp) {
  qp->wr_start(qp);
}

void wrapper_wr_rdma_write(
    struct ibv_qp_ex* qp,
    uint32_t rkey,
    uint64_t remote_addr) {
  qp->wr_rdma_write(qp, rkey, remote_addr);
}

void wrapper_wr_rdma_write_imm(
    struct ibv_qp_ex* qp,
    uint32_t rkey,
    uint64_t remote_addr,
    __be32 imm_data) {
  qp->wr_rdma_write_imm(qp, rkey, remote_addr, imm_data);
}

void wrapper_wr_set_sge_list(
    struct ibv_qp_ex* qp,
    size_t num_sge,
    const struct ibv_sge* sg_list) {
  qp->wr_set_sge_list(qp, num_sge, sg_list);
}

int wrapper_wr_complete(struct ibv_qp_ex* qp) {
  return qp->wr_complete(qp);
}

int buildIbvSymbols(IbvSymbols& symbols, const std::string& ibv_path) {
#ifdef IBVERBX_BUILD_RDMA_CORE
  // Direct linking mode - use wrapper functions to handle type conversions
  symbols.ibv_internal_get_device_list = &linked_get_device_list;
  symbols.ibv_internal_free_device_list = &linked_free_device_list;
  symbols.ibv_internal_get_device_name = &linked_get_device_name;
  symbols.ibv_internal_open_device = &linked_open_device;
  symbols.ibv_internal_close_device = &linked_close_device;
  symbols.ibv_internal_get_async_event = &linked_get_async_event;
  symbols.ibv_internal_ack_async_event = &linked_ack_async_event;
  symbols.ibv_internal_query_device = &linked_query_device;
  symbols.ibv_internal_query_port = &linked_query_port;
  symbols.ibv_internal_query_gid = &linked_query_gid;
  symbols.ibv_internal_query_qp = &linked_query_qp;
  symbols.ibv_internal_alloc_pd = &linked_alloc_pd;
  symbols.ibv_internal_alloc_parent_domain = &linked_alloc_parent_domain;
  symbols.ibv_internal_dealloc_pd = &linked_dealloc_pd;
  symbols.ibv_internal_reg_mr = &linked_reg_mr;

  symbols.ibv_internal_reg_mr_iova2 = &linked_reg_mr_iova2;
  symbols.ibv_internal_reg_dmabuf_mr = &linked_reg_dmabuf_mr;
  symbols.ibv_internal_query_ece = &linked_query_ece;
  symbols.ibv_internal_set_ece = &linked_set_ece;
  symbols.ibv_internal_is_fork_initialized = &linked_is_fork_initialized;

  symbols.ibv_internal_dereg_mr = &linked_dereg_mr;
  symbols.ibv_internal_create_cq = &linked_create_cq;
  symbols.ibv_internal_create_cq_ex = &linked_create_cq_ex;
  symbols.ibv_internal_destroy_cq = &linked_destroy_cq;
  symbols.ibv_internal_create_comp_channel = &linked_create_comp_channel;
  symbols.ibv_internal_destroy_comp_channel = &linked_destroy_comp_channel;
  symbols.ibv_internal_get_cq_event = &linked_get_cq_event;
  symbols.ibv_internal_ack_cq_events = &linked_ack_cq_events;
  symbols.ibv_internal_create_qp = &linked_create_qp;
  symbols.ibv_internal_modify_qp = &linked_modify_qp;
  symbols.ibv_internal_destroy_qp = &linked_destroy_qp;
  symbols.ibv_internal_fork_init = &ibv_fork_init;
  symbols.ibv_internal_event_type_str = &linked_event_type_str;

  // mlx5dv symbols
  symbols.mlx5dv_internal_is_supported = &linked_mlx5dv_is_supported;
  symbols.mlx5dv_internal_init_obj = &linked_mlx5dv_init_obj;
  symbols.mlx5dv_internal_get_data_direct_sysfs_path =
      &linked_mlx5dv_get_data_direct_sysfs_path;
  symbols.mlx5dv_internal_reg_dmabuf_mr = &linked_mlx5dv_reg_dmabuf_mr;
  symbols.mlx5dv_internal_create_qp = &linked_mlx5dv_create_qp;
  symbols.mlx5dv_internal_qp_ex_from_ibv_qp_ex =
      &linked_mlx5dv_qp_ex_from_ibv_qp_ex;

  // SRQ symbols
  symbols.ibv_internal_create_srq = &linked_create_srq;
  symbols.ibv_internal_create_srq_ex = &linked_create_srq_ex;
  symbols.ibv_internal_modify_srq = &linked_modify_srq;
  symbols.ibv_internal_query_srq = &linked_query_srq;
  symbols.ibv_internal_destroy_srq = &linked_destroy_srq;
  symbols.ibv_internal_post_srq_recv = &linked_post_srq_recv;

  // AH symbols
  symbols.ibv_internal_create_ah = &linked_create_ah;
  symbols.ibv_internal_destroy_ah = &linked_destroy_ah;

  // Extended QP symbols
  symbols.ibv_internal_create_qp_ex = &linked_create_qp_ex;
  symbols.ibv_internal_qp_to_qp_ex = &linked_qp_to_qp_ex;
  symbols.ibv_internal_wr_start = &linked_wr_start;
  symbols.ibv_internal_wr_rdma_write = &linked_wr_rdma_write;
  symbols.ibv_internal_wr_rdma_write_imm = &linked_wr_rdma_write_imm;
  symbols.ibv_internal_wr_set_sge_list = &linked_wr_set_sge_list;
  symbols.ibv_internal_wr_complete = &linked_wr_complete;
  symbols.mlx5dv_internal_wr_set_dc_addr = &linked_mlx5dv_wr_set_dc_addr;

  return 0;
#else
  // Dynamic loading mode - use dlopen/dlsym
  static void* ibvhandle = nullptr;
  static void* mlx5dvhandle = nullptr;
  void* tmp;
  void** cast;

  // Use folly::ScopedGuard to ensure resources are cleaned up upon failure
  auto guard = folly::makeGuard([&]() {
    if (ibvhandle != nullptr) {
      dlclose(ibvhandle);
    }
    if (mlx5dvhandle != nullptr) {
      dlclose(mlx5dvhandle);
    }
    symbols = {}; // Reset all function pointers to nullptr
  });

  if (!ibv_path.empty()) {
    ibvhandle = dlopen(ibv_path.c_str(), RTLD_NOW);
  }
  if (!ibvhandle) {
    ibvhandle = dlopen("libibverbs.so.1", RTLD_NOW);
    if (!ibvhandle) {
      XLOG(ERR) << "Failed to open libibverbs.so.1";
      return 1;
    }
  }

  // Load mlx5dv symbols if available, do not abort if failed
  mlx5dvhandle = dlopen("libmlx5.so", RTLD_NOW);
  if (!mlx5dvhandle) {
    mlx5dvhandle = dlopen("libmlx5.so.1", RTLD_NOW);
    if (!mlx5dvhandle) {
      XLOG(WARN)
          << "Failed to open libmlx5.so[.1]. Advance features like CX-8 Direct-NIC will be disabled.";
    }
  }

#define LOAD_SYM(handle, symbol, funcptr, version)                            \
  {                                                                           \
    cast = (void**)&funcptr;                                                  \
    tmp = dlvsym(handle, symbol, version);                                    \
    if (tmp == nullptr) {                                                     \
      XLOG(ERR) << fmt::format(                                               \
          "dlvsym failed on {} - {} version {}", symbol, dlerror(), version); \
      return 1;                                                               \
    }                                                                         \
    *cast = tmp;                                                              \
  }

#define LOAD_SYM_WARN_ONLY(handle, symbol, funcptr, version) \
  {                                                          \
    cast = (void**)&funcptr;                                 \
    tmp = dlvsym(handle, symbol, version);                   \
    if (tmp == nullptr) {                                    \
      XLOG(WARN) << fmt::format(                             \
          "dlvsym failed on {} - {} version {}, set null",   \
          symbol,                                            \
          dlerror(),                                         \
          version);                                          \
    }                                                        \
    *cast = tmp;                                             \
  }

#define LOAD_IBVERBS_SYM(symbol, funcptr) \
  LOAD_SYM(ibvhandle, symbol, funcptr, IBVERBS_VERSION)

#define LOAD_IBVERBS_SYM_VERSION(symbol, funcptr, version) \
  LOAD_SYM_WARN_ONLY(ibvhandle, symbol, funcptr, version)

#define LOAD_IBVERBS_SYM_WARN_ONLY(symbol, funcptr) \
  LOAD_SYM_WARN_ONLY(ibvhandle, symbol, funcptr, IBVERBS_VERSION)

// mlx5
#define LOAD_MLX5DV_SYM(symbol, funcptr)                              \
  if (mlx5dvhandle != nullptr) {                                      \
    LOAD_SYM_WARN_ONLY(mlx5dvhandle, symbol, funcptr, MLX5DV_VERSION) \
  }

#define LOAD_MLX5DV_SYM_VERSION(symbol, funcptr, version)      \
  if (mlx5dvhandle != nullptr) {                               \
    LOAD_SYM_WARN_ONLY(mlx5dvhandle, symbol, funcptr, version) \
  }

  LOAD_IBVERBS_SYM("ibv_get_device_list", symbols.ibv_internal_get_device_list);
  LOAD_IBVERBS_SYM(
      "ibv_free_device_list", symbols.ibv_internal_free_device_list);
  LOAD_IBVERBS_SYM("ibv_get_device_name", symbols.ibv_internal_get_device_name);
  LOAD_IBVERBS_SYM("ibv_open_device", symbols.ibv_internal_open_device);
  LOAD_IBVERBS_SYM("ibv_close_device", symbols.ibv_internal_close_device);
  LOAD_IBVERBS_SYM("ibv_get_async_event", symbols.ibv_internal_get_async_event);
  LOAD_IBVERBS_SYM("ibv_ack_async_event", symbols.ibv_internal_ack_async_event);
  LOAD_IBVERBS_SYM("ibv_query_device", symbols.ibv_internal_query_device);
  LOAD_IBVERBS_SYM("ibv_query_port", symbols.ibv_internal_query_port);
  LOAD_IBVERBS_SYM("ibv_query_gid", symbols.ibv_internal_query_gid);
  LOAD_IBVERBS_SYM("ibv_query_qp", symbols.ibv_internal_query_qp);
  LOAD_IBVERBS_SYM("ibv_alloc_pd", symbols.ibv_internal_alloc_pd);
  LOAD_IBVERBS_SYM_WARN_ONLY(
      "ibv_alloc_parent_domain", symbols.ibv_internal_alloc_parent_domain);
  LOAD_IBVERBS_SYM("ibv_dealloc_pd", symbols.ibv_internal_dealloc_pd);
  LOAD_IBVERBS_SYM("ibv_reg_mr", symbols.ibv_internal_reg_mr);
  // Cherry-pick the ibv_reg_mr_iova2 API from IBVERBS 1.8
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_reg_mr_iova2", symbols.ibv_internal_reg_mr_iova2, "IBVERBS_1.8");
  // Cherry-pick the ibv_reg_dmabuf_mr API from IBVERBS 1.12
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_reg_dmabuf_mr", symbols.ibv_internal_reg_dmabuf_mr, "IBVERBS_1.12");
  LOAD_IBVERBS_SYM("ibv_dereg_mr", symbols.ibv_internal_dereg_mr);
  LOAD_IBVERBS_SYM("ibv_create_cq", symbols.ibv_internal_create_cq);
  LOAD_IBVERBS_SYM("ibv_destroy_cq", symbols.ibv_internal_destroy_cq);
  LOAD_IBVERBS_SYM("ibv_create_qp", symbols.ibv_internal_create_qp);
  LOAD_IBVERBS_SYM("ibv_modify_qp", symbols.ibv_internal_modify_qp);
  LOAD_IBVERBS_SYM("ibv_destroy_qp", symbols.ibv_internal_destroy_qp);
  LOAD_IBVERBS_SYM("ibv_fork_init", symbols.ibv_internal_fork_init);
  LOAD_IBVERBS_SYM("ibv_event_type_str", symbols.ibv_internal_event_type_str);

  LOAD_IBVERBS_SYM_VERSION(
      "ibv_create_comp_channel",
      symbols.ibv_internal_create_comp_channel,
      "IBVERBS_1.0");
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_destroy_comp_channel",
      symbols.ibv_internal_destroy_comp_channel,
      "IBVERBS_1.0");
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_get_cq_event", symbols.ibv_internal_get_cq_event, "IBVERBS_1.0");
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_ack_cq_events", symbols.ibv_internal_ack_cq_events, "IBVERBS_1.0");
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_query_ece", symbols.ibv_internal_query_ece, "IBVERBS_1.10");
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_set_ece", symbols.ibv_internal_set_ece, "IBVERBS_1.10");
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_is_fork_initialized",
      symbols.ibv_internal_is_fork_initialized,
      "IBVERBS_1.13");

  LOAD_MLX5DV_SYM("mlx5dv_is_supported", symbols.mlx5dv_internal_is_supported);
  // Cherry-pick the mlx5dv_get_data_direct_sysfs_path API from MLX5 1.2
  LOAD_MLX5DV_SYM_VERSION(
      "mlx5dv_init_obj", symbols.mlx5dv_internal_init_obj, "MLX5_1.2");
  // Cherry-pick the mlx5dv_get_data_direct_sysfs_path API from MLX5 1.25
  LOAD_MLX5DV_SYM_VERSION(
      "mlx5dv_get_data_direct_sysfs_path",
      symbols.mlx5dv_internal_get_data_direct_sysfs_path,
      "MLX5_1.25");
  // Cherry-pick the ibv_reg_dmabuf_mr API from MLX5 1.25
  LOAD_MLX5DV_SYM_VERSION(
      "mlx5dv_reg_dmabuf_mr",
      symbols.mlx5dv_internal_reg_dmabuf_mr,
      "MLX5_1.25");

  // all symbols were loaded successfully, dismiss guard
  guard.dismiss();
  return 0;
#endif
}

} // namespace ibverbx
