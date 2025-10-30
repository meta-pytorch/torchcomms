// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/Ibverbx.h"

#ifdef IBVERBX_BUILD_RDMA_CORE
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#endif

#include <dlfcn.h>
#include <folly/ScopeGuard.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <folly/synchronization/CallOnce.h>
#include <stdexcept>
#include "comms/utils/cvars/nccl_cvars.h"

namespace ibverbx {

namespace {

IbvSymbols ibvSymbols;
folly::once_flag initIbvSymbolOnce;

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
#endif

bool mlx5dvDmaBufDataDirectLinkCapable(
    ibv_device* device,
    ibv_context* context) {
  if (ibvSymbols.mlx5dv_internal_is_supported == nullptr ||
      ibvSymbols.mlx5dv_internal_reg_dmabuf_mr == nullptr ||
      ibvSymbols.mlx5dv_internal_get_data_direct_sysfs_path == nullptr) {
    return false;
  }

  if (!ibvSymbols.mlx5dv_internal_is_supported(device)) {
    return false;
  }
  int dev_fail = 0;
  ibv_pd* pd = nullptr;
  pd = ibvSymbols.ibv_internal_alloc_pd(context);
  if (!pd) {
    XLOG(ERR) << "ibv_alloc_pd failed: " << folly::errnoStr(errno);
    return false;
  }

  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)ibvSymbols.ibv_internal_reg_dmabuf_mr(
      pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
  // supported (EBADF otherwise)
  (void)ibvSymbols.mlx5dv_internal_reg_dmabuf_mr(
      pd,
      0ULL /*offset*/,
      0ULL /*len*/,
      0ULL /*iova*/,
      -1 /*fd*/,
      0 /*flags*/,
      0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
  // supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  if (ibvSymbols.ibv_internal_dealloc_pd(pd) != 0) {
    XLOGF(
        WARN,
        "ibv_dealloc_pd failed: {} DMA-BUF support status: {}",
        folly::errnoStr(errno),
        dev_fail);
    return false;
  }
  if (dev_fail) {
    XLOGF(INFO, "Kernel DMA-BUF is not supported on device {}", device->name);
    return false;
  }

  char dataDirectDevicePath[PATH_MAX];
  snprintf(dataDirectDevicePath, PATH_MAX, "/sys");
  return ibvSymbols.mlx5dv_internal_get_data_direct_sysfs_path(
             context, dataDirectDevicePath + 4, PATH_MAX - 4) == 0;
}

} // namespace

int buildIbvSymbols(IbvSymbols& symbols) {
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
  symbols.ibv_internal_req_notify_cq = &linked_req_notify_cq;
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

  if (!NCCL_IBVERBS_PATH.empty()) {
    ibvhandle = dlopen(NCCL_IBVERBS_PATH.c_str(), RTLD_NOW);
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
  // TODO: ibv_req_notify_cq is found not in any version of IBVERBS
  LOAD_IBVERBS_SYM_VERSION(
      "ibv_req_notify_cq", symbols.ibv_internal_req_notify_cq, "IBVERBS_1.0");
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

folly::Expected<folly::Unit, Error> ibvInit() {
  static std::atomic<int> errNum{1};
  folly::call_once(
      initIbvSymbolOnce, [&]() { errNum = buildIbvSymbols(ibvSymbols); });
  if (errNum != 0) {
    return folly::makeUnexpected(Error(errNum));
  }
  return folly::unit;
}

folly::Expected<folly::Unit, Error>
ibvGetCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context) {
  int rc = ibvSymbols.ibv_internal_get_cq_event(channel, cq, cq_context);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

void ibvAckCqEvents(ibv_cq* cq, unsigned int nevents) {
  ibvSymbols.ibv_internal_ack_cq_events(cq, nevents);
}

/*** Error ***/

Error::Error() : errNum(errno), errStr(folly::errnoStr(errno)) {}
Error::Error(int errNum) : errNum(errNum), errStr(folly::errnoStr(errNum)) {}
Error::Error(int errNum, std::string errStr)
    : errNum(errNum), errStr(std::move(errStr)) {}

std::ostream& operator<<(std::ostream& out, Error const& err) {
  out << err.errStr << " (errno=" << err.errNum << ")";
  return out;
}

/*** IbvMr ***/

IbvMr::IbvMr(ibv_mr* mr) : mr_(mr) {}

IbvMr::IbvMr(IbvMr&& other) noexcept {
  mr_ = other.mr_;
  other.mr_ = nullptr;
}

IbvMr& IbvMr::operator=(IbvMr&& other) noexcept {
  mr_ = other.mr_;
  other.mr_ = nullptr;
  return *this;
}

IbvMr::~IbvMr() {
  if (mr_) {
    int rc = ibvSymbols.ibv_internal_dereg_mr(mr_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to deregister mr rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_mr* IbvMr::mr() const {
  return mr_;
}

/*** IbvPd ***/

IbvPd::IbvPd(ibv_pd* pd, Coordinator* coordinator, bool dataDirect)
    : pd_(pd), coordinator_(coordinator), dataDirect_(dataDirect) {}

IbvPd::IbvPd(IbvPd&& other) noexcept {
  pd_ = other.pd_;
  coordinator_ = other.coordinator_;
  dataDirect_ = other.dataDirect_;
  other.pd_ = nullptr;
  other.coordinator_ = nullptr;
}

IbvPd& IbvPd::operator=(IbvPd&& other) noexcept {
  pd_ = other.pd_;
  coordinator_ = other.coordinator_;
  dataDirect_ = other.dataDirect_;
  other.pd_ = nullptr;
  other.coordinator_ = nullptr;
  return *this;
}

IbvPd::~IbvPd() {
  if (pd_) {
    int rc = ibvSymbols.ibv_internal_dealloc_pd(pd_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to deallocate pd rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_pd* IbvPd::pd() const {
  return pd_;
}

bool IbvPd::useDataDirect() const {
  return dataDirect_;
}

folly::Expected<IbvMr, Error>
IbvPd::regMr(void* addr, size_t length, ibv_access_flags access) const {
  ibv_mr* mr;
  mr = ibvSymbols.ibv_internal_reg_mr(pd_, addr, length, access);
  if (!mr) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvMr(mr);
}

folly::Expected<IbvMr, Error> IbvPd::regDmabufMr(
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    ibv_access_flags access) const {
  ibv_mr* mr;
  if (dataDirect_) {
    mr = ibvSymbols.mlx5dv_internal_reg_dmabuf_mr(
        pd_,
        offset,
        length,
        iova,
        fd,
        access,
        MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  } else {
    mr = ibvSymbols.ibv_internal_reg_dmabuf_mr(
        pd_, offset, length, iova, fd, access);
  }
  if (!mr) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvMr(mr);
}

folly::Expected<IbvQp, Error> IbvPd::createQp(
    ibv_qp_init_attr* initAttr) const {
  ibv_qp* qp;
  qp = ibvSymbols.ibv_internal_create_qp(pd_, initAttr);
  if (!qp) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvQp(qp);
}

folly::Expected<IbvVirtualQp, Error> IbvPd::createVirtualQp(
    int totalQps,
    ibv_qp_init_attr* initAttr,
    IbvVirtualCq* sendCq,
    IbvVirtualCq* recvCq,
    int maxMsgCntPerQp,
    int maxMsgSize,
    LoadBalancingScheme loadBalancingScheme) const {
  std::vector<IbvQp> qps;
  qps.reserve(totalQps);

  if (sendCq == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "Empty sendCq being provided to createVirtualQp"));
  }

  if (recvCq == nullptr) {
    return folly::makeUnexpected(
        Error(EINVAL, "Empty recvCq being provided to createVirtualQp"));
  }

  // Overwrite the CQs in the initAttr to point to the virtual CQ
  initAttr->send_cq = sendCq->getPhysicalCqRef().cq();
  initAttr->recv_cq = recvCq->getPhysicalCqRef().cq();

  // First create all the data QPs
  for (int i = 0; i < totalQps; i++) {
    auto maybeQp = createQp(initAttr);
    if (maybeQp.hasError()) {
      return folly::makeUnexpected(maybeQp.error());
    }
    qps.emplace_back(std::move(*maybeQp));
  }

  // Create notify QP
  auto maybeNotifyQp = createQp(initAttr);
  if (maybeNotifyQp.hasError()) {
    return folly::makeUnexpected(maybeNotifyQp.error());
  }

  // Create the IbvVirtualQp instance
  IbvVirtualQp virtualQp(
      std::move(qps),
      std::move(*maybeNotifyQp),
      coordinator_,
      maxMsgCntPerQp,
      maxMsgSize,
      loadBalancingScheme);

  // Populate the physicalQpNumToVirtualQp_ map in the coordinator
  for (const auto& qp : virtualQp.getQpsRef()) {
    coordinator_->registerPhysicalQpNumToVirtualQp(qp.qp()->qp_num, &virtualQp);
  }
  coordinator_->registerPhysicalQpNumToVirtualQp(
      virtualQp.getNotifyQpRef().qp()->qp_num, &virtualQp);

  // Populate virtualQpNumToVirtualSendCq_ and virtualQpNumToVirtualRecvCq_ map
  // in the coordinator
  coordinator_->registerVirtualQpNumToVirtualSendCq(
      virtualQp.getVirtualQpNum(), sendCq);
  coordinator_->registerVirtualQpNumToVirtualRecvCq(
      virtualQp.getVirtualQpNum(), recvCq);

  return virtualQp;
}

/*** IbvCq ***/

IbvCq::IbvCq(ibv_cq* cq) : cq_(cq) {}

IbvCq::~IbvCq() {
  if (cq_) {
    int rc = ibvSymbols.ibv_internal_destroy_cq(cq_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy cq rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvCq::IbvCq(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  other.cq_ = nullptr;
}

IbvCq& IbvCq::operator=(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  other.cq_ = nullptr;
  return *this;
}

ibv_cq* IbvCq::cq() const {
  return cq_;
}

folly::Expected<folly::Unit, Error> IbvCq::reqNotifyCq(
    int solicited_only) const {
  int rc = ibvSymbols.ibv_internal_req_notify_cq(cq_, solicited_only);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

/*** IbvDevice ***/

// hcaList format examples:
// - Without port: "mlx5_0,mlx5_1,mlx5_2"
// - With port: "mlx5_0:1,mlx5_1:0,mlx5_2:1"
// - Prefix match: "mlx5"
// hcaPrefix: use "=" for exact match, "^" for exclude match, "" for prefix
// match. See guidelines:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca
folly::Expected<std::vector<IbvDevice>, Error> IbvDevice::ibvGetDeviceList(
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix,
    int defaultPort) {
  // Get device list
  ibv_device** devs{nullptr};
  int numDevs;
  devs = ibvSymbols.ibv_internal_get_device_list(&numDevs);
  if (!devs) {
    return folly::makeUnexpected(Error(errno));
  }
  auto devices =
      ibvFilterDeviceList(numDevs, devs, hcaList, hcaPrefix, defaultPort);
  // Free device list
  ibvSymbols.ibv_internal_free_device_list(devs);
  return devices;
}

std::vector<IbvDevice> IbvDevice::ibvFilterDeviceList(
    int numDevs,
    ibv_device** devs,
    const std::vector<std::string>& hcaList,
    const std::string& hcaPrefix,
    int defaultPort) {
  std::vector<IbvDevice> devices;

  if (hcaList.empty()) {
    devices.reserve(numDevs);
    for (int i = 0; i < numDevs; i++) {
      devices.emplace_back(devs[i], defaultPort);
    }
    return devices;
  }

  // Convert the provided list of HCA strings into a vector of RoceHca
  // objects, which enables efficient device filter operation
  std::vector<RoceHca> hcas;
  // Avoid copy triggered by resize
  hcas.reserve(hcaList.size());
  for (const auto& hca : hcaList) {
    // Copy value to each vector element so it can be freed automatically
    hcas.emplace_back(hca, defaultPort);
  }

  // Filter devices
  if (hcaPrefix == "=") {
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (hca.name == devs[i]->name) {
          devices.emplace_back(devs[i], hca.port);
          break;
        }
      }
    }
    return devices;
  } else if (hcaPrefix == "^") {
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (hca.name != devs[i]->name) {
          devices.emplace_back(devs[i], defaultPort);
          break;
        }
      }
    }
    return devices;
  } else {
    // Prefix match
    for (const auto& hca : hcas) {
      for (int i = 0; i < numDevs; i++) {
        if (strncmp(devs[i]->name, hca.name.c_str(), hca.name.length()) == 0) {
          devices.emplace_back(devs[i], hca.port);
          break;
        }
      }
    }
    return devices;
  }
}

IbvDevice::IbvDevice(ibv_device* ibvDevice, int port) : device_(ibvDevice) {
  port_ = port;
  context_ = ibvSymbols.ibv_internal_open_device(device_);
  if (!context_) {
    XLOGF(ERR, "Failed to open device {}", device_->name);
    throw std::runtime_error(
        fmt::format("Failed to open device {}", device_->name));
  }
  if ((mlx5dvDmaBufDataDirectLinkCapable(device_, context_))) {
    // Now check whether Data Direct has been disabled by the user
    dataDirect_ = NCCL_IB_DATA_DIRECT == 1;
    XLOGF(
        INFO,
        "NET/IB: Data Direct DMA Interface is detected for device: {} dataDirect: {}",
        device_->name,
        dataDirect_);
  }
}

IbvDevice::~IbvDevice() {
  if (context_) {
    int rc = ibvSymbols.ibv_internal_close_device(context_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to close device rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvDevice::IbvDevice(IbvDevice&& other) noexcept {
  device_ = other.device_;
  context_ = other.context_;
  port_ = other.port_;
  coordinator_ = std::move(other.coordinator_);
  dataDirect_ = other.dataDirect_;

  other.device_ = nullptr;
  other.context_ = nullptr;
}

IbvDevice& IbvDevice::operator=(IbvDevice&& other) noexcept {
  device_ = other.device_;
  context_ = other.context_;
  port_ = other.port_;
  coordinator_ = std::move(other.coordinator_);
  dataDirect_ = other.dataDirect_;

  other.device_ = nullptr;
  other.context_ = nullptr;
  return *this;
}

ibv_device* IbvDevice::device() const {
  return device_;
}

ibv_context* IbvDevice::context() const {
  return context_;
}

int IbvDevice::port() const {
  return port_;
}

folly::Expected<IbvPd, Error> IbvDevice::allocPd() {
  ibv_pd* pd;
  pd = ibvSymbols.ibv_internal_alloc_pd(context_);
  if (!pd) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvPd(pd, &coordinator_, dataDirect_);
}

folly::Expected<IbvPd, Error> IbvDevice::allocParentDomain(
    ibv_parent_domain_init_attr* attr) {
  ibv_pd* pd;

  if (ibvSymbols.ibv_internal_alloc_parent_domain == nullptr) {
    return folly::makeUnexpected(Error(ENOSYS));
  }

  pd = ibvSymbols.ibv_internal_alloc_parent_domain(context_, attr);

  if (!pd) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvPd(pd, &coordinator_, dataDirect_);
}

folly::Expected<ibv_device_attr, Error> IbvDevice::queryDevice() const {
  ibv_device_attr deviceAttr{};
  int rc = ibvSymbols.ibv_internal_query_device(context_, &deviceAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return deviceAttr;
}

folly::Expected<ibv_port_attr, Error> IbvDevice::queryPort(
    uint8_t portNum) const {
  ibv_port_attr portAttr{};
  int rc = ibvSymbols.ibv_internal_query_port(context_, portNum, &portAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return portAttr;
}

folly::Expected<ibv_gid, Error> IbvDevice::queryGid(
    uint8_t portNum,
    int gidIndex) const {
  ibv_gid gid{};
  int rc = ibvSymbols.ibv_internal_query_gid(context_, portNum, gidIndex, &gid);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return gid;
}

folly::Expected<IbvCq, Error> IbvDevice::createCq(
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) const {
  ibv_cq* cq;
  cq = ibvSymbols.ibv_internal_create_cq(
      context_, cqe, cq_context, channel, comp_vector);
  if (!cq) {
    return folly::makeUnexpected(Error(errno));
  }
  return IbvCq(cq);
}

folly::Expected<IbvVirtualCq, Error> IbvDevice::createVirtualCq(
    int cqe,
    void* cq_context,
    ibv_comp_channel* channel,
    int comp_vector) {
  auto maybeCq = createCq(cqe, cq_context, channel, comp_vector);
  if (maybeCq.hasError()) {
    return folly::makeUnexpected(maybeCq.error());
  }
  return IbvVirtualCq(std::move(*maybeCq), cqe, &coordinator_);
}

folly::Expected<IbvCq, Error> IbvDevice::createCq(
    ibv_cq_init_attr_ex* attr) const {
  ibv_cq_ex* cqEx;
  cqEx = ibvSymbols.ibv_internal_create_cq_ex(context_, attr);
  if (!cqEx) {
    return folly::makeUnexpected(Error(errno));
  }
  ibv_cq* cq = ibv_cq_ex_to_cq(cqEx);
  return IbvCq(cq);
}

folly::Expected<ibv_comp_channel*, Error> IbvDevice::createCompChannel() const {
  ibv_comp_channel* channel;
  channel = ibvSymbols.ibv_internal_create_comp_channel(context_);
  if (!channel) {
    return folly::makeUnexpected(Error(errno));
  }
  return channel;
}

folly::Expected<folly::Unit, Error> IbvDevice::destroyCompChannel(
    ibv_comp_channel* channel) const {
  int rc = ibvSymbols.ibv_internal_destroy_comp_channel(channel);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

/*** IbvQp ***/
IbvQp::IbvQp(ibv_qp* qp) : qp_(qp) {}

IbvQp::~IbvQp() {
  if (qp_) {
    int rc = ibvSymbols.ibv_internal_destroy_qp(qp_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy qp rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvQp::IbvQp(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  other.qp_ = nullptr;
}

IbvQp& IbvQp::operator=(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  other.qp_ = nullptr;
  return *this;
}

ibv_qp* IbvQp::qp() const {
  return qp_;
}

folly::Expected<folly::Unit, Error> IbvQp::modifyQp(
    ibv_qp_attr* attr,
    int attrMask) {
  int rc = ibvSymbols.ibv_internal_modify_qp(qp_, attr, attrMask);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<std::pair<ibv_qp_attr, ibv_qp_init_attr>, Error> IbvQp::queryQp(
    int attrMask) const {
  ibv_qp_attr qpAttr{};
  ibv_qp_init_attr initAttr{};
  int rc = ibvSymbols.ibv_internal_query_qp(qp_, &qpAttr, attrMask, &initAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return std::make_pair(qpAttr, initAttr);
}

void IbvQp::enquePhysicalSendWrStatus(int physicalWrId, int virtualWrId) {
  physicalSendWrStatus_.emplace_back(physicalWrId, virtualWrId);
}

void IbvQp::dequePhysicalSendWrStatus() {
  physicalSendWrStatus_.pop_front();
}

void IbvQp::dequePhysicalRecvWrStatus() {
  physicalRecvWrStatus_.pop_front();
}

void IbvQp::enquePhysicalRecvWrStatus(int physicalWrId, int virtualWrId) {
  physicalRecvWrStatus_.emplace_back(physicalWrId, virtualWrId);
}

bool IbvQp::isSendQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalSendWrStatus_.size() < maxMsgCntPerQp;
}

bool IbvQp::isRecvQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalRecvWrStatus_.size() < maxMsgCntPerQp;
}

/*** IbvVirtualQp ***/

IbvVirtualQp::IbvVirtualQp(
    std::vector<IbvQp>&& qps,
    IbvQp&& notifyQp,
    Coordinator* coordinator,
    int maxMsgCntPerQp,
    int maxMsgSize,
    LoadBalancingScheme loadBalancingScheme)
    : physicalQps_(std::move(qps)),
      coordinator_(coordinator),
      maxMsgCntPerQp_(maxMsgCntPerQp),
      maxMsgSize_(maxMsgSize),
      loadBalancingScheme_(loadBalancingScheme),
      notifyQp_(std::move(notifyQp)) {
  for (int i = 0; i < physicalQps_.size(); i++) {
    qpNumToIdx_[physicalQps_.at(i).qp()->qp_num] = i;
  }
}

size_t IbvVirtualQp::getTotalQps() const {
  return physicalQps_.size();
}

const std::vector<IbvQp>& IbvVirtualQp::getQpsRef() const {
  return physicalQps_;
}

std::vector<IbvQp>& IbvVirtualQp::getQpsRef() {
  return physicalQps_;
}

const IbvQp& IbvVirtualQp::getNotifyQpRef() const {
  return notifyQp_;
}

uint32_t IbvVirtualQp::getVirtualQpNum() const {
  return virtualQpNum_;
}

IbvVirtualQp::IbvVirtualQp(IbvVirtualQp&& other) noexcept
    : physicalQps_(std::move(other.physicalQps_)),
      qpNumToIdx_(std::move(other.qpNumToIdx_)),
      nextSendPhysicalQpIdx_(other.nextSendPhysicalQpIdx_),
      nextRecvPhysicalQpIdx_(other.nextRecvPhysicalQpIdx_),
      coordinator_(other.coordinator_),
      maxMsgCntPerQp_(other.maxMsgCntPerQp_),
      maxMsgSize_(other.maxMsgSize_),
      loadBalancingScheme_(other.loadBalancingScheme_),
      notifyQp_(std::move(other.notifyQp_)) {
  // Update all entries in coordinator that point to &other to point to this
  if (coordinator_) {
    coordinator_->updateVirtualQpMapping(&other, this);
  }

  other.physicalQps_.clear();
  other.nextSendPhysicalQpIdx_ = 0;
  other.nextRecvPhysicalQpIdx_ = 0;
  other.qpNumToIdx_.clear();
  other.maxMsgCntPerQp_ = 0;
  other.maxMsgSize_ = 0;
  other.coordinator_ = nullptr;
}

IbvVirtualQp& IbvVirtualQp::operator=(IbvVirtualQp&& other) noexcept {
  if (this != &other) {
    physicalQps_ = std::move(other.physicalQps_);
    notifyQp_ = std::move(other.notifyQp_);
    nextSendPhysicalQpIdx_ = other.nextSendPhysicalQpIdx_;
    nextRecvPhysicalQpIdx_ = other.nextRecvPhysicalQpIdx_;
    qpNumToIdx_ = std::move(other.qpNumToIdx_);
    maxMsgCntPerQp_ = other.maxMsgCntPerQp_;
    maxMsgSize_ = other.maxMsgSize_;
    coordinator_ = other.coordinator_;
    loadBalancingScheme_ = other.loadBalancingScheme_;

    // Update all entries in coordinator that point to &other to point to this
    if (coordinator_) {
      coordinator_->updateVirtualQpMapping(&other, this);
    }

    other.physicalQps_.clear();
    other.nextSendPhysicalQpIdx_ = 0;
    other.nextRecvPhysicalQpIdx_ = 0;
    other.qpNumToIdx_.clear();
    other.maxMsgCntPerQp_ = 0;
    other.maxMsgSize_ = 0;
    other.coordinator_ = nullptr;
  }
  return *this;
}

folly::Expected<folly::Unit, Error> IbvVirtualQp::modifyVirtualQp(
    ibv_qp_attr* attr,
    int attrMask,
    const IbvVirtualQpBusinessCard& businessCard) {
  // If businessCard is not empty, use it to modify QPs with specific
  // dest_qp_num values
  if (!businessCard.qpNums_.empty()) {
    // Make sure the businessCard has the same number of QPs as physicalQps_
    if (businessCard.qpNums_.size() != physicalQps_.size()) {
      return folly::makeUnexpected(Error(
          EINVAL, "BusinessCard QP count doesn't match physical QP count"));
    }

    // Modify each QP with its corresponding dest_qp_num from the businessCard
    for (auto i = 0; i < physicalQps_.size(); i++) {
      attr->dest_qp_num = businessCard.qpNums_.at(i);
      auto maybeModifyQp = physicalQps_.at(i).modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    attr->dest_qp_num = businessCard.notifyQpNum_;
    auto maybeModifyQp = notifyQp_.modifyQp(attr, attrMask);
    if (maybeModifyQp.hasError()) {
      return folly::makeUnexpected(maybeModifyQp.error());
    }
  } else {
    // If no businessCard provided, modify all QPs with the same attributes
    for (auto& qp : physicalQps_) {
      auto maybeModifyQp = qp.modifyQp(attr, attrMask);
      if (maybeModifyQp.hasError()) {
        return folly::makeUnexpected(maybeModifyQp.error());
      }
    }
    auto maybeModifyQp = notifyQp_.modifyQp(attr, attrMask);
    if (maybeModifyQp.hasError()) {
      return folly::makeUnexpected(maybeModifyQp.error());
    }
  }
  return folly::unit;
}

IbvVirtualQpBusinessCard IbvVirtualQp::getVirtualQpBusinessCard() const {
  std::vector<uint32_t> qpNums;
  qpNums.reserve(physicalQps_.size());
  for (auto& qp : physicalQps_) {
    qpNums.push_back(qp.qp()->qp_num);
  }
  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQp_.qp()->qp_num);
}

LoadBalancingScheme IbvVirtualQp::getLoadBalancingScheme() const {
  return loadBalancingScheme_;
}

/*** IbvVirtualQpBusinessCard ***/

IbvVirtualQpBusinessCard::IbvVirtualQpBusinessCard(
    std::vector<uint32_t> qpNums,
    uint32_t notifyQpNum)
    : qpNums_(std::move(qpNums)), notifyQpNum_(notifyQpNum) {}

folly::dynamic IbvVirtualQpBusinessCard::toDynamic() const {
  folly::dynamic obj = folly::dynamic::object;
  folly::dynamic qpNumsArray = folly::dynamic::array;

  // Use fixed-width string formatting to ensure consistent size
  // All uint32_t values will be formatted as 10-digit zero-padded strings
  for (const auto& qpNum : qpNums_) {
    std::string paddedQpNum = fmt::format("{:010d}", qpNum);
    qpNumsArray.push_back(paddedQpNum);
  }

  obj["qpNums"] = std::move(qpNumsArray);
  obj["notifyQpNum"] = fmt::format("{:010d}", notifyQpNum_);
  return obj;
}

folly::Expected<IbvVirtualQpBusinessCard, Error>
IbvVirtualQpBusinessCard::fromDynamic(const folly::dynamic& obj) {
  std::vector<uint32_t> qpNums;

  if (obj.count("qpNums") > 0 && obj["qpNums"].isArray()) {
    const auto& qpNumsArray = obj["qpNums"];
    qpNums.reserve(qpNumsArray.size());

    for (const auto& qpNum : qpNumsArray) {
      CHECK(qpNum.isString()) << "qp num is not string!";
      try {
        uint32_t qpNumValue =
            static_cast<uint32_t>(std::stoul(qpNum.asString()));
        qpNums.push_back(qpNumValue);
      } catch (const std::exception& e) {
        return folly::makeUnexpected(Error(
            EINVAL,
            fmt::format(
                "Invalid QP number string format: {}. Exception: {}",
                qpNum.asString(),
                e.what())));
      }
    }
  } else {
    return folly::makeUnexpected(
        Error(EINVAL, "Invalid qpNums array received from remote side"));
  }

  uint32_t notifyQpNum = 0; // Default value for backwards compatibility
  if (obj.count("notifyQpNum") > 0 && obj["notifyQpNum"].isString()) {
    try {
      notifyQpNum =
          static_cast<uint32_t>(std::stoul(obj["notifyQpNum"].asString()));
    } catch (const std::exception& e) {
      return folly::makeUnexpected(Error(
          EINVAL,
          fmt::format(
              "Invalid notifyQpNum string format: {}. Exception: {}",
              obj["notifyQpNum"].asString(),
              e.what())));
    }
  }

  return IbvVirtualQpBusinessCard(std::move(qpNums), notifyQpNum);
}

std::string IbvVirtualQpBusinessCard::serialize() const {
  return folly::toJson(toDynamic());
}

folly::Expected<IbvVirtualQpBusinessCard, Error>
IbvVirtualQpBusinessCard::deserialize(const std::string& jsonStr) {
  try {
    folly::dynamic obj = folly::parseJson(jsonStr);
    return fromDynamic(obj);
  } catch (const std::exception& e) {
    return folly::makeUnexpected(Error(
        EINVAL,
        fmt::format(
            "Failed to parse JSON in IbvVirtualQpBusinessCard Deserialize. Exception: {}",
            e.what())));
  }
}

/*** IbvVirtualCq ***/

IbvVirtualCq::IbvVirtualCq(
    IbvCq&& physicalCq,
    int maxCqe,
    Coordinator* coordinator)
    : physicalCq_(std::move(physicalCq)),
      maxCqe_(maxCqe),
      coordinator_(coordinator) {}

IbvVirtualCq::IbvVirtualCq(IbvVirtualCq&& other) noexcept {
  physicalCq_ = std::move(other.physicalCq_);
  pendingSendVirtualWcQue_ = std::move(other.pendingSendVirtualWcQue_);
  pendingRecvVirtualWcQue_ = std::move(other.pendingRecvVirtualWcQue_);
  maxCqe_ = other.maxCqe_;
  coordinator_ = other.coordinator_;

  // Update all entries in coordinator that point to &other to point to this
  if (coordinator_) {
    coordinator_->updateVirtualSendCqMapping(&other, this);
    coordinator_->updateVirtualRecvCqMapping(&other, this);
  }

  // Reset the moved-from object
  other.maxCqe_ = 0;
  other.coordinator_ = nullptr;
}

IbvVirtualCq& IbvVirtualCq::operator=(IbvVirtualCq&& other) noexcept {
  if (this != &other) {
    physicalCq_ = std::move(other.physicalCq_);
    pendingSendVirtualWcQue_ = std::move(other.pendingSendVirtualWcQue_);
    pendingRecvVirtualWcQue_ = std::move(other.pendingRecvVirtualWcQue_);
    maxCqe_ = other.maxCqe_;
    coordinator_ = other.coordinator_;

    // Update all entries in coordinator that point to &other to point to this
    if (coordinator_) {
      coordinator_->updateVirtualSendCqMapping(&other, this);
      coordinator_->updateVirtualRecvCqMapping(&other, this);
    }

    // Reset the moved-from object
    other.maxCqe_ = 0;
    other.coordinator_ = nullptr;
  }
  return *this;
}

IbvCq& IbvVirtualCq::getPhysicalCqRef() {
  return physicalCq_;
}

void IbvVirtualCq::enqueSendCq(VirtualWc virtualWc) {
  pendingSendVirtualWcQue_.push_back(std::move(virtualWc));
}

void IbvVirtualCq::enqueRecvCq(VirtualWc virtualWc) {
  pendingRecvVirtualWcQue_.push_back(std::move(virtualWc));
}

/*** Coordinator ***/

// Register APIs for mapping management
void Coordinator::registerVirtualQpNumToVirtualSendCq(
    int virtualQpNum,
    IbvVirtualCq* virtualCq) {
  virtualQpNumToVirtualSendCq_[virtualQpNum] = virtualCq;
}

void Coordinator::registerVirtualQpNumToVirtualRecvCq(
    int virtualQpNum,
    IbvVirtualCq* virtualCq) {
  virtualQpNumToVirtualRecvCq_[virtualQpNum] = virtualCq;
}

void Coordinator::registerPhysicalQpNumToVirtualQp(
    int physicalQpNum,
    IbvVirtualQp* virtualQp) {
  physicalQpNumToVirtualQp_[physicalQpNum] = virtualQp;
}

IbvVirtualQp* Coordinator::getVirtualQp(int physicalQpNum) const {
  return physicalQpNumToVirtualQp_.at(physicalQpNum);
}

// Access APIs for testing and internal use
const std::unordered_map<int, IbvVirtualCq*>& Coordinator::getVirtualSendCqMap()
    const {
  return virtualQpNumToVirtualSendCq_;
}

const std::unordered_map<int, IbvVirtualCq*>& Coordinator::getVirtualRecvCqMap()
    const {
  return virtualQpNumToVirtualRecvCq_;
}

const std::unordered_map<int, IbvVirtualQp*>& Coordinator::getPhysicalQpMap()
    const {
  return physicalQpNumToVirtualQp_;
}

// Update API for move operations
void Coordinator::updateVirtualQpMapping(
    IbvVirtualQp* oldPtr,
    IbvVirtualQp* newPtr) {
  for (auto& entry : physicalQpNumToVirtualQp_) {
    if (entry.second == oldPtr) {
      entry.second = newPtr;
    }
  }
}

void Coordinator::updateVirtualSendCqMapping(
    IbvVirtualCq* oldPtr,
    IbvVirtualCq* newPtr) {
  for (auto& entry : virtualQpNumToVirtualSendCq_) {
    if (entry.second == oldPtr) {
      entry.second = newPtr;
    }
  }
}

void Coordinator::updateVirtualRecvCqMapping(
    IbvVirtualCq* oldPtr,
    IbvVirtualCq* newPtr) {
  for (auto& entry : virtualQpNumToVirtualRecvCq_) {
    if (entry.second == oldPtr) {
      entry.second = newPtr;
    }
  }
}

/*** RoceHCA ***/

RoceHca::RoceHca(std::string hcaStr, int defaultPort) {
  std::string s = std::move(hcaStr);
  std::string delim = ":";

  std::vector<std::string> hcaStrPair;
  folly::split(':', s, hcaStrPair);
  if (hcaStrPair.size() == 1) {
    this->name = s;
    this->port = defaultPort;
  } else if (hcaStrPair.size() == 2) {
    this->name = hcaStrPair.at(0);
    this->port = std::stoi(hcaStrPair.at(1));
  }
}

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
