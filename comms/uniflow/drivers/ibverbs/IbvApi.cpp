// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/ibverbs/IbvApi.h"

#if !UNIFLOW_IBV_DIRECT || !UNIFLOW_MLX5_DIRECT
#include <dlfcn.h>
#endif

#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>

namespace uniflow {

// ---------------------------------------------------------------------------
// Function pointer declarations
// ---------------------------------------------------------------------------

#if UNIFLOW_IBV_DIRECT
// Static link: function pointers resolve directly to rdma-core symbols.
// Some ibverbs APIs (ibv_reg_mr, ibv_query_port) are macros in verbs.h,
// so we need passthrough wrappers to make them addressable (same approach
// as nccl ibvsymbols.cc).
static ibv_mr*
uniflow_ibv_reg_mr(ibv_pd* pd, void* addr, size_t length, int access) {
  return ibv_reg_mr(pd, addr, length, access);
}

static int uniflow_ibv_query_port(
    ibv_context* context,
    uint8_t port_num,
    ibv_port_attr* port_attr) {
  return ibv_query_port(context, port_num, port_attr);
}

#define UNIFLOW_IBV_FN(name, rettype, arglist) \
  constexpr rettype(*pfn_##name) arglist = ::name;
// Override for macro-based APIs:
#define UNIFLOW_IBV_FN_WRAP(name, wrapper, rettype, arglist) \
  rettype(*pfn_##name) arglist = wrapper;
#else
// Dynamic link: function pointers filled in by dlvsym during init.
#define UNIFLOW_IBV_FN(name, rettype, arglist) \
  rettype(*pfn_##name) arglist = nullptr;
#define UNIFLOW_IBV_FN_WRAP(name, wrapper, rettype, arglist) \
  rettype(*pfn_##name) arglist = nullptr;
#endif

namespace {

// NOLINTBEGIN(facebook-avoid-non-const-global-variables)
UNIFLOW_IBV_FN(ibv_get_device_list, ibv_device**, (int* num_devices))
UNIFLOW_IBV_FN(ibv_free_device_list, void, (ibv_device * *list))
UNIFLOW_IBV_FN(ibv_get_device_name, const char*, (ibv_device * device))
UNIFLOW_IBV_FN(ibv_open_device, ibv_context*, (ibv_device * device))
UNIFLOW_IBV_FN(ibv_close_device, int, (ibv_context * context))
UNIFLOW_IBV_FN(ibv_alloc_pd, ibv_pd*, (ibv_context * context))
UNIFLOW_IBV_FN(ibv_dealloc_pd, int, (ibv_pd * pd))
UNIFLOW_IBV_FN_WRAP(
    ibv_reg_mr,
    uniflow_ibv_reg_mr,
    ibv_mr*,
    (ibv_pd * pd, void* addr, size_t length, int access))
UNIFLOW_IBV_FN(
    ibv_reg_dmabuf_mr,
    ibv_mr*,
    (ibv_pd * pd,
     uint64_t offset,
     size_t length,
     uint64_t iova,
     int fd,
     int access))
UNIFLOW_IBV_FN(ibv_dereg_mr, int, (ibv_mr * mr))
UNIFLOW_IBV_FN(
    ibv_create_cq,
    ibv_cq*,
    (ibv_context * context,
     int cqe,
     void* cq_context,
     ibv_comp_channel* channel,
     int comp_vector))
UNIFLOW_IBV_FN(ibv_destroy_cq, int, (ibv_cq * cq))
UNIFLOW_IBV_FN(
    ibv_create_qp,
    ibv_qp*,
    (ibv_pd * pd, ibv_qp_init_attr* qp_init_attr))
UNIFLOW_IBV_FN(
    ibv_modify_qp,
    int,
    (ibv_qp * qp, ibv_qp_attr* attr, int attr_mask))
UNIFLOW_IBV_FN(ibv_destroy_qp, int, (ibv_qp * qp))
UNIFLOW_IBV_FN(
    ibv_query_device,
    int,
    (ibv_context * context, ibv_device_attr* device_attr))
UNIFLOW_IBV_FN_WRAP(
    ibv_query_port,
    uniflow_ibv_query_port,
    int,
    (ibv_context * context, uint8_t port_num, ibv_port_attr* port_attr))
UNIFLOW_IBV_FN(
    ibv_query_gid,
    int,
    (ibv_context * context, uint8_t port_num, int index, ibv_gid* gid))

#undef UNIFLOW_IBV_FN
#undef UNIFLOW_IBV_FN_WRAP

// --- MLX5 direct verbs function pointers ---
#if UNIFLOW_MLX5_DIRECT
// Static link: resolve directly to libmlx5 symbols.
#define UNIFLOW_MLX5_FN(name, rettype, arglist) \
  constexpr rettype(*pfn_##name) arglist = ::name;
#else
// Dynamic link: filled in by dlvsym during init.
#define UNIFLOW_MLX5_FN(name, rettype, arglist) \
  rettype(*pfn_##name) arglist = nullptr;
#endif

UNIFLOW_MLX5_FN(mlx5dv_is_supported, bool, (ibv_device * device))
UNIFLOW_MLX5_FN(
    mlx5dv_get_data_direct_sysfs_path,
    int,
    (ibv_context * context, char* buf, size_t buf_len))
UNIFLOW_MLX5_FN(
    mlx5dv_reg_dmabuf_mr,
    ibv_mr*,
    (ibv_pd * pd,
     uint64_t offset,
     size_t length,
     uint64_t iova,
     int fd,
     int access,
     int mlx5_access))

#undef UNIFLOW_MLX5_FN

std::once_flag g_initFlag;
Status g_initStatus{Ok()};
// NOLINTEND(facebook-avoid-non-const-global-variables)

void doInit() {
/// Load a versioned symbol from a dlopen handle into pfn_##name.
/// Required: fails init on missing symbol.
#define LOAD_SYM(handle, name, version)                                        \
  do {                                                                         \
    *reinterpret_cast<void**>(&pfn_##name) = dlvsym(handle, #name, version);   \
    if (pfn_##name == nullptr) {                                               \
      g_initStatus =                                                           \
          Err(ErrCode::DriverError,                                            \
              std::string("dlvsym failed on " #name " version ") + (version)); \
      return;                                                                  \
    }                                                                          \
  } while (0)

/// Optional: silently ignores missing symbol (pointer stays nullptr).
#define LOAD_SYM_OPTIONAL(handle, name, version) \
  *reinterpret_cast<void**>(&pfn_##name) = dlvsym(handle, #name, version)

#if !UNIFLOW_IBV_DIRECT
  constexpr const char* kIbverbsVersion = "IBVERBS_1.1";

  void* libHandle = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    libHandle = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (libHandle == nullptr) {
      g_initStatus =
          Err(ErrCode::DriverError, "Failed to open libibverbs.so[.1]");
      return;
    }
  }

  // Required symbols — all use IBVERBS_1.1
  LOAD_SYM(libHandle, ibv_get_device_list, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_free_device_list, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_get_device_name, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_open_device, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_close_device, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_alloc_pd, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_dealloc_pd, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_reg_mr, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_dereg_mr, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_create_cq, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_destroy_cq, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_create_qp, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_modify_qp, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_destroy_qp, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_query_device, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_query_port, kIbverbsVersion);
  LOAD_SYM(libHandle, ibv_query_gid, kIbverbsVersion);

  // Optional symbols — fail silently (pointer stays nullptr).
  LOAD_SYM_OPTIONAL(libHandle, ibv_reg_dmabuf_mr, "IBVERBS_1.12");

  // Deliberately never dlclose — the loaded object remains in memory
  // until the process terminates.
#endif

#if !UNIFLOW_MLX5_DIRECT
  // --- Load MLX5 direct verbs (optional) ---
  // libmlx5 is a separate shared object from libibverbs.
  // Failure to load is not fatal — mlx5dv APIs just stay nullptr.
  void* mlx5Handle = dlopen("libmlx5.so", RTLD_NOW | RTLD_LOCAL);
  if (mlx5Handle == nullptr) {
    mlx5Handle = dlopen("libmlx5.so.1", RTLD_NOW | RTLD_LOCAL);
  }
  if (mlx5Handle != nullptr) {
    LOAD_SYM(mlx5Handle, mlx5dv_is_supported, "MLX5_1.8");
    LOAD_SYM_OPTIONAL(
        mlx5Handle, mlx5dv_get_data_direct_sysfs_path, "MLX5_1.25");
    LOAD_SYM_OPTIONAL(mlx5Handle, mlx5dv_reg_dmabuf_mr, "MLX5_1.25");
  }
#endif

#undef LOAD_SYM
#undef LOAD_SYM_OPTIONAL
}

/// Ensure init() has been called.
#define IBV_ENSURE_INIT()           \
  do {                              \
    auto _s = init();               \
    if (_s.hasError()) {            \
      return std::move(_s).error(); \
    }                               \
  } while (0)

/// Check that a function pointer is resolved.
#define IBV_CHECK_FN(name)                                         \
  do {                                                             \
    if (pfn_##name == nullptr) {                                   \
      return Err(ErrCode::DriverError, #name " symbol not found"); \
    }                                                              \
  } while (0)

/// Check an already-obtained errno return; expect ret == 0. Return Status.
/// Used for hot-path ops vtable calls that bypass init/symbol checks.
#define IBV_CHECK_ERRNO(name, ret)                                \
  do {                                                            \
    if ((ret) != 0) {                                             \
      return Err(                                                 \
          ErrCode::DriverError,                                   \
          std::string(#name "() failed: ") + std::strerror(ret)); \
    }                                                             \
    return Ok();                                                  \
  } while (0)

/// Call pfn_##name(__VA_ARGS__); expect ret == 0 (returns errno on failure).
/// Error message includes strerror(ret). Matches nccl IBV_CHECK_INT_ERRNO.
#define IBV_CHECK_INT_ERRNO(name, ...)  \
  do {                                  \
    IBV_ENSURE_INIT();                  \
    IBV_CHECK_FN(name);                 \
    int _ret = pfn_##name(__VA_ARGS__); \
    IBV_CHECK_ERRNO(name, _ret);        \
  } while (0)

/// Call pfn_##name(__VA_ARGS__); expect ret != error_retval.
/// No errno info. Matches nccl IBV_CHECK_INT.
#define IBV_CHECK_INT(name, error_retval, ...)             \
  do {                                                     \
    IBV_ENSURE_INIT();                                     \
    IBV_CHECK_FN(name);                                    \
    int _ret = pfn_##name(__VA_ARGS__);                    \
    if (_ret == (error_retval)) {                          \
      return Err(ErrCode::DriverError, #name "() failed"); \
    }                                                      \
    return Ok();                                           \
  } while (0)

/// Call pfn_##name(__VA_ARGS__); expect ret != nullptr.
/// Error message includes strerror(errno). Matches nccl
/// IBV_CHECK_PTR_ERRNO.
#define IBV_CHECK_PTR_ERRNO(name, ...)                              \
  do {                                                              \
    IBV_ENSURE_INIT();                                              \
    IBV_CHECK_FN(name);                                             \
    auto _ptr = pfn_##name(__VA_ARGS__);                            \
    if (_ptr == nullptr) {                                          \
      return Err(                                                   \
          ErrCode::DriverError,                                     \
          std::string(#name "() failed: ") + std::strerror(errno)); \
    }                                                               \
    return _ptr;                                                    \
  } while (0)

/// Call pfn_##name(__VA_ARGS__); expect ret != nullptr.
/// No errno info. Matches nccl IBV_CHECK_PTR.
#define IBV_CHECK_PTR(name, ...)                           \
  do {                                                     \
    IBV_ENSURE_INIT();                                     \
    IBV_CHECK_FN(name);                                    \
    auto _ptr = pfn_##name(__VA_ARGS__);                   \
    if (_ptr == nullptr) {                                 \
      return Err(ErrCode::DriverError, #name "() failed"); \
    }                                                      \
    return _ptr;                                           \
  } while (0)

} // namespace

// ---------------------------------------------------------------------------
// IbvApi implementation
// ---------------------------------------------------------------------------

Status IbvApi::init() {
  std::call_once(g_initFlag, doInit);
  return g_initStatus;
}

// --- Device management ---

Result<ibv_device**> IbvApi::getDeviceList(int* numDevices) {
  IBV_CHECK_PTR_ERRNO(ibv_get_device_list, numDevices);
}

Status IbvApi::freeDeviceList(ibv_device** list) {
  IBV_ENSURE_INIT();
  IBV_CHECK_FN(ibv_free_device_list);
  pfn_ibv_free_device_list(list);
  return Ok();
}

Result<const char*> IbvApi::getDeviceName(ibv_device* device) {
  IBV_CHECK_PTR(ibv_get_device_name, device);
}

Result<ibv_context*> IbvApi::openDevice(ibv_device* device) {
  IBV_CHECK_PTR(ibv_open_device, device);
}

Status IbvApi::closeDevice(ibv_context* context) {
  IBV_CHECK_INT(ibv_close_device, -1, context);
}

// --- Protection domain ---

Result<ibv_pd*> IbvApi::allocPd(ibv_context* context) {
  IBV_CHECK_PTR_ERRNO(ibv_alloc_pd, context);
}

Status IbvApi::deallocPd(ibv_pd* pd) {
  IBV_CHECK_INT_ERRNO(ibv_dealloc_pd, pd);
}

// --- Memory registration ---

Result<ibv_mr*>
IbvApi::regMr(ibv_pd* pd, void* addr, size_t length, int access) {
  IBV_CHECK_PTR_ERRNO(ibv_reg_mr, pd, addr, length, access);
}

Status IbvApi::deregMr(ibv_mr* mr) {
  IBV_CHECK_INT_ERRNO(ibv_dereg_mr, mr);
}

Result<bool> IbvApi::isDmaBufSupported(ibv_pd* pd) {
  IBV_ENSURE_INIT();
  if (pfn_ibv_reg_dmabuf_mr == nullptr) {
    return false;
  }
  // Probe with a dummy call (fd=-1). The call will fail, but the errno
  // tells us whether the API is supported:
  //   EBADF           → supported (fd is just invalid)
  //   EOPNOTSUPP      → not supported by the kernel
  //   EPROTONOSUPPORT → not supported by the kernel
  errno = 0;
  pfn_ibv_reg_dmabuf_mr(
      pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  return errno != EOPNOTSUPP && errno != EPROTONOSUPPORT;
}

Result<ibv_mr*> IbvApi::regDmabufMr(
    ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access) {
  IBV_CHECK_PTR_ERRNO(ibv_reg_dmabuf_mr, pd, offset, length, iova, fd, access);
}

// --- Completion queue ---

Result<ibv_cq*> IbvApi::createCq(
    ibv_context* context,
    int cqe,
    void* cqContext,
    ibv_comp_channel* channel,
    int compVector) {
  IBV_CHECK_PTR_ERRNO(
      ibv_create_cq, context, cqe, cqContext, channel, compVector);
}

Status IbvApi::destroyCq(ibv_cq* cq) {
  IBV_CHECK_INT_ERRNO(ibv_destroy_cq, cq);
}

Result<int> IbvApi::pollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) {
  // Hot path: use ops vtable directly, skip init/symbol checks.
  int done = cq->context->ops.poll_cq(cq, numEntries, wc);
  if (done < 0) {
    return Err(
        ErrCode::DriverError, "ibv_poll_cq() returned " + std::to_string(done));
  }
  return done;
}

// --- Queue pair ---

Result<ibv_qp*> IbvApi::createQp(ibv_pd* pd, ibv_qp_init_attr* attr) {
  IBV_CHECK_PTR_ERRNO(ibv_create_qp, pd, attr);
}

Status IbvApi::destroyQp(ibv_qp* qp) {
  IBV_CHECK_INT_ERRNO(ibv_destroy_qp, qp);
}

Status IbvApi::modifyQp(ibv_qp* qp, ibv_qp_attr* attr, int attrMask) {
  IBV_CHECK_INT_ERRNO(ibv_modify_qp, qp, attr, attrMask);
}

// --- Data path (hot path — uses ops vtable) ---

Status IbvApi::postSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** badWr) {
  int ret = qp->context->ops.post_send(qp, wr, badWr);
  IBV_CHECK_ERRNO(ibv_post_send, ret);
}

Status IbvApi::postRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** badWr) {
  int ret = qp->context->ops.post_recv(qp, wr, badWr);
  IBV_CHECK_ERRNO(ibv_post_recv, ret);
}

// --- Query ---

Status IbvApi::queryDevice(ibv_context* context, ibv_device_attr* attr) {
  IBV_CHECK_INT_ERRNO(ibv_query_device, context, attr);
}

Status
IbvApi::queryPort(ibv_context* context, uint8_t portNum, ibv_port_attr* attr) {
  IBV_CHECK_INT_ERRNO(ibv_query_port, context, portNum, attr);
}

Status IbvApi::queryGid(
    ibv_context* context,
    uint8_t portNum,
    int index,
    ibv_gid* gid) {
  IBV_CHECK_INT_ERRNO(ibv_query_gid, context, portNum, index, gid);
}

// --- MLX5 direct verbs ---

Result<bool> IbvApi::mlx5dvIsSupported(ibv_device* device) {
  IBV_ENSURE_INIT();
  return pfn_mlx5dv_is_supported && pfn_mlx5dv_is_supported(device);
}

Status IbvApi::mlx5dvGetDataDirectSysfsPath(
    ibv_context* context,
    char* buf,
    size_t bufLen) {
  IBV_CHECK_INT_ERRNO(mlx5dv_get_data_direct_sysfs_path, context, buf, bufLen);
}

Result<ibv_mr*> IbvApi::mlx5dvRegDmabufMr(
    ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access,
    int mlx5Access) {
  IBV_CHECK_PTR_ERRNO(
      mlx5dv_reg_dmabuf_mr, pd, offset, length, iova, fd, access, mlx5Access);
}

} // namespace uniflow
