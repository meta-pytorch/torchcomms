// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/IbverbsLazy.h"

#include <dlfcn.h>
#include <glog/logging.h>

#include <mutex>

namespace comms::pipes {

namespace {

// ---- ibverbs function pointer types ----
using IbvRegMrIova2Fn =
    struct ibv_mr* (*)(struct ibv_pd*, void*, size_t, uint64_t, unsigned int);
using IbvRegDmabufMrFn =
    struct ibv_mr* (*)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);

IbvRegMrIova2Fn gIbvRegMrIova2 = nullptr;
IbvRegDmabufMrFn gIbvRegDmabufMr = nullptr;

std::once_flag gIbvLoadFlag;
int gIbvLoadResult = -1;

void do_load_ibverbs() {
  void* handle = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!handle) {
    handle = dlopen("libibverbs.so.1", RTLD_NOW);
  }
  if (!handle) {
    LOG(ERROR) << "IbverbsLazy: failed to dlopen libibverbs.so.1: "
               << dlerror();
    gIbvLoadResult = 1;
    return;
  }

  gIbvRegMrIova2 = reinterpret_cast<IbvRegMrIova2Fn>(
      dlvsym(handle, "ibv_reg_mr_iova2", "IBVERBS_1.8"));
  if (!gIbvRegMrIova2) {
    LOG(WARNING) << "IbverbsLazy: ibv_reg_mr_iova2 not available: "
                 << dlerror();
  }

  gIbvRegDmabufMr = reinterpret_cast<IbvRegDmabufMrFn>(
      dlvsym(handle, "ibv_reg_dmabuf_mr", "IBVERBS_1.12"));
  if (!gIbvRegDmabufMr) {
    LOG(WARNING) << "IbverbsLazy: ibv_reg_dmabuf_mr not available: "
                 << dlerror();
  }

  gIbvLoadResult = 0;
}

int load_ibverbs_lazy() {
  std::call_once(gIbvLoadFlag, do_load_ibverbs);
  return gIbvLoadResult;
}

// ---- mlx5dv function pointer types ----
using Mlx5dvIsSupportedFn = int (*)(struct ibv_device*);
using Mlx5dvGetDataDirectSysfsPathFn =
    int (*)(struct ibv_context*, char*, size_t);

Mlx5dvIsSupportedFn gMlx5dvIsSupported = nullptr;
Mlx5dvGetDataDirectSysfsPathFn gMlx5dvGetDataDirectSysfsPath = nullptr;

std::once_flag gMlx5LoadFlag;
bool gMlx5Loaded = false;

void do_load_mlx5() {
  void* handle = dlopen("libmlx5.so", RTLD_NOW);
  if (!handle) {
    handle = dlopen("libmlx5.so.1", RTLD_NOW);
  }
  if (!handle) {
    LOG(WARNING) << "IbverbsLazy: failed to dlopen libmlx5.so[.1]: "
                 << dlerror()
                 << ". Data Direct NIC discovery will be disabled.";
    return;
  }

  gMlx5dvIsSupported = reinterpret_cast<Mlx5dvIsSupportedFn>(
      dlsym(handle, "mlx5dv_is_supported"));
  if (!gMlx5dvIsSupported) {
    LOG(WARNING) << "IbverbsLazy: mlx5dv_is_supported not available: "
                 << dlerror();
  }

  // mlx5dv_get_data_direct_sysfs_path — available since MLX5_1.25
  gMlx5dvGetDataDirectSysfsPath =
      reinterpret_cast<Mlx5dvGetDataDirectSysfsPathFn>(
          dlvsym(handle, "mlx5dv_get_data_direct_sysfs_path", "MLX5_1.25"));
  if (!gMlx5dvGetDataDirectSysfsPath) {
    LOG(WARNING)
        << "IbverbsLazy: mlx5dv_get_data_direct_sysfs_path not available: "
        << dlerror();
  }

  gMlx5Loaded = true;
}

void load_mlx5_lazy() {
  std::call_once(gMlx5LoadFlag, do_load_mlx5);
}

} // namespace

struct ibv_mr* lazy_ibv_reg_mr_iova2(
    struct ibv_pd* pd,
    void* addr,
    size_t length,
    uint64_t iova,
    unsigned int access) {
  load_ibverbs_lazy();
  if (!gIbvRegMrIova2) {
    return nullptr;
  }
  return gIbvRegMrIova2(pd, addr, length, iova, access);
}

struct ibv_mr* lazy_ibv_reg_dmabuf_mr(
    struct ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access) {
  load_ibverbs_lazy();
  if (!gIbvRegDmabufMr) {
    return nullptr;
  }
  return gIbvRegDmabufMr(pd, offset, length, iova, fd, access);
}

int lazy_mlx5dv_is_supported(struct ibv_device* device) {
  load_mlx5_lazy();
  if (!gMlx5dvIsSupported) {
    return 0;
  }
  return gMlx5dvIsSupported(device);
}

int lazy_mlx5dv_get_data_direct_sysfs_path(
    struct ibv_context* ctx,
    char* buf,
    size_t buf_len) {
  load_mlx5_lazy();
  if (!gMlx5dvGetDataDirectSysfsPath) {
    return -1;
  }
  return gMlx5dvGetDataDirectSysfsPath(ctx, buf, buf_len);
}

} // namespace comms::pipes
