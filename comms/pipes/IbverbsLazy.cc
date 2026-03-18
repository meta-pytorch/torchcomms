// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/IbverbsLazy.h"

#include <dlfcn.h>
#include <glog/logging.h>

#include <mutex>

namespace comms::pipes {

namespace {

// Function pointer types for ibverbs functions we load via dlopen.
using IbvRegMrIova2Fn =
    struct ibv_mr* (*)(struct ibv_pd*, void*, size_t, uint64_t, unsigned int);

using IbvRegDmabufMrFn =
    struct ibv_mr* (*)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);

// Loaded function pointers (populated by load_ibverbs_lazy).
IbvRegMrIova2Fn gIbvRegMrIova2 = nullptr;
IbvRegDmabufMrFn gIbvRegDmabufMr = nullptr;

std::once_flag gLoadFlag;
int gLoadResult = -1;

void do_load() {
  void* handle = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!handle) {
    // Not already loaded — open fresh
    handle = dlopen("libibverbs.so.1", RTLD_NOW);
  }
  if (!handle) {
    LOG(ERROR) << "IbverbsLazy: failed to dlopen libibverbs.so.1: "
               << dlerror();
    gLoadResult = 1;
    return;
  }

  // ibv_reg_mr_iova2 — available since IBVERBS 1.8
  gIbvRegMrIova2 = reinterpret_cast<IbvRegMrIova2Fn>(
      dlvsym(handle, "ibv_reg_mr_iova2", "IBVERBS_1.8"));
  if (!gIbvRegMrIova2) {
    LOG(WARNING) << "IbverbsLazy: ibv_reg_mr_iova2 not available: "
                 << dlerror();
  }

  // ibv_reg_dmabuf_mr — available since IBVERBS 1.12
  gIbvRegDmabufMr = reinterpret_cast<IbvRegDmabufMrFn>(
      dlvsym(handle, "ibv_reg_dmabuf_mr", "IBVERBS_1.12"));
  if (!gIbvRegDmabufMr) {
    LOG(WARNING) << "IbverbsLazy: ibv_reg_dmabuf_mr not available: "
                 << dlerror();
  }

  gLoadResult = 0;
}

int load_ibverbs_lazy() {
  std::call_once(gLoadFlag, do_load);
  return gLoadResult;
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

} // namespace comms::pipes
