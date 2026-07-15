// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PipesGdaHost - host-side `pipes_gda_*` impl for AMD/HIP
// =============================================================================
// Implements the host-side `pipes_gda_*` API for two NIC backends, selected at
// build time:
//   - mlx5 (default):  uses mlx5 direct verbs (mlx5dv_*) and the static
//                      libibverbs.
//   - bnxt:            uses Broadcom direct verbs (bnxt_re_dv_*) loaded from
//                      libbnxt_re-rdmav34.so via dlopen, and the system
//                      libibverbs.so.1 (PABI 34, supports kernel uverbs
//                      ABI 8 — required for bnxt_re).
//
// What's NIC-agnostic (shared code, ~50% of the file):
//   - HSA agent / pool discovery + `hsa_amd_memory_lock_to_pool`
//   - `pipes_gda_gpu_*` lifecycle (HIP device tracking, host-pinned alloc)
//   - QP / AH attribute setters (build IBV_QP_* mask incrementally)
//   - DMA-buf export via `hsa_amd_portable_export_dmabuf`
//
// What's NIC-specific (gated by `#ifdef NIC_BNXT`):
//   - ibverbs wrappers (BNXT routes through SysIbv → system libibverbs;
//     mlx5 calls the static libibverbs directly).
//   - `pipes_gda_verbs_qp_modify` (BNXT uses bnxt_re_dv.modify_qp's extended
//     5-arg signature; mlx5 calls plain ibv_modify_qp).
//   - `pipes_gda_gpu_verbs_create_qp_hl`:
//       BNXT  - bnxt_re_dv (cq_mem_alloc → umem_reg → create_cq → qp_mem_alloc
//               → alloc_db_region → create_qp → init_obj). SQ/CQ/RQ buffers in
//               GPU uncached memory; MSN table at SQ tail.
//       mlx5  - libibverbs create_qp/cq + mlx5dv_init_obj for raw layout, HSA
//               UAR-to-GPU mapping for BlueFlame, host-pinned SQ/CQ +
//               hipHostRegister.
//   - destroy / group variants follow the same split.
// =============================================================================

#ifdef __HIP_PLATFORM_AMD__

#include "pipes_gda/PipesGdaHost.h" // @manual

// pipes_gda_gpu_dev_verbs_qp is declared in this header; needed for
// constructing the device-side QP descriptor below.
#include "pipes_gda/PipesGdaDev.h" // @manual

#ifdef NIC_BNXT
// BNXT direct-verbs typedefs (struct definitions and function pointer types).
// The actual symbols are loaded lazily from libbnxt_re via dlopen.
#include "nic/BnxtReDv.h" // @manual
#else
// mlx5 direct verbs — implementation detail of the mlx5 backend.
#include <infiniband/mlx5dv.h>
#endif

#include <dlfcn.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace {

// ===========================================================================
// HSA Runtime Helpers — NIC-agnostic.
// ===========================================================================

struct HsaAgentInfo {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t pool;
};

static std::vector<HsaAgentInfo> g_hsaGpuAgents;
static std::vector<HsaAgentInfo> g_hsaCpuAgents;
static std::once_flag g_hsaInitFlag;
static bool g_hsaInitSuccess = false;

static hsa_status_t hsaPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_memory_pool_global_flag_t flag{};
  hsa_status_t st = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  if (st != HSA_STATUS_SUCCESS) {
    return st;
  }
  if (flag ==
      (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
       HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t hsaAgentCallback(hsa_agent_t agent, void*) {
  hsa_device_type_t devType{};
  hsa_status_t st = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &devType);
  if (st != HSA_STATUS_SUCCESS) {
    return st;
  }
  if (devType == HSA_DEVICE_TYPE_GPU) {
    g_hsaGpuAgents.emplace_back();
    g_hsaGpuAgents.back().agent = agent;
    st = hsa_amd_agent_iterate_memory_pools(
        agent, hsaPoolCallback, &g_hsaGpuAgents.back().pool);
  } else if (devType == HSA_DEVICE_TYPE_CPU) {
    g_hsaCpuAgents.emplace_back();
    g_hsaCpuAgents.back().agent = agent;
    st = hsa_amd_agent_iterate_memory_pools(
        agent, hsaPoolCallback, &g_hsaCpuAgents.back().pool);
  }
  return st;
}

static bool ensureHsaInitialized() {
  std::call_once(g_hsaInitFlag, []() {
    hsa_status_t st = hsa_init();
    if (st != HSA_STATUS_SUCCESS) {
      return;
    }
    st = hsa_iterate_agents(hsaAgentCallback, nullptr);
    if (st != HSA_STATUS_SUCCESS && st != HSA_STATUS_INFO_BREAK) {
      return;
    }
    g_hsaInitSuccess = true;
  });
  return g_hsaInitSuccess;
}

// hsa_amd_memory_lock_to_pool gives a GPU-accessible alias for a host
// pointer. Required for both:
//   - mlx5: BlueFlame UAR doorbell page (was already using this).
//   - bnxt: alloc_db_region's MMIO doorbell page. The hipHostRegister
//     path that BNXT previously used produces a GPU-side mapping that
//     is READ-ONLY on this ROCm release; the kernel's atomic store to
//     `qp->nic.bnxt.dbr` then faults with `(nil)` / `Write access to
//     a read-only page`. hsa_amd_memory_lock_to_pool produces a
//     writable GPU alias, matching how mlx5's BlueFlame UAR is mapped.
static bool
hsaMemoryLockToGpu(void* hostPtr, size_t size, void** gpuPtr, int gpuId) {
  if (!ensureHsaInitialized()) {
    return false;
  }
  if (gpuId < 0 || static_cast<size_t>(gpuId) >= g_hsaGpuAgents.size()) {
    return false;
  }
  if (g_hsaCpuAgents.empty()) {
    return false;
  }
  hsa_status_t st = hsa_amd_memory_lock_to_pool(
      hostPtr,
      size,
      &g_hsaGpuAgents[gpuId].agent,
      1,
      g_hsaCpuAgents[0].pool,
      0,
      gpuPtr);
  return st == HSA_STATUS_SUCCESS;
}

#ifdef NIC_BNXT
// ===========================================================================
// SysIbv — dlopen wrapper for system libibverbs.so.1 (PABI 34)
// ===========================================================================
//
// A statically-linked rdma-core (PABI 59) bundles a BNXT provider that only
// supports kernel uverbs ABI 1. The kernel bnxt_re module
// on AMD/BNXT hosts reports uverbs ABI 8, so the static provider rejects every
// device. The system /lib64/libibverbs.so.1 (PABI 34) +
// /lib64/libbnxt_re-rdmav34.so DO support ABI 8 — we dlopen them at runtime
// and route ALL ibverbs calls in the BNXT path through these.

struct SysIbv {
  void* handle{nullptr};
  struct ibv_device** (*get_device_list)(int*){nullptr};
  void (*free_device_list)(struct ibv_device**){nullptr};
  const char* (*get_device_name)(struct ibv_device*){nullptr};
  struct ibv_context* (*open_device)(struct ibv_device*){nullptr};
  int (*close_device)(struct ibv_context*){nullptr};
  struct ibv_pd* (*alloc_pd)(struct ibv_context*){nullptr};
  int (*dealloc_pd)(struct ibv_pd*){nullptr};
  int (*query_device)(struct ibv_context*, struct ibv_device_attr*){nullptr};
  int (*query_port)(struct ibv_context*, uint8_t, struct ibv_port_attr*){
      nullptr};
  int (*query_gid)(struct ibv_context*, uint8_t, int, union ibv_gid*){nullptr};
  struct ibv_mr* (*reg_mr)(struct ibv_pd*, void*, size_t, int){nullptr};
  int (*dereg_mr)(struct ibv_mr*){nullptr};
  int (*modify_qp)(struct ibv_qp*, struct ibv_qp_attr*, int){nullptr};
};

static SysIbv* getSysIbv() {
  static SysIbv s_ibv{};
  static std::once_flag s_once;
  static bool s_loaded = false;
  std::call_once(s_once, []() {
    // Force the SYSTEM libibverbs (PABI 34, supports kernel uverbs ABI 8).
    // A statically-linked libibverbs earlier in the loader search order
    // (PABI 59) would otherwise win the search.
    void* lib = dlopen("/lib64/libibverbs.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!lib) {
      lib = dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!lib) {
      fprintf(
          stderr,
          "PipesGdaHost(BNXT): dlopen libibverbs.so.1 failed: %s\n",
          dlerror());
      return;
    }
    s_ibv.handle = lib;
#define LOAD(sym)                                                       \
  do {                                                                  \
    s_ibv.sym =                                                         \
        reinterpret_cast<decltype(s_ibv.sym)>(dlsym(lib, "ibv_" #sym)); \
    if (!s_ibv.sym) {                                                   \
      fprintf(stderr, "SysIbv missing ibv_" #sym "\n");                 \
    }                                                                   \
  } while (0)
    LOAD(get_device_list);
    LOAD(free_device_list);
    LOAD(get_device_name);
    LOAD(open_device);
    LOAD(close_device);
    LOAD(alloc_pd);
    LOAD(dealloc_pd);
    LOAD(query_device);
    LOAD(query_port);
    LOAD(query_gid);
    LOAD(reg_mr);
    LOAD(dereg_mr);
    LOAD(modify_qp);
#undef LOAD
    if (!s_ibv.get_device_list || !s_ibv.open_device || !s_ibv.alloc_pd ||
        !s_ibv.query_port || !s_ibv.query_gid || !s_ibv.reg_mr ||
        !s_ibv.modify_qp) {
      return;
    }
    s_loaded = true;
  });
  return s_loaded ? &s_ibv : nullptr;
}

// ===========================================================================
// BNXT direct-verbs lazy loader
// ===========================================================================

struct bnxt_re_dv_funcs* getBnxtReDv() {
  static bnxt_re_dv_funcs s_funcs{};
  static std::once_flag s_dlOnce;
  static bool s_loaded = false;
  std::call_once(s_dlOnce, []() {
    void* lib = dlopen("libbnxt_re-rdmav34.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) {
      lib = dlopen("libbnxt_re-rdmav34.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!lib) {
      lib = dlopen("libbnxt_re.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!lib) {
      fprintf(
          stderr,
          "PipesGdaHost(BNXT): failed to dlopen libbnxt_re-rdmav34.so: %s\n",
          dlerror());
      return;
    }
    s_funcs.dl_handle = lib;
    s_funcs.init_obj = reinterpret_cast<bnxt_re_dv_init_obj_fn>(
        dlsym(lib, "bnxt_re_dv_init_obj"));
    s_funcs.alloc_db_region = reinterpret_cast<bnxt_re_dv_alloc_db_region_fn>(
        dlsym(lib, "bnxt_re_dv_alloc_db_region"));
    s_funcs.free_db_region = reinterpret_cast<bnxt_re_dv_free_db_region_fn>(
        dlsym(lib, "bnxt_re_dv_free_db_region"));
    s_funcs.umem_reg = reinterpret_cast<bnxt_re_dv_umem_reg_fn>(
        dlsym(lib, "bnxt_re_dv_umem_reg"));
    s_funcs.umem_dereg = reinterpret_cast<bnxt_re_dv_umem_dereg_fn>(
        dlsym(lib, "bnxt_re_dv_umem_dereg"));
    s_funcs.cq_mem_alloc = reinterpret_cast<bnxt_re_dv_cq_mem_alloc_fn>(
        dlsym(lib, "bnxt_re_dv_cq_mem_alloc"));
    s_funcs.create_cq = reinterpret_cast<bnxt_re_dv_create_cq_fn>(
        dlsym(lib, "bnxt_re_dv_create_cq"));
    s_funcs.destroy_cq = reinterpret_cast<bnxt_re_dv_destroy_cq_fn>(
        dlsym(lib, "bnxt_re_dv_destroy_cq"));
    s_funcs.qp_mem_alloc = reinterpret_cast<bnxt_re_dv_qp_mem_alloc_fn>(
        dlsym(lib, "bnxt_re_dv_qp_mem_alloc"));
    s_funcs.create_qp = reinterpret_cast<bnxt_re_dv_create_qp_fn>(
        dlsym(lib, "bnxt_re_dv_create_qp"));
    s_funcs.destroy_qp = reinterpret_cast<bnxt_re_dv_destroy_qp_fn>(
        dlsym(lib, "bnxt_re_dv_destroy_qp"));
    s_funcs.modify_qp = reinterpret_cast<bnxt_re_dv_modify_qp_fn>(
        dlsym(lib, "bnxt_re_dv_modify_qp"));

    if (!s_funcs.init_obj || !s_funcs.alloc_db_region || !s_funcs.umem_reg ||
        !s_funcs.cq_mem_alloc || !s_funcs.create_cq || !s_funcs.qp_mem_alloc ||
        !s_funcs.create_qp || !s_funcs.modify_qp) {
      fprintf(
          stderr,
          "PipesGdaHost(BNXT): libbnxt_re missing required bnxt_re_dv_* "
          "symbols\n");
      s_funcs = bnxt_re_dv_funcs{};
      return;
    }
    s_loaded = true;
  });
  return s_loaded ? &s_funcs : nullptr;
}

// Export a GPU device buffer as a DMA-BUF for NIC peer-to-peer DMA. Used by
// the BNXT QP create path to register SQ/CQ/RQ GPU buffers via umem_reg.
static int
exportGpuBufferDmabuf(void* ptr, size_t size, int* outFd, uint64_t* outOffset) {
  using ExportDmabufFn = int (*)(const void*, size_t, int*, uint64_t*);
  static ExportDmabufFn exportFn = nullptr;
  static std::once_flag dlOnce;
  std::call_once(dlOnce, []() {
    void* lib = dlopen("libhsa-runtime64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) {
      lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (lib) {
      exportFn = reinterpret_cast<ExportDmabufFn>(
          dlsym(lib, "hsa_amd_portable_export_dmabuf"));
    }
  });
  if (!exportFn) {
    return -1;
  }
  *outFd = -1;
  *outOffset = 0;
  return exportFn(ptr, size, outFd, outOffset);
}
#endif // NIC_BNXT

// ===========================================================================
// AMD-internal QP control block — tracks per-QP resources we registered so
// destroy_qp_hl can undo them. The mlx5 path registers SQ/CQ/DBREC pages via
// hipHostRegister and maps the BlueFlame UAR through HSA. The BNXT path
// allocates GPU device memory for SQ/CQ/RQ and gets a host-mapped DBR from
// alloc_db_region.
// ===========================================================================

#ifdef NIC_BNXT
struct AmdBnxtQpInternal {
  void* sq_umem{nullptr};
  void* rq_umem{nullptr};
  void* cq_umem{nullptr};
  bnxt_re_dv_db_region_attr* db_region{nullptr};

  // SQ/CQ/RQ in GPU device memory (uncached) for ~100ns local-HBM polls.
  void* sq_buf{nullptr};
  size_t sq_buf_size{0};
  void* cq_buf{nullptr};
  size_t cq_buf_size{0};
  void* rq_buf{nullptr};
  size_t rq_buf_size{0};

  // MMIO doorbell page registered via `hipHostRegister(Default)` so the GPU
  // can issue 64-bit atomic stores. Released via `hipHostUnregister` in
  // `freeQpResources`. Pointer is `dbRegion->dbr`, which is stored
  // independently in the `bnxt_re_dv_db_region_attr*` referenced by the QP.
  bool dbr_host_registered{false};
};
#else
struct AmdQpInternal {
  void* uar_bf_host{nullptr};
  size_t uar_bf_size{0};
  void* registered_sq_buf{nullptr};
  void* registered_cq_buf{nullptr};
  void* registered_sq_dbrec_page{nullptr};
  void* registered_cq_dbrec_page{nullptr}; // nullptr if same page as SQ
};
#endif

// ===========================================================================
// AMD-internal QP attribute / address-handle structs — IDENTICAL layout
// across backends so the (opaque) `pipes_gda_verbs_qp_attr*` /
// `pipes_gda_verbs_ah_attr*` produced by either backend are
// indistinguishable at the call site.
// ===========================================================================

struct AmdQpAttrImpl {
  ibv_qp_attr attr{};
  int attr_mask{0};
  pipes_gda_verbs_ah_attr* ah_attr_ref{nullptr};
};

struct AmdAhAttrImpl {
  pipes_gda_verbs_addr_type addr_type{PIPES_GDA_VERBS_ADDR_TYPE_IPv6};
  pipes_gda_verbs_gid gid{};
  uint16_t dlid{0};
  int sgid_index{0};
  uint8_t hop_limit{0};
  uint8_t sl{0};
  uint8_t traffic_class{0};
};

inline AmdQpAttrImpl* castQpAttr(pipes_gda_verbs_qp_attr* a) {
  return reinterpret_cast<AmdQpAttrImpl*>(a);
}
inline AmdAhAttrImpl* castAhAttr(pipes_gda_verbs_ah_attr* a) {
  return reinterpret_cast<AmdAhAttrImpl*>(a);
}

} // namespace

// ===========================================================================
// pipes_gda_gpu - GPU context
// ===========================================================================

struct pipes_gda_gpu {
  int hip_device_id{0};
};

namespace pipes_gda {

pipes_gda_error_t pipes_gda_gpu_create(
    const char* /*gpu_pci_bus_id*/,
    pipes_gda_gpu** out_gpu) {
  if (!out_gpu) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  if (!ensureHsaInitialized()) {
    return PIPES_GDA_ERROR_INITIALIZATION;
  }
  int devId = -1;
  if (hipGetDevice(&devId) != hipSuccess) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  auto* gpu = new (std::nothrow) pipes_gda_gpu();
  if (!gpu) {
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  gpu->hip_device_id = devId;
  *out_gpu = gpu;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_gpu_destroy(pipes_gda_gpu* gpu) {
  delete gpu;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_gpu_mem_alloc(
    pipes_gda_gpu* /*gpu*/,
    std::size_t size,
    std::size_t /*alignment*/,
    int /*mem_type*/,
    int /*access_type*/,
    void** out_ptr,
    void** out_gpu_ptr) {
  // Use host-pinned memory; HIP maps it 1:1 to the GPU address space.
  void* p = nullptr;
  hipError_t err = hipHostMalloc(&p, size, hipHostMallocDefault);
  if (err != hipSuccess) {
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  if (out_ptr) {
    *out_ptr = p;
  }
  if (out_gpu_ptr) {
    *out_gpu_ptr = p;
  }
  return PIPES_GDA_SUCCESS;
}

// ===========================================================================
// ibverbs wrappers
//
// BNXT: route through SysIbv (system /lib64/libibverbs.so.1) — the
// statically-linked libibverbs (PABI 59) cannot register the system BNXT
// provider (PABI 34).
// mlx5: direct calls into the statically-linked libibverbs.
// ===========================================================================

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_get_device_list(
    int* num_devices,
    ibv_device*** out_list) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  *out_list = iv->get_device_list(num_devices);
#else
  *out_list = ibv_get_device_list(num_devices);
#endif
  return *out_list ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_free_device_list(
    ibv_device** list) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv || !iv->free_device_list) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  iv->free_device_list(list);
#else
  ibv_free_device_list(list);
#endif
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_get_device_name(
    ibv_device* dev,
    const char** out_name) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv || !iv->get_device_name) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  *out_name = iv->get_device_name(dev);
#else
  *out_name = ibv_get_device_name(dev);
#endif
  return *out_name ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_open_device(
    ibv_device* dev,
    ibv_context** out_ctx) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  *out_ctx = iv->open_device(dev);
#else
  *out_ctx = ibv_open_device(dev);
#endif
  return *out_ctx ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_close_device(ibv_context* ctx) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv || !iv->close_device) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  return iv->close_device(ctx) == 0 ? PIPES_GDA_SUCCESS
                                    : PIPES_GDA_ERROR_DRIVER;
#else
  return ibv_close_device(ctx) == 0 ? PIPES_GDA_SUCCESS
                                    : PIPES_GDA_ERROR_DRIVER;
#endif
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_alloc_pd(
    ibv_context* ctx,
    ibv_pd** out_pd) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  *out_pd = iv->alloc_pd(ctx);
#else
  *out_pd = ibv_alloc_pd(ctx);
#endif
  return *out_pd ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_dealloc_pd(ibv_pd* pd) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv || !iv->dealloc_pd) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  return iv->dealloc_pd(pd) == 0 ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
#else
  return ibv_dealloc_pd(pd) == 0 ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
#endif
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_query_device(
    ibv_context* ctx,
    ibv_device_attr* attr) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv || !iv->query_device) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  return iv->query_device(ctx, attr) == 0 ? PIPES_GDA_SUCCESS
                                          : PIPES_GDA_ERROR_DRIVER;
#else
  return ibv_query_device(ctx, attr) == 0 ? PIPES_GDA_SUCCESS
                                          : PIPES_GDA_ERROR_DRIVER;
#endif
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_query_port(
    ibv_context* ctx,
    uint8_t port,
    ibv_port_attr* attr) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  return iv->query_port(ctx, port, attr) == 0 ? PIPES_GDA_SUCCESS
                                              : PIPES_GDA_ERROR_DRIVER;
#else
  return ibv_query_port(ctx, port, attr) == 0 ? PIPES_GDA_SUCCESS
                                              : PIPES_GDA_ERROR_DRIVER;
#endif
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_query_gid(
    ibv_context* ctx,
    uint8_t port,
    int index,
    union ibv_gid* gid) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  return iv->query_gid(ctx, port, index, gid) == 0 ? PIPES_GDA_SUCCESS
                                                   : PIPES_GDA_ERROR_DRIVER;
#else
  return ibv_query_gid(ctx, port, index, gid) == 0 ? PIPES_GDA_SUCCESS
                                                   : PIPES_GDA_ERROR_DRIVER;
#endif
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_reg_mr(
    ibv_pd* pd,
    void* addr,
    std::size_t length,
    int access,
    ibv_mr** out_mr) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  *out_mr = iv->reg_mr(pd, addr, length, access);
#else
  *out_mr = ibv_reg_mr(pd, addr, length, access);
#endif
  return *out_mr ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
}

pipes_gda_error_t pipes_gda_verbs_wrapper_ibv_dereg_mr(ibv_mr* mr) {
#ifdef NIC_BNXT
  auto* iv = getSysIbv();
  if (!iv || !iv->dereg_mr) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  return iv->dereg_mr(mr) == 0 ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
#else
  return ibv_dereg_mr(mr) == 0 ? PIPES_GDA_SUCCESS : PIPES_GDA_ERROR_DRIVER;
#endif
}

// ===========================================================================
// QP attribute setters (NIC-agnostic — forward to ibv_qp_attr)
// ===========================================================================

pipes_gda_error_t pipes_gda_verbs_qp_attr_create(
    pipes_gda_verbs_qp_attr** out_attr) {
  if (!out_attr) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  auto* impl = new (std::nothrow) AmdQpAttrImpl();
  if (!impl) {
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  *out_attr = reinterpret_cast<pipes_gda_verbs_qp_attr*>(impl);
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_destroy(
    pipes_gda_verbs_qp_attr* attr) {
  delete castQpAttr(attr);
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_next_state(
    pipes_gda_verbs_qp_attr* attr,
    pipes_gda_verbs_qp_state state) {
  auto* impl = castQpAttr(attr);
  impl->attr.qp_state = static_cast<ibv_qp_state>(state);
  impl->attr_mask |= IBV_QP_STATE;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_path_mtu(
    pipes_gda_verbs_qp_attr* attr,
    pipes_gda_mtu mtu) {
  auto* impl = castQpAttr(attr);
  impl->attr.path_mtu = static_cast<ibv_mtu>(mtu);
  impl->attr_mask |= IBV_QP_PATH_MTU;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_port_num(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t port) {
  auto* impl = castQpAttr(attr);
  impl->attr.port_num = port;
  impl->attr_mask |= IBV_QP_PORT;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_dest_qp_num(
    pipes_gda_verbs_qp_attr* attr,
    uint32_t dest_qp_num) {
  auto* impl = castQpAttr(attr);
  impl->attr.dest_qp_num = dest_qp_num;
  impl->attr_mask |= IBV_QP_DEST_QPN;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_rq_psn(
    pipes_gda_verbs_qp_attr* attr,
    uint32_t psn) {
  auto* impl = castQpAttr(attr);
  impl->attr.rq_psn = psn;
  impl->attr_mask |= IBV_QP_RQ_PSN;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_sq_psn(
    pipes_gda_verbs_qp_attr* attr,
    uint32_t psn) {
  auto* impl = castQpAttr(attr);
  impl->attr.sq_psn = psn;
  impl->attr_mask |= IBV_QP_SQ_PSN;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_min_rnr_timer(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v) {
  auto* impl = castQpAttr(attr);
  impl->attr.min_rnr_timer = v;
  impl->attr_mask |= IBV_QP_MIN_RNR_TIMER;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_ack_timeout(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v) {
  auto* impl = castQpAttr(attr);
  impl->attr.timeout = v;
  impl->attr_mask |= IBV_QP_TIMEOUT;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_retry_cnt(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v) {
  auto* impl = castQpAttr(attr);
  impl->attr.retry_cnt = v;
  impl->attr_mask |= IBV_QP_RETRY_CNT;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_rnr_retry(
    pipes_gda_verbs_qp_attr* attr,
    uint8_t v) {
  auto* impl = castQpAttr(attr);
  impl->attr.rnr_retry = v;
  impl->attr_mask |= IBV_QP_RNR_RETRY;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_allow_remote_read(
    pipes_gda_verbs_qp_attr* attr,
    bool allow) {
  auto* impl = castQpAttr(attr);
  if (allow) {
    impl->attr.qp_access_flags |= IBV_ACCESS_REMOTE_READ;
  } else {
    impl->attr.qp_access_flags &= ~IBV_ACCESS_REMOTE_READ;
  }
  impl->attr_mask |= IBV_QP_ACCESS_FLAGS;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_allow_remote_write(
    pipes_gda_verbs_qp_attr* attr,
    bool allow) {
  auto* impl = castQpAttr(attr);
  if (allow) {
    impl->attr.qp_access_flags |= IBV_ACCESS_REMOTE_WRITE;
  } else {
    impl->attr.qp_access_flags &= ~IBV_ACCESS_REMOTE_WRITE;
  }
  impl->attr_mask |= IBV_QP_ACCESS_FLAGS;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_allow_remote_atomic(
    pipes_gda_verbs_qp_attr* attr,
    int atomic_mode) {
  auto* impl = castQpAttr(attr);
  if (atomic_mode != PIPES_GDA_VERBS_QP_ATOMIC_MODE_NONE) {
    impl->attr.qp_access_flags |= IBV_ACCESS_REMOTE_ATOMIC;
  } else {
    impl->attr.qp_access_flags &= ~IBV_ACCESS_REMOTE_ATOMIC;
  }
  impl->attr_mask |= IBV_QP_ACCESS_FLAGS;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_qp_attr_set_ah_attr(
    pipes_gda_verbs_qp_attr* attr,
    pipes_gda_verbs_ah_attr* ah_attr) {
  castQpAttr(attr)->ah_attr_ref = ah_attr;
  return PIPES_GDA_SUCCESS;
}

// ===========================================================================
// Address handle attribute setters (NIC-agnostic)
// ===========================================================================

pipes_gda_error_t pipes_gda_verbs_ah_attr_create(
    ibv_context* /*ctx*/,
    pipes_gda_verbs_ah_attr** out_attr) {
  if (!out_attr) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  auto* impl = new (std::nothrow) AmdAhAttrImpl();
  if (!impl) {
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  *out_attr = reinterpret_cast<pipes_gda_verbs_ah_attr*>(impl);
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_destroy(
    pipes_gda_verbs_ah_attr* attr) {
  delete castAhAttr(attr);
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_addr_type(
    pipes_gda_verbs_ah_attr* attr,
    pipes_gda_verbs_addr_type t) {
  castAhAttr(attr)->addr_type = t;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_dlid(
    pipes_gda_verbs_ah_attr* attr,
    uint16_t dlid) {
  castAhAttr(attr)->dlid = dlid;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_gid(
    pipes_gda_verbs_ah_attr* attr,
    const pipes_gda_verbs_gid& gid) {
  castAhAttr(attr)->gid = gid;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_sgid_index(
    pipes_gda_verbs_ah_attr* attr,
    int idx) {
  castAhAttr(attr)->sgid_index = idx;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_hop_limit(
    pipes_gda_verbs_ah_attr* attr,
    uint8_t hop) {
  castAhAttr(attr)->hop_limit = hop;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_sl(
    pipes_gda_verbs_ah_attr* attr,
    uint8_t sl) {
  castAhAttr(attr)->sl = sl;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_verbs_ah_attr_set_traffic_class(
    pipes_gda_verbs_ah_attr* attr,
    uint8_t tc) {
  castAhAttr(attr)->traffic_class = tc;
  return PIPES_GDA_SUCCESS;
}

// ===========================================================================
// QP modify
// ===========================================================================

// Translate PIPES_GDA_VERBS_QP_ATTR_* (caller's DOCA-style mask) to the
// corresponding IBV_QP_* bits. Each enum lives in its own bit-position space,
// so we can't just OR the caller's mask into the kernel mask.
static int translatePipesGdaMaskToIbv(int pipesMask) {
  int ibvMask = 0;
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_NEXT_STATE) {
    ibvMask |= IBV_QP_STATE;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_PKEY_INDEX) {
    ibvMask |= IBV_QP_PKEY_INDEX;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_PORT_NUM) {
    ibvMask |= IBV_QP_PORT;
  }
  if (pipesMask &
      (PIPES_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
       PIPES_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
       PIPES_GDA_VERBS_QP_ATTR_ALLOW_REMOTE_ATOMIC)) {
    ibvMask |= IBV_QP_ACCESS_FLAGS;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_PATH_MTU) {
    ibvMask |= IBV_QP_PATH_MTU;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_DEST_QP_NUM) {
    ibvMask |= IBV_QP_DEST_QPN;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_RQ_PSN) {
    ibvMask |= IBV_QP_RQ_PSN;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_AH_ATTR) {
    ibvMask |= IBV_QP_AV;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_MIN_RNR_TIMER) {
    ibvMask |= IBV_QP_MIN_RNR_TIMER;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_SQ_PSN) {
    ibvMask |= IBV_QP_SQ_PSN;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_ACK_TIMEOUT) {
    ibvMask |= IBV_QP_TIMEOUT;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_RETRY_CNT) {
    ibvMask |= IBV_QP_RETRY_CNT;
  }
  if (pipesMask & PIPES_GDA_VERBS_QP_ATTR_RNR_RETRY) {
    ibvMask |= IBV_QP_RNR_RETRY;
  }
  return ibvMask;
}

pipes_gda_error_t pipes_gda_verbs_qp_modify(
    ibv_qp* qp,
    pipes_gda_verbs_qp_attr* attr,
    int attr_mask) {
  auto* impl = castQpAttr(attr);
  ibv_qp_attr ibvAttr = impl->attr;
  // Combine setter-accumulated IBV_QP_* flags with the user-supplied
  // PIPES_GDA mask (translated to IBV space).
  int mask =
      impl->attr_mask | translatePipesGdaMaskToIbv(static_cast<int>(attr_mask));

  // Apply the AH only when the caller explicitly includes AH_ATTR in the
  // current modify. The AH pointer is stored on the attr struct by an
  // earlier `set_ah_attr` call but stays live across subsequent modifies.
  if (impl->ah_attr_ref && (attr_mask & PIPES_GDA_VERBS_QP_ATTR_AH_ATTR) != 0) {
    auto* ah = castAhAttr(impl->ah_attr_ref);
    ibvAttr.ah_attr.is_global = 1;
    ibvAttr.ah_attr.dlid = ah->dlid;
    ibvAttr.ah_attr.sl = ah->sl;
    ibvAttr.ah_attr.port_num = impl->attr.port_num;
    std::memcpy(ibvAttr.ah_attr.grh.dgid.raw, ah->gid.raw, sizeof(ah->gid.raw));
    ibvAttr.ah_attr.grh.sgid_index = ah->sgid_index;
    ibvAttr.ah_attr.grh.hop_limit = ah->hop_limit;
    ibvAttr.ah_attr.grh.traffic_class = ah->traffic_class;
    mask |= IBV_QP_AV;
  }

  // RC QP responder/initiator depths are required by the kernel for the
  // INIT->RTR and RTR->RTS transitions, but NVIDIA DOCA's public API does
  // not expose setters for `max_dest_rd_atomic` / `max_rd_atomic`. We set
  // both to 16.
  if ((mask & IBV_QP_STATE) != 0) {
    if (ibvAttr.qp_state == IBV_QPS_RTR) {
      ibvAttr.max_dest_rd_atomic = 16;
      mask |= IBV_QP_MAX_DEST_RD_ATOMIC;
    } else if (ibvAttr.qp_state == IBV_QPS_RTS) {
      ibvAttr.max_rd_atomic = 16;
      mask |= IBV_QP_MAX_QP_RD_ATOMIC;
    }
  }

#ifdef NIC_BNXT
  // BNXT extended signature: (qp, attr, mask, type, value).
  // type=0, value=0 → no extension fields applied.
  auto* dv = getBnxtReDv();
  if (!dv) {
    return PIPES_GDA_ERROR_DRIVER;
  }
  int rc = dv->modify_qp(qp, &ibvAttr, mask, 0, 0);
#else
  int rc = ibv_modify_qp(qp, &ibvAttr, mask);
#endif
  if (rc != 0) {
    fprintf(
        stderr,
        "pipes_gda_verbs_qp_modify: modify_qp failed rc=%d errno=%d (%s) "
        "qp_state=%d mask=0x%x port=%u pkey=%u path_mtu=%d dest_qpn=%u "
        "rq_psn=%u sq_psn=%u\n",
        rc,
        errno,
        std::strerror(errno),
        static_cast<int>(ibvAttr.qp_state),
        mask,
        ibvAttr.port_num,
        ibvAttr.pkey_index,
        static_cast<int>(ibvAttr.path_mtu),
        ibvAttr.dest_qp_num,
        ibvAttr.rq_psn,
        ibvAttr.sq_psn);
    fflush(stderr);
    return PIPES_GDA_ERROR_DRIVER;
  }
  // Clear the setter-accumulated mask + AH ref so a subsequent modify on
  // the same `pipes_gda_verbs_qp_attr` only includes flags the caller re-
  // sets.
  impl->attr_mask = 0;
  impl->ah_attr_ref = nullptr;
  return PIPES_GDA_SUCCESS;
}

// ===========================================================================
// GPU verbs QP creation
// ===========================================================================

#ifdef NIC_BNXT

pipes_gda_error_t pipes_gda_gpu_verbs_create_qp_hl(
    const pipes_gda_gpu_verbs_qp_init_attr_hl* attr,
    pipes_gda_gpu_verbs_qp_hl** out_qp) {
  if (!attr || !out_qp || !attr->ibpd) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  if (!ensureHsaInitialized()) {
    return PIPES_GDA_ERROR_INITIALIZATION;
  }
  auto* dv = getBnxtReDv();
  if (!dv) {
    return PIPES_GDA_ERROR_DRIVER;
  }

  ibv_pd* pd = attr->ibpd;
  ibv_context* ctx = pd->context;

  void* cqMemHandle = nullptr;
  void* cqUmem = nullptr;
  ibv_cq* cq = nullptr;
  void* sqUmem = nullptr;
  void* rqUmem = nullptr;
  ibv_qp* qp = nullptr;
  bnxt_re_dv_db_region_attr* dbRegion = nullptr;
  void* sqBuf = nullptr;
  size_t sqBufSize = 0;
  void* rqBuf = nullptr;
  size_t rqBufSize = 0;
  void* cqBuf = nullptr;
  size_t cqBufSize = 0;
  bool dbrRegistered = false; // hipHostUnregister on unwind if true
  size_t pageSize = sysconf(_SC_PAGESIZE);

  auto unwind = [&]() {
    if (dbRegion && dbrRegistered) {
      hipHostUnregister(dbRegion->dbr);
    }
    if (qp && dv->destroy_qp) {
      dv->destroy_qp(qp);
    }
    if (sqUmem && dv->umem_dereg) {
      dv->umem_dereg(sqUmem);
    }
    if (rqUmem && dv->umem_dereg) {
      dv->umem_dereg(rqUmem);
    }
    if (dbRegion && dv->free_db_region) {
      dv->free_db_region(ctx, dbRegion);
    }
    if (cq && dv->destroy_cq) {
      dv->destroy_cq(cq);
    }
    if (cqUmem && dv->umem_dereg) {
      dv->umem_dereg(cqUmem);
    }
    if (cqBuf) {
      hipFree(cqBuf);
    }
    if (sqBuf) {
      hipFree(sqBuf);
    }
    if (rqBuf) {
      hipFree(rqBuf);
    }
  };

  // ---- Step 1: Allocate CQ memory descriptor ----
  // Force ncqe=1 to use BNXT's CQE compression mode: a single CQE slot
  // gets overwritten on each completion with a toggling phase bit. Our
  // device-side pollCqAt only reads CQE[0]'s phase bit, so any depth > 1
  // would produce non-deterministic polls.
  bnxt_re_dv_cq_attr cqAttr{};
  cqMemHandle = dv->cq_mem_alloc(ctx, /*num_cqe=*/1, &cqAttr);
  if (!cqMemHandle) {
    fprintf(stderr, "[BNXT] cq_mem_alloc failed (sq_nwqe=%u)\n", attr->sq_nwqe);
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }
  cqAttr.ncqe = 1;
  cqBufSize = static_cast<size_t>(cqAttr.ncqe) * cqAttr.cqe_size;
  if (cqBufSize == 0) {
    cqBufSize = 64;
  }
  // CQ buffer in GPU device memory (uncached): GPU poll loop reads local
  // HBM (~100ns) instead of PCIe-mapped host memory (~1us).
  hipError_t herr =
      hipExtMallocWithFlags(&cqBuf, cqBufSize, hipDeviceMallocFinegrained);
  if (herr != hipSuccess || !cqBuf) {
    fprintf(
        stderr,
        "[BNXT] hipExtMallocWithFlags(CQ %zu) failed: %s\n",
        cqBufSize,
        hipGetErrorString(herr));
    unwind();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  hipMemset(cqBuf, 0, cqBufSize);

  // ---- Step 2: Register CQ buffer with bnxt_re_dv (umem_reg) ----
  int cqDmabufFd = -1;
  uint64_t cqDmabufOffset = 0;
  exportGpuBufferDmabuf(cqBuf, cqBufSize, &cqDmabufFd, &cqDmabufOffset);

  bnxt_re_dv_umem_reg_attr cqUmemAttr{};
  cqUmemAttr.addr = cqBuf;
  cqUmemAttr.size = cqBufSize;
  cqUmemAttr.access_flags = IBV_ACCESS_LOCAL_WRITE;
  cqUmemAttr.dmabuf_fd = (cqDmabufFd >= 0) ? cqDmabufFd : 0;
  cqUmem = dv->umem_reg(ctx, &cqUmemAttr);
  if (cqDmabufFd >= 0) {
    close(cqDmabufFd);
  }
  if (!cqUmem) {
    fprintf(
        stderr,
        "[BNXT] cq umem_reg failed errno=%d (%s)\n",
        errno,
        std::strerror(errno));
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 3: Create CQ via bnxt_re_dv ----
  bnxt_re_dv_cq_init_attr cqInitAttr{};
  cqInitAttr.cq_handle = reinterpret_cast<uint64_t>(cqMemHandle);
  cqInitAttr.umem_handle = cqUmem;
  cqInitAttr.cq_umem_offset = 0;
  cqInitAttr.ncqe = cqAttr.ncqe;
  cq = dv->create_cq(ctx, &cqInitAttr);
  if (!cq) {
    fprintf(
        stderr,
        "[BNXT] create_cq failed errno=%d (%s)\n",
        errno,
        std::strerror(errno));
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 4: Allocate QP memory descriptor (returns SQ size incl. MSN
  // table tail and PSN-entry size) ----
  ibv_qp_init_attr qpInitAttr{};
  qpInitAttr.send_cq = cq;
  qpInitAttr.recv_cq = cq;
  qpInitAttr.cap.max_send_wr = attr->sq_nwqe;
  qpInitAttr.cap.max_recv_wr = 1;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.sq_sig_all = 0;

  bnxt_re_dv_qp_mem_info memInfo{};
  if (dv->qp_mem_alloc(pd, &qpInitAttr, &memInfo) != 0) {
    fprintf(
        stderr,
        "[BNXT] qp_mem_alloc failed errno=%d (%s)\n",
        errno,
        std::strerror(errno));
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // BNXT hardware tracks the SQ tail with a single epoch bit that toggles
  // on wrap. Mask-based wrap detection assumes sq_slots is a power of 2;
  // `qp_mem_alloc` may return a non-power-of-2 (e.g., 15616). Round
  // sq_slots up and grow sq_len to match (so the SQ region + MSN-table
  // tail still fit in the allocated buffer). Pairs with the per-slot
  // wrap in `BnxtNicBackend::getBnxtWqeSlot` so 3-slot WQEs that straddle
  // the buffer end don't trash the MSN table.
  {
    uint32_t v = memInfo.sq_slots;
    uint32_t rounded = 1;
    while (rounded < v) {
      rounded <<= 1;
    }
    if (rounded != memInfo.sq_slots) {
      uint32_t extraSlots = rounded - memInfo.sq_slots;
      memInfo.sq_slots = rounded;
      memInfo.sq_len += extraSlots * 16u;
    }
  }

  sqBufSize = memInfo.sq_len;
  if (sqBufSize == 0) {
    sqBufSize = pageSize;
  }
  rqBufSize = memInfo.rq_len;

  // SQ in GPU device memory. Use `hipDeviceMallocFinegrained` to match
  // rocSHMEM's QP buffer allocator (`HIPAllocatorFinegrained`).
  // We previously used `hipDeviceMallocUncached`, but on AMD MI300X
  // multiple-warp WQE writes to that memory type fault with `Memory
  // access fault by GPU on (nil)` / `Write access to a read-only page`.
  // Fine-grained coherent memory accepts the concurrent writes correctly
  // and is what rocSHMEM (the validated baseline) uses for SQ/CQ/RQ.
  herr = hipExtMallocWithFlags(&sqBuf, sqBufSize, hipDeviceMallocFinegrained);
  if (herr != hipSuccess || !sqBuf) {
    fprintf(
        stderr,
        "[BNXT] hipExtMallocWithFlags(SQ %zu) failed: %s\n",
        sqBufSize,
        hipGetErrorString(herr));
    unwind();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  hipMemset(sqBuf, 0, sqBufSize);

  if (rqBufSize > 0) {
    herr = hipExtMallocWithFlags(&rqBuf, rqBufSize, hipDeviceMallocFinegrained);
    if (herr != hipSuccess || !rqBuf) {
      fprintf(
          stderr,
          "[BNXT] hipExtMallocWithFlags(RQ %zu) failed: %s\n",
          rqBufSize,
          hipGetErrorString(herr));
      unwind();
      return PIPES_GDA_ERROR_NO_MEMORY;
    }
    hipMemset(rqBuf, 0, rqBufSize);
  }

  // bnxt_re_dv treats `qp_mem_alloc` as a size-negotiator; the caller is
  // expected to allocate the actual SQ/RQ buffers and write their VAs
  // back into `mem_info.{sq,rq}_va` before calling `create_qp`.
  memInfo.sq_va = reinterpret_cast<uint64_t>(sqBuf);
  memInfo.rq_va = reinterpret_cast<uint64_t>(rqBuf);

  // ---- Step 5: Register SQ + RQ buffers with bnxt_re_dv (with dma-buf
  // for NIC peer-to-peer DMA into GPU memory) ----
  int sqDmabufFd = -1;
  uint64_t sqDmabufOffset = 0;
  exportGpuBufferDmabuf(sqBuf, sqBufSize, &sqDmabufFd, &sqDmabufOffset);

  bnxt_re_dv_umem_reg_attr sqUmemAttr{};
  sqUmemAttr.addr = sqBuf;
  sqUmemAttr.size = sqBufSize;
  sqUmemAttr.access_flags = IBV_ACCESS_LOCAL_WRITE;
  sqUmemAttr.dmabuf_fd = (sqDmabufFd >= 0) ? sqDmabufFd : 0;
  sqUmem = dv->umem_reg(ctx, &sqUmemAttr);
  if (sqDmabufFd >= 0) {
    close(sqDmabufFd);
  }
  if (!sqUmem) {
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  if (rqBuf) {
    int rqDmabufFd = -1;
    uint64_t rqDmabufOffset = 0;
    exportGpuBufferDmabuf(rqBuf, rqBufSize, &rqDmabufFd, &rqDmabufOffset);

    bnxt_re_dv_umem_reg_attr rqUmemAttr{};
    rqUmemAttr.addr = rqBuf;
    rqUmemAttr.size = rqBufSize;
    rqUmemAttr.access_flags = IBV_ACCESS_LOCAL_WRITE;
    rqUmemAttr.dmabuf_fd = (rqDmabufFd >= 0) ? rqDmabufFd : 0;
    rqUmem = dv->umem_reg(ctx, &rqUmemAttr);
    if (rqDmabufFd >= 0) {
      close(rqDmabufFd);
    }
    if (!rqUmem) {
      unwind();
      return PIPES_GDA_ERROR_DRIVER;
    }
  }

  // ---- Step 6: Allocate doorbell region (host-mapped 64-bit register) ----
  dbRegion = dv->alloc_db_region(ctx);
  if (!dbRegion || !dbRegion->dbr) {
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 7: Create QP via bnxt_re_dv ----
  bnxt_re_dv_qp_init_attr dvQpAttr{};
  dvQpAttr.qp_type = IBV_QPT_RC;
  dvQpAttr.max_send_wr = attr->sq_nwqe;
  dvQpAttr.max_recv_wr = 1;
  dvQpAttr.max_send_sge = 1;
  dvQpAttr.max_recv_sge = 1;
  dvQpAttr.send_cq = cq;
  dvQpAttr.recv_cq = cq;
  dvQpAttr.qp_handle = memInfo.qp_handle;
  dvQpAttr.dbr_handle = dbRegion;
  dvQpAttr.sq_umem_handle = sqUmem;
  dvQpAttr.sq_umem_offset = 0;
  dvQpAttr.sq_len = memInfo.sq_len;
  dvQpAttr.sq_slots = memInfo.sq_slots;
  dvQpAttr.sq_wqe_sz = memInfo.sq_wqe_sz;
  dvQpAttr.sq_psn_sz = memInfo.sq_psn_sz;
  dvQpAttr.sq_npsn = memInfo.sq_npsn;
  dvQpAttr.rq_umem_handle = rqUmem;
  dvQpAttr.rq_umem_offset = 0;
  dvQpAttr.rq_len = memInfo.rq_len;
  dvQpAttr.rq_slots = memInfo.rq_slots;
  dvQpAttr.rq_wqe_sz = memInfo.rq_wqe_sz;
  // Propagate comp_mask from mem_info — carries optional DV feature bits
  // that the create_qp path may gate on.
  dvQpAttr.comp_mask = memInfo.comp_mask;

  qp = dv->create_qp(pd, &dvQpAttr);
  if (!qp) {
    fprintf(
        stderr,
        "[BNXT] create_qp failed errno=%d (%s)\n",
        errno,
        std::strerror(errno));
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 8: Initialize obj (populate CQN + QP exports) ----
  bnxt_re_dv_qp dvQpOut{};
  bnxt_re_dv_cq dvCqOut{};
  bnxt_re_dv_obj obj{};
  obj.qp.in = qp;
  obj.qp.out = &dvQpOut;
  obj.cq.in = cq;
  obj.cq.out = &dvCqOut;
  if (dv->init_obj(&obj, BNXT_RE_DV_OBJ_QP | BNXT_RE_DV_OBJ_CQ) != 0) {
    fprintf(
        stderr,
        "[BNXT] init_obj failed errno=%d (%s)\n",
        errno,
        std::strerror(errno));
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 9: SQ/CQ are already GPU device pointers from
  // hipExtMallocWithFlags — no hipHostGetDevicePointer needed.
  void* gpuSqBuf = sqBuf;
  void* gpuCqBuf = cqBuf;

  // Map MMIO doorbell to GPU via `hipHostRegister(dbr, pagesize, Default)`
  // + `hipHostGetDevicePointer`, mirroring rocSHMEM's BNXT GDA backend.
  // This is the validated
  // production path on BNXT MI300X: GPU writes directly to MMIO from the
  // device-side `ringDoorbell` — no CPU relay involved.
  long pageSizeForDbr = sysconf(_SC_PAGESIZE);
  if (pageSizeForDbr <= 0) {
    pageSizeForDbr = 4096;
  }
  hipError_t dbrRegErr = hipHostRegister(
      dbRegion->dbr,
      static_cast<size_t>(pageSizeForDbr),
      hipHostRegisterDefault);
  if (dbrRegErr != hipSuccess) {
    fprintf(
        stderr,
        "[BNXT] hipHostRegister(dbr=%p, pagesize=%ld, Default) failed: %s\n",
        dbRegion->dbr,
        pageSizeForDbr,
        hipGetErrorString(dbrRegErr));
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }
  dbrRegistered = true;
  void* gpuDbrPtr = nullptr;
  hipError_t dbrGetErr = hipHostGetDevicePointer(&gpuDbrPtr, dbRegion->dbr, 0);
  if (dbrGetErr != hipSuccess || gpuDbrPtr == nullptr) {
    fprintf(
        stderr,
        "[BNXT] hipHostGetDevicePointer(dbr=%p) failed: %s (gpuDbrPtr=%p)\n",
        dbRegion->dbr,
        hipGetErrorString(dbrGetErr),
        gpuDbrPtr);
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }
  uint64_t* gpuDbr = reinterpret_cast<uint64_t*>(gpuDbrPtr);

  // ---- Step 10: Compute MSN table location (tail of SQ buffer) ----
  size_t msnOffset =
      memInfo.sq_len - static_cast<size_t>(memInfo.sq_psn_sz) * memInfo.sq_npsn;
  void* gpuMsnTbl = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(gpuSqBuf) + msnOffset);

  uint32_t psnSzLog2 = 0;
  for (uint32_t v = memInfo.sq_psn_sz; v > 1; v >>= 1) {
    psnSzLog2++;
  }

  // ---- Step 11: Build the device-side QP descriptor ----
  pipes_gda_gpu_dev_verbs_qp hostQp{};
  hostQp.sq_wqe_daddr = reinterpret_cast<uint8_t*>(gpuSqBuf);
  hostQp.sq_db = nullptr; // unused on BNXT (we use nic.bnxt.dbr)
  hostQp.sq_dbrec = nullptr; // unused on BNXT
  hostQp.sq_wqe_num = memInfo.sq_slots;
  hostQp.sq_wqe_mask = hostQp.sq_wqe_num ? (hostQp.sq_wqe_num - 1) : 0;
  hostQp.sq_num = qp->qp_num;
  hostQp.sq_num_shift8 = qp->qp_num << 8;
  hostQp.sq_num_shift8_be = __builtin_bswap32(hostQp.sq_num_shift8 | 3);
  hostQp.sq_rsvd_index = 0;
  hostQp.sq_ready_index = 0;
  hostQp.nic_handler = PIPES_GDA_VERBS_NIC_HANDLER_GPU_SM_DB;
  hostQp.mem_type = PIPES_GDA_VERBS_MEM_TYPE_GPU;

  hostQp.cq_sq.cqe_daddr = reinterpret_cast<uint8_t*>(gpuCqBuf);
  hostQp.cq_sq.cq_num = dvCqOut.cqn;
  hostQp.cq_sq.cqe_num = cqAttr.ncqe;
  hostQp.cq_sq.dbrec = nullptr; // unused on BNXT
  hostQp.cq_sq.cqe_ci = 0;
  hostQp.cq_sq.cqe_mask = cqAttr.ncqe ? (cqAttr.ncqe - 1) : 0;
  hostQp.cq_sq.cqe_size = cqAttr.cqe_size;
  hostQp.cq_sq.cqe_rsvd = 0;
  hostQp.cq_sq.mem_type = PIPES_GDA_VERBS_MEM_TYPE_GPU;

  // BNXT-specific QP extension fields.
  hostQp.nic.bnxt.sq_depth = memInfo.sq_slots;
  hostQp.nic.bnxt.sq_head = 0;
  hostQp.nic.bnxt.sq_tail = 0;
  hostQp.nic.bnxt.sq_flags = 0;
  hostQp.nic.bnxt.sq_id = qp->qp_num;
  hostQp.nic.bnxt.msntbl = gpuMsnTbl;
  hostQp.nic.bnxt.msn = 0;
  hostQp.nic.bnxt.msn_tbl_sz = memInfo.sq_npsn;
  hostQp.nic.bnxt.psn = 0;
  hostQp.nic.bnxt.psn_sz_log2 = psnSzLog2;
  // Query the port's active MTU. MSN PSN computation depends on packet
  // count = ceil(msg_len / mtu); using a wrong MTU causes PSN drift on
  // multi-packet writes and forces NIC retransmits.
  ibv_port_attr portAttr{};
  uint32_t activeMtuBytes = 4096;
  // BNXT: route through `pipes_gda_verbs_wrapper_ibv_query_port` (which
  // dispatches to `SysIbv->query_port`) rather than calling the
  // statically-linked `ibv_query_port` directly — `ctx` was created via the
  // SysIbv (system `libibverbs.so.1`, PABI 34) path, so calling the
  // statically-linked libibverbs (PABI 59) on it risks an ABI mismatch.
  if (ctx &&
      pipes_gda_verbs_wrapper_ibv_query_port(ctx, /*port_num=*/1, &portAttr) ==
          PIPES_GDA_SUCCESS) {
    switch (portAttr.active_mtu) {
      case IBV_MTU_256:
        activeMtuBytes = 256;
        break;
      case IBV_MTU_512:
        activeMtuBytes = 512;
        break;
      case IBV_MTU_1024:
        activeMtuBytes = 1024;
        break;
      case IBV_MTU_2048:
        activeMtuBytes = 2048;
        break;
      case IBV_MTU_4096:
        activeMtuBytes = 4096;
        break;
    }
  }
  hostQp.nic.bnxt.mtu = activeMtuBytes;
  hostQp.nic.bnxt.dbr = gpuDbr;
  hostQp.nic.bnxt.cq_buf = gpuCqBuf;
  hostQp.nic.bnxt.cq_depth = cqAttr.ncqe;
  hostQp.nic.bnxt.sq_lock = 0;

  pipes_gda_gpu_dev_verbs_qp* gpuQp = nullptr;
  if (hipMalloc(&gpuQp, sizeof(pipes_gda_gpu_dev_verbs_qp)) != hipSuccess) {
    unwind();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  if (hipMemcpy(
          gpuQp,
          &hostQp,
          sizeof(pipes_gda_gpu_dev_verbs_qp),
          hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(gpuQp);
    unwind();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 12: Assemble the public handle ----
  auto* internal = new (std::nothrow) AmdBnxtQpInternal();
  if (!internal) {
    hipFree(gpuQp);
    unwind();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  internal->sq_umem = sqUmem;
  internal->rq_umem = rqUmem;
  internal->cq_umem = cqUmem;
  internal->db_region = dbRegion;
  internal->sq_buf = sqBuf;
  internal->sq_buf_size = sqBufSize;
  internal->cq_buf = cqBuf;
  internal->cq_buf_size = cqBufSize;
  internal->rq_buf = rqBuf;
  internal->rq_buf_size = rqBufSize;
  internal->dbr_host_registered = dbrRegistered;

  auto* out = new (std::nothrow) pipes_gda_gpu_verbs_qp_hl();
  if (!out) {
    delete internal;
    hipFree(gpuQp);
    unwind();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  out->qp = qp;
  out->cq = cq;
  out->gpu_qp = gpuQp;
  out->amd_internal = internal;
  out->qp_gverbs = reinterpret_cast<pipes_gda_gpu_verbs_qp*>(out);
  *out_qp = out;

  // We've taken ownership of these via `internal`; clear locals so unwind
  // (if invoked by future code paths) doesn't double-free.
  qp = nullptr;
  cq = nullptr;
  sqUmem = nullptr;
  rqUmem = nullptr;
  cqUmem = nullptr;
  dbRegion = nullptr;
  sqBuf = nullptr;
  cqBuf = nullptr;
  rqBuf = nullptr;
  dbrRegistered = false; // ownership transferred to `internal`
  return PIPES_GDA_SUCCESS;
}

#else // !NIC_BNXT (mlx5)

// One QP = ibv_create_cq + ibv_create_qp (basic libibverbs setup) + raw
// mlx5dv inspection to grab the SQ/CQ buffers and BlueFlame UAR + HSA
// mapping of the UAR page to GPU address space + hipHostRegister of the
// SQ/CQ buffers + construction of the device-side
// `pipes_gda_gpu_dev_verbs_qp` descriptor.
pipes_gda_error_t pipes_gda_gpu_verbs_create_qp_hl(
    const pipes_gda_gpu_verbs_qp_init_attr_hl* attr,
    pipes_gda_gpu_verbs_qp_hl** out_qp) {
  if (!attr || !out_qp || !attr->ibpd) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  if (!ensureHsaInitialized()) {
    return PIPES_GDA_ERROR_INITIALIZATION;
  }

  // ---- Step 1: create CQ + QP via libibverbs ----
  ibv_pd* pd = attr->ibpd;
  ibv_context* ctx = pd->context;

  ibv_cq* cq = ibv_create_cq(ctx, attr->sq_nwqe, nullptr, nullptr, 0);
  if (!cq) {
    return PIPES_GDA_ERROR_DRIVER;
  }

  ibv_qp_init_attr qpInitAttr = {};
  qpInitAttr.send_cq = cq;
  qpInitAttr.recv_cq = cq;
  qpInitAttr.cap.max_send_wr = attr->sq_nwqe;
  qpInitAttr.cap.max_recv_wr = 1;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.sq_sig_all = 0;

  ibv_qp* qp = ibv_create_qp(pd, &qpInitAttr);
  if (!qp) {
    ibv_destroy_cq(cq);
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 2: query mlx5dv to get raw QP/CQ layout ----
  mlx5dv_obj dvObj = {};
  mlx5dv_qp dvQp = {};
  mlx5dv_cq dvCq = {};
  dvObj.qp.in = qp;
  dvObj.qp.out = &dvQp;
  dvObj.cq.in = cq;
  dvObj.cq.out = &dvCq;
  if (mlx5dv_init_obj(&dvObj, MLX5DV_OBJ_QP | MLX5DV_OBJ_CQ) != 0) {
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 3: map BlueFlame UAR page to GPU via HSA ----
  int hipDevId = -1;
  if (hipGetDevice(&hipDevId) != hipSuccess) {
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    return PIPES_GDA_ERROR_DRIVER;
  }
  if (!dvQp.bf.reg || dvQp.bf.size == 0) {
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    return PIPES_GDA_ERROR_DRIVER;
  }

  void* uarBfHost = dvQp.bf.reg;
  size_t uarBfSize = static_cast<size_t>(dvQp.bf.size) * 2;
  void* gpuUarBf = nullptr;
  if (!hsaMemoryLockToGpu(uarBfHost, uarBfSize, &gpuUarBf, hipDevId)) {
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    return PIPES_GDA_ERROR_INITIALIZATION;
  }

  // ---- Step 4: initialize CQ owner bits ----
  {
    uint8_t* cqBuf = reinterpret_cast<uint8_t*>(dvCq.buf);
    for (uint32_t i = 0; i < dvCq.cqe_cnt; i++) {
      cqBuf[i * dvCq.cqe_size + dvCq.cqe_size - 1] = 0x01;
    }
  }

  // ---- Step 5: register SQ, CQ, DBREC pages with HIP ----
  size_t pageSize = sysconf(_SC_PAGESIZE);
  size_t sqSize = static_cast<size_t>(dvQp.sq.wqe_cnt) * dvQp.sq.stride;
  size_t cqSize = static_cast<size_t>(dvCq.cqe_cnt) * dvCq.cqe_size;

  void* gpuSqBuf = nullptr;
  void* gpuCqBuf = nullptr;
  void* gpuSqDbrec = nullptr;
  void* gpuCqDbrec = nullptr;

  bool sqBufReg = false;
  bool cqBufReg = false;
  bool sqDbrecReg = false;
  bool cqDbrecReg = false;

  void* sqDbrecPage = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(dvQp.dbrec) & ~(pageSize - 1));
  void* cqDbrecPage = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(dvCq.dbrec) & ~(pageSize - 1));
  bool cqDbrecPageIsDifferent = (cqDbrecPage != sqDbrecPage);

  auto unwindOnError = [&]() {
    if (cqDbrecReg) {
      hipHostUnregister(cqDbrecPage);
    }
    if (sqDbrecReg) {
      hipHostUnregister(sqDbrecPage);
    }
    if (cqBufReg) {
      hipHostUnregister(dvCq.buf);
    }
    if (sqBufReg) {
      hipHostUnregister(dvQp.sq.buf);
    }
    hsa_amd_memory_unlock(uarBfHost);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
  };

  if (hipHostRegister(dvQp.sq.buf, sqSize, hipHostRegisterDefault) !=
      hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }
  sqBufReg = true;
  if (hipHostGetDevicePointer(&gpuSqBuf, dvQp.sq.buf, 0) != hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }

  if (hipHostRegister(dvCq.buf, cqSize, hipHostRegisterDefault) != hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }
  cqBufReg = true;
  if (hipHostGetDevicePointer(&gpuCqBuf, dvCq.buf, 0) != hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }

  if (hipHostRegister(sqDbrecPage, pageSize, hipHostRegisterDefault) !=
      hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }
  sqDbrecReg = true;
  void* gpuSqDbrecPage = nullptr;
  if (hipHostGetDevicePointer(&gpuSqDbrecPage, sqDbrecPage, 0) != hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }
  size_t sqDbrecOffset = reinterpret_cast<uintptr_t>(dvQp.dbrec) -
      reinterpret_cast<uintptr_t>(sqDbrecPage);
  gpuSqDbrec = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(gpuSqDbrecPage) + sqDbrecOffset);

  if (cqDbrecPageIsDifferent) {
    if (hipHostRegister(cqDbrecPage, pageSize, hipHostRegisterDefault) !=
        hipSuccess) {
      unwindOnError();
      return PIPES_GDA_ERROR_DRIVER;
    }
    cqDbrecReg = true;
  }
  void* gpuCqDbrecPage = nullptr;
  if (hipHostGetDevicePointer(&gpuCqDbrecPage, cqDbrecPage, 0) != hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }
  size_t cqDbrecOffset = reinterpret_cast<uintptr_t>(dvCq.dbrec) -
      reinterpret_cast<uintptr_t>(cqDbrecPage);
  gpuCqDbrec = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(gpuCqDbrecPage) + cqDbrecOffset);

  // ---- Step 6: build the device-side QP descriptor ----
  pipes_gda_gpu_dev_verbs_qp hostQp = {};
  hostQp.sq_wqe_daddr = reinterpret_cast<uint8_t*>(gpuSqBuf);
  hostQp.sq_dbrec = reinterpret_cast<__be32*>(gpuSqDbrec);
  hostQp.sq_db = reinterpret_cast<uint64_t*>(gpuUarBf);
  hostQp.sq_wqe_num = dvQp.sq.wqe_cnt;
  hostQp.sq_wqe_mask = hostQp.sq_wqe_num - 1;
  hostQp.sq_num = qp->qp_num;
  hostQp.sq_num_shift8 = qp->qp_num << 8;
  hostQp.sq_num_shift8_be = __builtin_bswap32(hostQp.sq_num_shift8 | 3);
  hostQp.sq_rsvd_index = 0;
  hostQp.sq_ready_index = 0;
  hostQp.nic_handler = PIPES_GDA_VERBS_NIC_HANDLER_GPU_SM_BF;
  hostQp.mem_type = PIPES_GDA_VERBS_MEM_TYPE_GPU;

  hostQp.cq_sq.cqe_daddr = reinterpret_cast<uint8_t*>(gpuCqBuf);
  hostQp.cq_sq.cq_num = dvCq.cqn;
  hostQp.cq_sq.cqe_num = dvCq.cqe_cnt;
  hostQp.cq_sq.dbrec = reinterpret_cast<__be32*>(gpuCqDbrec);
  hostQp.cq_sq.cqe_ci = 0;
  hostQp.cq_sq.cqe_mask = dvCq.cqe_cnt - 1;
  hostQp.cq_sq.cqe_size = dvCq.cqe_size;
  hostQp.cq_sq.cqe_rsvd = 0;
  hostQp.cq_sq.mem_type = PIPES_GDA_VERBS_MEM_TYPE_GPU;

  pipes_gda_gpu_dev_verbs_qp* gpuQp = nullptr;
  if (hipMalloc(&gpuQp, sizeof(pipes_gda_gpu_dev_verbs_qp)) != hipSuccess) {
    unwindOnError();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  if (hipMemcpy(
          gpuQp,
          &hostQp,
          sizeof(pipes_gda_gpu_dev_verbs_qp),
          hipMemcpyHostToDevice) != hipSuccess) {
    hipFree(gpuQp);
    unwindOnError();
    return PIPES_GDA_ERROR_DRIVER;
  }

  // ---- Step 7: assemble the public handle ----
  auto* internal = new (std::nothrow) AmdQpInternal();
  if (!internal) {
    hipFree(gpuQp);
    unwindOnError();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  internal->uar_bf_host = uarBfHost;
  internal->uar_bf_size = uarBfSize;
  internal->registered_sq_buf = dvQp.sq.buf;
  internal->registered_cq_buf = dvCq.buf;
  internal->registered_sq_dbrec_page = sqDbrecPage;
  internal->registered_cq_dbrec_page =
      cqDbrecPageIsDifferent ? cqDbrecPage : nullptr;

  auto* out = new (std::nothrow) pipes_gda_gpu_verbs_qp_hl();
  if (!out) {
    delete internal;
    hipFree(gpuQp);
    unwindOnError();
    return PIPES_GDA_ERROR_NO_MEMORY;
  }
  out->qp = qp;
  out->cq = cq;
  out->gpu_qp = gpuQp;
  out->amd_internal = internal;
  out->qp_gverbs = reinterpret_cast<pipes_gda_gpu_verbs_qp*>(out);
  *out_qp = out;
  return PIPES_GDA_SUCCESS;
}

#endif // NIC_BNXT (mlx5)

pipes_gda_error_t pipes_gda_gpu_verbs_get_qp_dev(
    pipes_gda_gpu_verbs_qp* qp_gverbs,
    pipes_gda_gpu_dev_verbs_qp** out_dev_qp) {
  if (!qp_gverbs || !out_dev_qp) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  auto* qp_hl = reinterpret_cast<pipes_gda_gpu_verbs_qp_hl*>(qp_gverbs);
  *out_dev_qp = qp_hl->gpu_qp;
  return PIPES_GDA_SUCCESS;
}

namespace {

// Free the per-QP internal resources (registrations, NIC objects). Used by
// destroy_qp_hl and destroy_qp_group_hl. Does NOT free the qp_hl handle
// itself — caller decides whether to delete or treat it as embedded in a
// group.
static void freeQpResources(pipes_gda_gpu_verbs_qp_hl* qp) {
  if (!qp) {
    return;
  }
#ifdef NIC_BNXT
  auto* dv = getBnxtReDv();
  auto* internal = static_cast<AmdBnxtQpInternal*>(qp->amd_internal);
  if (internal) {
    ibv_context* ctx = nullptr;
    if (qp->qp) {
      ctx = qp->qp->context;
    } else if (qp->cq) {
      ctx = qp->cq->context;
    }
    // BNXT teardown: destroy QP before freeing DB region to avoid driver hang.
    // Skip modify_qp(RESET) — it causes destroy_qp to hang on bnxt_re.
    if (qp->qp && dv && dv->destroy_qp) {
      dv->destroy_qp(qp->qp);
    }
    // Unregister the GPU mapping of the MMIO doorbell page BEFORE freeing
    // the bnxt_re DB region (the DB region owns the underlying mmap).
    if (internal->dbr_host_registered && internal->db_region) {
      hipHostUnregister(internal->db_region->dbr);
    }
    if (internal->db_region && dv && dv->free_db_region && ctx) {
      dv->free_db_region(ctx, internal->db_region);
    }
    if (internal->sq_umem && dv && dv->umem_dereg) {
      dv->umem_dereg(internal->sq_umem);
    }
    if (internal->rq_umem && dv && dv->umem_dereg) {
      dv->umem_dereg(internal->rq_umem);
    }
    if (qp->cq && dv && dv->destroy_cq) {
      dv->destroy_cq(qp->cq);
    }
    if (internal->cq_umem && dv && dv->umem_dereg) {
      dv->umem_dereg(internal->cq_umem);
    }
    if (internal->cq_buf) {
      hipFree(internal->cq_buf);
    }
    if (internal->sq_buf) {
      hipFree(internal->sq_buf);
    }
    if (internal->rq_buf) {
      hipFree(internal->rq_buf);
    }
    delete internal;
  }
#else
  auto* internal = static_cast<AmdQpInternal*>(qp->amd_internal);
  if (internal) {
    if (internal->registered_cq_dbrec_page) {
      hipHostUnregister(internal->registered_cq_dbrec_page);
    }
    if (internal->registered_sq_dbrec_page) {
      hipHostUnregister(internal->registered_sq_dbrec_page);
    }
    if (internal->registered_cq_buf) {
      hipHostUnregister(internal->registered_cq_buf);
    }
    if (internal->registered_sq_buf) {
      hipHostUnregister(internal->registered_sq_buf);
    }
    if (internal->uar_bf_host) {
      hsa_amd_memory_unlock(internal->uar_bf_host);
    }
    delete internal;
  }
  if (qp->qp) {
    ibv_destroy_qp(qp->qp);
  }
  if (qp->cq) {
    ibv_destroy_cq(qp->cq);
  }
#endif
  if (qp->gpu_qp) {
    hipFree(qp->gpu_qp);
  }
}

} // namespace

pipes_gda_error_t pipes_gda_gpu_verbs_destroy_qp_hl(
    pipes_gda_gpu_verbs_qp_hl* qp) {
  if (!qp) {
    return PIPES_GDA_SUCCESS;
  }
  freeQpResources(qp);
  delete qp;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_gpu_verbs_create_qp_group_hl(
    const pipes_gda_gpu_verbs_qp_init_attr_hl* attr,
    pipes_gda_gpu_verbs_qp_group_hl** out_grp) {
  if (!out_grp) {
    return PIPES_GDA_ERROR_INVALID_VALUE;
  }
  auto* grp = new (std::nothrow) pipes_gda_gpu_verbs_qp_group_hl();
  if (!grp) {
    return PIPES_GDA_ERROR_NO_MEMORY;
  }

  pipes_gda_gpu_verbs_qp_hl* mainQp = nullptr;
  pipes_gda_error_t err = pipes_gda_gpu_verbs_create_qp_hl(attr, &mainQp);
  if (err != PIPES_GDA_SUCCESS) {
    delete grp;
    return err;
  }
  pipes_gda_gpu_verbs_qp_hl* compQp = nullptr;
  err = pipes_gda_gpu_verbs_create_qp_hl(attr, &compQp);
  if (err != PIPES_GDA_SUCCESS) {
    pipes_gda_gpu_verbs_destroy_qp_hl(mainQp);
    delete grp;
    return err;
  }

  // Move-by-copy: the create_qp_hl returns heap-allocated handles. We
  // copy their fields into the group and then free the wrappers. After
  // copy, fix up qp_gverbs to point at the new (in-group) qp_hl
  // addresses so pipes_gda_gpu_verbs_get_qp_dev recovers the right
  // handle.
  grp->qp_main = *mainQp;
  grp->qp_companion = *compQp;
  grp->qp_main.qp_gverbs =
      reinterpret_cast<pipes_gda_gpu_verbs_qp*>(&grp->qp_main);
  grp->qp_companion.qp_gverbs =
      reinterpret_cast<pipes_gda_gpu_verbs_qp*>(&grp->qp_companion);
  delete mainQp;
  delete compQp;

  *out_grp = grp;
  return PIPES_GDA_SUCCESS;
}

pipes_gda_error_t pipes_gda_gpu_verbs_destroy_qp_group_hl(
    pipes_gda_gpu_verbs_qp_group_hl* g) {
  if (!g) {
    return PIPES_GDA_SUCCESS;
  }
  // Free the resources owned by the embedded handles. We can't call
  // destroy_qp_hl directly because it would try to delete the wrapper;
  // instead reuse the shared free helper.
  freeQpResources(&g->qp_main);
  freeQpResources(&g->qp_companion);
  delete g;
  return PIPES_GDA_SUCCESS;
}

} // namespace pipes_gda

// ===========================================================================
// DMA-BUF export — NIC-agnostic.
// ===========================================================================
//
// HSA equivalent of NVIDIA's `cuMemGetAddressRange + doca_gpu_dmabuf_fd`
// pipeline. Loads `hsa_amd_portable_export_dmabuf` lazily via dlsym since
// it isn't always exposed by older HSA headers.

namespace comms::prims {

std::optional<DmaBufExport>
export_gpu_dmabuf_aligned(void* ptr, std::size_t size, DmaBufExportKind kind) {
  // mlx5 Data-Direct (BAR1 PCIe mapping) is NVIDIA-only; the AMD HSA export has
  // no equivalent, so report Pcie as unavailable. (Data-Direct is also disabled
  // in AMD NIC discovery, so this is never requested at runtime.)
  if (kind == DmaBufExportKind::Pcie) {
    return std::nullopt;
  }
  // Skip dmabuf for non-GPU pointers. HSA's
  // `hsa_amd_portable_export_dmabuf` will export host-pinned
  // (`hipHostMalloc`) memory as a dmabuf, but the resulting fd
  // represents host pages — passing it to `ibv_reg_dmabuf_mr` causes
  // silent RDMA corruption AND can SIGSEGV inside libmlx5/kernel
  // mlx5_ib.
  hipPointerAttribute_t attrs{};
  if (hipPointerGetAttributes(&attrs, ptr) != hipSuccess ||
      attrs.type != hipMemoryTypeDevice) {
    return std::nullopt;
  }

  using ExportDmabufFn = int (*)(const void*, size_t, int*, uint64_t*);
  static ExportDmabufFn exportDmabuf = nullptr;
  static std::once_flag dlOpenOnce;
  std::call_once(dlOpenOnce, []() {
    void* lib = dlopen("libhsa-runtime64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) {
      lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (lib) {
      exportDmabuf = reinterpret_cast<ExportDmabufFn>(
          dlsym(lib, "hsa_amd_portable_export_dmabuf"));
    }
  });

  if (!exportDmabuf) {
    return std::nullopt;
  }

  int fd = -1;
  uint64_t dmabufOffset = 0;
  int hsaStatus = exportDmabuf(ptr, size, &fd, &dmabufOffset);
  if (hsaStatus != 0 || fd < 0) {
    return std::nullopt;
  }

  // Skip dmabuf if HSA returned a non-page-aligned offset — `ibv_reg_dmabuf_mr`
  // would SIGSEGV inside libmlx5 on AMD MI300X.
  if (dmabufOffset % static_cast<uint64_t>(sysconf(_SC_PAGESIZE)) != 0) {
    close(fd);
    return std::nullopt;
  }

  uintptr_t alignedBase = reinterpret_cast<uintptr_t>(ptr) - dmabufOffset;
  DmaBufExport ex;
  ex.fd = fd;
  ex.alignment.alignedBase = reinterpret_cast<void*>(alignedBase);
  ex.alignment.alignedSize = size + dmabufOffset;
  ex.alignment.dmabufOffset = dmabufOffset;
  return ex;
}

} // namespace comms::prims

#endif // __HIP_PLATFORM_AMD__
