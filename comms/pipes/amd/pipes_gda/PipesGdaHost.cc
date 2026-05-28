// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// PipesGdaHost implementation
// =============================================================================
//
// Implementations for the host-side `pipes_gda_*` API declared in
// `PipesGdaHost.h`. The heavy lifting (HSA UAR-to-GPU mapping, raw mlx5dv
// QP/CQ setup, device-side `pipes_gda_gpu_dev_verbs_qp` descriptor
// construction) handles the AMD/HIP IBGDA fast-path on top of mlx5
// direct verbs.
// =============================================================================

#ifdef __HIP_PLATFORM_AMD__

#include "pipes_gda/PipesGdaHost.h" // @manual

// pipes_gda_gpu_dev_verbs_qp is declared in this header; needed for
// constructing the device-side QP descriptor below.
#include "pipes_gda/PipesGdaDev.h"

#include <dlfcn.h> // dlopen / dlsym for hsa_amd_portable_export_dmabuf
#include <unistd.h> // sysconf, _SC_PAGESIZE

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

namespace {

// ===========================================================================
// HSA Runtime Helpers (UAR-to-GPU mapping)
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

// ===========================================================================
// AMD-internal QP control block — tracks per-QP resources we registered via
// hipHostRegister so we can undo them in destroy_qp_hl.
// ===========================================================================

struct AmdQpInternal {
  void* uar_bf_host{nullptr};
  size_t uar_bf_size{0};
  void* registered_sq_buf{nullptr};
  void* registered_cq_buf{nullptr};
  void* registered_sq_dbrec_page{nullptr};
  void* registered_cq_dbrec_page{nullptr}; // nullptr if same page as SQ
};

// ===========================================================================
// AMD-internal QP attribute / address-handle structs
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
// QP attribute setters (trivial — forward to ibv_qp_attr)
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
// Address handle attribute setters
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
// corresponding IBV_QP_* bits expected by `ibv_modify_qp`. Each enum lives
// in its own bit-position space (e.g. PIPES_GDA_..._PKEY_INDEX = 1<<1
// while IBV_QP_CUR_STATE = 1<<1), so we cannot simply OR the caller's
// mask into the kernel mask — that would set unrelated IBV bits and the
// kernel would either reject the call or read uninitialized fields.
// This handles fields like `pkey_index` that have a valid zero-init
// default and no explicit setter; without translation, NVIDIA DOCA's
// "always-write-pkey_index-from-struct" semantics are unreachable from
// the AMD shim. See third-party/nvidia-doca/gpunetio/main/src/
// doca_verbs_qp.cpp::rst2init for the NVIDIA reference.
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
  // PIPES_GDA mask (translated to IBV space). The translation is required
  // because the two enums use different bit positions — a raw OR would
  // pollute the IBV mask with unrelated bits (e.g. PIPES_GDA's PKEY_INDEX
  // = 1<<1 collides with IBV_QP_CUR_STATE), causing the kernel to reject
  // the modify or read garbage from uninitialized struct fields.
  int mask =
      impl->attr_mask | translatePipesGdaMaskToIbv(static_cast<int>(attr_mask));

  // Apply the AH only when the caller explicitly includes AH_ATTR in the
  // current modify. The AH pointer is stored on the attr struct by an
  // earlier `set_ah_attr` call but stays live across subsequent modifies
  // (the unified DOCA path reuses one `qpAttr` across RST->INIT->RTR->RTS).
  // Adding `IBV_QP_AV` to a transition that doesn't allow it (e.g. RTR->RTS)
  // makes the kernel return EINVAL.
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
  // not expose setters for `max_dest_rd_atomic` / `max_rd_atomic` (NVIDIA
  // configures them via direct mlx5_devx WQEs, bypassing `ibv_modify_qp`).
  // The deleted `MultipeerIbgdaTransportAmd::connectQp` set both to 16,
  // matching what NCCL's `ncclIbRtrQp` / `ncclIbRtsQp` use today.
  if ((mask & IBV_QP_STATE) != 0) {
    if (ibvAttr.qp_state == IBV_QPS_RTR) {
      ibvAttr.max_dest_rd_atomic = 16;
      mask |= IBV_QP_MAX_DEST_RD_ATOMIC;
    } else if (ibvAttr.qp_state == IBV_QPS_RTS) {
      ibvAttr.max_rd_atomic = 16;
      mask |= IBV_QP_MAX_QP_RD_ATOMIC;
    }
  }

  int rc = ibv_modify_qp(qp, &ibvAttr, mask);
  if (rc != 0) {
    // Surface the actual `ibv_modify_qp` failure. Without this, callers see
    // only the generic `PIPES_GDA_ERROR_DRIVER` (= 5), which the DOCA error-
    // name table prints as the misleading `DOCA_ERROR_AGAIN`.
    fprintf(
        stderr,
        "pipes_gda_verbs_qp_modify: ibv_modify_qp failed rc=%d errno=%d (%s) "
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
  // sets. The unified `MultipeerIbgdaTransport.cc` reuses one `qpAttr`
  // across the RST->INIT, INIT->RTR, and RTR->RTS transitions; without
  // clearing, the RTR modify would carry stale `IBV_QP_PORT` /
  // `IBV_QP_ACCESS_FLAGS` bits from the prior INIT setters, and the RTS
  // modify would carry the stale AH ref — both invalid for those
  // transitions on some kernels.
  impl->attr_mask = 0;
  impl->ah_attr_ref = nullptr;
  return PIPES_GDA_SUCCESS;
}

// ===========================================================================
// GPU verbs QP creation
// ===========================================================================
//
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

  // Track which pages got registered so the error path can unregister
  // exactly those (avoids hipHostUnregister() on never-registered pages).
  bool sqBufReg = false;
  bool cqBufReg = false;
  bool sqDbrecReg = false;
  bool cqDbrecReg = false;

  void* sqDbrecPage = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(dvQp.dbrec) & ~(pageSize - 1));
  void* cqDbrecPage = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(dvCq.dbrec) & ~(pageSize - 1));
  bool cqDbrecPageIsDifferent = (cqDbrecPage != sqDbrecPage);

  // Cleanup helper used by every error return below to undo partial
  // registration. Capture by reference so the flags reflect progress.
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
  hostQp.sq_wqe_num = static_cast<uint16_t>(dvQp.sq.wqe_cnt);
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
  // qp_gverbs is a tagged self-pointer so pipes_gda_gpu_verbs_get_qp_dev
  // can recover the qp_hl handle from it.
  out->qp_gverbs = reinterpret_cast<pipes_gda_gpu_verbs_qp*>(out);
  *out_qp = out;
  return PIPES_GDA_SUCCESS;
}

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

pipes_gda_error_t pipes_gda_gpu_verbs_destroy_qp_hl(
    pipes_gda_gpu_verbs_qp_hl* qp) {
  if (!qp) {
    return PIPES_GDA_SUCCESS;
  }
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
  if (qp->gpu_qp) {
    hipFree(qp->gpu_qp);
  }
  if (qp->qp) {
    ibv_destroy_qp(qp->qp);
  }
  if (qp->cq) {
    ibv_destroy_cq(qp->cq);
  }
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
  // instead inline the cleanup below.
  for (pipes_gda_gpu_verbs_qp_hl* qp : {&g->qp_main, &g->qp_companion}) {
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
    if (qp->gpu_qp) {
      hipFree(qp->gpu_qp);
    }
    if (qp->qp) {
      ibv_destroy_qp(qp->qp);
    }
    if (qp->cq) {
      ibv_destroy_cq(qp->cq);
    }
  }
  delete g;
  return PIPES_GDA_SUCCESS;
}

} // namespace pipes_gda

// ===========================================================================
// DMA-BUF export
// ===========================================================================
//
// HSA equivalent of NVIDIA's `cuMemGetAddressRange + doca_gpu_dmabuf_fd`
// pipeline. Loads `hsa_amd_portable_export_dmabuf` lazily via dlsym since
// it isn't always exposed by older HSA headers.

namespace comms::pipes {

std::optional<DmaBufExport>
export_gpu_dmabuf_aligned(pipes_gda_gpu* /*gpu*/, void* ptr, std::size_t size) {
  // Skip dmabuf for non-GPU pointers. HSA's
  // `hsa_amd_portable_export_dmabuf` will export host-pinned
  // (`hipHostMalloc`) memory as a dmabuf, but the resulting fd
  // represents host pages — passing it to `ibv_reg_dmabuf_mr` causes
  // silent RDMA corruption AND can SIGSEGV inside libmlx5/kernel
  // mlx5_ib. The deleted `MultipeerIbgdaTransportAmd::registerRdmaBuffer`
  // gated dmabuf with `isDevicePointer(ptr)` for exactly this reason.
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

  // Skip dmabuf if HSA returned a non-page-aligned offset. The deleted
  // `MultipeerIbgdaTransportAmd::registerRdmaBuffer` had this exact
  // check; omitting it lets `ibv_reg_dmabuf_mr` SIGSEGV inside libmlx5
  // on AMD MI300X (kernel path expects a page-multiple offset).
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

} // namespace comms::pipes

#endif // __HIP_PLATFORM_AMD__
