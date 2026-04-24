#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "MultipeerIbgdaTransportAmd.h"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/mlx5dv.h>

#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>
#include <glog/logging.h>
#include <unistd.h>

#include "MultipeerIbgdaTransportAmdHip.h"
#include "verbs/VerbsUtils.h"

namespace pipes_gda {

namespace {

// =============================================================================
// HSA Runtime Helpers (UAR-to-GPU mapping)
// =============================================================================

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
  if (st != HSA_STATUS_SUCCESS)
    return st;
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
  if (st != HSA_STATUS_SUCCESS)
    return st;
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
    if (st != HSA_STATUS_SUCCESS)
      return;
    st = hsa_iterate_agents(hsaAgentCallback, nullptr);
    if (st != HSA_STATUS_SUCCESS && st != HSA_STATUS_INFO_BREAK)
      return;
    g_hsaInitSuccess = true;
  });
  return g_hsaInitSuccess;
}

static bool
hsaMemoryLockToGpu(void* hostPtr, size_t size, void** gpuPtr, int gpuId) {
  if (!ensureHsaInitialized())
    return false;
  if (gpuId < 0 || static_cast<size_t>(gpuId) >= g_hsaGpuAgents.size())
    return false;
  if (g_hsaCpuAgents.empty())
    return false;
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

// Check if a pointer is device (GPU VRAM) memory.
// Uses hipPointerGetAttributes which distinguishes:
//   hipMemoryTypeDevice = GPU VRAM (allocated via hipMalloc)
//   hipMemoryTypeHost   = host pinned memory (allocated via hipHostMalloc)
// Note: hipMemoryTypeHost is returned for all host-side allocations including
// pinned memory. This is the correct classification for RDMA registration
// purposes: only hipMemoryTypeDevice pointers should use DMA-buf export.
static bool isDevicePointer(void* ptr) {
  hipPointerAttribute_t attrs = {};
  hipError_t err = hipPointerGetAttributes(&attrs, ptr);
  if (err != hipSuccess)
    return false;
  return attrs.type == hipMemoryTypeDevice;
}

// Register a buffer with RDMA.
// For GPU memory: tries DMA-buf first (no peer_mem needed), then ibv_reg_mr.
// For host memory: uses ibv_reg_mr directly (dmabuf is for GPU memory only).
static ibv_mr*
registerRdmaBuffer(ibv_pd* pd, void* ptr, size_t size, int accessFlags) {
  ibv_mr* mr = nullptr;

  // For host memory (hipHostMalloc), use ibv_reg_mr directly.
  // DMA-buf export is only valid for GPU VRAM (hipMemoryTypeDevice) —
  // using it on host-pinned memory would export wrong physical pages,
  // causing silent RDMA corruption. The isDevicePointer check above
  // relies on hipPointerGetAttributes returning hipMemoryTypeHost for
  // all host-side allocations.
  if (!isDevicePointer(ptr)) {
    mr = ibv_reg_mr(pd, ptr, size, accessFlags);
    if (mr) {
      VLOG(1) << "Registered host buffer via ibv_reg_mr";
    } else {
      LOG(WARNING) << "ibv_reg_mr failed for host buffer (errno=" << errno
                   << ": " << strerror(errno) << ")";
    }
    return mr;
  }

  // GPU memory: try DMA-buf first (preferred, no peer_mem module needed)
  using ExportDmabufFn = int (*)(const void*, size_t, int*, uint64_t*);
  static ExportDmabufFn exportDmabuf = nullptr;
  static bool triedLoad = false;
  if (!triedLoad) {
    triedLoad = true;
    void* hsaLib = dlopen("libhsa-runtime64.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!hsaLib)
      hsaLib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (hsaLib) {
      exportDmabuf = reinterpret_cast<ExportDmabufFn>(
          dlsym(hsaLib, "hsa_amd_portable_export_dmabuf"));
    }
  }

  if (exportDmabuf) {
    int dmabufFd = -1;
    uint64_t dmabufOffset = 0;
    int hsaStatus = exportDmabuf(ptr, size, &dmabufFd, &dmabufOffset);
    if (hsaStatus == 0 && dmabufFd >= 0) {
      if (dmabufOffset % sysconf(_SC_PAGESIZE) != 0) {
        LOG(WARNING) << "dmabuf offset " << dmabufOffset
                     << " is not page-aligned, skipping ibv_reg_dmabuf_mr"
                     << " to avoid RDMA corruption";
        close(dmabufFd);
      } else {
        mr = ibv_reg_dmabuf_mr(
            pd,
            dmabufOffset,
            size,
            reinterpret_cast<uint64_t>(ptr),
            dmabufFd,
            accessFlags);
        if (mr) {
          VLOG(1) << "Registered GPU buffer via dmabuf (fd=" << dmabufFd << ")";
          close(dmabufFd);
          return mr;
        }
        LOG(WARNING) << "ibv_reg_dmabuf_mr failed (errno=" << errno
                     << "), falling back to ibv_reg_mr";
        close(dmabufFd);
      }
    }
  }

  // Fallback: ibv_reg_mr (requires amd-peer-mem kernel module for GPU memory)
  mr = ibv_reg_mr(pd, ptr, size, accessFlags);
  if (mr) {
    VLOG(1) << "Registered GPU buffer via legacy ibv_reg_mr (peer_mem)";
  } else {
    LOG(WARNING) << "ibv_reg_mr failed for GPU buffer (errno=" << errno << ": "
                 << strerror(errno) << ")";
  }
  return mr;
}

} // namespace

// =============================================================================
// MultipeerIbgdaTransportAmd Implementation
// =============================================================================

MultipeerIbgdaTransportAmd::MultipeerIbgdaTransportAmd(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbgdaTransportAmdConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  initCommon();
}

MultipeerIbgdaTransportAmd::MultipeerIbgdaTransportAmd(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbgdaTransportAmdConfig& config,
    const std::vector<int>& targetRanks)
    : myRank_(myRank),
      nRanks_(nRanks),
      targetRanks_(targetRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  initCommon();
}

void MultipeerIbgdaTransportAmd::initCommon() {
  if (myRank_ < 0 || myRank_ >= nRanks_)
    throw std::invalid_argument("Invalid rank");
  if (nRanks_ < 2)
    throw std::invalid_argument("Need at least 2 ranks");

  // Validate targetRanks if provided (filtered mode)
  if (!targetRanks_.empty()) {
    std::unordered_set<int> seen;
    for (int rank : targetRanks_) {
      if (rank < 0 || rank >= nRanks_)
        throw std::invalid_argument(
            "targetRanks contains out-of-range rank: " + std::to_string(rank));
      if (rank == myRank_)
        throw std::invalid_argument(
            "targetRanks must not contain myRank (" + std::to_string(myRank_) +
            ")");
      if (!seen.insert(rank).second)
        throw std::invalid_argument(
            "targetRanks contains duplicate rank: " + std::to_string(rank));
    }
  }

  hipError_t err = hipSetDevice(config_.hipDevice);
  if (err != hipSuccess)
    throw std::runtime_error(
        "Failed to set HIP device: " + std::string(hipGetErrorString(err)));
  // Force HIP context init
  hipFree(0);

  const int nPeers = numPeers();

  try {
    openIbDevice();

    peerResources_.resize(nPeers);
    for (int i = 0; i < nPeers; i++) {
      if (!createQpAndCq(i))
        throw std::runtime_error(
            "Failed to create QP for peer " + std::to_string(i));
    }

    // Allocate sink buffer for atomic return values (discarded).
    // Use host-pinned memory so ibv_reg_mr works without amd-peer-mem module.
    hipError_t hipErr =
        hipHostMalloc(&sinkBuffer_, sizeof(uint64_t), hipHostMallocDefault);
    if (hipErr != hipSuccess)
      throw std::runtime_error("Failed to allocate sink buffer");
    memset(sinkBuffer_, 0, sizeof(uint64_t));

    int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
    sinkMr_ = ibv_reg_mr(ibvPd_, sinkBuffer_, sizeof(uint64_t), accessFlags);
    if (!sinkMr_)
      throw std::runtime_error("Failed to register sink MR");
  } catch (...) {
    cleanup();
    throw;
  }

  VLOG(1) << "MultipeerIbgdaTransportAmd: rank " << myRank_ << "/" << nRanks_
          << " initialized with " << nPeers << " peers on HIP device "
          << config_.hipDevice;
}

MultipeerIbgdaTransportAmd::~MultipeerIbgdaTransportAmd() {
  cleanup();
}

void MultipeerIbgdaTransportAmd::openIbDevice() {
  char pciBusId[32] = {};
  if (hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), config_.hipDevice) !=
      hipSuccess)
    throw std::runtime_error("Failed to get GPU PCI bus ID");
  gpuPciBusId_ = pciBusId;

  std::string nicName = tests::findClosestNic(gpuPciBusId_);
  if (nicName.empty())
    nicName = tests::findFirstNicDevice();
  if (nicName.empty())
    throw std::runtime_error("No suitable NIC found");

  VLOG(1) << "MultipeerIbgdaTransportAmd: GPU " << gpuPciBusId_ << " -> NIC "
          << nicName;

  int numDevices = 0;
  ibv_device** deviceList = ibv_get_device_list(&numDevices);
  if (!deviceList)
    throw std::runtime_error("No IB devices found");

  ibv_device* target = nullptr;
  for (int i = 0; i < numDevices; i++) {
    if (nicName == deviceList[i]->name) {
      target = deviceList[i];
      break;
    }
  }
  if (!target) {
    ibv_free_device_list(deviceList);
    throw std::runtime_error("NIC not found: " + nicName);
  }

  ibvCtx_ = ibv_open_device(target);
  ibv_free_device_list(deviceList);
  if (!ibvCtx_)
    throw std::runtime_error("Failed to open IB device");

  ibvPd_ = ibv_alloc_pd(ibvCtx_);
  if (!ibvPd_)
    throw std::runtime_error("Failed to allocate PD");

  if (ibv_query_gid(ibvCtx_, 1, config_.gidIndex, &localGid_) != 0)
    throw std::runtime_error("Failed to query GID");

  ibv_port_attr portAttr{};
  if (ibv_query_port(ibvCtx_, 1, &portAttr) != 0)
    throw std::runtime_error("Failed to query port");
  if (portAttr.state != IBV_PORT_ACTIVE)
    throw std::runtime_error("Port not active");
  localMtu_ = portAttr.active_mtu;
}

bool MultipeerIbgdaTransportAmd::createQpAndCq(int peerIndex) {
  auto& res = peerResources_[peerIndex];

  res.cq = ibv_create_cq(ibvCtx_, config_.qpDepth, nullptr, nullptr, 0);
  if (!res.cq)
    return false;

  ibv_qp_init_attr qpInitAttr = {};
  qpInitAttr.send_cq = res.cq;
  qpInitAttr.recv_cq = res.cq;
  qpInitAttr.cap.max_send_wr = config_.qpDepth;
  qpInitAttr.cap.max_recv_wr = 1;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.sq_sig_all = 0;

  res.qp = ibv_create_qp(ibvPd_, &qpInitAttr);
  if (!res.qp)
    return false;

  res.qpNum = res.qp->qp_num;
  return true;
}

bool MultipeerIbgdaTransportAmd::connectQp(
    int peerIndex,
    const IbgdaTransportExchInfoAmd& peerInfo) {
  auto& res = peerResources_[peerIndex];
  ibv_qp_attr attr = {};
  int flags;

  // RESET -> INIT
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;
  attr.pkey_index = 0;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(res.qp, &attr, flags) != 0)
    return false;

  // INIT -> RTR
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;
  attr.dest_qp_num = peerInfo.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer = config_.minRnrTimer;
  attr.ah_attr.dlid = 0;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = 1;
  attr.ah_attr.is_global = 1;
  memcpy(&attr.ah_attr.grh.dgid, peerInfo.gid, 16);
  attr.ah_attr.grh.sgid_index = config_.gidIndex;
  attr.ah_attr.grh.hop_limit = 255;
  flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(res.qp, &attr, flags) != 0)
    return false;

  // RTR -> RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = config_.timeout;
  attr.retry_cnt = config_.retryCount;
  attr.rnr_retry = config_.rnrRetry;
  attr.max_rd_atomic = 16;
  flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
      IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
  if (ibv_modify_qp(res.qp, &attr, flags) != 0)
    return false;

  return true;
}

bool MultipeerIbgdaTransportAmd::exportQpToGpu(int peerIndex) {
  auto& res = peerResources_[peerIndex];

  // Query mlx5dv QP/CQ layout
  mlx5dv_obj dvObj = {};
  mlx5dv_qp dvQp = {};
  mlx5dv_cq dvCq = {};
  dvObj.qp.in = res.qp;
  dvObj.qp.out = &dvQp;
  dvObj.cq.in = res.cq;
  dvObj.cq.out = &dvCq;
  if (mlx5dv_init_obj(&dvObj, MLX5DV_OBJ_QP | MLX5DV_OBJ_CQ) != 0)
    return false;

  // Map BlueFlame UAR to GPU via HSA
  int hipDevId = -1;
  if (hipGetDevice(&hipDevId) != hipSuccess)
    return false;

  if (!dvQp.bf.reg || dvQp.bf.size == 0)
    return false;

  res.uarBfHostPtr = dvQp.bf.reg;
  res.uarBfSize = static_cast<size_t>(dvQp.bf.size) * 2;
  void* gpuUarBf = nullptr;
  if (!hsaMemoryLockToGpu(res.uarBfHostPtr, res.uarBfSize, &gpuUarBf, hipDevId))
    return false;
  res.gpuUarBf = gpuUarBf;

  // Initialize CQ owner bits
  {
    uint8_t* cqBuf = reinterpret_cast<uint8_t*>(dvCq.buf);
    for (uint32_t i = 0; i < dvCq.cqe_cnt; i++) {
      cqBuf[i * dvCq.cqe_size + dvCq.cqe_size - 1] = 0x01;
    }
  }

  size_t pageSize = sysconf(_SC_PAGESIZE);
  size_t sqSize = static_cast<size_t>(dvQp.sq.wqe_cnt) * dvQp.sq.stride;

  // Register SQ, CQ, and DBREC buffers with HIP
  void* gpuSqBuf = nullptr;
  void* gpuCqBuf = nullptr;
  void* gpuSqDbrec = nullptr;
  void* gpuCqDbrec = nullptr;

  hipHostRegister(dvQp.sq.buf, sqSize, hipHostRegisterDefault);
  res.registeredSqBuf = dvQp.sq.buf;
  hipHostGetDevicePointer(&gpuSqBuf, dvQp.sq.buf, 0);

  size_t cqSize = static_cast<size_t>(dvCq.cqe_cnt) * dvCq.cqe_size;
  hipHostRegister(dvCq.buf, cqSize, hipHostRegisterDefault);
  res.registeredCqBuf = dvCq.buf;
  hipHostGetDevicePointer(&gpuCqBuf, dvCq.buf, 0);

  // SQ DBREC
  void* sqDbrecPage = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(dvQp.dbrec) & ~(pageSize - 1));
  hipHostRegister(sqDbrecPage, pageSize, hipHostRegisterDefault);
  res.registeredSqDbrecPage = sqDbrecPage;
  void* gpuSqDbrecPage = nullptr;
  hipHostGetDevicePointer(&gpuSqDbrecPage, sqDbrecPage, 0);
  size_t sqDbrecOffset = reinterpret_cast<uintptr_t>(dvQp.dbrec) -
      reinterpret_cast<uintptr_t>(sqDbrecPage);
  gpuSqDbrec = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(gpuSqDbrecPage) + sqDbrecOffset);

  // CQ DBREC
  void* cqDbrecPage = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(dvCq.dbrec) & ~(pageSize - 1));
  if (cqDbrecPage != sqDbrecPage) {
    hipHostRegister(cqDbrecPage, pageSize, hipHostRegisterDefault);
    res.registeredCqDbrecPage = cqDbrecPage;
  }
  void* gpuCqDbrecPage = nullptr;
  hipHostGetDevicePointer(&gpuCqDbrecPage, cqDbrecPage, 0);
  size_t cqDbrecOffset = reinterpret_cast<uintptr_t>(dvCq.dbrec) -
      reinterpret_cast<uintptr_t>(cqDbrecPage);
  gpuCqDbrec = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(gpuCqDbrecPage) + cqDbrecOffset);

  // Build QP struct on host then copy to GPU
  pipes_gda_gpu_dev_verbs_qp hostQp = {};
  hostQp.sq_wqe_daddr = reinterpret_cast<uint8_t*>(gpuSqBuf);
  hostQp.sq_dbrec = reinterpret_cast<__be32*>(gpuSqDbrec);
  hostQp.sq_db = reinterpret_cast<uint64_t*>(gpuUarBf);
  hostQp.sq_wqe_num = static_cast<uint16_t>(dvQp.sq.wqe_cnt);
  hostQp.sq_wqe_mask = hostQp.sq_wqe_num - 1;
  hostQp.sq_num = res.qp->qp_num;
  hostQp.sq_num_shift8 = res.qp->qp_num << 8;
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

  hipError_t err = hipMalloc(&res.gpuQp, sizeof(pipes_gda_gpu_dev_verbs_qp));
  if (err != hipSuccess)
    return false;

  err = hipMemcpy(
      res.gpuQp,
      &hostQp,
      sizeof(pipes_gda_gpu_dev_verbs_qp),
      hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    hipFree(res.gpuQp);
    res.gpuQp = nullptr;
    return false;
  }

  VLOG(1) << "MultipeerIbgdaTransportAmd: exported QP " << peerIndex
          << " to GPU (SQ wqe_cnt=" << dvQp.sq.wqe_cnt
          << ", CQ cqe_cnt=" << dvCq.cqe_cnt << ")";
  return true;
}

void MultipeerIbgdaTransportAmd::exchange() {
  const int nPeers = numPeers();

  if (nRanks_ > kMaxRanksAmd)
    throw std::runtime_error("Too many ranks for allGather exchange");

  // Build local exchange info.
  // Even in filtered mode, we allGather with ALL ranks so each rank can
  // find the QPN that its peer created for it.
  std::vector<IbgdaTransportExchInfoAllAmd> allInfo(nRanks_);
  auto& myInfo = allInfo[myRank_];
  memcpy(myInfo.gid, localGid_.raw, sizeof(myInfo.gid));
  myInfo.gidIndex = config_.gidIndex;
  myInfo.mtu = localMtu_;

  ibv_port_attr portAttr{};
  if (ibv_query_port(ibvCtx_, 1, &portAttr) == 0) {
    myInfo.lid = portAttr.lid;
  }

  // Fill in the QPN that each peer should use to reach us
  for (int pi = 0; pi < nPeers; pi++) {
    int peerRank = peerIndexToRank(pi);
    myInfo.qpnForRank[peerRank] = peerResources_[pi].qpNum;
  }
  myInfo.qpnForRank[myRank_] = 0;

  // AllGather exchange (all ranks participate)
  auto result = bootstrap_
                    ->allGather(
                        allInfo.data(),
                        sizeof(IbgdaTransportExchInfoAllAmd),
                        myRank_,
                        nRanks_)
                    .get();
  if (result != 0)
    throw std::runtime_error("allGather failed");

  // Connect QPs to peers (only target peers in filtered mode)
  for (int pi = 0; pi < nPeers; pi++) {
    int peerRank = peerIndexToRank(pi);
    const auto& peerInfo = allInfo[peerRank];

    IbgdaTransportExchInfoAmd exchInfo{};
    exchInfo.qpn = peerInfo.qpnForRank[myRank_];
    memcpy(exchInfo.gid, peerInfo.gid, sizeof(exchInfo.gid));
    exchInfo.gidIndex = peerInfo.gidIndex;
    exchInfo.lid = peerInfo.lid;
    exchInfo.mtu = peerInfo.mtu;

    if (!connectQp(pi, exchInfo))
      throw std::runtime_error(
          "Failed to connect QP to peer " + std::to_string(peerRank));
  }

  // Export QPs to GPU
  for (int pi = 0; pi < nPeers; pi++) {
    if (!exportQpToGpu(pi))
      throw std::runtime_error(
          "Failed to export QP to GPU for peer " + std::to_string(pi));
  }

  // Build P2pIbgdaTransportDevice array on GPU via HIP helper
  peerTransportSize_ = getP2pIbgdaTransportDeviceSizeAmd();

  std::vector<P2pIbgdaTransportBuildParamsAmd> buildParams(nPeers);
  NetworkLKey sinkLkey(HostLKey(sinkMr_->lkey));
  for (int pi = 0; pi < nPeers; pi++) {
    buildParams[pi] = {peerResources_[pi].gpuQp, sinkLkey, sinkBuffer_};
  }

  peerTransportsGpu_ = static_cast<P2pIbgdaTransportDevice*>(
      buildDeviceTransportsOnGpuAmd(buildParams.data(), nPeers));
  if (!peerTransportsGpu_)
    throw std::runtime_error("Failed to build device transports on GPU");

  VLOG(1) << "MultipeerIbgdaTransportAmd: rank " << myRank_
          << " exchange complete, connected to " << nPeers << " peers"
          << (targetRanks_.empty() ? " (all)" : " (filtered)");
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransportAmd::getP2pTransportDevice(
    int peerRank) const {
  int pi = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbgdaTransportDevice*>(
      reinterpret_cast<char*>(peerTransportsGpu_) + pi * peerTransportSize_);
}

P2pIbgdaTransportDevice* MultipeerIbgdaTransportAmd::getDeviceTransportPtr()
    const {
  return peerTransportsGpu_;
}

int MultipeerIbgdaTransportAmd::numPeers() const {
  return targetRanks_.empty() ? (nRanks_ - 1)
                              : static_cast<int>(targetRanks_.size());
}

int MultipeerIbgdaTransportAmd::myRank() const {
  return myRank_;
}

int MultipeerIbgdaTransportAmd::nRanks() const {
  return nRanks_;
}

IbgdaLocalBuffer MultipeerIbgdaTransportAmd::registerBuffer(
    void* ptr,
    std::size_t size) {
  if (!ptr || size == 0)
    throw std::invalid_argument("Invalid buffer");

  auto it = registeredBuffers_.find(ptr);
  if (it != registeredBuffers_.end())
    return IbgdaLocalBuffer(ptr, NetworkLKeys{HostLKey(it->second->lkey)});

  int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  ibv_mr* mr = registerRdmaBuffer(ibvPd_, ptr, size, accessFlags);
  if (!mr)
    throw std::runtime_error(
        "Failed to register buffer with RDMA. "
        "Tried dmabuf (hsa_amd_portable_export_dmabuf) and "
        "legacy ibv_reg_mr. Check kernel/driver support.");

  registeredBuffers_.emplace(ptr, mr);

  VLOG(1) << "MultipeerIbgdaTransportAmd: registered buffer ptr=" << ptr
          << " size=" << size << " lkey=" << mr->lkey;

  return IbgdaLocalBuffer(ptr, NetworkLKeys{HostLKey(mr->lkey)});
}

void MultipeerIbgdaTransportAmd::deregisterBuffer(void* ptr) {
  auto it = registeredBuffers_.find(ptr);
  if (it == registeredBuffers_.end())
    return;
  ibv_dereg_mr(it->second);
  registeredBuffers_.erase(it);
}

std::vector<IbgdaRemoteBuffer> MultipeerIbgdaTransportAmd::exchangeBuffer(
    const IbgdaLocalBuffer& localBuf) {
  const int nPeers = numPeers();

  auto it = registeredBuffers_.find(localBuf.ptr);
  if (it == registeredBuffers_.end())
    throw std::runtime_error("Buffer not registered");

  // AllGather with ALL ranks (even in filtered mode)
  std::vector<IbgdaBufferExchInfo> allInfo(nRanks_);
  allInfo[myRank_] = IbgdaBufferExchInfo{
      reinterpret_cast<uint64_t>(localBuf.ptr),
      HostRKey(it->second->rkey),
  };

  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IbgdaBufferExchInfo), myRank_, nRanks_)
          .get();
  if (result != 0)
    throw std::runtime_error("exchangeBuffer allGather failed");

  // Return only buffers for our target peers
  std::vector<IbgdaRemoteBuffer> peerBuffers(nPeers);
  for (int pi = 0; pi < nPeers; pi++) {
    int peerRank = peerIndexToRank(pi);
    peerBuffers[pi] = allInfo[peerRank].toRemoteBuffer();
  }

  return peerBuffers;
}

int MultipeerIbgdaTransportAmd::rankToPeerIndex(int rank) const {
  if (targetRanks_.empty()) {
    // Unfiltered: dense mapping skipping self
    return (rank < myRank_) ? rank : (rank - 1);
  }
  // Filtered: find rank in targetRanks_
  for (int i = 0; i < static_cast<int>(targetRanks_.size()); i++) {
    if (targetRanks_[i] == rank)
      return i;
  }
  return -1; // rank not in target list
}

int MultipeerIbgdaTransportAmd::peerIndexToRank(int peerIndex) const {
  if (targetRanks_.empty()) {
    return (peerIndex < myRank_) ? peerIndex : (peerIndex + 1);
  }
  return targetRanks_[peerIndex];
}

void MultipeerIbgdaTransportAmd::cleanup() {
  if (peerTransportsGpu_) {
    freeDeviceTransportsOnGpuAmd(peerTransportsGpu_);
    peerTransportsGpu_ = nullptr;
  }

  for (auto& res : peerResources_) {
    if (res.gpuQp) {
      hipFree(res.gpuQp);
      res.gpuQp = nullptr;
    }
    if (res.gpuUarBf) {
      hsa_amd_memory_unlock(res.uarBfHostPtr);
      res.gpuUarBf = nullptr;
    }
    // Unregister hipHostRegister'd buffers (reverse order of registration)
    if (res.registeredCqDbrecPage) {
      hipHostUnregister(res.registeredCqDbrecPage);
      res.registeredCqDbrecPage = nullptr;
    }
    if (res.registeredSqDbrecPage) {
      hipHostUnregister(res.registeredSqDbrecPage);
      res.registeredSqDbrecPage = nullptr;
    }
    if (res.registeredCqBuf) {
      hipHostUnregister(res.registeredCqBuf);
      res.registeredCqBuf = nullptr;
    }
    if (res.registeredSqBuf) {
      hipHostUnregister(res.registeredSqBuf);
      res.registeredSqBuf = nullptr;
    }
    if (res.qp) {
      ibv_destroy_qp(res.qp);
      res.qp = nullptr;
    }
    if (res.cq) {
      ibv_destroy_cq(res.cq);
      res.cq = nullptr;
    }
  }
  peerResources_.clear();

  for (auto& [_, mr] : registeredBuffers_)
    ibv_dereg_mr(mr);
  registeredBuffers_.clear();

  if (sinkMr_) {
    ibv_dereg_mr(sinkMr_);
    sinkMr_ = nullptr;
  }
  if (sinkBuffer_) {
    hipHostFree(sinkBuffer_);
    sinkBuffer_ = nullptr;
  }
  if (ibvPd_) {
    ibv_dealloc_pd(ibvPd_);
    ibvPd_ = nullptr;
  }
  if (ibvCtx_) {
    ibv_close_device(ibvCtx_);
    ibvCtx_ = nullptr;
  }
}

} // namespace pipes_gda
#endif
