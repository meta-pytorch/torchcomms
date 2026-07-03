// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// NIC -> CPU path bandwidth benchmark for Grace-Blackwell (GB300).
//
// Goal: empirically show that an RDMA write whose landing buffer is plain
// Grace (CPU) memory is capped by the NIC's PCIe link to Grace, while a
// landing buffer exposed behind the GPU travels NIC -> GPU -> C2C -> Grace and
// reaches a different (higher) rate. What decides the path is the DMA address
// the NIC targets, i.e. how the receiver's landing buffer is allocated and
// registered. Sweeps receiver-side configs and reports RDMA_WRITE bandwidth:
//
//   host              posix_memalign + regMr           NIC -> Grace
//   system-ats        malloc + regMr (ATS translate)   NIC -> Grace
//   cuda-host-reg     malloc + cudaHostRegister(Mapped)
//                     + cudaHostGetDevicePointer + regMr(devptr)
//                                                       NIC -> GPU -> C2C?
//                                                       (test)
//   egm               VMM HOST_NUMA + dmabuf + regDmabufMr   (opt-in; dmabuf
//                     export of host-NUMA mem is unsupported on some drivers)
//   hbm               VMM DEVICE (gpuDirectRDMACapable) + dmabuf   NIC -> GPU
//   HBM
//
// The SENDER sources writes from GPU HBM so its egress is never the bottleneck;
// only the receiver's landing buffer varies.
//
// NIC selection mirrors ncclx/ctran: comms::prims::GpuNicDiscovery picks the
// PCIe-affinity NIC for the local GPU (raw index 0 lands on the wrong plane on
// GB300). The RoCEv2 GID index is auto-detected from sysfs (like ncclx's
// ncclIbGetGidIndex) instead of hardcoding 3.
//
// GB300/aarch64 (SMMU): device memory cannot be registered as an IB MR via
// plain regMr; device buffers use cuMemCreate(gpuDirectRDMACapable=1) + dmabuf.
//
// Env knobs: NIC_NAME (force NIC), DATA_DIRECT (0/1/2, default 0=regular NIC),
// GID_INDEX / NCCL_IB_GID_INDEX (force GID), EGM_NUMA_NODE, DEST_CONFIGS.

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/prims/transport/rdma/NicDiscovery.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ibverbx;
using meta::comms::BenchmarkEnvironment;
using meta::comms::BenchmarkTestFixture;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

//------------------------------------------------------------------------------
// Constants / config
//------------------------------------------------------------------------------

constexpr uint8_t kPortNum = 1;
constexpr uint32_t kTotalQps = 8;
constexpr uint32_t kMaxMsgCntPerQp = 128;
constexpr uint32_t kMaxMsgSize = 524288;
constexpr uint32_t kMaxOutstandingWrs = 128;
constexpr uint32_t kMaxSge = 1;
constexpr uint32_t kVirtualCqSize = 32768;

constexpr int kBwTxDepth = 64;

constexpr int kSenderRank = 0;
constexpr int kReceiverRank = 1;

const std::vector<size_t> kMsgSizes = {
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024};
constexpr size_t kMaxBufferSize = 256 * 1024 * 1024;

//------------------------------------------------------------------------------
// Error-check helpers
//------------------------------------------------------------------------------

#define CUDART_CHECK(expr)                                     \
  do {                                                         \
    cudaError_t _err = (expr);                                 \
    if (_err != cudaSuccess) {                                 \
      throw std::runtime_error(                                \
          std::string("CUDA runtime error: ") + #expr + ": " + \
          cudaGetErrorString(_err));                           \
    }                                                          \
  } while (0)

#define CU_CHECK(expr)                                        \
  do {                                                        \
    CUresult _err = (expr);                                   \
    if (_err != CUDA_SUCCESS) {                               \
      const char* _name = nullptr;                            \
      cuGetErrorName(_err, &_name);                           \
      throw std::runtime_error(                               \
          std::string("CUDA driver error: ") + #expr + ": " + \
          (_name ? _name : "unknown"));                       \
    }                                                         \
  } while (0)

static int envInt(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

//------------------------------------------------------------------------------
// NIC selection (PCIe affinity, like ncclx/ctran) + RoCEv2 GID auto-detect
//------------------------------------------------------------------------------

// Pick the topologically-closest backend NIC for the local GPU. Falls back to
// NIC_NAME env if set. DATA_DIRECT selects regular (0) vs data-direct (1/2).
static std::string resolveNicName(int cudaDev) {
  const char* forced = std::getenv("NIC_NAME");
  if (forced && std::strlen(forced) > 0) {
    return std::string(forced);
  }
  using comms::prims::DataDirectMode;
  int dd = envInt("DATA_DIRECT", 0);
  DataDirectMode mode = (dd == 2) ? DataDirectMode::Both
      : (dd == 1)                 ? DataDirectMode::Only
                                  : DataDirectMode::Disabled;
  comms::prims::GpuNicDiscovery disc(cudaDev, /*ibHcaEnv=*/"", mode);
  const auto& cands = disc.getCandidates();
  std::cout << "[nic] GPU " << cudaDev << " candidates (DATA_DIRECT=" << dd
            << "):" << std::endl;
  for (const auto& c : cands) {
    std::cout << "  " << c.name << " pcie=" << c.pcie
              << " path=" << comms::prims::pathTypeToString(c.pathType)
              << " bw=" << c.bandwidthGbps << "Gbps dd=" << c.isDataDirect
              << std::endl;
  }
  auto best = disc.getBestAffinityNics();
  if (best.empty()) {
    throw std::runtime_error("GpuNicDiscovery found no affinity NIC");
  }
  return best.at(0).name;
}

// True if this is a usable RoCEv2 IPv4-mapped, non-link-local GID.
static bool isRoceV2Ipv4Gid(const ibv_gid& gid) {
  const uint8_t* r = gid.raw;
  for (int i = 0; i < 10; ++i) {
    if (r[i] != 0) {
      return false;
    }
  }
  return r[10] == 0xff && r[11] == 0xff;
}

// Auto-detect the RoCEv2 GID index by scanning the sysfs gid-type table, like
// ncclx's ncclIbGetGidIndex. Honors GID_INDEX / NCCL_IB_GID_INDEX overrides.
static int detectRoceV2GidIndex(const IbvDevice& dev) {
  int ov = envInt("GID_INDEX", envInt("NCCL_IB_GID_INDEX", -1));
  if (ov >= 0) {
    return ov;
  }
  const char* name = dev.device()->name;
  auto port = dev.queryPort(kPortNum);
  int tblLen = (port && port->gid_tbl_len > 0) ? port->gid_tbl_len : 8;
  int best = -1;
  for (int i = 0; i < tblLen; ++i) {
    std::string path = std::string("/sys/class/infiniband/") + name +
        "/ports/" + std::to_string(kPortNum) + "/gid_attrs/types/" +
        std::to_string(i);
    std::ifstream f(path);
    std::string type;
    if (!std::getline(f, type)) {
      continue;
    }
    if (type.find("RoCE v2") == std::string::npos) {
      continue;
    }
    auto gid = dev.queryGid(kPortNum, i);
    if (gid && isRoceV2Ipv4Gid(*gid)) {
      best = i; // keep the highest matching index
    }
  }
  return best >= 0 ? best : 3;
}

//------------------------------------------------------------------------------
// VMM device (HBM) allocation registerable as an IB MR on GB300/aarch64
//------------------------------------------------------------------------------

struct VmmBuffer {
  CUdeviceptr ptr{0};
  CUmemGenericAllocationHandle handle{};
  size_t size{0};
  int dmabufFd{-1};

  void reset() {
    if (dmabufFd >= 0) {
      close(dmabufFd);
      dmabufFd = -1;
    }
    if (ptr) {
      cuMemUnmap(ptr, size);
      cuMemRelease(handle);
      cuMemAddressFree(ptr, size);
      ptr = 0;
    }
  }
};

static VmmBuffer allocHbmRdma(size_t size, int cudaDev) {
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, cudaDev));

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  int rdmaFlag = 0;
  cuDeviceGetAttribute(
      &rdmaFlag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      dev);
  if (rdmaFlag) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  size_t gran = 0;
  CU_CHECK(cuMemGetAllocationGranularity(
      &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  VmmBuffer b;
  b.size = ((size + gran - 1) / gran) * gran;
  CU_CHECK(cuMemCreate(&b.handle, b.size, &prop, 0));
  CU_CHECK(cuMemAddressReserve(&b.ptr, b.size, gran, 0, 0));
  CU_CHECK(cuMemMap(b.ptr, b.size, 0, b.handle, 0));

  CUmemAccessDesc acc{};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = dev;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(b.ptr, b.size, &acc, 1));
  CUDART_CHECK(cudaMemset(reinterpret_cast<void*>(b.ptr), 0, size));
  // Device memory registers via dmabuf on GB300/aarch64 (SMMU).
  CU_CHECK(cuMemGetHandleForAddressRange(
      &b.dmabufFd, b.ptr, b.size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  return b;
}

//------------------------------------------------------------------------------
// Destination config
//------------------------------------------------------------------------------

enum class DestConfig { Host, SystemAts, CudaHostRegister, Egm, Hbm };

static const char* destName(DestConfig c) {
  switch (c) {
    case DestConfig::Host:
      return "host";
    case DestConfig::SystemAts:
      return "system-ats";
    case DestConfig::CudaHostRegister:
      return "cuda-host-reg";
    case DestConfig::Egm:
      return "egm";
    case DestConfig::Hbm:
      return "hbm";
  }
  return "?";
}

static const char* destPath(DestConfig c) {
  switch (c) {
    case DestConfig::Host:
    case DestConfig::SystemAts:
      return "NIC->Grace";
    case DestConfig::CudaHostRegister:
      return "NIC->GPU->C2C? (test)";
    case DestConfig::Egm:
      return "NIC->GPU->C2C->Grace";
    case DestConfig::Hbm:
      return "NIC->GPU HBM";
  }
  return "?";
}

static bool parseOne(const std::string& tok, DestConfig& out) {
  if (tok == "host") {
    out = DestConfig::Host;
  } else if (tok == "system-ats") {
    out = DestConfig::SystemAts;
  } else if (tok == "cuda-host-reg") {
    out = DestConfig::CudaHostRegister;
  } else if (tok == "egm") {
    out = DestConfig::Egm;
  } else if (tok == "hbm") {
    out = DestConfig::Hbm;
  } else {
    return false;
  }
  return true;
}

static std::vector<DestConfig> parseConfigs() {
  const char* v = std::getenv("DEST_CONFIGS");
  if (!v || std::strlen(v) == 0) {
    // egm excluded by default: dmabuf export of host-NUMA memory is unsupported
    // on some drivers. Opt in with DEST_CONFIGS=...,egm.
    return {
        DestConfig::Host,
        DestConfig::SystemAts,
        DestConfig::CudaHostRegister,
        DestConfig::Hbm};
  }
  std::vector<DestConfig> out;
  std::stringstream ss(v);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    DestConfig c;
    if (parseOne(tok, c)) {
      out.push_back(c);
    }
  }
  return out;
}

//------------------------------------------------------------------------------
// IbvEndPoint (single NIC, VirtualQp)
//------------------------------------------------------------------------------

class IbvEndPoint {
 public:
  explicit IbvEndPoint(const std::string& nicName);

  ibv_qp_init_attr makeIbvQpInitAttr();
  ibv_qp_attr makeQpAttrInit();
  ibv_qp_attr makeQpAttrRtr(ibv_gid remoteGid);
  static ibv_qp_attr makeQpAttrRts();
  void changeVirtualQpStateToRts(
      ibv_gid remoteGid,
      const IbvVirtualQpBusinessCard& remoteBusinessCard);

  IbvDevice device;
  IbvPd pd;
  IbvVirtualCq cq;
  IbvVirtualQp qp;
  int gidIndex{3};
};

IbvEndPoint::IbvEndPoint(const std::string& nicName)
    : device(([&nicName]() {
        if (!ibvInit()) {
          throw std::runtime_error("ibvInit() failed");
        }
        std::vector<std::string> hca = {nicName};
        auto devices = IbvDevice::ibvGetDeviceList(hca, "=");
        if (!devices || devices->empty()) {
          throw std::runtime_error("NIC '" + nicName + "' not found");
        }
        return std::move(devices->at(0));
      })()),
      pd(([this]() {
        auto maybePd = device.allocPd();
        if (!maybePd) {
          throw std::runtime_error("Failed to allocate protection domain");
        }
        return std::move(*maybePd);
      })()),
      cq(([this]() {
        auto maybeCq =
            device.createVirtualCq(kVirtualCqSize, nullptr, nullptr, 0);
        if (!maybeCq) {
          throw std::runtime_error("Failed to create virtual CQ");
        }
        return std::move(*maybeCq);
      })()),
      qp([this]() {
        auto initAttr = makeIbvQpInitAttr();
        auto maybeQp = pd.createVirtualQp(
            kTotalQps,
            &initAttr,
            &cq,
            kMaxMsgCntPerQp,
            kMaxMsgSize,
            LoadBalancingScheme::SPRAY);
        if (!maybeQp) {
          throw std::runtime_error("Failed to create virtual QP");
        }
        return std::move(*maybeQp);
      }()) {
  gidIndex = detectRoceV2GidIndex(device);
}

ibv_qp_init_attr IbvEndPoint::makeIbvQpInitAttr() {
  ibv_qp_init_attr initAttr{};
  initAttr.send_cq = cq.getPhysicalCqsRef().at(0).cq();
  initAttr.recv_cq = cq.getPhysicalCqsRef().at(0).cq();
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.sq_sig_all = 0;
  initAttr.cap.max_send_wr = kMaxOutstandingWrs;
  initAttr.cap.max_recv_wr = kMaxOutstandingWrs;
  initAttr.cap.max_send_sge = kMaxSge;
  initAttr.cap.max_recv_sge = kMaxSge;
  initAttr.cap.max_inline_data = 0;
  return initAttr;
}

ibv_qp_attr IbvEndPoint::makeQpAttrInit() {
  ibv_qp_attr qpAttr = {
      .qp_state = IBV_QPS_INIT,
      .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_WRITE,
      .pkey_index = 0,
      .port_num = kPortNum,
  };
  return qpAttr;
}

ibv_qp_attr IbvEndPoint::makeQpAttrRtr(ibv_gid remoteGid) {
  ibv_qp_attr qpAttr{};
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = IBV_MTU_4096;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  qpAttr.ah_attr.is_global = 1;
  qpAttr.ah_attr.grh.dgid.global.subnet_prefix = remoteGid.global.subnet_prefix;
  qpAttr.ah_attr.grh.dgid.global.interface_id = remoteGid.global.interface_id;
  qpAttr.ah_attr.grh.flow_label = 0;
  qpAttr.ah_attr.grh.sgid_index = gidIndex;
  qpAttr.ah_attr.grh.hop_limit = 255;
  qpAttr.ah_attr.grh.traffic_class = 0;
  qpAttr.ah_attr.sl = 0;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = kPortNum;
  return qpAttr;
}

ibv_qp_attr IbvEndPoint::makeQpAttrRts() {
  ibv_qp_attr qpAttr{};
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = 14;
  qpAttr.retry_cnt = 7;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  return qpAttr;
}

void IbvEndPoint::changeVirtualQpStateToRts(
    ibv_gid remoteGid,
    const IbvVirtualQpBusinessCard& remoteBusinessCard) {
  {
    auto qpAttr = makeQpAttrInit();
    if (!qp.modifyVirtualQp(
            &qpAttr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                IBV_QP_ACCESS_FLAGS)) {
      throw std::runtime_error("Failed to modify QP to INIT");
    }
  }
  {
    auto qpAttr = makeQpAttrRtr(remoteGid);
    if (!qp.modifyVirtualQp(
            &qpAttr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER,
            remoteBusinessCard)) {
      throw std::runtime_error("Failed to modify QP to RTR");
    }
  }
  {
    auto qpAttr = makeQpAttrRts();
    if (!qp.modifyVirtualQp(
            &qpAttr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
      throw std::runtime_error("Failed to modify QP to RTS");
    }
  }
}

//------------------------------------------------------------------------------
// Connection exchange (business card + gid) via bootstrap allGather
//------------------------------------------------------------------------------

struct ConnInfo {
  uint8_t gid[16];
  char businessCardJson[1024];
  size_t businessCardJsonLen;
};

static ConnInfo makeLocalConnInfo(IbvEndPoint& ep) {
  ConnInfo info{};
  auto gid = ep.device.queryGid(kPortNum, ep.gidIndex);
  if (!gid) {
    throw std::runtime_error("Failed to query GID");
  }
  std::memcpy(info.gid, &(*gid), 16);

  auto card = ep.qp.getVirtualQpBusinessCard();
  std::string json = card.serialize();
  if (json.size() >= sizeof(info.businessCardJson)) {
    throw std::runtime_error("Business card JSON too large");
  }
  std::memcpy(info.businessCardJson, json.c_str(), json.size());
  info.businessCardJsonLen = json.size();
  return info;
}

//------------------------------------------------------------------------------
// Receiver-side destination buffer (one per config)
//------------------------------------------------------------------------------

struct DestBuffer {
  uint64_t addr{0};
  uint32_t rkey{0};
  std::optional<IbvMr> mr;

  DestConfig config{};
  void* hostPtr{nullptr};
  bool hostRegistered{false};
  VmmBuffer vmm; // egm (host-numa) or hbm (device); owns its dmabuf fd
  void* copyDst{nullptr};

  DestBuffer() = default;
  DestBuffer(const DestBuffer&) = delete;
  DestBuffer& operator=(const DestBuffer&) = delete;

  ~DestBuffer() {
    mr.reset();
    vmm.reset();
    if (hostRegistered && hostPtr) {
      cudaHostUnregister(hostPtr);
    }
    if (hostPtr) {
      free(hostPtr);
    }
    if (copyDst) {
      free(copyDst);
    }
  }
};

static ibv_access_flags kAccess = static_cast<ibv_access_flags>(
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

// EGM: Grace LPDDR exposed behind the GPU via the CUDA VMM driver API.
static void allocEgm(DestBuffer& d, size_t size, int cudaDev, int numaNode) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = numaNode;

  size_t gran = 0;
  CU_CHECK(cuMemGetAllocationGranularity(
      &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  d.vmm.size = ((size + gran - 1) / gran) * gran;

  CU_CHECK(cuMemCreate(&d.vmm.handle, d.vmm.size, &prop, 0));
  CU_CHECK(cuMemAddressReserve(&d.vmm.ptr, d.vmm.size, gran, 0, 0));
  CU_CHECK(cuMemMap(d.vmm.ptr, d.vmm.size, 0, d.vmm.handle, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = cudaDev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(d.vmm.ptr, d.vmm.size, &access, 1));

  CU_CHECK(cuMemGetHandleForAddressRange(
      &d.vmm.dmabufFd,
      d.vmm.ptr,
      d.vmm.size,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      0));
}

static std::unique_ptr<DestBuffer>
makeDestBuffer(DestConfig config, IbvPd& pd, size_t size, int cudaDev) {
  const int numaNode = envInt("EGM_NUMA_NODE", 0);

  auto d = std::make_unique<DestBuffer>();
  d->config = config;

  switch (config) {
    case DestConfig::Host: {
      if (posix_memalign(&d->hostPtr, 4096, size) != 0) {
        throw std::runtime_error("posix_memalign failed");
      }
      std::memset(d->hostPtr, 0, size);
      auto mr = pd.regMr(d->hostPtr, size, kAccess);
      if (!mr) {
        throw std::runtime_error("regMr(host) failed");
      }
      d->mr = std::move(*mr);
      d->addr = reinterpret_cast<uint64_t>(d->hostPtr);
      break;
    }
    case DestConfig::SystemAts: {
      d->hostPtr = malloc(size);
      if (!d->hostPtr) {
        throw std::runtime_error("malloc failed");
      }
      std::memset(d->hostPtr, 0, size);
      auto mr = pd.regMr(d->hostPtr, size, kAccess);
      if (!mr) {
        throw std::runtime_error("regMr(system-ats) failed");
      }
      d->mr = std::move(*mr);
      d->addr = reinterpret_cast<uint64_t>(d->hostPtr);
      break;
    }
    case DestConfig::CudaHostRegister: {
      if (posix_memalign(&d->hostPtr, 4096, size) != 0) {
        throw std::runtime_error("posix_memalign failed");
      }
      std::memset(d->hostPtr, 0, size);
      CUDART_CHECK(cudaSetDevice(cudaDev));
      CUDART_CHECK(cudaHostRegister(
          d->hostPtr, size, cudaHostRegisterMapped | cudaHostRegisterPortable));
      d->hostRegistered = true;
      void* devPtr = nullptr;
      CUDART_CHECK(cudaHostGetDevicePointer(&devPtr, d->hostPtr, 0));
      auto mr = pd.regMr(devPtr, size, kAccess);
      if (!mr) {
        throw std::runtime_error("regMr(cuda-host-reg devptr) failed");
      }
      d->mr = std::move(*mr);
      d->addr = reinterpret_cast<uint64_t>(devPtr);
      break;
    }
    case DestConfig::Egm: {
      CUDART_CHECK(cudaSetDevice(cudaDev));
      allocEgm(*d, size, cudaDev, numaNode);
      auto mr = pd.regDmabufMr(
          /*offset=*/0,
          d->vmm.size,
          /*iova=*/static_cast<uint64_t>(d->vmm.ptr),
          d->vmm.dmabufFd,
          kAccess);
      if (!mr) {
        throw std::runtime_error("regDmabufMr(egm) failed");
      }
      d->mr = std::move(*mr);
      d->addr = static_cast<uint64_t>(d->vmm.ptr);
      break;
    }
    case DestConfig::Hbm: {
      CUDART_CHECK(cudaSetDevice(cudaDev));
      d->vmm = allocHbmRdma(size, cudaDev);
      auto mr = pd.regDmabufMr(
          /*offset=*/0,
          d->vmm.size,
          /*iova=*/static_cast<uint64_t>(d->vmm.ptr),
          d->vmm.dmabufFd,
          kAccess);
      if (!mr) {
        throw std::runtime_error("regDmabufMr(hbm vmm) failed");
      }
      d->mr = std::move(*mr);
      d->addr = static_cast<uint64_t>(d->vmm.ptr);
      if (posix_memalign(&d->copyDst, 4096, size) != 0) {
        throw std::runtime_error("posix_memalign(copyDst) failed");
      }
      break;
    }
  }
  return d;
}

//------------------------------------------------------------------------------
// Sender-side HBM source buffer (fast egress so it isn't the bottleneck)
//------------------------------------------------------------------------------

struct SourceBuffer {
  VmmBuffer vmm;
  std::optional<IbvMr> mr;

  void* ptr() const {
    return reinterpret_cast<void*>(vmm.ptr);
  }

  SourceBuffer() = default;
  SourceBuffer(const SourceBuffer&) = delete;
  SourceBuffer& operator=(const SourceBuffer&) = delete;
  ~SourceBuffer() {
    mr.reset();
    vmm.reset();
  }
};

static std::unique_ptr<SourceBuffer>
makeSourceBuffer(IbvPd& pd, size_t size, int cudaDev) {
  auto s = std::make_unique<SourceBuffer>();
  CUDART_CHECK(cudaSetDevice(cudaDev));
  s->vmm = allocHbmRdma(size, cudaDev);
  auto mr = pd.regDmabufMr(
      /*offset=*/0,
      s->vmm.size,
      /*iova=*/static_cast<uint64_t>(s->vmm.ptr),
      s->vmm.dmabufFd,
      kAccess);
  if (!mr) {
    throw std::runtime_error("regDmabufMr(source hbm vmm) failed");
  }
  s->mr = std::move(*mr);
  return s;
}

//------------------------------------------------------------------------------
// Bandwidth loop (pipelined RDMA_WRITE, sender-timed)
//------------------------------------------------------------------------------

struct DestKeys {
  int ok;
  uint32_t rkey;
  uint64_t addr;
};

static double runSenderBandwidth(
    IbvEndPoint& ep,
    SourceBuffer& src,
    const DestKeys& dest,
    size_t msgSize) {
  // Scale iterations so each size moves ~kTargetBytes total: enough samples for
  // a stable BW while bounding per-size wall time (avoids TcpStore barrier
  // timeouts at large sizes / low bandwidth).
  const uint64_t kTargetBytes = 8ULL << 30; // ~8 GB
  const int iters = static_cast<int>(
      std::clamp<uint64_t>(kTargetBytes / msgSize, 20ULL, 2000ULL));
  const int warmup = std::max(3, iters / 10);

  IbvVirtualSendWr sendWr;
  sendWr.localAddr = src.ptr();
  sendWr.length = static_cast<uint32_t>(msgSize);
  sendWr.remoteAddr = dest.addr;
  sendWr.opcode = IBV_WR_RDMA_WRITE;
  sendWr.sendFlags = IBV_SEND_SIGNALED;
  int32_t deviceId = ep.qp.getQpsRef().at(0).getDeviceId();
  sendWr.deviceKeys[deviceId] =
      MemoryRegionKeys{.lkey = src.mr->mr()->lkey, .rkey = dest.rkey};

  uint64_t wrId = 0;

  for (int i = 0; i < warmup; ++i) {
    sendWr.wrId = wrId++;
    ep.qp.postSend(sendWr);
    while (true) {
      auto wcs = ep.cq.pollCq();
      if (wcs && !wcs->empty()) {
        if (wcs->at(0).status != IBV_WC_SUCCESS) {
          throw std::runtime_error("warmup WC failed");
        }
        break;
      }
    }
  }

  using Clock = std::chrono::high_resolution_clock;
  uint64_t scnt = 0;
  uint64_t ccnt = 0;
  auto start = Clock::now();

  while (scnt < static_cast<uint64_t>(iters) ||
         ccnt < static_cast<uint64_t>(iters)) {
    while (scnt < static_cast<uint64_t>(iters) &&
           (scnt - ccnt) < static_cast<uint64_t>(kBwTxDepth)) {
      sendWr.wrId = wrId++;
      ep.qp.postSend(sendWr);
      ++scnt;
    }
    if (ccnt < static_cast<uint64_t>(iters)) {
      auto wcs = ep.cq.pollCq();
      if (wcs && !wcs->empty()) {
        for (const auto& wc : *wcs) {
          if (wc.status != IBV_WC_SUCCESS) {
            throw std::runtime_error("bw WC failed");
          }
          ++ccnt;
        }
      }
    }
  }
  auto end = Clock::now();

  double us = std::chrono::duration<double, std::micro>(end - start).count();
  uint64_t totalBytes = static_cast<uint64_t>(iters) * msgSize;
  return (us > 0) ? static_cast<double>(totalBytes) / (us * 1000.0) : 0.0;
}

static double measureHbmToCpuCopy(DestBuffer& dest, size_t msgSize) {
  const int iters = 50;
  void* devPtr = reinterpret_cast<void*>(dest.vmm.ptr);
  CUDART_CHECK(
      cudaMemcpy(dest.copyDst, devPtr, msgSize, cudaMemcpyDeviceToHost));
  using Clock = std::chrono::high_resolution_clock;
  auto start = Clock::now();
  for (int i = 0; i < iters; ++i) {
    CUDART_CHECK(
        cudaMemcpy(dest.copyDst, devPtr, msgSize, cudaMemcpyDeviceToHost));
  }
  auto end = Clock::now();
  double us = std::chrono::duration<double, std::micro>(end - start).count();
  uint64_t totalBytes = static_cast<uint64_t>(iters) * msgSize;
  return (us > 0) ? static_cast<double>(totalBytes) / (us * 1000.0) : 0.0;
}

static std::string formatSize(size_t bytes) {
  std::stringstream ss;
  if (bytes >= 1024 * 1024) {
    ss << (bytes / (1024 * 1024)) << "MB";
  } else if (bytes >= 1024) {
    ss << (bytes / 1024) << "KB";
  } else {
    ss << bytes << "B";
  }
  return ss.str();
}

//------------------------------------------------------------------------------
// Test
//------------------------------------------------------------------------------

class NicToCpuPathFixture : public BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    ncclCvarInit();
  }

  template <typename T>
  std::vector<T> allGatherStruct(const T& local) {
    std::vector<T> all(worldSize);
    all[globalRank] = local;
    auto res =
        bootstrap->allGather(all.data(), sizeof(T), globalRank, worldSize);
    if (std::move(res).get() != 0) {
      throw std::runtime_error("allGather failed");
    }
    return all;
  }
};

TEST_F(NicToCpuPathFixture, NicToCpuPathBandwidth) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  const bool isSender = (globalRank == kSenderRank);
  const int cudaDev = localRank;

  CU_CHECK(cuInit(0));
  CUDART_CHECK(cudaSetDevice(cudaDev));

  std::string nicName = resolveNicName(cudaDev);
  auto ep = std::make_unique<IbvEndPoint>(nicName);
  std::cout << "[rank " << globalRank << "] NIC='" << nicName
            << "' gidIndex=" << ep->gidIndex << " cudaDev=" << cudaDev
            << std::endl;

  // Exchange connection info and bring the QP to RTS.
  auto allConn = allGatherStruct(makeLocalConnInfo(*ep));
  const ConnInfo& remote = allConn[isSender ? kReceiverRank : kSenderRank];
  std::string remoteJson(remote.businessCardJson, remote.businessCardJsonLen);
  auto remoteCard = IbvVirtualQpBusinessCard::deserialize(remoteJson);
  if (!remoteCard) {
    throw std::runtime_error("Failed to deserialize remote business card");
  }
  ibv_gid remoteGid{};
  std::memcpy(&remoteGid, remote.gid, 16);
  ep->changeVirtualQpStateToRts(remoteGid, *remoteCard);

  // Sender sources from GPU HBM so its egress is never the bottleneck; only the
  // receiver's landing buffer varies.
  std::unique_ptr<SourceBuffer> src;
  if (isSender) {
    src = makeSourceBuffer(ep->pd, kMaxBufferSize, cudaDev);
  }

  auto configs = parseConfigs();

  if (isSender) {
    std::cout
        << "\n============================================================================\n"
        << " GB300 NIC -> CPU path bandwidth (RDMA_WRITE, sender source = HBM)\n"
        << "============================================================================\n"
        << " NIC=" << nicName << " gidIndex=" << ep->gidIndex
        << " cudaDev=" << cudaDev
        << " EGM_NUMA_NODE=" << envInt("EGM_NUMA_NODE", 0) << "\n"
        << "----------------------------------------------------------------------------\n"
        << std::left << std::setw(16) << "Config" << std::setw(10) << "MsgSize"
        << std::right << std::setw(14) << "BW(GB/s)" << "  " << std::left
        << "Path\n"
        << "----------------------------------------------------------------------------\n";
  }

  for (DestConfig config : configs) {
    std::unique_ptr<DestBuffer> dest;
    DestKeys localKeys{};
    localKeys.ok = 1;
    if (!isSender) {
      try {
        dest = makeDestBuffer(config, ep->pd, kMaxBufferSize, cudaDev);
        localKeys.rkey = dest->mr->mr()->rkey;
        localKeys.addr = dest->addr;
      } catch (const std::exception& e) {
        localKeys.ok = 0;
        XLOGF(ERR, "[{}] dest build failed: {}", destName(config), e.what());
      }
    }
    auto allKeys = allGatherStruct(localKeys);
    const DestKeys& keys = allKeys[kReceiverRank];

    if (!keys.ok) {
      if (isSender) {
        std::cout << std::left << std::setw(16) << destName(config)
                  << std::setw(10) << "-" << std::right << std::setw(14)
                  << "FAILED" << "  " << std::left << destPath(config)
                  << " (registration failed)" << std::endl;
      }
      continue;
    }

    for (size_t msgSize : kMsgSizes) {
      bootstrap->barrierAll();
      double bw = 0;
      if (isSender) {
        bw = runSenderBandwidth(*ep, *src, keys, msgSize);
      }
      bootstrap->barrierAll();

      if (isSender) {
        std::cout << std::left << std::setw(16) << destName(config)
                  << std::setw(10) << formatSize(msgSize) << std::right
                  << std::setw(14) << std::fixed << std::setprecision(2) << bw
                  << "  " << std::left << destPath(config) << std::endl;
      } else if (config == DestConfig::Hbm && msgSize == kMsgSizes.back()) {
        double copyBw = measureHbmToCpuCopy(*dest, msgSize);
        std::cout << "[hbm] extra HBM->CPU (C2C) copy BW at "
                  << formatSize(msgSize) << ": " << std::fixed
                  << std::setprecision(2) << copyBw << " GB/s" << std::endl;
      }
      bootstrap->barrierAll();
    }
  }

  if (isSender) {
    std::cout
        << "============================================================================\n"
        << " host / system-ats land in Grace directly; cuda-host-reg / egm / hbm\n"
        << " route through the GPU aperture. hbm needs the HBM->CPU copy to reach Grace.\n"
        << "============================================================================\n\n";
  }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  if (!meta::comms::isTcpEnvironment()) {
    ::testing::AddGlobalTestEnvironment(new BenchmarkEnvironment());
  }
  return RUN_ALL_TESTS();
}
