// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <fmt/core.h>
#include <glog/logging.h>

#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace comms::prims {

namespace {

constexpr uint8_t kDefaultIbPort = 1;
constexpr uint8_t kDefaultIbHopLimit = 255;

std::string errnoString(int err) {
  return std::strerror(err);
}

[[noreturn]] void throwIbrcUnimplemented(const char* what) {
  throw std::runtime_error(
      std::string("MultipeerIbrcTransport: ") + what +
      " is not implemented yet");
}

bool isPowerOfTwo(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

#ifdef __HIP_PLATFORM_AMD__
using GpuError = hipError_t;
constexpr GpuError kGpuSuccess = hipSuccess;

const char* gpuGetErrorString(GpuError err) {
  return hipGetErrorString(err);
}

GpuError gpuHostAlloc(void** ptr, std::size_t bytes) {
  return hipHostMalloc(ptr, bytes, hipHostMallocMapped);
}

GpuError gpuHostGetDevicePointer(void** devicePtr, void* hostPtr) {
  return hipHostGetDevicePointer(devicePtr, hostPtr, 0);
}

GpuError gpuFreeHost(void* ptr) {
  return hipHostFree(ptr);
}

GpuError gpuSetDevice(int device) {
  return hipSetDevice(device);
}
#else
using GpuError = cudaError_t;
constexpr GpuError kGpuSuccess = cudaSuccess;

const char* gpuGetErrorString(GpuError err) {
  return cudaGetErrorString(err);
}

GpuError gpuHostAlloc(void** ptr, std::size_t bytes) {
  return cudaHostAlloc(ptr, bytes, cudaHostAllocMapped);
}

GpuError gpuHostGetDevicePointer(void** devicePtr, void* hostPtr) {
  return cudaHostGetDevicePointer(devicePtr, hostPtr, 0);
}

GpuError gpuFreeHost(void* ptr) {
  return cudaFreeHost(ptr);
}

GpuError gpuSetDevice(int device) {
  return cudaSetDevice(device);
}
#endif

void checkGpu(GpuError err, const std::string& what) {
  if (err != kGpuSuccess) {
    throw std::runtime_error(
        fmt::format("{}: {}", what, gpuGetErrorString(err)));
  }
}

std::size_t checkedMul(std::size_t a, std::size_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::overflow_error(
        fmt::format("MultipeerIbrcTransport: {} size overflow", label));
  }
  return a * b;
}

std::size_t checkedAdd(std::size_t a, std::size_t b, const char* label) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::overflow_error(
        fmt::format("MultipeerIbrcTransport: {} size overflow", label));
  }
  return a + b;
}

std::size_t alignUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    throw std::invalid_argument(
        "MultipeerIbrcTransport: alignment must be non-zero");
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return checkedAdd(value, alignment - remainder, "aligned control block");
}

} // namespace

MultipeerIbrcTransport::MappedAllocation::~MappedAllocation() {
  reset();
}

MultipeerIbrcTransport::MappedAllocation::MappedAllocation(
    MappedAllocation&& other) noexcept
    : host(std::exchange(other.host, nullptr)),
      device(std::exchange(other.device, nullptr)),
      bytes(std::exchange(other.bytes, 0)) {}

MultipeerIbrcTransport::MappedAllocation&
MultipeerIbrcTransport::MappedAllocation::operator=(
    MappedAllocation&& other) noexcept {
  if (this != &other) {
    reset();
    host = std::exchange(other.host, nullptr);
    device = std::exchange(other.device, nullptr);
    bytes = std::exchange(other.bytes, 0);
  }
  return *this;
}

void MultipeerIbrcTransport::MappedAllocation::reset() noexcept {
  if (host == nullptr) {
    return;
  }
  const GpuError err = gpuFreeHost(host);
  if (err != kGpuSuccess) {
    LOG(ERROR) << "MultipeerIbrcTransport: gpuFreeHost failed for " << bytes
               << " bytes at " << host << ": " << gpuGetErrorString(err);
  }
  host = nullptr;
  device = nullptr;
  bytes = 0;
}

MultipeerIbrcTransport::MultipeerIbrcTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbTransportConfig& config)
    : MultiPeerIbTransport<MultipeerIbrcTransport>(
          myRank,
          nRanks,
          std::move(bootstrap),
          config) {
  if (config.numQpsPerPeerPerNic < 1 ||
      config.numQpsPerPeerPerNic > kMaxQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "numQpsPerPeerPerNic must be in [1, {}], got {}",
            kMaxQpsPerPeerPerNic,
            config.numQpsPerPeerPerNic));
  }

  peerResources_.resize(nRanks_ - 1);

  try {
    // Pin GPU work to config_.cudaDevice.
    checkGpu(
        gpuSetDevice(config_.cudaDevice),
        "MultipeerIbrcTransport: set CUDA device");
    openNics();
    initializeControlResources();
    if (config_.ibLazyConnect) {
      peerMaterialized_.resize(nRanks_ - 1, false);
    }
  } catch (const std::exception&) {
    cleanup();
    throw;
  }
}

MultipeerIbrcTransport::~MultipeerIbrcTransport() {
  cleanup();
}

void MultipeerIbrcTransport::exchange() {
  if (!config_.ibLazyConnect) {
    exchangeAndConnectQps();
    allocateCmdQueuesForAllPeers();
    VLOG(1) << "MultipeerIbrcTransport: rank " << myRank_ << " allocated "
            << allocatedCmdQueueCount() << " command queues";
  } else {
    VLOG(1)
        << "MultipeerIbrcTransport: rank " << myRank_
        << " lazy exchange complete (per-peer QPs and command queues deferred "
           "to materializePeer)";
  }

  throwIbrcUnimplemented("command-ring/device transport path");
}

void MultipeerIbrcTransport::cleanup() {
  for (int peerIndex = 0; peerIndex < static_cast<int>(peerResources_.size());
       ++peerIndex) {
    cleanupPeerCmdQueues(peerIndex);
    cleanupPeerQps(peerIndex);
  }

  auto& symbols = ibverbx::ibvSymbols;
  if (symbols.ibv_internal_dereg_mr != nullptr) {
    for (auto& [_, cached] : registeredBuffers_) {
      for (int n = 0; n < numNics_; ++n) {
        if (cached.mrs[n] != nullptr) {
          int rc = symbols.ibv_internal_dereg_mr(cached.mrs[n]);
          if (rc != 0) {
            LOG(WARNING) << "Failed to deregister IBRC MR on NIC " << n
                         << ": rc=" << rc;
          }
          cached.mrs[n] = nullptr;
        }
      }
    }
  }
  registeredBuffers_.clear();

  statusHostByNic_.clear();
  statusDeviceByNic_.clear();
  statusControl_.reset();

  closeNics();
}

void MultipeerIbrcTransport::initializeControlResources() {
  if (!isPowerOfTwo(cmdQueueDepth_)) {
    throw std::invalid_argument(
        "MultipeerIbrcTransport: command queue depth must be a power of two");
  }
  if (numNics_ <= 0) {
    throw std::invalid_argument(
        "MultipeerIbrcTransport: numNics must be positive");
  }
  // The progress loop self-limits inflight work to one command-queue ring
  // (cmdQueueDepth_ descriptors), and each descriptor posts at most
  // kIbrcMaxWrsPerDescriptor WRs (RDMA_WRITE + ATOMIC) to its own QP. Requiring
  // the SQ/CQ (sized to qpDepth) to cover a full ring makes overrun
  // structurally impossible without per-post accounting.
  const std::size_t wrsPerRing =
      checkedMul(cmdQueueDepth_, kIbrcMaxWrsPerDescriptor, "qp depth");
  if (config_.qpDepth < wrsPerRing) {
    throw std::invalid_argument(
        fmt::format(
            "MultipeerIbrcTransport: qpDepth ({}) must be >= cmdQueueDepth ({}) "
            "* {}",
            config_.qpDepth,
            cmdQueueDepth_,
            kIbrcMaxWrsPerDescriptor));
  }

  const std::size_t statusBytes = checkedMul(
      static_cast<std::size_t>(numNics_), sizeof(IbrcNicStatus), "NIC status");
  statusControl_ = allocateMapped(statusBytes, "NIC status block");
  auto* const statusHostBase = static_cast<IbrcNicStatus*>(statusControl_.host);
  auto* const statusDeviceBase =
      static_cast<IbrcNicStatus*>(statusControl_.device);
  statusHostByNic_.resize(numNics_);
  statusDeviceByNic_.resize(numNics_);
  for (int nic = 0; nic < numNics_; ++nic) {
    statusHostByNic_.at(nic) = statusHostBase + nic;
    statusDeviceByNic_.at(nic) = statusDeviceBase + nic;
  }

  const std::size_t descBytes = checkedMul(
      static_cast<std::size_t>(cmdQueueDepth_),
      sizeof(IbrcDesc),
      "command queue descriptor");
  // Place pi and ci on separate cache lines (kIbrcCacheLineBytes) to avoid
  // false-sharing across the host-mapped GPU<->CPU boundary: the GPU fetch_adds
  // pi while the CPU writes ci on every reservation/completion.
  cmdQueuePiOffset_ = alignUp(descBytes, kIbrcCacheLineBytes);
  cmdQueueCiOffset_ =
      checkedAdd(cmdQueuePiOffset_, kIbrcCacheLineBytes, "command queue pi");
  cmdQueueControlBytes_ =
      checkedAdd(cmdQueueCiOffset_, kIbrcCacheLineBytes, "command queue ci");
}

void MultipeerIbrcTransport::cleanupPeerCmdQueues(int peerIndex) noexcept {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size())) {
    return;
  }
  auto& peer = peerResources_[peerIndex];
  peer.cmdQueues.clear();
  peer.cmdQueuesAllocated = false;
}

void MultipeerIbrcTransport::allocateCmdQueuesForAllPeers() {
  for (int peerIndex = 0; peerIndex < static_cast<int>(peerResources_.size());
       ++peerIndex) {
    allocatePeerCmdQueues(peerIndex);
  }
}

void MultipeerIbrcTransport::allocatePeerCmdQueues(int peerIndex) {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size())) {
    throw std::invalid_argument(
        fmt::format("allocatePeerCmdQueues: invalid peerIndex={}", peerIndex));
  }

  auto& peer = peerResources_[peerIndex];
  if (peer.cmdQueuesAllocated) {
    return;
  }

  const int numQps = config_.numQpsPerPeerPerNic;
  const std::size_t cmdQueuesPerPeer = checkedMul(
      static_cast<std::size_t>(numNics_),
      static_cast<std::size_t>(numQps),
      "command queues per peer");
  if (cmdQueuesPerPeer == 0) {
    throw std::overflow_error(
        "MultipeerIbrcTransport: command queues per peer must be non-zero");
  }

  std::vector<IbrcCmdQueueHost> cmdQueues;
  cmdQueues.reserve(cmdQueuesPerPeer);

  for (int q = 0; q < numQps; ++q) {
    for (int nic = 0; nic < numNics_; ++nic) {
      IbrcCmdQueueHost cmdQueue;
      cmdQueue.control =
          allocateMapped(cmdQueueControlBytes_, "command queue control block");
      cmdQueue.cmdStates.resize(cmdQueueDepth_);

      auto* const hostBase = static_cast<std::byte*>(cmdQueue.control.host);
      auto* const deviceBase = static_cast<std::byte*>(cmdQueue.control.device);
      cmdQueue.descsHost = reinterpret_cast<IbrcDesc*>(hostBase);
      cmdQueue.piHost =
          reinterpret_cast<uint64_t*>(hostBase + cmdQueuePiOffset_);
      cmdQueue.ciHost =
          reinterpret_cast<uint64_t*>(hostBase + cmdQueueCiOffset_);
      cmdQueue.device.descs = reinterpret_cast<IbrcDesc*>(deviceBase);
      cmdQueue.device.pi =
          reinterpret_cast<uint64_t*>(deviceBase + cmdQueuePiOffset_);
      cmdQueue.device.ci =
          reinterpret_cast<uint64_t*>(deviceBase + cmdQueueCiOffset_);
      cmdQueue.device.status = statusDeviceByNic_.at(nic);
      cmdQueue.device.depth = cmdQueueDepth_;
      cmdQueue.device.mask = cmdQueueDepth_ - 1;

      cmdQueue.nic = static_cast<uint32_t>(nic);
      cmdQueue.qpSlot = static_cast<uint32_t>(q);

      for (uint32_t slot = 0; slot < cmdQueueDepth_; ++slot) {
        cmdQueue.descsHost[slot].ready_seq = kIbrcInvalidReadySeq;
      }
      cmdQueues.push_back(std::move(cmdQueue));
    }
  }

  peer.cmdQueues = std::move(cmdQueues);
  peer.cmdQueuesAllocated = true;
}

std::size_t MultipeerIbrcTransport::allocatedCmdQueueCount() const {
  std::size_t count = 0;
  for (const auto& peer : peerResources_) {
    count = checkedAdd(count, peer.cmdQueues.size(), "allocated command queue");
  }
  return count;
}

MultipeerIbrcTransport::MappedAllocation MultipeerIbrcTransport::allocateMapped(
    std::size_t bytes,
    const char* label) {
  if (bytes == 0) {
    throw std::invalid_argument(
        fmt::format("MultipeerIbrcTransport: {} size must be non-zero", label));
  }

  MappedAllocation allocation;
  allocation.bytes = bytes;
  checkGpu(
      gpuHostAlloc(&allocation.host, bytes),
      fmt::format(
          "MultipeerIbrcTransport: mapped host allocation for {}", label));
  if (allocation.host == nullptr) {
    throw std::runtime_error(
        fmt::format(
            "MultipeerIbrcTransport: mapped host allocation returned null for {}",
            label));
  }
  std::memset(allocation.host, 0, bytes);
  checkGpu(
      gpuHostGetDevicePointer(&allocation.device, allocation.host),
      fmt::format(
          "MultipeerIbrcTransport: mapped device pointer lookup for {}",
          label));
  if (allocation.device == nullptr) {
    throw std::runtime_error(
        fmt::format(
            "MultipeerIbrcTransport: mapped device pointer returned null for {}",
            label));
  }

  return allocation;
}

void MultipeerIbrcTransport::destroyPeerQps(
    std::vector<PeerQpResource>& qpResources) noexcept {
  auto& symbols = ibverbx::ibvSymbols;
  for (auto& qpResource : qpResources) {
    if (qpResource.qp != nullptr &&
        symbols.ibv_internal_destroy_qp != nullptr) {
      int rc = symbols.ibv_internal_destroy_qp(qpResource.qp);
      if (rc != 0) {
        LOG(WARNING) << "Failed to destroy IBRC QP nic=" << qpResource.nic
                     << " qpSlot=" << qpResource.qpSlot << ": rc=" << rc;
      }
      qpResource.qp = nullptr;
    }
  }
  for (auto& qpResource : qpResources) {
    if (qpResource.cq != nullptr &&
        symbols.ibv_internal_destroy_cq != nullptr) {
      int rc = symbols.ibv_internal_destroy_cq(qpResource.cq);
      if (rc != 0) {
        LOG(WARNING) << "Failed to destroy IBRC CQ nic=" << qpResource.nic
                     << " qpSlot=" << qpResource.qpSlot << ": rc=" << rc;
      }
      qpResource.cq = nullptr;
    }
  }
  qpResources.clear();
}

void MultipeerIbrcTransport::closeNics() noexcept {
  auto& symbols = ibverbx::ibvSymbols;
  for (auto& nic : nics_) {
    if (nic.ibvPd != nullptr && symbols.ibv_internal_dealloc_pd != nullptr) {
      int rc = symbols.ibv_internal_dealloc_pd(nic.ibvPd);
      if (rc != 0) {
        LOG(WARNING) << "Failed to dealloc IBRC PD on NIC " << nic.deviceName
                     << ": rc=" << rc;
      }
      nic.ibvPd = nullptr;
    }
  }
  for (auto& nic : nics_) {
    if (nic.ibvCtx != nullptr && symbols.ibv_internal_close_device != nullptr) {
      int rc = symbols.ibv_internal_close_device(nic.ibvCtx);
      if (rc != 0) {
        LOG(WARNING) << "Failed to close IBRC device " << nic.deviceName
                     << ": rc=" << rc;
      }
      nic.ibvCtx = nullptr;
    }
  }
  nics_.clear();
}

void MultipeerIbrcTransport::cleanupPeerQps(int peerIndex) noexcept {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size())) {
    return;
  }
  destroyPeerQps(peerResources_[peerIndex].qpResources);
  peerResources_[peerIndex].qpsConnected = false;
}

void MultipeerIbrcTransport::createPeerQps(int peerIndex) {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size())) {
    throw std::invalid_argument(
        fmt::format("createPeerQps: invalid peerIndex={}", peerIndex));
  }
  auto& peer = peerResources_[peerIndex];
  if (!peer.qpResources.empty()) {
    return;
  }

  const int numQps = config_.numQpsPerPeerPerNic;
  std::vector<PeerQpResource> qpResources;
  qpResources.reserve(static_cast<std::size_t>(numNics_) * numQps);
  auto& symbols = ibverbx::ibvSymbols;

  try {
    for (int q = 0; q < numQps; ++q) {
      for (int nic = 0; nic < numNics_; ++nic) {
        errno = 0;
        ibverbx::ibv_cq* cq = symbols.ibv_internal_create_cq(
            nics_[nic].ibvCtx,
            static_cast<int>(config_.qpDepth),
            nullptr,
            nullptr,
            0);
        if (cq == nullptr) {
          const int savedErrno = errno;
          throw std::runtime_error(
              fmt::format(
                  "Failed to create IBRC CQ for peerIndex={} nic={} qpSlot={}: "
                  "errno={} ({})",
                  peerIndex,
                  nic,
                  q,
                  savedErrno,
                  errnoString(savedErrno)));
        }

        PeerQpResource qpResource;
        qpResource.cq = cq;
        qpResource.nic = nic;
        qpResource.qpSlot = q;
        qpResources.push_back(qpResource);
        auto& createdQpResource = qpResources.back();

        ibverbx::ibv_qp_init_attr initAttr{};
        initAttr.send_cq = cq;
        initAttr.recv_cq = cq;
        initAttr.cap.max_send_wr = config_.qpDepth;
        initAttr.cap.max_recv_wr = 1;
        initAttr.cap.max_send_sge = 1;
        initAttr.cap.max_recv_sge = 1;
        initAttr.cap.max_inline_data = 0;
        initAttr.qp_type = ibverbx::IBV_QPT_RC;
        initAttr.sq_sig_all = 0;

        errno = 0;
        createdQpResource.qp =
            symbols.ibv_internal_create_qp(nics_[nic].ibvPd, &initAttr);
        if (createdQpResource.qp == nullptr) {
          const int savedErrno = errno;
          throw std::runtime_error(
              fmt::format(
                  "Failed to create IBRC QP for peerIndex={} nic={} qpSlot={}: "
                  "errno={} ({})",
                  peerIndex,
                  nic,
                  q,
                  savedErrno,
                  errnoString(savedErrno)));
        }
      }
    }
  } catch (const std::exception&) {
    destroyPeerQps(qpResources);
    throw;
  }

  peer.qpResources = std::move(qpResources);
}

MultipeerIbrcTransport::PeerQpResource&
MultipeerIbrcTransport::qpResourceAt(int peerIndex, int nic, int qpSlot) {
  return const_cast<PeerQpResource&>(
      static_cast<const MultipeerIbrcTransport&>(*this).qpResourceAt(
          peerIndex, nic, qpSlot));
}

const MultipeerIbrcTransport::PeerQpResource&
MultipeerIbrcTransport::qpResourceAt(int peerIndex, int nic, int qpSlot) const {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size()) ||
      nic < 0 || nic >= numNics_ || qpSlot < 0 ||
      qpSlot >= config_.numQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "qpResourceAt: invalid peerIndex={} nic={} qpSlot={}",
            peerIndex,
            nic,
            qpSlot));
  }
  const auto& qpResources = peerResources_[peerIndex].qpResources;
  const int slot = qpSlot * numNics_ + nic;
  if (slot >= static_cast<int>(qpResources.size())) {
    throw std::runtime_error(
        fmt::format(
            "qpResourceAt: peerIndex={} has {} QP resource(s), missing slot {}",
            peerIndex,
            qpResources.size(),
            slot));
  }
  return qpResources[slot];
}

PeerQpPayload MultipeerIbrcTransport::buildLocalQpPayload(int peerIndex) const {
  const int numQps = config_.numQpsPerPeerPerNic;
  PeerQpPayload payload{};
  payload.gidIndex = gidIndex_;
  payload.mtu = static_cast<int>(localMtu_);
  payload.numNics = numNics_;
  payload.numQpsPerPeerPerNic = numQps;

  auto& symbols = ibverbx::ibvSymbols;
  for (int n = 0; n < numNics_; ++n) {
    std::memcpy(
        payload.nicInfo[n].gid,
        nics_[n].localGid.raw,
        sizeof(payload.nicInfo[n].gid));
    ibverbx::ibv_port_attr portAttr{};
    if (symbols.ibv_internal_query_port(
            nics_[n].ibvCtx, kDefaultIbPort, &portAttr) == 0) {
      payload.nicInfo[n].lid = portAttr.lid;
    } else {
      LOG(WARNING) << "Failed to query port for IBRC LID on NIC " << n;
    }
    for (int q = 0; q < numQps; ++q) {
      payload.nicInfo[n].qpns[q] = qpResourceAt(peerIndex, n, q).qp->qp_num;
    }
  }
  return payload;
}

void MultipeerIbrcTransport::connectPeerQp(
    PeerQpResource& qpResource,
    uint32_t remoteQpn,
    const uint8_t* remoteGid,
    uint16_t remoteLid,
    int remoteMtu) {
  if (qpResource.qp == nullptr) {
    throw std::runtime_error("connectPeerQp: QP resource is null");
  }

  auto& symbols = ibverbx::ibvSymbols;
  auto modifyQp = [&](const char* state, ibverbx::ibv_qp_attr& attr, int mask) {
    errno = 0;
    int rc = symbols.ibv_internal_modify_qp(qpResource.qp, &attr, mask);
    if (rc != 0) {
      const int savedErrno = errno;
      throw std::runtime_error(
          fmt::format(
              "Failed to modify IBRC QP {} to {} (nic={} qpSlot={} "
              "remoteQpn={}): rc={} errno={} ({})",
              qpResource.qp->qp_num,
              state,
              qpResource.nic,
              qpResource.qpSlot,
              remoteQpn,
              rc,
              savedErrno,
              errnoString(savedErrno)));
    }
  };

  ibverbx::ibv_qp_attr initAttr{};
  initAttr.qp_state = ibverbx::IBV_QPS_INIT;
  initAttr.pkey_index = 0;
  initAttr.port_num = kDefaultIbPort;
  initAttr.qp_access_flags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ |
      ibverbx::IBV_ACCESS_REMOTE_ATOMIC;
  modifyQp(
      "INIT",
      initAttr,
      ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_PKEY_INDEX |
          ibverbx::IBV_QP_PORT | ibverbx::IBV_QP_ACCESS_FLAGS);

  ibverbx::ibv_qp_attr rtrAttr{};
  rtrAttr.qp_state = ibverbx::IBV_QPS_RTR;
  // path_mtu = min(local, remote), guarding an unset (0) or invalid remote MTU
  // that would otherwise select an invalid ibv_mtu(0) and fail modify_qp.
  rtrAttr.path_mtu = (remoteMtu >= 1 && remoteMtu < static_cast<int>(localMtu_))
      ? static_cast<ibverbx::ibv_mtu>(remoteMtu)
      : localMtu_;
  rtrAttr.dest_qp_num = remoteQpn;
  rtrAttr.rq_psn = 0;
  rtrAttr.max_dest_rd_atomic = 1;
  rtrAttr.min_rnr_timer = config_.minRnrTimer;
  rtrAttr.ah_attr.dlid = remoteLid;
  rtrAttr.ah_attr.sl = config_.serviceLevel;
  rtrAttr.ah_attr.src_path_bits = 0;
  rtrAttr.ah_attr.port_num = kDefaultIbPort;
  rtrAttr.ah_attr.static_rate = 0;
  if (nics_[qpResource.nic].linkLayer == ibverbx::IBV_LINK_LAYER_ETHERNET) {
    rtrAttr.ah_attr.is_global = 1;
    std::memcpy(
        rtrAttr.ah_attr.grh.dgid.raw,
        remoteGid,
        sizeof(rtrAttr.ah_attr.grh.dgid.raw));
    rtrAttr.ah_attr.grh.flow_label = 0;
    rtrAttr.ah_attr.grh.sgid_index = static_cast<uint8_t>(gidIndex_);
    rtrAttr.ah_attr.grh.hop_limit = kDefaultIbHopLimit;
    rtrAttr.ah_attr.grh.traffic_class = config_.trafficClass;
  } else {
    rtrAttr.ah_attr.is_global = 0;
  }
  modifyQp(
      "RTR",
      rtrAttr,
      ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_AV | ibverbx::IBV_QP_PATH_MTU |
          ibverbx::IBV_QP_DEST_QPN | ibverbx::IBV_QP_RQ_PSN |
          ibverbx::IBV_QP_MAX_DEST_RD_ATOMIC | ibverbx::IBV_QP_MIN_RNR_TIMER);

  ibverbx::ibv_qp_attr rtsAttr{};
  rtsAttr.qp_state = ibverbx::IBV_QPS_RTS;
  rtsAttr.sq_psn = 0;
  rtsAttr.timeout = config_.timeout;
  rtsAttr.retry_cnt = config_.retryCount;
  rtsAttr.rnr_retry = config_.rnrRetry;
  rtsAttr.max_rd_atomic = 1;
  modifyQp(
      "RTS",
      rtsAttr,
      ibverbx::IBV_QP_STATE | ibverbx::IBV_QP_SQ_PSN | ibverbx::IBV_QP_TIMEOUT |
          ibverbx::IBV_QP_RETRY_CNT | ibverbx::IBV_QP_RNR_RETRY |
          ibverbx::IBV_QP_MAX_QP_RD_ATOMIC);
}

void MultipeerIbrcTransport::connectPeerQps(
    int peerIndex,
    const PeerQpPayload& remotePayload) {
  if (remotePayload.numNics != numNics_) {
    throw std::runtime_error(
        fmt::format(
            "IBRC peerIndex={} numNics={} vs local {}",
            peerIndex,
            remotePayload.numNics,
            numNics_));
  }
  if (remotePayload.numQpsPerPeerPerNic != config_.numQpsPerPeerPerNic) {
    throw std::runtime_error(
        fmt::format(
            "IBRC peerIndex={} numQps={} vs local {}",
            peerIndex,
            remotePayload.numQpsPerPeerPerNic,
            config_.numQpsPerPeerPerNic));
  }

  const int numQps = config_.numQpsPerPeerPerNic;
  for (int nic = 0; nic < numNics_; ++nic) {
    for (int q = 0; q < numQps; ++q) {
      connectPeerQp(
          qpResourceAt(peerIndex, nic, q),
          remotePayload.nicInfo[nic].qpns[q],
          remotePayload.nicInfo[nic].gid,
          remotePayload.nicInfo[nic].lid,
          remotePayload.mtu);
    }
  }
  peerResources_[peerIndex].qpsConnected = true;
}

void MultipeerIbrcTransport::exchangeAndConnectQps() {
  const int numPeers = nRanks_ - 1;
  const int numQps = config_.numQpsPerPeerPerNic;

  for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
    createPeerQps(peerIndex);
  }

  IbTransportExchInfoAll myInfo{};
  myInfo.gidIndex = gidIndex_;
  myInfo.mtu = localMtu_;
  myInfo.numNics = numNics_;
  myInfo.numQpsPerPeerPerNic = numQps;

  auto& symbols = ibverbx::ibvSymbols;
  for (int n = 0; n < numNics_; ++n) {
    std::memcpy(
        myInfo.nicInfo[n].gid,
        nics_[n].localGid.raw,
        sizeof(myInfo.nicInfo[n].gid));
    ibverbx::ibv_port_attr portAttr{};
    if (symbols.ibv_internal_query_port(
            nics_[n].ibvCtx, kDefaultIbPort, &portAttr) == 0) {
      myInfo.nicInfo[n].lid = portAttr.lid;
    } else {
      LOG(WARNING) << "Failed to query port for IBRC LID on NIC " << n;
    }
  }

  const int totalQpsPerPeer = numNics_ * numQps;
  for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
    const int peerRank = peerIndexToRank(peerIndex);
    for (int nic = 0; nic < numNics_; ++nic) {
      for (int q = 0; q < numQps; ++q) {
        myInfo.nicInfo[nic].qpnForRank[peerRank][q] =
            qpResourceAt(peerIndex, nic, q).qp->qp_num;
      }
    }
  }

  VLOG(1) << "MultipeerIbrcTransport: rank " << myRank_
          << " performing allGather QP exchange (" << totalQpsPerPeer
          << " slots/peer = " << numNics_ << " NICs x " << numQps << " QPs)";

  std::vector<IbTransportExchInfoAll> allInfo = allGatherExchInfo(myInfo);
  validatePeerTopology(allInfo);

  for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
    const auto& peerInfo = allInfo[peerIndexToRank(peerIndex)];
    for (int nic = 0; nic < numNics_; ++nic) {
      for (int q = 0; q < numQps; ++q) {
        connectPeerQp(
            qpResourceAt(peerIndex, nic, q),
            peerInfo.nicInfo[nic].qpnForRank[myRank_][q],
            peerInfo.nicInfo[nic].gid,
            peerInfo.nicInfo[nic].lid,
            static_cast<int>(peerInfo.mtu));
      }
    }
    peerResources_[peerIndex].qpsConnected = true;
  }
}

void MultipeerIbrcTransport::doMaterializePeer(int peerRank) {
  const int peerIndex = rankToPeerIndex(peerRank);

  createPeerQps(peerIndex);

  auto localQp = buildLocalQpPayload(peerIndex);
  auto remoteQp = exchangeWithPeer(peerRank, localQp, kIbPeerQpExchangeTag);
  connectPeerQps(peerIndex, remoteQp);
  allocatePeerCmdQueues(peerIndex);

  throwIbrcUnimplemented("lazy command-ring/device transport path");
}

void MultipeerIbrcTransport::cleanupPeerOnFailure(int peerIndex) {
  cleanupPeerCmdQueues(peerIndex);
  cleanupPeerQps(peerIndex);
  if (peerIndex >= 0 &&
      peerIndex < static_cast<int>(peerMaterialized_.size())) {
    peerMaterialized_[peerIndex] = false;
  }
}

} // namespace comms::prims
