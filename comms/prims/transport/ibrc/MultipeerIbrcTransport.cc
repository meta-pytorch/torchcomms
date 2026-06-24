// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"

#include <endian.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
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
// Device-handle construction lives in MultipeerIbrcTransportCuda.cu: the full
// P2pIbrcTransportDevice pulls hip_bf16 (via the send/recv CopyOp) which a host
// .cc can't compile on AMD, so it's built in device context there (mirrors
// IBGDA's MultipeerIbgdaTransportCuda). This .cc only needs the host-callable
// builder declarations.
#include "comms/prims/transport/ibrc/MultipeerIbrcTransportCuda.cuh"

namespace comms::prims {

namespace {

constexpr uint8_t kDefaultIbPort = 1;
constexpr uint8_t kDefaultIbHopLimit = 255;

std::string errnoString(int err) {
  return std::strerror(err);
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

GpuError gpuMemset(void* ptr, int value, std::size_t bytes) {
  return hipMemset(ptr, value, bytes);
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

GpuError gpuMemset(void* ptr, int value, std::size_t bytes) {
  return cudaMemset(ptr, value, bytes);
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

uint32_t keyForIbvPostSend(uint32_t deviceOrderKey) {
#if defined(NIC_BNXT) || defined(NIC_IONIC)
  return deviceOrderKey;
#else
  return be32toh(deviceOrderKey);
#endif
}

constexpr uint16_t kSupportedIbrcFlags =
    IBRC_HAS_SIGNAL | IBRC_SIGNAL_ADD | IBRC_HAS_COUNTER;

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
  const int numQpsPerPeerPerNic = config.numQpsPerPeerPerNic();
  if (config.maxGroups < 1) {
    throw std::invalid_argument("maxGroups must be >= 1");
  }
  if (config.qpsPerBlockPerNic < 1) {
    throw std::invalid_argument("qpsPerBlockPerNic must be >= 1");
  }
  if (config.maxGroups > kMaxIbGroups) {
    throw std::invalid_argument(
        fmt::format(
            "maxGroups must be <= {}, got {}", kMaxIbGroups, config.maxGroups));
  }
  if (config.qpsPerBlockPerNic > kMaxIbQpsPerBlockPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "qpsPerBlockPerNic must be <= {}, got {}",
            kMaxIbQpsPerBlockPerNic,
            config.qpsPerBlockPerNic));
  }
  if (numQpsPerPeerPerNic > kMaxIbQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "maxGroups * qpsPerBlockPerNic must be <= {}, got {} * {} = {}",
            kMaxIbQpsPerPeerPerNic,
            config.maxGroups,
            config.qpsPerBlockPerNic,
            numQpsPerPeerPerNic));
  }
  if (!config.ibLazyConnect &&
      numQpsPerPeerPerNic > kMaxEagerExchangeQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "eager IBRC allGather exchange supports at most {} QPs per "
            "(peer,NIC); got {}. Enable ibLazyConnect for larger "
            "maxGroups * qpsPerBlockPerNic shapes.",
            kMaxEagerExchangeQpsPerPeerPerNic,
            numQpsPerPeerPerNic));
  }

  peerResources_.resize(nRanks_ - 1);
  peerQueuesPublished_ = std::make_unique<std::atomic<bool>[]>(nRanks_ - 1);

  try {
    // Pin GPU work to config_.cudaDevice.
    checkGpu(
        gpuSetDevice(config_.cudaDevice),
        "MultipeerIbrcTransport: set CUDA device");
    openNics();
    initializeControlResources();
    initializeDeviceTransportSlots();
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
    allocateSignalCounterResources(
        IbCounterStorage::HostPinned, /*allocateDiscardSignal=*/false);
    // Allocate + collectively exchange send/recv staging before building the
    // device transports, so updatePeerDeviceTransport can embed the resulting
    // IbSendRecvState. Delegated to the shared base; IBRC uses a host-mapped
    // NIC_DONE counter (CPU proxy writes it) via the HostPinned counter
    // storage.
    allocateSendRecvBuffersEager(IbCounterStorage::HostPinned);
    exchangeSendRecvBuffersEager();
    allocateCmdQueuesForAllPeers();
    VLOG(1) << "MultipeerIbrcTransport: rank " << myRank_ << " allocated "
            << allocatedCmdQueueCount() << " command queues";
    startProgressThread();
  } else {
    VLOG(1)
        << "MultipeerIbrcTransport: rank " << myRank_
        << " lazy exchange complete (per-peer QPs and command queues deferred "
           "to materializePeer)";
  }
}

void MultipeerIbrcTransport::cleanup() {
  stopProgressThread();

  for (int peerIndex = 0; peerIndex < static_cast<int>(peerResources_.size());
       ++peerIndex) {
    cleanupPeerCmdQueues(peerIndex);
    cleanupPeerQps(peerIndex);
  }
  cleanupSendRecvBuffers();
  cleanupSignalCounterResources();

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
  p2pTransportDevices_.reset();

  closeNics();
}

void MultipeerIbrcTransport::startProgressThread() {
  if (progressThread_.joinable()) {
    return;
  }
  stopProgress_.store(false, std::memory_order_release);
  progressThread_ = std::thread([this] { progressLoop(); });
}

void MultipeerIbrcTransport::stopProgressThread() noexcept {
  stopProgress_.store(true, std::memory_order_release);
  if (!progressThread_.joinable()) {
    return;
  }
  try {
    progressThread_.join();
  } catch (const std::exception& ex) {
    LOG(ERROR) << "MultipeerIbrcTransport: failed to join progress thread: "
               << ex.what();
  }
}

void MultipeerIbrcTransport::progressLoop() noexcept {
  while (!stopProgress_.load(std::memory_order_acquire)) {
    bool progressed = false;
    try {
      progressed = progressOnce();
    } catch (const std::exception& ex) {
      LOG(ERROR) << "MultipeerIbrcTransport: progress thread failed: "
                 << ex.what();
      publishTransportError(EIO, "progress error");
      return;
    } catch (...) {
      LOG(ERROR) << "MultipeerIbrcTransport: progress thread failed";
      publishTransportError(EIO, "progress error");
      return;
    }

    if (!progressed) {
      std::this_thread::yield();
    }
  }
}

bool MultipeerIbrcTransport::progressOnce() {
  bool progressed = false;
  for (int peerIndex = 0; peerIndex < static_cast<int>(peerResources_.size());
       ++peerIndex) {
    // Acquire pairs with the release in allocatePeerCmdQueues: skip peers whose
    // cmdQueues is not fully built (avoids racing a lazy move-assignment).
    if (!peerQueuesPublished_[peerIndex].load(std::memory_order_acquire)) {
      continue;
    }
    auto& peer = peerResources_[peerIndex];
    for (auto& cmdQueue : peer.cmdQueues) {
      progressed |= pollCmdQueueCompletions(peerIndex, cmdQueue);
      progressed |= pollOneCmdQueueDescriptor(peerIndex, cmdQueue);
    }
  }
  return progressed;
}

bool MultipeerIbrcTransport::pollOneCmdQueueDescriptor(
    int peerIndex,
    IbrcCmdQueueHost& cmdQueue) {
  const uint64_t seq = cmdQueue.nextToPoll;
  if (__atomic_load_n(cmdQueue.piHost, __ATOMIC_ACQUIRE) <= seq) {
    return false;
  }

  const uint32_t slot = static_cast<uint32_t>(seq & cmdQueue.device.mask);
  IbrcDesc& descSlot = cmdQueue.descsHost[slot];
  if (__atomic_load_n(&descSlot.ready_seq, __ATOMIC_ACQUIRE) != seq) {
    return false;
  }

  IbrcDesc desc = descSlot;
  auto& state = cmdQueue.cmdStates.at(slot);
  state.seq = seq;
  state.flags = desc.flags;
  state.counterAddr = desc.counter_addr;
  state.counterValue = desc.counter_value;
  state.peerCompleted = false;

  __atomic_store_n(&descSlot.ready_seq, kIbrcInvalidReadySeq, __ATOMIC_RELEASE);
  postDescriptor(peerIndex, cmdQueue, desc, seq);
  cmdQueue.nextToPoll = seq + 1;
  return true;
}

bool MultipeerIbrcTransport::pollCmdQueueCompletions(
    int peerIndex,
    IbrcCmdQueueHost& cmdQueue) {
  auto& qpResource = qpResourceAt(
      peerIndex,
      static_cast<int>(cmdQueue.nic),
      static_cast<int>(cmdQueue.qpSlot));
  bool progressed = false;

  ibverbx::ibv_wc completions[kIbrcCqPollBatch]{};
  const int n = qpResource.cq->context->ops.poll_cq(
      qpResource.cq, static_cast<int>(kIbrcCqPollBatch), completions);
  if (n < 0) {
    publishQueueError(
        peerIndex, cmdQueue, errno == 0 ? EIO : errno, "ibv_poll_cq failed");
    return false;
  }

  for (int i = 0; i < n; ++i) {
    const auto& wc = completions[i];
    if (wc.status != ibverbx::IBV_WC_SUCCESS) {
      publishQueueError(
          peerIndex, cmdQueue, static_cast<uint32_t>(wc.status), "CQE error");
      continue;
    }

    // Mark the descriptor's peer-facing WR complete; retirement (advancing
    // nextToComplete / publishing ci) happens strictly in seq order in
    // drainCompletedCommands(), so a no-WR descriptor can never retire ahead
    // of an in-flight peer WR at a lower seq.
    auto& state = cmdQueue.cmdStates.at(wc.wr_id & cmdQueue.device.mask);
    if (state.seq != wc.wr_id) {
      publishQueueError(peerIndex, cmdQueue, EPROTO, "stale CQE state");
      continue;
    }
    state.peerCompleted = true;
    progressed = true;
  }

  return drainCompletedCommands(peerIndex, cmdQueue) || progressed;
}

bool MultipeerIbrcTransport::drainCompletedCommands(
    int peerIndex,
    IbrcCmdQueueHost& cmdQueue) {
  bool progressed = false;
  while (true) {
    auto& state =
        cmdQueue.cmdStates.at(cmdQueue.nextToComplete & cmdQueue.device.mask);
    if (state.seq != cmdQueue.nextToComplete || !state.peerCompleted) {
      return progressed;
    }

    if ((state.flags & IBRC_HAS_COUNTER) != 0) {
      __atomic_fetch_add(
          reinterpret_cast<uint64_t*>(state.counterAddr),
          state.counterValue,
          __ATOMIC_RELEASE);
    }

    state = IbrcCmdState{};
    ++cmdQueue.nextToComplete;
    __atomic_store_n(
        cmdQueue.ciHost, cmdQueue.nextToComplete, __ATOMIC_RELEASE);
    progressed = true;
  }
}

void MultipeerIbrcTransport::postDescriptor(
    int peerIndex,
    IbrcCmdQueueHost& cmdQueue,
    const IbrcDesc& desc,
    uint64_t seq) {
  const auto op = static_cast<IbrcOp>(desc.op);
  const bool hasSignal = (desc.flags & IBRC_HAS_SIGNAL) != 0;
  const bool hasCounter = (desc.flags & IBRC_HAS_COUNTER) != 0;
  const bool hasData = op == IbrcOp::PUT && desc.bytes > 0;
  const uint16_t unsupportedFlags = desc.flags & ~kSupportedIbrcFlags;

  if (op != IbrcOp::PUT && op != IbrcOp::SIGNAL) {
    publishQueueError(peerIndex, cmdQueue, EINVAL, "unsupported descriptor op");
    return;
  }
  if (unsupportedFlags != 0) {
    publishQueueError(
        peerIndex, cmdQueue, ENOTSUP, "unsupported descriptor flags");
    return;
  }
  if (op == IbrcOp::SIGNAL && desc.bytes != 0) {
    publishQueueError(
        peerIndex, cmdQueue, EINVAL, "SIGNAL descriptor cannot carry data");
    return;
  }
  if (hasSignal && (desc.flags & IBRC_SIGNAL_ADD) == 0) {
    publishQueueError(
        peerIndex, cmdQueue, ENOTSUP, "only signal add is supported");
    return;
  }
  if (desc.bytes > std::numeric_limits<uint32_t>::max()) {
    publishQueueError(
        peerIndex,
        cmdQueue,
        EMSGSIZE,
        "descriptor bytes exceed verbs SGE size");
    return;
  }

  auto& qpResource = qpResourceAt(
      peerIndex,
      static_cast<int>(cmdQueue.nic),
      static_cast<int>(cmdQueue.qpSlot));
  auto& state = cmdQueue.cmdStates.at(seq & cmdQueue.device.mask);
  if (hasCounter) {
    if (desc.counter_addr == 0) {
      publishQueueError(peerIndex, cmdQueue, EINVAL, "counter address is null");
      return;
    }
    const auto counterAddr = static_cast<uintptr_t>(desc.counter_addr);
    if (counterAddr % alignof(uint64_t) != 0) {
      publishQueueError(
          peerIndex, cmdQueue, EINVAL, "counter address is unaligned");
      return;
    }
  }
  if (!hasData && !hasSignal && !hasCounter) {
    publishQueueError(peerIndex, cmdQueue, EINVAL, "empty descriptor");
    return;
  }
  if (!hasData && !hasSignal) {
    state.peerCompleted = true;
    drainCompletedCommands(peerIndex, cmdQueue);
    return;
  }
  if (hasSignal &&
      (qpResource.signalAtomicSinkMr == nullptr ||
       qpResource.signalAtomicSink == nullptr)) {
    publishQueueError(peerIndex, cmdQueue, EINVAL, "missing signal sink MR");
    return;
  }

  ibverbx::ibv_sge dataSge{};
  ibverbx::ibv_send_wr dataWr{};
  ibverbx::ibv_sge signalSge{};
  ibverbx::ibv_send_wr signalWr{};
  ibverbx::ibv_send_wr* firstWr = nullptr;
  ibverbx::ibv_send_wr* finalWr = nullptr;

  if (hasData) {
    dataSge.addr = desc.local_addr;
    dataSge.length = static_cast<uint32_t>(desc.bytes);
    dataSge.lkey = keyForIbvPostSend(desc.lkey_device_order);

    dataWr.wr_id = seq;
    dataWr.sg_list = &dataSge;
    dataWr.num_sge = 1;
    dataWr.opcode = ibverbx::IBV_WR_RDMA_WRITE;
    dataWr.send_flags = hasSignal ? 0 : ibverbx::IBV_SEND_SIGNALED;
    dataWr.wr.rdma.remote_addr = desc.remote_addr;
    dataWr.wr.rdma.rkey = keyForIbvPostSend(desc.rkey_device_order);
    firstWr = &dataWr;
    finalWr = &dataWr;
  }

  if (hasSignal) {
    signalSge.addr =
        reinterpret_cast<uint64_t>(qpResource.signalAtomicSink.get());
    signalSge.length = sizeof(uint64_t);
    signalSge.lkey = qpResource.signalAtomicSinkMr->lkey;

    signalWr.wr_id = seq;
    signalWr.sg_list = &signalSge;
    signalWr.num_sge = 1;
    signalWr.opcode = ibverbx::IBV_WR_ATOMIC_FETCH_AND_ADD;
    signalWr.send_flags = ibverbx::IBV_SEND_SIGNALED | ibverbx::IBV_SEND_FENCE;
    signalWr.wr.atomic.remote_addr = desc.signal_addr;
    signalWr.wr.atomic.compare_add = desc.signal_value;
    signalWr.wr.atomic.rkey = keyForIbvPostSend(desc.signal_rkey_device_order);

    if (firstWr == nullptr) {
      firstWr = &signalWr;
    } else {
      finalWr->next = &signalWr;
    }
  }

  ibverbx::ibv_send_wr* badWr = nullptr;
  const int rc =
      qpResource.qp->context->ops.post_send(qpResource.qp, firstWr, &badWr);
  if (rc != 0) {
    publishQueueError(
        peerIndex,
        cmdQueue,
        rc > 0 ? rc : (errno == 0 ? EIO : errno),
        "ibv_post_send failed");
  }
}

void MultipeerIbrcTransport::publishQueueError(
    int peerIndex,
    const IbrcCmdQueueHost& cmdQueue,
    uint32_t errorCode,
    const char* reason) noexcept {
  const int peerRank = peerIndex < myRank_ ? peerIndex : peerIndex + 1;
  const auto queueIndex = static_cast<uint32_t>(
      ((static_cast<uint64_t>(peerIndex) *
            static_cast<uint64_t>(config_.numQpsPerPeerPerNic()) +
        static_cast<uint64_t>(cmdQueue.qpSlot)) *
       static_cast<uint64_t>(numNics_)) +
      static_cast<uint64_t>(cmdQueue.nic));

  LOG(ERROR) << "MultipeerIbrcTransport: " << reason << " peerRank=" << peerRank
             << " queue=" << queueIndex << " nic=" << cmdQueue.nic
             << " qpSlot=" << cmdQueue.qpSlot << " code=" << errorCode;
  for (auto* status : statusHostByNic_) {
    if (status == nullptr) {
      continue;
    }
    __atomic_store_n(&status->error_queue, queueIndex, __ATOMIC_RELAXED);
    __atomic_store_n(&status->error_code, errorCode, __ATOMIC_RELAXED);
    __atomic_store_n(&status->error, 1, __ATOMIC_RELEASE);
  }
  stopProgress_.store(true, std::memory_order_release);
}

void MultipeerIbrcTransport::publishTransportError(
    uint32_t errorCode,
    const char* reason) noexcept {
  LOG(ERROR) << "MultipeerIbrcTransport: " << reason << " code=" << errorCode;
  for (auto* status : statusHostByNic_) {
    if (status == nullptr) {
      continue;
    }
    __atomic_store_n(&status->error_queue, kIbrcUnknownQueue, __ATOMIC_RELAXED);
    __atomic_store_n(&status->error_code, errorCode, __ATOMIC_RELAXED);
    __atomic_store_n(&status->error, 1, __ATOMIC_RELEASE);
  }
  stopProgress_.store(true, std::memory_order_release);
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
  // Unpublish before clearing so progressOnce can't walk a half-cleared vector.
  peerQueuesPublished_[peerIndex].store(false, std::memory_order_release);
  auto& peer = peerResources_[peerIndex];
  peer.cmdQueues.clear();
  peer.cmdQueueDevices.reset();
  peer.blockQpState.reset();
  peer.cmdQueuesAllocated = false;
  if (p2pTransportDevices_.host != nullptr) {
    updatePeerDeviceTransport(peerIndex);
  }
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

  const int numQps = config_.numQpsPerPeerPerNic();
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
  std::vector<IbrcCmdQueueDevice> deviceCmdQueues;
  deviceCmdQueues.reserve(cmdQueuesPerPeer);

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
      deviceCmdQueues.push_back(cmdQueue.device);
      cmdQueues.push_back(std::move(cmdQueue));
    }
  }

  peer.cmdQueueDevices = allocateMapped(
      deviceCmdQueues.size() * sizeof(IbrcCmdQueueDevice),
      "per-peer command queue device descriptors");
  std::memcpy(
      peer.cmdQueueDevices.host,
      deviceCmdQueues.data(),
      deviceCmdQueues.size() * sizeof(IbrcCmdQueueDevice));
  peer.blockQpState = allocateMapped(
      static_cast<std::size_t>(config_.maxGroups) * sizeof(IbrcBlockQpState),
      "per-peer block QP state");
  peer.cmdQueues = std::move(cmdQueues);
  peer.cmdQueuesAllocated = true;
  updatePeerDeviceTransport(peerIndex);
  // Publish last (release): progressOnce may now iterate this peer's cmdQueues.
  peerQueuesPublished_[peerIndex].store(true, std::memory_order_release);
}

void MultipeerIbrcTransport::initializeDeviceTransportSlots() {
  const std::size_t numPeers = static_cast<std::size_t>(nRanks_ - 1);
  p2pTransportDevices_ = allocateMapped(
      numPeers * ibrcDeviceSlotSize(), "P2pIbrcTransportDevice slots");
  constructIbrcDeviceSlots(
      p2pTransportDevices_.host, static_cast<int>(numPeers));
}

void MultipeerIbrcTransport::updatePeerDeviceTransport(int peerIndex) noexcept {
  if (p2pTransportDevices_.host == nullptr || peerIndex < 0 ||
      peerIndex >= static_cast<int>(peerResources_.size())) {
    return;
  }

  auto& peer = peerResources_[peerIndex];
  if (!peer.cmdQueuesAllocated || peer.cmdQueueDevices.device == nullptr) {
    constructIbrcDeviceSlots(
        static_cast<char*>(p2pTransportDevices_.host) +
            peerIndex * ibrcDeviceSlotSize(),
        1);
    return;
  }

  IbgdaRemoteBuffer remoteSignalBuf{};
  IbgdaLocalBuffer localSignalBuf{};
  if (config_.numSignalSlots > 0) {
    remoteSignalBuf = slotRemoteSignalView(peerIndex);
    localSignalBuf = slotLocalSignalView(peerIndex);
  }
  IbgdaLocalBuffer counterDeviceBuf{};
  IbgdaLocalBuffer counterHostBuf{};
  if (config_.numCounterSlots > 0) {
    counterDeviceBuf = slotCounterDeviceView(peerIndex);
    counterHostBuf = slotCounterHostView(peerIndex);
  }

  writeIbrcDeviceSlot(
      p2pTransportDevices_.host,
      peerIndex,
      DeviceSpan<IbrcCmdQueueDevice>(
          static_cast<IbrcCmdQueueDevice*>(peer.cmdQueueDevices.device),
          static_cast<uint32_t>(peer.cmdQueues.size())),
      static_cast<uint32_t>(numNics_),
      static_cast<uint32_t>(config_.maxGroups),
      static_cast<uint32_t>(config_.qpsPerBlockPerNic),
      DeviceSpan<IbrcBlockQpState>(
          static_cast<IbrcBlockQpState*>(peer.blockQpState.device),
          static_cast<uint32_t>(config_.maxGroups)),
      remoteSignalBuf,
      localSignalBuf,
      counterDeviceBuf,
      counterHostBuf,
      config_.numSignalSlots,
      config_.numCounterSlots,
      sendRecvStateForPeer(peerIndex));
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

// Send/recv staging allocation, exchange, cleanup, and IbSendRecvState
// construction are provided by MultiPeerIbTransportBase. IBRC delegates via
// allocateSendRecvBuffersEager(IbCounterStorage::HostPinned) /
// exchangeSendRecvBuffersEager() / cleanupSendRecvBuffers() /
// sendRecvStateForPeer().

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
    if (qpResource.signalAtomicSinkMr != nullptr &&
        symbols.ibv_internal_dereg_mr != nullptr) {
      int rc = symbols.ibv_internal_dereg_mr(qpResource.signalAtomicSinkMr);
      if (rc != 0) {
        LOG(WARNING) << "Failed to deregister IBRC signal sink MR nic="
                     << qpResource.nic << " qpSlot=" << qpResource.qpSlot
                     << ": rc=" << rc;
      }
      qpResource.signalAtomicSinkMr = nullptr;
    }
    qpResource.signalAtomicSink.reset();
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

  const int numQps = config_.numQpsPerPeerPerNic();
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
        qpResources.push_back(std::move(qpResource));
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

        auto signalAtomicSink = std::make_unique<uint64_t>(0);
        if (symbols.ibv_internal_reg_mr == nullptr) {
          throw std::runtime_error("ibv_reg_mr is unavailable");
        }
        errno = 0;
        createdQpResource.signalAtomicSinkMr = symbols.ibv_internal_reg_mr(
            nics_[nic].ibvPd,
            signalAtomicSink.get(),
            sizeof(uint64_t),
            ibverbx::IBV_ACCESS_LOCAL_WRITE);
        if (createdQpResource.signalAtomicSinkMr == nullptr) {
          const int savedErrno = errno;
          throw std::runtime_error(
              fmt::format(
                  "Failed to register IBRC signal sink MR for peerIndex={} "
                  "nic={} qpSlot={}: errno={} ({})",
                  peerIndex,
                  nic,
                  q,
                  savedErrno,
                  errnoString(savedErrno)));
        }
        createdQpResource.signalAtomicSink = std::move(signalAtomicSink);
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
      qpSlot >= config_.numQpsPerPeerPerNic()) {
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
  const int numQps = config_.numQpsPerPeerPerNic();
  PeerQpPayload payload{};
  payload.gidIndex = gidIndex_;
  payload.mtu = static_cast<int>(localMtu_);
  payload.numNics = numNics_;
  payload.numQpsPerPeerPerNic = numQps;
  payload.maxGroups = config_.maxGroups;
  payload.qpsPerBlockPerNic = config_.qpsPerBlockPerNic;

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
  if (remotePayload.numQpsPerPeerPerNic != config_.numQpsPerPeerPerNic()) {
    throw std::runtime_error(
        fmt::format(
            "IBRC peerIndex={} numQps={} vs local {}",
            peerIndex,
            remotePayload.numQpsPerPeerPerNic,
            config_.numQpsPerPeerPerNic()));
  }
  if (remotePayload.maxGroups != config_.maxGroups ||
      remotePayload.qpsPerBlockPerNic != config_.qpsPerBlockPerNic) {
    throw std::runtime_error(
        fmt::format(
            "IBRC peerIndex={} block-owned QP shape maxGroups={} "
            "qpsPerBlockPerNic={} vs local {} {}",
            peerIndex,
            remotePayload.maxGroups,
            remotePayload.qpsPerBlockPerNic,
            config_.maxGroups,
            config_.qpsPerBlockPerNic));
  }

  const int numQps = config_.numQpsPerPeerPerNic();
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
  const int numQps = config_.numQpsPerPeerPerNic();

  for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
    createPeerQps(peerIndex);
  }

  IbTransportExchInfoAll myInfo{};
  myInfo.gidIndex = gidIndex_;
  myInfo.mtu = localMtu_;
  myInfo.numNics = numNics_;
  myInfo.numQpsPerPeerPerNic = numQps;
  myInfo.maxGroups = config_.maxGroups;
  myInfo.qpsPerBlockPerNic = config_.qpsPerBlockPerNic;

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

P2pIbrcTransportDevice* MultipeerIbrcTransport::getP2pTransportDeviceSlot(
    int peerRank) const {
  if (config_.ibLazyConnect) {
    LOG_FIRST_N(WARNING, 1)
        << "MultipeerIbrcTransport: lazy mode is enabled but Transport[] "
        << "array is being built with possibly unmaterialized IBRC slots. "
        << "Call get_device_handle(peers) before kernels access lazy peers.";
  }
  if (p2pTransportDevices_.device == nullptr) {
    throw std::runtime_error(
        "getP2pTransportDeviceSlot: IBRC device transport slots are not initialized");
  }
  const int peerIndex = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbrcTransportDevice*>(
      static_cast<char*>(p2pTransportDevices_.device) +
      peerIndex * ibrcDeviceSlotSize());
}

P2pIbrcTransportDevice* MultipeerIbrcTransport::getP2pTransportDevice(
    int peerRank) const {
  // IBRC builds every peer slot eagerly in initializeDeviceTransportSlots(), so
  // the per-peer accessor is just slot pointer arithmetic (no materialization,
  // hence no lazy warning unlike getP2pTransportDeviceSlot()).
  if (p2pTransportDevices_.device == nullptr) {
    throw std::runtime_error(
        "getP2pTransportDevice: IBRC device transport slots are not initialized");
  }
  const int peerIndex = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbrcTransportDevice*>(
      static_cast<char*>(p2pTransportDevices_.device) +
      peerIndex * ibrcDeviceSlotSize());
}

void MultipeerIbrcTransport::doMaterializePeer(int peerRank) {
  const int peerIndex = rankToPeerIndex(peerRank);

  createPeerQps(peerIndex);

  auto localQp = buildLocalQpPayload(peerIndex);
  auto remoteQp = exchangeWithPeer(peerRank, localQp, kIbPeerQpExchangeTag);
  connectPeerQps(peerIndex, remoteQp);
  PeerBufferPayload localBuf{};
  allocatePeerSignalCounterResources(
      peerIndex,
      localBuf,
      IbCounterStorage::HostPinned,
      /*allocateDiscardSignal=*/false);
  // Allocate this peer's send/recv rings on demand (HostPinned NIC_DONE
  // counter, CPU-proxy written) and publish them on the same bilateral round,
  // so sendRecvStateForPeer(peerIndex) is populated when allocatePeerCmdQueues
  // -> updatePeerDeviceTransport bakes it into the device slot.
  allocateSendRecvBufferForPeer(
      peerIndex, localBuf, IbCounterStorage::HostPinned);
  auto remoteBuf =
      exchangeWithPeer(peerRank, localBuf, kIbPeerBufferExchangeTag);
  applyRemoteSignalCounterResources(
      peerIndex, remoteBuf, /*hasDiscardSignal=*/false);
  applyRemoteSendRecvBuffer(peerIndex, remoteBuf);
  allocatePeerCmdQueues(peerIndex);
  startProgressThread();
  peerMaterialized_[peerIndex] = true;
}

void MultipeerIbrcTransport::cleanupPeerOnFailure(int peerIndex) {
  // Quiesce progress before teardown; next doMaterializePeer restarts it.
  stopProgressThread();
  cleanupPeerCmdQueues(peerIndex);
  cleanupPeerQps(peerIndex);
  cleanupPeerSignalCounterResources(peerIndex);
  cleanupSendRecvBufferForPeer(peerIndex);
  if (peerIndex >= 0 &&
      peerIndex < static_cast<int>(peerMaterialized_.size())) {
    peerMaterialized_[peerIndex] = false;
  }
}

} // namespace comms::prims
