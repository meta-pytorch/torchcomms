// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"

#include <endian.h>

#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
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

struct QpSlotRange {
  int begin;
  int end;
};

QpSlotRange qpSlotRange(
    const MultipeerIbTransportConfig& config,
    uint32_t beginChannel,
    uint32_t endChannel) {
  const auto slot = [&](uint32_t channel) {
    return static_cast<int>(ibQpSlotWithinNic(
        channel,
        IbDirection::Send,
        static_cast<uint32_t>(config.fixedChannelDirectionCount()),
        static_cast<uint32_t>(config.qpsPerConnection),
        0));
  };
  return {.begin = slot(beginChannel), .end = slot(endChannel)};
}

#ifdef __HIP_PLATFORM_AMD__
class HipStreamCaptureModeGuard {
 public:
  HipStreamCaptureModeGuard() {
    const hipError_t error = hipThreadExchangeStreamCaptureMode(&previous_);
    if (error != hipSuccess) {
      throw std::runtime_error(
          "hipThreadExchangeStreamCaptureMode failed: " +
          std::string(hipGetErrorString(error)));
    }
  }

  ~HipStreamCaptureModeGuard() {
    const hipError_t error = hipThreadExchangeStreamCaptureMode(&previous_);
    if (error != hipSuccess) {
      LOG(ERROR) << "Failed to restore HIP stream capture mode: "
                 << hipGetErrorString(error);
    }
  }

 private:
  hipStreamCaptureMode previous_{hipStreamCaptureModeRelaxed};
};
#endif

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

void appendCpuRange(std::vector<int>& cpus, int first, int last) {
  if (first < 0 || last < first || first >= CPU_SETSIZE) {
    return;
  }
  const int end = last < CPU_SETSIZE ? last : CPU_SETSIZE - 1;
  for (int cpu = first; cpu <= end; ++cpu) {
    cpus.push_back(cpu);
  }
}

std::vector<int> cpusFromLocalCpuList(const std::string& cpulist) {
  std::vector<int> cpus;
  std::stringstream stream(cpulist);
  std::string token;
  while (std::getline(stream, token, ',')) {
    int first = -1;
    int last = -1;
    if (std::sscanf(token.c_str(), " %d - %d", &first, &last) == 2) {
      appendCpuRange(cpus, first, last);
    } else if (std::sscanf(token.c_str(), " %d", &first) == 1) {
      appendCpuRange(cpus, first, first);
    }
  }
  return cpus;
}

std::string cpuListToString(const std::vector<int>& cpus) {
  std::string result;
  for (std::size_t i = 0; i < cpus.size();) {
    const int first = cpus[i];
    int last = first;
    ++i;
    while (i < cpus.size() && cpus[i] == last + 1) {
      last = cpus[i];
      ++i;
    }

    if (!result.empty()) {
      result += ",";
    }
    result += std::to_string(first);
    if (last != first) {
      result += "-";
      result += std::to_string(last);
    }
  }
  return result;
}

void pinCurrentThreadToCpus(const std::vector<int>& cpus) noexcept {
  if (cpus.empty()) {
    return;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (const int cpu : cpus) {
    CPU_SET(cpu, &cpuset);
  }
  const int rc =
      pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  if (rc != 0) {
    LOG(WARNING) << "Failed to pin IBRC progress thread to CPUs "
                 << cpuListToString(cpus) << ": " << errnoString(rc);
  } else {
    VLOG(1) << "Pinned IBRC progress thread to CPUs " << cpuListToString(cpus);
  }
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
  const int directionCount = config_.fixedChannelDirectionCount();
  const int numQpsPerPeerPerNic = config_.fixedChannelMainQpsPerPeerPerNic();
  if (config_.max_num_channels < 1) {
    throw std::invalid_argument("max_num_channels must be >= 1");
  }
  if (config_.qpsPerConnection < 1) {
    throw std::invalid_argument("qpsPerConnection must be >= 1");
  }
  if (config_.max_num_channels > kMaxIbGroups) {
    throw std::invalid_argument(
        fmt::format(
            "max_num_channels must be <= {}, got {}",
            kMaxIbGroups,
            config_.max_num_channels));
  }
  if (config_.qpsPerConnection > kMaxIbQpsPerBlockPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "qpsPerConnection must be <= {}, got {}",
            kMaxIbQpsPerBlockPerNic,
            config_.qpsPerConnection));
  }
  if (numQpsPerPeerPerNic > kMaxIbQpsPerPeerPerNic) {
    throw std::invalid_argument(
        fmt::format(
            "max_num_channels * directionCount * qpsPerConnection must be <= {}, got {} * {} * {} = {}",
            kMaxIbQpsPerPeerPerNic,
            config_.max_num_channels,
            directionCount,
            config_.qpsPerConnection,
            numQpsPerPeerPerNic));
  }
  if (sendRecvBuffersEnabled()) {
    validateSendRecvConfig();
  }
  peerResources_.resize(nRanks_ - 1);
  publishedCmdQueueCounts_ =
      std::make_unique<std::atomic<uint32_t>[]>(nRanks_ - 1);
  for (int peerIndex = 0; peerIndex < nRanks_ - 1; ++peerIndex) {
    publishedCmdQueueCounts_[peerIndex].store(0, std::memory_order_relaxed);
  }

  try {
    // Pin GPU work to config_.cudaDevice.
    checkGpu(
        gpuSetDevice(config_.cudaDevice),
        "MultipeerIbrcTransport: set CUDA device");
    openNics();
    if (numNics_ * config_.qpsPerConnection >
        kIbMaxQpLanesPerChannelDirection) {
      throw std::invalid_argument(
          fmt::format(
              "numNics * qpsPerConnection must be <= {}, got {} * {}",
              kIbMaxQpLanesPerChannelDirection,
              numNics_,
              config_.qpsPerConnection));
    }
    progressCpus_ = selectProgressCpus();
    initializeControlResources();
    initializeDeviceTransportSlots();
  } catch (const std::exception&) {
    cleanup();
    throw;
  }
}

MultipeerIbrcTransport::~MultipeerIbrcTransport() {
  cleanup();
}

void MultipeerIbrcTransport::exchange() {
  VLOG(1) << "MultipeerIbrcTransport: rank " << myRank_
          << " exchange complete (per-peer QPs and command queues deferred "
             "to materializePeerChannelRange)";
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
  cmdQueueDevices_.reset();
  channelStates_.reset();

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

std::vector<int> MultipeerIbrcTransport::selectProgressCpus() const {
  std::vector<int> selectedCpus;
  std::string selectedNic;
  std::string missingNic;
  std::string missingReason;
  for (const auto& nic : nics_) {
    if (nic.deviceName.empty()) {
      continue;
    }
    const std::string path =
        "/sys/class/infiniband/" + nic.deviceName + "/device/local_cpulist";
    std::ifstream file(path);
    std::string cpulist;
    if (!file || !std::getline(file, cpulist)) {
      VLOG(1) << "Could not read IBRC progress CPU locality from " << path;
      if (missingNic.empty()) {
        missingNic = nic.deviceName;
        missingReason = "could not be read";
      }
      continue;
    }

    std::vector<int> cpus = cpusFromLocalCpuList(cpulist);
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    if (cpus.empty()) {
      if (missingNic.empty()) {
        missingNic = nic.deviceName;
        missingReason = fmt::format(
            "local_cpulist={} did not resolve to any usable CPUs", cpulist);
      }
      continue;
    }

    VLOG(1) << "Selected IBRC progress CPUs " << cpuListToString(cpus)
            << " from " << nic.deviceName << " local_cpulist=" << cpulist;
    if (selectedCpus.empty()) {
      selectedCpus = cpus;
      selectedNic = nic.deviceName;
    } else if (cpus != selectedCpus) {
      throw std::runtime_error(
          fmt::format(
              "IBRC selected NICs have different CPU locality: {} resolved to "
              "CPUs {}, but {} resolved to CPUs {}. A single progress thread "
              "requires a single NIC-local CPU affinity mask.",
              selectedNic,
              cpuListToString(selectedCpus),
              nic.deviceName,
              cpuListToString(cpus)));
    }
  }
  if (!selectedCpus.empty() && !missingNic.empty()) {
    throw std::runtime_error(
        fmt::format(
            "IBRC selected NICs do not all expose CPU locality: {} resolved to "
            "CPUs {}, but {} {}. A single progress thread requires a single "
            "NIC-local CPU affinity mask.",
            selectedNic,
            cpuListToString(selectedCpus),
            missingNic,
            missingReason));
  }
  return selectedCpus;
}

void MultipeerIbrcTransport::progressLoop() noexcept {
  pinCurrentThreadToCpus(progressCpus_);
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
    const uint32_t queueCount =
        publishedCmdQueueCounts_[peerIndex].load(std::memory_order_acquire);
    if (queueCount == 0) {
      continue;
    }
    auto& peer = peerResources_[peerIndex];
    CHECK_LE(queueCount, peer.cmdQueues.size());
    for (uint32_t queue = 0; queue < queueCount; ++queue) {
      auto& cmdQueue = peer.cmdQueues[queue];
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
            static_cast<uint64_t>(config_.fixedChannelMainQpsPerPeerPerNic()) +
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
  poisonTransport();
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
  poisonTransport();
}

void MultipeerIbrcTransport::onTerminalMaterializationFailure() noexcept {
  publishTransportError(
      ECANCELED, "terminal peer/channel materialization failure");
  stopProgressThread();
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
  publishedCmdQueueCounts_[peerIndex].store(0, std::memory_order_release);
  auto& peer = peerResources_[peerIndex];
  clearPeerDeviceRange(peerIndex, 0, config_.max_num_channels);
  peer.cmdQueues.clear();
  peer.completionRanges.clear();
  peer.channelLayout = IbChannelLayout{};
  peer.deviceSlotPublished = false;
  if (p2pTransportDevices_.host != nullptr) {
    updatePeerDeviceTransport(peerIndex);
  }
}

void MultipeerIbrcTransport::allocatePeerCmdQueueRange(
    int peerIndex,
    uint32_t beginChannel,
    uint32_t endChannel,
    const IbChannelLayout& rangeLayout) {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size())) {
    throw std::invalid_argument(
        fmt::format(
            "allocatePeerCmdQueueRange: invalid peerIndex={}", peerIndex));
  }
  if (beginChannel >= endChannel || endChannel > channelCapacity()) {
    throw std::invalid_argument(
        fmt::format(
            "allocatePeerCmdQueueRange: invalid range=[{}, {}) capacity={}",
            beginChannel,
            endChannel,
            channelCapacity()));
  }

  auto& peer = peerResources_[peerIndex];
  const auto slots = qpSlotRange(config_, beginChannel, endChannel);
  const std::size_t queueBegin =
      static_cast<std::size_t>(slots.begin) * numNics_;
  const std::size_t queueEnd = static_cast<std::size_t>(slots.end) * numNics_;
  const std::size_t queueCount = queueEnd - queueBegin;
  CHECK_EQ(peer.cmdQueues.size(), cmdQueuesPerPeer_);
  CHECK_EQ(
      publishedCmdQueueCounts_[peerIndex].load(std::memory_order_acquire),
      queueBegin);

  std::vector<IbrcCmdQueueHost> cmdQueues;
  cmdQueues.reserve(queueCount);
  std::vector<IbrcCmdQueueDevice> deviceCmdQueues;
  deviceCmdQueues.reserve(queueCount);

  for (int q = slots.begin; q < slots.end; ++q) {
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

  CHECK_EQ(deviceCmdQueues.size(), queueCount);
  CHECK_GE(rangeLayout.pipelineDepth, 0);
  const std::size_t completionSlotCount = checkedMul(
      static_cast<std::size_t>(endChannel - beginChannel),
      static_cast<std::size_t>(rangeLayout.pipelineDepth),
      "channel-range send completion slots");
  const std::size_t completionSlotBytes = checkedMul(
      completionSlotCount,
      sizeof(IbSendCompletionSlot),
      "channel-range send completion slots");
  MappedAllocation completionRange;
  if (completionSlotBytes != 0) {
    completionRange = allocateMapped(
        completionSlotBytes, "channel-range send completion slots");
  }
  auto* const completionSlots = completionSlotBytes == 0
      ? nullptr
      : static_cast<IbSendCompletionSlot*>(completionRange.device);
  std::vector<IbChannel> channels;
  channels.reserve(endChannel - beginChannel);
  const std::size_t pipelineDepth =
      static_cast<std::size_t>(rangeLayout.pipelineDepth);
  for (uint32_t channel = beginChannel; channel < endChannel; ++channel) {
    IbSendCompletionSlot* channelCompletionSlots = pipelineDepth == 0
        ? nullptr
        : completionSlots +
            static_cast<std::size_t>(channel - beginChannel) * pipelineDepth;
    channels.push_back(makeIbChannel(
        rangeLayout,
        static_cast<int>(channel - beginChannel),
        channelCompletionSlots));
  }

  for (std::size_t queue = 0; queue < queueCount; ++queue) {
    CHECK(peer.cmdQueues[queueBegin + queue].control.host == nullptr);
    peer.cmdQueues[queueBegin + queue] = std::move(cmdQueues[queue]);
  }
  if (completionSlotBytes != 0) {
    peer.completionRanges.push_back(std::move(completionRange));
  }
  populatePeerDeviceRange(
      peerIndex, beginChannel, endChannel, deviceCmdQueues, channels);
  if (beginChannel == 0) {
    peer.channelLayout = config_.lazyChannels ? sendRecvChannelGeometry()
                                              : channelLayoutForPeer(peerIndex);
    peer.deviceSlotPublished = true;
    updatePeerDeviceTransport(peerIndex);
  }
  CHECK_LE(queueEnd, std::numeric_limits<uint32_t>::max());
  publishedCmdQueueCounts_[peerIndex].store(
      static_cast<uint32_t>(queueEnd), std::memory_order_release);
}

void MultipeerIbrcTransport::initializeDeviceTransportSlots() {
  const std::size_t numPeers = static_cast<std::size_t>(nRanks_ - 1);
  const std::size_t qpsPerPeerPerNic =
      static_cast<std::size_t>(config_.fixedChannelMainQpsPerPeerPerNic());
  cmdQueuesPerPeer_ = checkedMul(
      static_cast<std::size_t>(numNics_),
      qpsPerPeerPerNic,
      "fixed command queue descriptors per peer");
  cmdQueueDevices_ = allocateMapped(
      checkedMul(
          checkedMul(
              numPeers, cmdQueuesPerPeer_, "fixed command queue descriptors"),
          sizeof(IbrcCmdQueueDevice),
          "fixed command queue descriptors"),
      "fixed command queue device descriptors");
  channelStates_ = allocateMapped(
      checkedMul(
          checkedMul(
              numPeers,
              static_cast<std::size_t>(config_.max_num_channels),
              "fixed channel states"),
          sizeof(IbChannel),
          "fixed channel states"),
      "fixed channel states");
  p2pTransportDevices_ = allocateMapped(
      numPeers * ibrcDeviceSlotSize(), "P2pIbrcTransportDevice slots");
  for (int peerIndex = 0; peerIndex < static_cast<int>(numPeers); ++peerIndex) {
    auto& peer = peerResources_[peerIndex];
    peer.qpResources.resize(cmdQueuesPerPeer_);
    peer.cmdQueues.resize(cmdQueuesPerPeer_);
    peer.completionRanges.reserve(config_.max_num_channels);
    updatePeerDeviceTransport(peerIndex);
  }
}

void MultipeerIbrcTransport::populatePeerDeviceRange(
    int peerIndex,
    int channelBegin,
    int channelEnd,
    const std::vector<IbrcCmdQueueDevice>& rangeCmdQueues,
    const std::vector<IbChannel>& rangeChannels) {
  CHECK_GE(peerIndex, 0);
  CHECK_LT(peerIndex, static_cast<int>(peerResources_.size()));
  CHECK_GE(channelBegin, 0);
  CHECK_LE(channelBegin, channelEnd);
  CHECK_LE(channelEnd, config_.max_num_channels);
  CHECK_EQ(
      rangeChannels.size(),
      static_cast<std::size_t>(channelEnd - channelBegin));
  CHECK(cmdQueueDevices_.host != nullptr);
  CHECK(channelStates_.host != nullptr);

  const std::size_t queuesPerChannel = checkedMul(
      checkedMul(
          static_cast<std::size_t>(config_.fixedChannelDirectionCount()),
          static_cast<std::size_t>(config_.qpsPerConnection),
          "command queues per channel"),
      static_cast<std::size_t>(numNics_),
      "command queues per channel");
  const std::size_t queueBegin = ibCommandQueueSlot(
      static_cast<uint32_t>(channelBegin),
      IbDirection::Send,
      static_cast<uint32_t>(config_.fixedChannelDirectionCount()),
      static_cast<uint32_t>(config_.qpsPerConnection),
      0,
      static_cast<uint32_t>(numNics_),
      0);
  const std::size_t queueCount = checkedMul(
      static_cast<std::size_t>(channelEnd - channelBegin),
      queuesPerChannel,
      "command queue range");
  CHECK_EQ(rangeCmdQueues.size(), queueCount);
  auto* const peerCmdQueueDevices =
      static_cast<IbrcCmdQueueDevice*>(cmdQueueDevices_.host) +
      static_cast<std::size_t>(peerIndex) * cmdQueuesPerPeer_;
  if (queueCount != 0) {
    std::memcpy(
        peerCmdQueueDevices + queueBegin,
        rangeCmdQueues.data(),
        queueCount * sizeof(IbrcCmdQueueDevice));
  }

  auto* const peerChannels = static_cast<IbChannel*>(channelStates_.host) +
      static_cast<std::size_t>(peerIndex) * config_.max_num_channels;
  if (channelBegin != channelEnd) {
    std::memcpy(
        peerChannels + channelBegin,
        rangeChannels.data(),
        rangeChannels.size() * sizeof(IbChannel));
  }
}

void MultipeerIbrcTransport::clearPeerDeviceRange(
    int peerIndex,
    int channelBegin,
    int channelEnd) noexcept {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size()) ||
      channelBegin < 0 || channelBegin > channelEnd ||
      channelEnd > config_.max_num_channels) {
    return;
  }
  const std::size_t queuesPerChannel = static_cast<std::size_t>(numNics_) *
      config_.fixedChannelDirectionCount() * config_.qpsPerConnection;
  const std::size_t queueBegin = ibCommandQueueSlot(
      static_cast<uint32_t>(channelBegin),
      IbDirection::Send,
      static_cast<uint32_t>(config_.fixedChannelDirectionCount()),
      static_cast<uint32_t>(config_.qpsPerConnection),
      0,
      static_cast<uint32_t>(numNics_),
      0);
  const std::size_t queueCount =
      static_cast<std::size_t>(channelEnd - channelBegin) * queuesPerChannel;
  if (cmdQueueDevices_.host != nullptr && queueCount != 0) {
    auto* const peerCmdQueueDevices =
        static_cast<IbrcCmdQueueDevice*>(cmdQueueDevices_.host) +
        static_cast<std::size_t>(peerIndex) * cmdQueuesPerPeer_;
    std::memset(
        peerCmdQueueDevices + queueBegin,
        0,
        queueCount * sizeof(IbrcCmdQueueDevice));
  }
  if (channelStates_.host != nullptr && channelBegin != channelEnd) {
    auto* const peerChannels = static_cast<IbChannel*>(channelStates_.host) +
        static_cast<std::size_t>(peerIndex) * config_.max_num_channels;
    std::memset(
        peerChannels + channelBegin,
        0,
        static_cast<std::size_t>(channelEnd - channelBegin) *
            sizeof(IbChannel));
  }
}

void MultipeerIbrcTransport::updatePeerDeviceTransport(int peerIndex) noexcept {
  if (p2pTransportDevices_.host == nullptr || peerIndex < 0 ||
      peerIndex >= static_cast<int>(peerResources_.size())) {
    return;
  }

  auto& peer = peerResources_[peerIndex];
  IbgdaRemoteBuffer remoteSignalBuf{};
  IbgdaLocalBuffer localSignalBuf{};
  if (peer.deviceSlotPublished && config_.numSignalSlots > 0) {
    remoteSignalBuf = slotRemoteSignalView(peerIndex);
    localSignalBuf = slotLocalSignalView(peerIndex);
  }
  IbgdaLocalBuffer counterDeviceBuf{};
  IbgdaLocalBuffer counterHostBuf{};
  if (peer.deviceSlotPublished && config_.numCounterSlots > 0) {
    counterDeviceBuf = slotCounterDeviceView(peerIndex);
    counterHostBuf = slotCounterHostView(peerIndex);
  }
  const IbChannelLayout channelLayout =
      peer.deviceSlotPublished ? peer.channelLayout : IbChannelLayout{};
  const int numSignalSlots =
      peer.deviceSlotPublished ? config_.numSignalSlots : 0;
  const int numCounterSlots =
      peer.deviceSlotPublished ? config_.numCounterSlots : 0;
  const uint32_t maxChannels = peer.deviceSlotPublished
      ? static_cast<uint32_t>(config_.max_num_channels)
      : 0;
  auto* const peerCmdQueueDevices =
      static_cast<IbrcCmdQueueDevice*>(cmdQueueDevices_.device) +
      static_cast<std::size_t>(peerIndex) * cmdQueuesPerPeer_;
  auto* const peerChannels = static_cast<IbChannel*>(channelStates_.device) +
      static_cast<std::size_t>(peerIndex) * config_.max_num_channels;

  writeIbrcDeviceSlot(
      p2pTransportDevices_.host,
      peerIndex,
      DeviceSpan<IbrcCmdQueueDevice>(
          peerCmdQueueDevices, static_cast<uint32_t>(cmdQueuesPerPeer_)),
      static_cast<uint32_t>(numNics_),
      maxChannels,
      static_cast<uint32_t>(config_.qpsPerConnection),
      static_cast<uint32_t>(config_.fixedChannelDirectionCount()),
      DeviceSpan<IbChannel>(
          peerChannels, static_cast<uint32_t>(config_.max_num_channels)),
      remoteSignalBuf,
      localSignalBuf,
      counterDeviceBuf,
      counterHostBuf,
      numSignalSlots,
      numCounterSlots,
      channelLayout);
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

// Peer-lazy send/recv staging allocation, exchange, cleanup, and channel-layout
// construction are provided by MultiPeerIbTransportBase.

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
}

void MultipeerIbrcTransport::createPeerQps(
    int peerIndex,
    uint32_t beginChannel,
    uint32_t endChannel) {
  if (peerIndex < 0 || peerIndex >= static_cast<int>(peerResources_.size())) {
    throw std::invalid_argument(
        fmt::format("createPeerQps: invalid peerIndex={}", peerIndex));
  }
  if (beginChannel >= endChannel || endChannel > channelCapacity()) {
    throw std::invalid_argument(
        fmt::format(
            "createPeerQps: invalid range=[{}, {}) capacity={}",
            beginChannel,
            endChannel,
            channelCapacity()));
  }
  auto& peer = peerResources_[peerIndex];
  const auto slots = qpSlotRange(config_, beginChannel, endChannel);
  const std::size_t resourceBegin =
      static_cast<std::size_t>(slots.begin) * numNics_;
  const std::size_t resourceCount =
      static_cast<std::size_t>(slots.end - slots.begin) * numNics_;
  CHECK_EQ(peer.qpResources.size(), cmdQueuesPerPeer_);
  std::vector<PeerQpResource> qpResources;
  qpResources.reserve(resourceCount);
  auto& symbols = ibverbx::ibvSymbols;

  try {
    for (int q = slots.begin; q < slots.end; ++q) {
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

  for (std::size_t resource = 0; resource < resourceCount; ++resource) {
    auto& destination = peer.qpResources[resourceBegin + resource];
    auto& source = qpResources[resource];
    CHECK(destination.cq == nullptr);
    CHECK(destination.qp == nullptr);
    destination.cq = std::exchange(source.cq, nullptr);
    destination.qp = std::exchange(source.qp, nullptr);
    destination.signalAtomicSinkMr =
        std::exchange(source.signalAtomicSinkMr, nullptr);
    destination.signalAtomicSink = std::move(source.signalAtomicSink);
    destination.nic = source.nic;
    destination.qpSlot = source.qpSlot;
  }
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
      qpSlot >= config_.fixedChannelMainQpsPerPeerPerNic()) {
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

PeerQpPayload MultipeerIbrcTransport::buildLocalQpPayload(
    int peerIndex,
    uint32_t beginChannel,
    uint32_t endChannel) const {
  const auto slots = qpSlotRange(config_, beginChannel, endChannel);
  PeerQpPayload payload{};
  populatePeerGeometry(payload);
  payload.gidIndex = gidIndex_;
  payload.mtu = static_cast<int>(localMtu_);

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
    for (int q = slots.begin; q < slots.end; ++q) {
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
    uint32_t beginChannel,
    uint32_t endChannel,
    const PeerQpPayload& remotePayload) {
  validatePeerGeometry(peerIndexToRank(peerIndex), remotePayload);

  const auto slots = qpSlotRange(config_, beginChannel, endChannel);
  for (int nic = 0; nic < numNics_; ++nic) {
    for (int q = slots.begin; q < slots.end; ++q) {
      connectPeerQp(
          qpResourceAt(peerIndex, nic, q),
          remotePayload.nicInfo[nic].qpns[q],
          remotePayload.nicInfo[nic].gid,
          remotePayload.nicInfo[nic].lid,
          remotePayload.mtu);
    }
  }
}

P2pIbrcTransportDevice* MultipeerIbrcTransport::getP2pTransportDeviceSlot(
    int peerRank) const {
  throwIfMaterializationFailed();
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
    int peerRank) {
  throwIfMaterializationFailed();
  if (!isPeerMaterialized(peerRank)) {
    queuePeerForMaterialization(peerRank, channelCapacity());
    connectPeers();
  }
  if (p2pTransportDevices_.device == nullptr) {
    throw std::runtime_error(
        "getP2pTransportDevice: IBRC device transport slots are not initialized");
  }
  const int peerIndex = rankToPeerIndex(peerRank);
  return reinterpret_cast<P2pIbrcTransportDevice*>(
      static_cast<char*>(p2pTransportDevices_.device) +
      peerIndex * ibrcDeviceSlotSize());
}

void MultipeerIbrcTransport::materializePeerChannelRange(
    int peerRank,
    uint32_t beginChannel,
    uint32_t endChannel) {
  if (beginChannel >= endChannel || endChannel > channelCapacity()) {
    throw std::invalid_argument(
        fmt::format(
            "materializePeerChannelRange: invalid range=[{}, {}) capacity={}",
            beginChannel,
            endChannel,
            channelCapacity()));
  }
  if (!config_.lazyChannels &&
      (beginChannel != 0 || endChannel != channelCapacity())) {
    throw std::runtime_error(
        "IBRC eager materialization requires the full channel range");
  }
  const int peerIndex = rankToPeerIndex(peerRank);
#ifdef __HIP_PLATFORM_AMD__
  HipStreamCaptureModeGuard captureModeGuard;
#else
  meta::comms::StreamCaptureModeGuard captureModeGuard{
      cudaStreamCaptureModeRelaxed};
#endif

  createPeerQps(peerIndex, beginChannel, endChannel);

  const auto localQp = buildLocalQpPayload(peerIndex, beginChannel, endChannel);
  const auto remoteQp =
      exchangeWithPeer(peerRank, localQp, kIbPeerQpExchangeTag);
  connectPeerQps(peerIndex, beginChannel, endChannel, remoteQp);

  PeerBufferPayload localBuf{};
  if (beginChannel == 0) {
    allocatePeerSignalCounterResources(
        peerIndex, localBuf, IbCounterStorage::HostPinned);
  }

  IbChannelLayout channelLayout;
  if (config_.lazyChannels) {
    channelLayout = allocateSendRecvChannelRange(
        peerIndex,
        beginChannel,
        endChannel,
        localBuf,
        IbCounterStorage::HostPinned);
  } else {
    allocateSendRecvBufferForPeer(
        peerIndex, localBuf, IbCounterStorage::HostPinned);
    channelLayout = channelLayoutForPeer(peerIndex);
  }

  const auto remoteBuf =
      exchangeWithPeer(peerRank, localBuf, kIbPeerBufferExchangeTag);
  if (config_.lazyChannels) {
    applyRemoteSendRecvChannelRange(channelLayout, remoteBuf);
  } else {
    applyRemoteSendRecvBuffer(peerIndex, remoteBuf);
    channelLayout = channelLayoutForPeer(peerIndex);
  }
  if (beginChannel == 0) {
    applyRemoteSignalCounterResources(peerIndex, remoteBuf);
  }

  allocatePeerCmdQueueRange(peerIndex, beginChannel, endChannel, channelLayout);
  startProgressThread();

  VLOG(1) << "MultipeerIbrcTransport: rank " << myRank_ << " materialized peer "
          << peerRank << " channels [" << beginChannel << ", " << endChannel
          << ")";
}

} // namespace comms::prims
