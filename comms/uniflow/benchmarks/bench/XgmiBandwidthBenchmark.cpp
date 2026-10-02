// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/XgmiBandwidthBenchmark.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string_view>
#include <system_error>

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/SegmentHelper.h"
#include "comms/uniflow/benchmarks/Stats.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"
#include "comms/uniflow/transport/p2p/P2pTransport.h"

namespace uniflow::benchmark {

namespace {

constexpr uint8_t kFillByte = 0xAB;

/// RAII device buffer: a plain allocation, which P2P shares through HIP IPC,
/// or cuMem VMM chunks with a POSIX-fd handle type mapped back to back in one
/// reservation, which it shares as one exported fd per chunk. size() is the
/// requested size in every case, so all modes register and copy the same
/// extent.
class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  DeviceAllocation(DeviceAllocation&&) = delete;
  DeviceAllocation& operator=(DeviceAllocation&&) = delete;

  ~DeviceAllocation() {
    if (driverApi_ == nullptr) {
      if (ptr_ != nullptr) {
        (void)cudaFree(ptr_);
      }
      return;
    }
    const auto base = reinterpret_cast<uintptr_t>(ptr_);
    for (size_t i = 0; i < mappedChunks_; ++i) {
      (void)driverApi_->cuMemUnmap(
          toDevicePtr(base + i * chunkSize_), chunkSize_);
    }
    if (ptr_ != nullptr) {
      (void)driverApi_->cuMemAddressFree(
          toDevicePtr(base), chunkSize_ * chunkCount_);
    }
    for (const auto handle : handles_) {
      (void)driverApi_->cuMemRelease(handle);
    }
  }

  Status init(size_t size) {
    auto err = cudaMalloc(&ptr_, size);
    if (err != cudaSuccess) {
      ptr_ = nullptr;
      return Err(
          ErrCode::MemoryRegistrationError,
          fmt::format(
              "cudaMalloc({}) failed: {}", size, cudaGetErrorString(err)));
    }
    size_ = size;
    return Ok();
  }

  /// Chunks of chunkSize (0: a single chunk), each rounded up to the
  /// allocation granularity, mapped read-write for deviceId only; importers
  /// grant their own access.
  Status initVmm(
      CudaDriverApi& driverApi,
      int deviceId,
      size_t size,
      size_t chunkSize) {
    driverApi_ = &driverApi;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity = 0;
    CHECK_EXPR(driverApi_->cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const size_t requested = chunkSize == 0 ? size : std::min(chunkSize, size);
    chunkSize_ = (requested + granularity - 1) / granularity * granularity;
    chunkCount_ = (size + chunkSize_ - 1) / chunkSize_;
    CUdeviceptr reservation{};
    CHECK_EXPR(driverApi_->cuMemAddressReserve(
        &reservation, chunkSize_ * chunkCount_, granularity, CUdeviceptr{}, 0));
    // CUdeviceptr is an integer on NVIDIA.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    ptr_ = reinterpret_cast<void*>(reservation);
    const auto base = reinterpret_cast<uintptr_t>(ptr_);
    handles_.reserve(chunkCount_);
    for (size_t i = 0; i < chunkCount_; ++i) {
      CUmemGenericAllocationHandle handle{};
      CHECK_EXPR(driverApi_->cuMemCreate(&handle, chunkSize_, &prop, 0));
      handles_.push_back(handle);
      CHECK_EXPR(driverApi_->cuMemMap(
          toDevicePtr(base + i * chunkSize_), chunkSize_, 0, handle, 0));
      ++mappedChunks_;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = deviceId;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_EXPR(driverApi_->cuMemSetAccess(
        reservation, chunkSize_ * chunkCount_, &access, 1));
    size_ = size;
    return Ok();
  }

  void* ptr() const {
    return ptr_;
  }
  size_t size() const {
    return size_;
  }
  size_t chunkCount() const {
    return chunkCount_;
  }

 private:
  void* ptr_{nullptr};
  size_t size_{0};
  CudaDriverApi* driverApi_{nullptr};
  size_t chunkSize_{0};
  size_t chunkCount_{1};
  std::vector<CUmemGenericAllocationHandle> handles_;
  size_t mappedChunks_{0};
};

/// Holds all resources for a benchmark transport session.
struct TransportSession {
  TransportSession() = default;
  ~TransportSession() {
    if (transport) {
      transport->shutdown();
    }
  }
  TransportSession(const TransportSession&) = delete;
  TransportSession& operator=(const TransportSession&) = delete;
  TransportSession(TransportSession&&) = delete;
  TransportSession& operator=(TransportSession&&) = delete;

  XgmiOptions options;
  std::shared_ptr<CudaDriverApi> driverApi; // allocates the VMM buffers
  std::unique_ptr<ScopedEventBaseThread> evbThread;
  std::unique_ptr<P2pTransportFactory> factory;
  std::unique_ptr<DeviceAllocation> srcAlloc;
  std::unique_ptr<DeviceAllocation> dstAlloc;
  std::unique_ptr<Transport> transport;
  std::unique_ptr<RegisteredSegment> localReg;
  std::unique_ptr<RemoteRegisteredSegment> remoteReg;
  // The peer imports this registration. A VMM registration owns the exported
  // fds the peer pulls with pidfd_getfd, so it must outlive the peer's import:
  // it is kept until the next size (or the end of the sweep), after both ranks
  // pass the teardown barrier.
  std::unique_ptr<RegistrationHandle> dstReg;
  // Size of the buffers above, for the teardown log.
  size_t bufferSize{0};
};

// P2P exports VMM memory only if the chunk fds fit under RLIMIT_NOFILE with
// this many descriptors to spare (kFdHeadroom in P2pVmm.cpp).
constexpr size_t kP2pFdHeadroom = 256;

bool hasVmm(const XgmiOptions& options) {
  return options.srcMemory == XgmiMemory::Vmm ||
      options.dstMemory == XgmiMemory::Vmm;
}

std::string_view memoryName(XgmiMemory memory) {
  return memory == XgmiMemory::Vmm ? "vmm" : "device";
}

/// Transport column of the results: "xgmi" for the device/device baseline,
/// otherwise suffixed with the buffer pair and the VMM chunk size, plus a
/// disabled VMM switch, so every arm stays distinguishable in one CSV.
std::string transportLabel(const XgmiOptions& options) {
  std::string label{"xgmi"};
  if (hasVmm(options)) {
    label += fmt::format(
        "_{}_{}", memoryName(options.srcMemory), memoryName(options.dstMemory));
    if (options.vmmChunkSize != 0) {
      label += fmt::format("_c{}", options.vmmChunkSize);
    }
  }
  if (!options.p2pEnableVmm) {
    label += "_vmm_off";
  }
  return label;
}

/// Whether P2P can export VMM buffers totalling this many chunks without
/// exceeding its fd budget. Past it P2P would fall back to HIP IPC, which
/// cannot share VMM memory and on some runtimes hands out a handle the peer
/// cannot safely open, so the benchmark refuses the size instead.
bool fitsP2pFdBudget(size_t chunks) {
  rlimit limit{};
  if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: getrlimit(RLIMIT_NOFILE) failed");
    return false;
  }
  if (limit.rlim_cur == RLIM_INFINITY) {
    return true;
  }
  std::error_code ec;
  size_t openFds = 0;
  for (std::filesystem::directory_iterator it{"/proc/self/fd", ec};
       !ec && it != std::filesystem::directory_iterator{};
       it.increment(ec)) {
    ++openFds;
  }
  if (ec) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: counting open fds failed: {}", ec.message());
    return false;
  }
  if (openFds + chunks + kP2pFdHeadroom > limit.rlim_cur) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: {} VMM chunk fds with {} open and {} spare "
        "exceed the RLIMIT_NOFILE soft limit {}; use a larger "
        "--xgmi-vmm-chunk-size or raise the limit",
        chunks,
        openFds,
        kP2pFdHeadroom,
        limit.rlim_cur);
    return false;
  }
  return true;
}

/// Sharing mode P2P must have picked for the configured memory; a registration
/// that fell back to the other mode would measure the wrong export path.
P2pSharingMode expectedSharingMode(XgmiMemory memory) {
  return memory == XgmiMemory::Vmm ? P2pSharingMode::PosixFd
                                   : P2pSharingMode::Ipc;
}

bool hasSharingMode(const RegistrationHandle& handle, P2pSharingMode mode) {
  const auto* p2p = dynamic_cast<const P2pRegistrationHandle*>(&handle);
  return p2p != nullptr && p2p->sharingMode() == mode;
}

double elapsedUs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count();
}

/// Create factory, create transport, and connect to peer.
///
/// Mirrors NVLinkBandwidthBenchmark's setup, against the AMD P2P factory:
/// getTopology -> exchange -> createTransport -> bind -> exchange -> connect.
std::unique_ptr<TransportSession> setupConnection(
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    const XgmiOptions& options) {
  if (peers.empty()) {
    UNIFLOW_LOG_ERROR("XgmiBandwidthBenchmark: setupConnection: no peers");
    return nullptr;
  }

  auto session = std::make_unique<TransportSession>();
  session->options = options;
  if (hasVmm(options)) {
    session->driverApi = std::make_shared<CudaDriverApi>();
  }

  CudaApi cudaApi;
  auto setDevStatus = cudaApi.setDevice(bootstrap.localRank);
  if (!setDevStatus) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: setDevice failed: {}",
        setDevStatus.error().toString());
    return nullptr;
  }

  session->evbThread = std::make_unique<ScopedEventBaseThread>("bench-evb");
#if defined(__HIP_PLATFORM_AMD__)
  session->factory = std::make_unique<P2pTransportFactory>(
      bootstrap.localRank,
      session->evbThread->getEventBase(),
      nullptr,
      nullptr,
      options.p2pEnableVmm);
#else
  session->factory = std::make_unique<P2pTransportFactory>(
      bootstrap.localRank, session->evbThread->getEventBase());
#endif

  auto localTopology = session->factory->getTopology();
  auto remoteTopologyResult =
      exchangeMetadata(*peers[0].ctrl, localTopology, bootstrap.isRank0());
  if (!remoteTopologyResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: topology exchange failed: {}",
        remoteTopologyResult.error().toString());
    return nullptr;
  }

  auto transportResult = session->factory->createTransport(
      std::move(remoteTopologyResult).value());
  if (!transportResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: createTransport failed: {}",
        transportResult.error().toString());
    return nullptr;
  }
  session->transport = std::move(transportResult).value();

  auto localInfo = session->transport->bind();
  auto remoteInfoResult =
      exchangeMetadata(*peers[0].ctrl, localInfo, bootstrap.isRank0());
  if (!remoteInfoResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: transport info exchange failed: {}",
        remoteInfoResult.error().toString());
    return nullptr;
  }

  auto connectStatus =
      session->transport->connect(std::move(remoteInfoResult).value());
  if (!connectStatus) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: connect failed: {}",
        connectStatus.error().toString());
    return nullptr;
  }

  return session;
}

/// Free the previous size's buffers. After a completed setup, also time the
/// P2P teardown each rank pays: unmapping the peer's buffer it imported, and
/// deregistering its own buffer the peer imported.
void releaseBuffers(TransportSession& session) {
  if (session.remoteReg != nullptr) {
    const auto unimportStart = std::chrono::steady_clock::now();
    session.remoteReg.reset();
    const double unimportUs = elapsedUs(unimportStart);
    const auto deregisterStart = std::chrono::steady_clock::now();
    session.dstReg.reset();
    const double deregisterUs = elapsedUs(deregisterStart);
    UNIFLOW_LOG_INFO(
        "XgmiBandwidthBenchmark: teardown size={} unimport_us={:.1f} "
        "dst_deregister_us={:.1f}",
        session.bufferSize,
        unimportUs,
        deregisterUs);
  }
  session.remoteReg.reset();
  session.dstReg.reset();
  session.localReg.reset();
  session.srcAlloc.reset();
  session.dstAlloc.reset();
}

/// Allocate device buffers of the given size, register segments, and exchange
/// handles with the peer. Re-done per message size so the registered extent
/// matches the transfer size, as the NVIDIA benchmark does.
bool setupBuffersForSize(
    TransportSession& session,
    size_t bufferSize,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_ERROR("XgmiBandwidthBenchmark: setupBuffersForSize: no peers");
    return false;
  }

  releaseBuffers(session);
  session.bufferSize = bufferSize;

  auto allocate = [&](DeviceAllocation& alloc, XgmiMemory memory) {
    return memory == XgmiMemory::Vmm ? alloc.initVmm(
                                           *session.driverApi,
                                           bootstrap.localRank,
                                           bufferSize,
                                           session.options.vmmChunkSize)
                                     : alloc.init(bufferSize);
  };

  session.srcAlloc = std::make_unique<DeviceAllocation>();
  auto srcStatus = allocate(*session.srcAlloc, session.options.srcMemory);
  if (srcStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: src allocation failed: {}",
        srcStatus.error().message());
    return false;
  }

  session.dstAlloc = std::make_unique<DeviceAllocation>();
  auto dstStatus = allocate(*session.dstAlloc, session.options.dstMemory);
  if (dstStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: dst allocation failed: {}",
        dstStatus.error().message());
    return false;
  }

  auto srcMemsetErr =
      cudaMemset(session.srcAlloc->ptr(), kFillByte, session.srcAlloc->size());
  if (srcMemsetErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: memset(src) failed: {}",
        cudaGetErrorString(srcMemsetErr));
    return false;
  }
  auto dstMemsetErr =
      cudaMemset(session.dstAlloc->ptr(), 0, session.dstAlloc->size());
  if (dstMemsetErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: memset(dst) failed: {}",
        cudaGetErrorString(dstMemsetErr));
    return false;
  }
  auto syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: deviceSynchronize failed: {}",
        cudaGetErrorString(syncErr));
    return false;
  }

  size_t vmmChunks = 0;
  if (session.options.srcMemory == XgmiMemory::Vmm) {
    vmmChunks += session.srcAlloc->chunkCount();
  }
  if (session.options.dstMemory == XgmiMemory::Vmm) {
    vmmChunks += session.dstAlloc->chunkCount();
  }
  if (vmmChunks != 0 && !fitsP2pFdBudget(vmmChunks)) {
    return false;
  }

  Segment srcSeg(
      session.srcAlloc->ptr(),
      session.srcAlloc->size(),
      MemoryType::VRAM,
      bootstrap.localRank);
  const auto srcRegisterStart = std::chrono::steady_clock::now();
  auto srcRegResult = session.factory->registerSegment(srcSeg);
  const double srcRegisterUs = elapsedUs(srcRegisterStart);
  if (!srcRegResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: registerSegment(src) failed: {}",
        srcRegResult.error().toString());
    return false;
  }

  Segment dstSeg(
      session.dstAlloc->ptr(),
      session.dstAlloc->size(),
      MemoryType::VRAM,
      bootstrap.localRank);
  const auto dstRegisterStart = std::chrono::steady_clock::now();
  auto dstRegResult = session.factory->registerSegment(dstSeg);
  const double dstRegisterUs = elapsedUs(dstRegisterStart);
  if (!dstRegResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: registerSegment(dst) failed: {}",
        dstRegResult.error().toString());
    return false;
  }

  if (!hasSharingMode(
          *srcRegResult.value(),
          expectedSharingMode(session.options.srcMemory)) ||
      !hasSharingMode(
          *dstRegResult.value(),
          expectedSharingMode(session.options.dstMemory))) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: P2P did not share the src={} dst={} buffers "
        "as configured (vmm as POSIX fds, device as IPC handles)",
        memoryName(session.options.srcMemory),
        memoryName(session.options.dstMemory));
    return false;
  }

  auto dstPayload = dstRegResult.value()->serialize();
  session.dstReg = std::move(dstRegResult).value();
  auto remotePayloadResult =
      exchangeMetadata(*peers[0].ctrl, dstPayload, bootstrap.isRank0());
  if (!remotePayloadResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: handle exchange failed: {}",
        remotePayloadResult.error().toString());
    return false;
  }

  // Cross-process: the peer's payload names a different PID, so this is the
  // real ipcOpenMemHandle (or pidfd_getfd and VMM import) path rather than the
  // same-process shortcut. That is the whole reason this benchmark runs as two
  // ranks.
  const auto importStart = std::chrono::steady_clock::now();
  auto remoteHandleResult = session.factory->importSegment(
      session.dstAlloc->size(), std::move(remotePayloadResult).value());
  const double importUs = elapsedUs(importStart);
  if (!remoteHandleResult) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: importSegment failed: {}",
        remoteHandleResult.error().toString());
    return false;
  }
  UNIFLOW_LOG_INFO(
      "XgmiBandwidthBenchmark: setup src={} dst={} p2p_vmm={} size={} "
      "src_chunks={} dst_chunks={} src_register_us={:.1f} "
      "dst_register_us={:.1f} import_us={:.1f}",
      memoryName(session.options.srcMemory),
      memoryName(session.options.dstMemory),
      session.options.p2pEnableVmm ? "on" : "off",
      bufferSize,
      session.srcAlloc->chunkCount(),
      session.dstAlloc->chunkCount(),
      srcRegisterUs,
      dstRegisterUs,
      importUs);

  session.localReg = std::make_unique<RegisteredSegment>(
      SegmentTest::makeRegistered(srcSeg, std::move(srcRegResult.value())));

  auto* p2pRemote = dynamic_cast<P2pRemoteRegistrationHandle*>(
      remoteHandleResult.value().get());
  if (!p2pRemote) {
    UNIFLOW_LOG_ERROR("XgmiBandwidthBenchmark: failed to cast remote handle");
    return false;
  }

  session.remoteReg =
      std::make_unique<RemoteRegisteredSegment>(SegmentTest::makeRemote(
          p2pRemote->mappedPtr(),
          p2pRemote->mappedSize(),
          std::move(remoteHandleResult.value())));

  return true;
}

/// Full-size put+get before timing, then verify the bytes actually moved.
///
/// This is correctness insurance, not warmup: a transport that silently moved
/// nothing would otherwise report excellent bandwidth.
bool prefaultAndVerify(TransportSession& session, size_t maxSize) {
  TransferRequest req{
      .local = session.localReg->span(size_t{0}, maxSize),
      .remote = session.remoteReg->span(size_t{0}, maxSize),
  };

  // 1. Put: src (0xAB) -> peer's mapped buffer.
  auto putStatus = session.transport->put({&req, 1}).get();
  if (putStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: put pre-fault failed: {}",
        putStatus.error().message());
    return false;
  }

  // 2. Zero src so the get below has to overwrite it to pass.
  auto memsetErr = cudaMemset(session.srcAlloc->ptr(), 0, maxSize);
  if (memsetErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: verification memset failed: {}",
        cudaGetErrorString(memsetErr));
    return false;
  }
  // Must complete before the get below, or the verification races the zeroing
  // and can pass on stale 0xAB rather than on bytes the get actually moved.
  auto zeroSyncErr = cudaDeviceSynchronize();
  if (zeroSyncErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: verification deviceSynchronize failed: {}",
        cudaGetErrorString(zeroSyncErr));
    return false;
  }

  // 3. Get: peer's buffer (holds 0xAB) -> src.
  auto getStatus = session.transport->get({&req, 1}).get();
  if (getStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: get pre-fault failed: {}",
        getStatus.error().message());
    return false;
  }

  // 4. Read back and verify.
  constexpr size_t kCheckSize = 64;
  const size_t checkSize = std::min(maxSize, kCheckSize);
  uint8_t hostBuf[kCheckSize] = {};
  auto copyErr = cudaMemcpy(
      hostBuf, session.srcAlloc->ptr(), checkSize, cudaMemcpyDeviceToHost);
  if (copyErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: verification memcpy failed: {}",
        cudaGetErrorString(copyErr));
    return false;
  }

  for (size_t i = 0; i < checkSize; ++i) {
    if (hostBuf[i] != kFillByte) {
      UNIFLOW_LOG_ERROR(
          "XgmiBandwidthBenchmark: data verification failed at byte {}: "
          "expected {:#x}, got {:#x}",
          i,
          kFillByte,
          hostBuf[i]);
      return false;
    }
  }

  // Check the tail too. A head-only check cannot see a short copy or an
  // undersized mapping, and would report full bandwidth for it. That matters
  // more on this path than on the NVIDIA one: importSegment cannot bounds-check
  // (ipcOpenMemHandle does not report the mapping size) and registerSegment
  // does allocBase + offset math the VMM path does not have.
  if (maxSize > kCheckSize) {
    uint8_t tailBuf[kCheckSize] = {};
    auto* tail =
        static_cast<uint8_t*>(session.srcAlloc->ptr()) + maxSize - kCheckSize;
    auto tailErr =
        cudaMemcpy(tailBuf, tail, kCheckSize, cudaMemcpyDeviceToHost);
    if (tailErr != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "XgmiBandwidthBenchmark: tail verification memcpy failed: {}",
          cudaGetErrorString(tailErr));
      return false;
    }
    for (size_t i = 0; i < kCheckSize; ++i) {
      if (tailBuf[i] != kFillByte) {
        UNIFLOW_LOG_ERROR(
            "XgmiBandwidthBenchmark: tail verification failed at offset {}: "
            "expected {:#x}, got {:#x}",
            maxSize - kCheckSize + i,
            kFillByte,
            tailBuf[i]);
        return false;
      }
    }
  }

  // Refill src for the timed loop.
  auto refillErr =
      cudaMemset(session.srcAlloc->ptr(), kFillByte, session.srcAlloc->size());
  if (refillErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: refill memset failed: {}",
        cudaGetErrorString(refillErr));
    return false;
  }
  auto refillSyncErr = cudaDeviceSynchronize();
  if (refillSyncErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: refill deviceSynchronize failed: {}",
        cudaGetErrorString(refillSyncErr));
    return false;
  }

  UNIFLOW_LOG_INFO("XgmiBandwidthBenchmark: pre-faulted, data verified");
  return true;
}

/// Sweep message sizes, measure put/get bandwidth, collect results.
std::vector<BenchmarkResult> runBenchmarkLoop(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    TransportSession& session,
    const std::string& benchmarkName,
    bool isActiveRank) {
  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;
  const std::string transport = transportLabel(session.options);

  // Latch rather than return: the peer is already entering the sweep.
  bool setupFailed = false;

  cudaStream_t benchStream = nullptr;
  if (isActiveRank) {
    auto streamErr =
        cudaStreamCreateWithFlags(&benchStream, cudaStreamNonBlocking);
    if (streamErr != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "XgmiBandwidthBenchmark: streamCreate failed: {}",
          cudaGetErrorString(streamErr));
      setupFailed = true;
    }
  }

  auto runDirection = [&](const std::string& dir) {
    // A rank that gives up on its own work latches this and keeps meeting the
    // peer at every rendezvous below, doing no work and recording no results.
    // Returning instead would leave the peer waiting at one.
    //
    // A failing barrier() is the exception and does return: the rendezvous
    // channel is gone, and every later one would only time out in turn.
    bool aborted = setupFailed;

    for (auto size : sizes) {
      // Called even when aborted: this does an exchangeMetadata, so skipping
      // it on one rank desynchronizes exactly like skipping a barrier.
      if (!setupBuffersForSize(session, size, peers, bootstrap)) {
        UNIFLOW_LOG_ERROR(
            "XgmiBandwidthBenchmark: setupBuffersForSize failed for size {}",
            size);
        aborted = true;
      }

      if (!aborted && isActiveRank && !prefaultAndVerify(session, size)) {
        UNIFLOW_LOG_ERROR(
            "XgmiBandwidthBenchmark: prefault failed for size {}", size);
        aborted = true;
      }

      const int totalIterations = config.warmupIterations + config.iterations;
      std::vector<double> latenciesUs;
      latenciesUs.reserve(config.iterations);

      auto barrierStatus = barrier(peers, bootstrap);
      if (!barrierStatus) {
        UNIFLOW_LOG_ERROR(
            "XgmiBandwidthBenchmark: barrier failed: {}",
            barrierStatus.error().toString());
        return;
      }

      if (!aborted && isActiveRank) {
        TransferRequest singleReq{
            .local = session.localReg->span(size_t{0}, size),
            .remote = session.remoteReg->span(size_t{0}, size),
        };
        std::vector<TransferRequest> batchReqs(config.loopCount, singleReq);

        for (int iter = 0; iter < totalIterations; ++iter) {
          auto start = std::chrono::steady_clock::now();

          RequestOptions opts;
          opts.stream = benchStream;
          Status opStatus;
          if (dir == "put") {
            opStatus = session.transport->put(batchReqs, opts).get();
          } else {
            opStatus = session.transport->get(batchReqs, opts).get();
          }

          if (opStatus.hasError()) {
            UNIFLOW_LOG_ERROR(
                "XgmiBandwidthBenchmark: {} failed at size {}: {}",
                dir,
                size,
                opStatus.error().message());
            aborted = true;
            break;
          }

          auto end = std::chrono::steady_clock::now();

          if (iter >= config.warmupIterations) {
            double elapsedUs =
                std::chrono::duration<double, std::micro>(end - start).count();
            latenciesUs.push_back(elapsedUs / config.loopCount);
          }
        }
      } // isActiveRank

      // Teardown barrier: nobody frees this size's buffers until both ranks are
      // done with them.
      auto teardownBarrier = barrier(peers, bootstrap);
      if (!teardownBarrier) {
        UNIFLOW_LOG_ERROR(
            "XgmiBandwidthBenchmark: teardown barrier failed: {}",
            teardownBarrier.error().toString());
        return;
      }

      if (aborted || !isActiveRank) {
        continue;
      }

      auto stats = Stats::compute(std::move(latenciesUs));
      double bandwidthGBs = (stats.avg > 0)
          ? (static_cast<double>(size) / (stats.avg * 1e-6)) / 1e9
          : 0;

      BenchmarkResult result{
          .benchmarkName = benchmarkName,
          .transport = transport,
          .direction = dir,
          .messageSize = size,
          .iterations = config.iterations,
          .bandwidthGBs = bandwidthGBs,
          .latency = stats,
      };
      results.push_back(result);

      UNIFLOW_LOG_INFO(
          "XgmiBandwidthBenchmark: {} size={} avg={:.1f}us bw={:.2f}GB/s",
          dir,
          size,
          stats.avg,
          bandwidthGBs);
    }
  };

  if (config.direction == "put" || config.direction == "both") {
    runDirection("put");
  }
  if (config.direction == "get" || config.direction == "both") {
    runDirection("get");
  }

  if (benchStream) {
    (void)cudaStreamDestroy(benchStream); // [[nodiscard]] under HIP
  }

  return results;
}

} // namespace

std::vector<BenchmarkResult> XgmiBandwidthBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("XgmiBandwidthBenchmark: no peers, skipping");
    return {};
  }

  if (config.loopCount < 1) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: loopCount must be >= 1, got {}",
        config.loopCount);
    return {};
  }

  if (!options_.p2pEnableVmm && hasVmm(options_)) {
    UNIFLOW_LOG_ERROR(
        "XgmiBandwidthBenchmark: VMM buffers need P2P VMM sharing enabled; "
        "HIP IPC cannot share them");
    return {};
  }

  auto session = setupConnection(peers, bootstrap, options_);
  if (!session) {
    return {};
  }

  const bool isActiveRank = config.bidirectional || bootstrap.isRank0();

  UNIFLOW_LOG_INFO(
      "XgmiBandwidthBenchmark: rank {} setup complete, sweeping sizes {}-{}"
      " (src={}, dst={}, vmm_chunk_size={}, p2p_vmm={}, loopCount={}, "
      "{}directional, active={})",
      bootstrap.rank,
      config.minSize,
      config.maxSize,
      memoryName(options_.srcMemory),
      memoryName(options_.dstMemory),
      options_.vmmChunkSize,
      options_.p2pEnableVmm ? "on" : "off",
      config.loopCount,
      config.bidirectional ? "bi" : "uni",
      isActiveRank);

  auto results = runBenchmarkLoop(
      config, peers, bootstrap, *session, name(), isActiveRank);
  // The sweep ends after a teardown barrier, so both ranks are done with the
  // last size; releasing it here logs its teardown like the earlier sizes.
  releaseBuffers(*session);

  auto shutdownBarrier = barrier(peers, bootstrap);
  if (!shutdownBarrier) {
    UNIFLOW_LOG_WARN(
        "XgmiBandwidthBenchmark: shutdown barrier failed: {}",
        shutdownBarrier.error().toString());
  }
  // No explicit shutdown(): ~TransportSession does it, and its body runs before
  // any member is destroyed. Calling both only duplicated the log line.
  return results;
}

} // namespace uniflow::benchmark
