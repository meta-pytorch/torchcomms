// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/NVLinkBandwidthBenchmark.h"

#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/SegmentHelper.h"
#include "comms/uniflow/benchmarks/Stats.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/nvlink/NVLinkRegistrationHandle.h"
#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"

namespace uniflow::benchmark {

namespace {

/// RAII wrapper for cuMem VMM GPU memory allocation.
class VmmAllocation {
 public:
  VmmAllocation() = default;

  Status init(
      CudaDriverApi& driverApi,
      int deviceId,
      size_t requestedSize,
      CUmemAllocationHandleType handleType) {
    driverApi_ = &driverApi;

    CUdevice device;
    CHECK_RETURN(driverApi_->cuDeviceGet(&device, deviceId));

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = handleType;

    CHECK_RETURN(driverApi_->cuMemGetAllocationGranularity(
        &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    size_ = ((requestedSize + granularity_ - 1) / granularity_) * granularity_;

    CHECK_RETURN(driverApi_->cuMemCreate(&allocHandle_, size_, &prop, 0));
    created_ = true;

    CHECK_RETURN(
        driverApi_->cuMemAddressReserve(&ptr_, size_, granularity_, 0, 0));
    reserved_ = true;

    CHECK_RETURN(driverApi_->cuMemMap(ptr_, size_, 0, allocHandle_, 0));
    mapped_ = true;

    CUmemAccessDesc accessDesc{};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_RETURN(driverApi_->cuMemSetAccess(ptr_, size_, &accessDesc, 1));

    return Ok();
  }

  ~VmmAllocation() {
    if (!driverApi_) {
      return;
    }
    if (mapped_) {
      driverApi_->cuMemUnmap(ptr_, size_);
    }
    if (reserved_) {
      driverApi_->cuMemAddressFree(ptr_, size_);
    }
    if (created_) {
      driverApi_->cuMemRelease(allocHandle_);
    }
  }

  VmmAllocation(const VmmAllocation&) = delete;
  VmmAllocation& operator=(const VmmAllocation&) = delete;

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  void* ptr() const {
    return reinterpret_cast<void*>(ptr_);
  }
  size_t size() const {
    return size_;
  }

 private:
  CudaDriverApi* driverApi_{nullptr};
  CUmemGenericAllocationHandle allocHandle_{};
  CUdeviceptr ptr_{0};
  size_t size_{0};
  size_t granularity_{0};
  bool created_{false};
  bool reserved_{false};
  bool mapped_{false};
};

/// Holds all resources for a benchmark transport session.
struct TransportSession {
  ~TransportSession() {
    if (transport) {
      transport->shutdown();
    }
  }

  std::shared_ptr<CudaDriverApi> driverApi;
  std::unique_ptr<ScopedEventBaseThread> evbThread;
  std::unique_ptr<NVLinkTransportFactory> factory;
  VmmAllocation srcAlloc;
  VmmAllocation dstAlloc;
  std::unique_ptr<Transport> transport;
  std::unique_ptr<RegisteredSegment> localReg;
  std::unique_ptr<RemoteRegisteredSegment> remoteReg;
};

/// Create factory, allocate VMM buffers, create transport, connect,
/// register segments, and exchange handles with peer.
/// Returns nullptr on failure.
std::unique_ptr<TransportSession> setupTransport(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  auto session = std::make_unique<TransportSession>();

  // --- Device & buffer init ---
  CudaApi cudaApi;
  auto setDevStatus = cudaApi.setDevice(bootstrap.localRank);
  if (!setDevStatus) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: cudaSetDevice failed: {}",
        setDevStatus.error().toString());
    return nullptr;
  }

  session->driverApi = std::make_shared<CudaDriverApi>();
  session->evbThread = std::make_unique<ScopedEventBaseThread>("bench-evb");
  session->factory = std::make_unique<NVLinkTransportFactory>(
      bootstrap.localRank, session->evbThread->getEventBase());

  const auto handleType = session->factory->handleType();

  auto srcStatus = session->srcAlloc.init(
      *session->driverApi, bootstrap.localRank, config.maxSize, handleType);
  if (srcStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: src VmmAllocation failed: {}",
        srcStatus.error().message());
    return nullptr;
  }

  auto dstStatus = session->dstAlloc.init(
      *session->driverApi, bootstrap.localRank, config.maxSize, handleType);
  if (dstStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: dst VmmAllocation failed: {}",
        dstStatus.error().message());
    return nullptr;
  }

  // Fill source with pattern, zero destination.
  auto srcMemsetErr =
      cudaMemset(session->srcAlloc.ptr(), 0xAB, session->srcAlloc.size());
  if (srcMemsetErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: cudaMemset(src) failed: {}",
        cudaGetErrorString(srcMemsetErr));
    return nullptr;
  }
  auto dstMemsetErr =
      cudaMemset(session->dstAlloc.ptr(), 0, session->dstAlloc.size());
  if (dstMemsetErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: cudaMemset(dst) failed: {}",
        cudaGetErrorString(dstMemsetErr));
    return nullptr;
  }
  auto syncErr = cudaDeviceSynchronize();
  if (syncErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: cudaDeviceSynchronize failed: {}",
        cudaGetErrorString(syncErr));
    return nullptr;
  }

  // --- Transport connect ---
  auto localTopology = session->factory->getTopology();
  auto remoteTopologyResult =
      exchangeMetadata(*peers[0].ctrl, localTopology, bootstrap.isRank0());
  if (!remoteTopologyResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: topology exchange failed: {}",
        remoteTopologyResult.error().toString());
    return nullptr;
  }

  auto transportResult = session->factory->createTransport(
      std::move(remoteTopologyResult).value());
  if (!transportResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: createTransport failed: {}",
        transportResult.error().toString());
    return nullptr;
  }
  session->transport = std::move(transportResult).value();

  auto localInfo = session->transport->bind();
  auto remoteInfoResult =
      exchangeMetadata(*peers[0].ctrl, localInfo, bootstrap.isRank0());
  if (!remoteInfoResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: transport info exchange failed: {}",
        remoteInfoResult.error().toString());
    return nullptr;
  }

  auto connectStatus =
      session->transport->connect(std::move(remoteInfoResult).value());
  if (!connectStatus) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: connect failed: {}",
        connectStatus.error().toString());
    return nullptr;
  }

  // --- Segment registration ---
  Segment srcSeg(
      session->srcAlloc.ptr(),
      session->srcAlloc.size(),
      MemoryType::VRAM,
      bootstrap.localRank);
  auto srcRegResult = session->factory->registerSegment(srcSeg);
  if (!srcRegResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: registerSegment(src) failed: {}",
        srcRegResult.error().toString());
    return nullptr;
  }

  Segment dstSeg(
      session->dstAlloc.ptr(),
      session->dstAlloc.size(),
      MemoryType::VRAM,
      bootstrap.localRank);
  auto dstRegResult = session->factory->registerSegment(dstSeg);
  if (!dstRegResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: registerSegment(dst) failed: {}",
        dstRegResult.error().toString());
    return nullptr;
  }

  auto dstPayload = dstRegResult.value()->serialize();
  auto remotePayloadResult =
      exchangeMetadata(*peers[0].ctrl, dstPayload, bootstrap.isRank0());
  if (!remotePayloadResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: handle exchange failed: {}",
        remotePayloadResult.error().toString());
    return nullptr;
  }

  auto remoteHandleResult = session->factory->importSegment(
      session->dstAlloc.size(), std::move(remotePayloadResult).value());
  if (!remoteHandleResult) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: importSegment failed: {}",
        remoteHandleResult.error().toString());
    return nullptr;
  }

  session->localReg = std::make_unique<RegisteredSegment>(
      SegmentTest::makeRegistered(srcSeg, std::move(srcRegResult.value())));

  auto* nvlinkRemote = dynamic_cast<NVLinkRemoteRegistrationHandle*>(
      remoteHandleResult.value().get());
  if (!nvlinkRemote) {
    UNIFLOW_LOG_ERROR("NVLinkBandwidthBenchmark: failed to cast remote handle");
    return nullptr;
  }

  session->remoteReg =
      std::make_unique<RemoteRegisteredSegment>(SegmentTest::makeRemote(
          nvlinkRemote->mappedPtr(),
          nvlinkRemote->mappedSize(),
          std::move(remoteHandleResult.value())));

  return session;
}

/// Pre-fault VMM memory paths with a full-size put+get to populate page tables,
/// then verify data correctness.
bool prefaultAndVerify(TransportSession& session, size_t maxSize) {
  TransferRequest req{
      .local = session.localReg->span(size_t{0}, maxSize),
      .remote = session.remoteReg->span(size_t{0}, maxSize),
  };

  // 1. Put: srcAlloc (0xAB) → remote mapped buffer.
  auto putStatus = session.transport->put({&req, 1}).get();
  if (putStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: put pre-fault failed: {}",
        putStatus.error().message());
    return false;
  }

  // 2. Zero srcAlloc so we can verify the get overwrites it.
  auto memsetErr = cudaMemset(session.srcAlloc.ptr(), 0, maxSize);
  if (memsetErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: verification cudaMemset failed: {}",
        cudaGetErrorString(memsetErr));
    return false;
  }
  cudaDeviceSynchronize();

  // 3. Get: remote mapped buffer (should contain 0xAB) → srcAlloc.
  auto getStatus = session.transport->get({&req, 1}).get();
  if (getStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: get pre-fault failed: {}",
        getStatus.error().message());
    return false;
  }

  // 4. Read back srcAlloc and verify it contains 0xAB.
  constexpr size_t kCheckSize = 64;
  uint8_t hostBuf[kCheckSize] = {};
  auto copyErr = cudaMemcpy(
      hostBuf,
      session.srcAlloc.ptr(),
      std::min(maxSize, kCheckSize),
      cudaMemcpyDeviceToHost);
  if (copyErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: verification cudaMemcpy failed: {}",
        cudaGetErrorString(copyErr));
    return false;
  }

  for (size_t i = 0; i < std::min(maxSize, kCheckSize); ++i) {
    if (hostBuf[i] != 0xAB) {
      UNIFLOW_LOG_ERROR(
          "NVLinkBandwidthBenchmark: data verification failed at byte {}: "
          "expected 0xAB, got {:#x}",
          i,
          hostBuf[i]);
      return false;
    }
  }

  // Refill srcAlloc with 0xAB for the benchmark loop.
  auto refillErr =
      cudaMemset(session.srcAlloc.ptr(), 0xAB, session.srcAlloc.size());
  if (refillErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: refill cudaMemset failed: {}",
        cudaGetErrorString(refillErr));
    return false;
  }
  auto refillSyncErr = cudaDeviceSynchronize();
  if (refillSyncErr != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: refill cudaDeviceSynchronize failed: {}",
        cudaGetErrorString(refillSyncErr));
    return false;
  }

  UNIFLOW_LOG_INFO(
      "NVLinkBandwidthBenchmark: memory pre-faulted, data verified");
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

  // Use a non-blocking CUDA stream to avoid implicit synchronization
  // with the default stream, reducing per-call overhead.
  cudaStream_t benchStream = nullptr;
  if (isActiveRank) {
    auto streamErr =
        cudaStreamCreateWithFlags(&benchStream, cudaStreamNonBlocking);
    if (streamErr != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "NVLinkBandwidthBenchmark: cudaStreamCreateWithFlags failed: {}",
          cudaGetErrorString(streamErr));
      return {};
    }
  }

  auto runDirection = [&](const std::string& dir) {
    for (auto size : sizes) {
      const int totalIterations = config.warmupIterations + config.iterations;
      std::vector<double> latenciesUs;
      latenciesUs.reserve(config.iterations);

      // Prepare transfer requests outside the timed region.
      // Build loopCount identical requests so they can be batched into a
      // single transport call, pipelining memcpys with one event — matching
      // nvbandwidth's methodology.
      TransferRequest singleReq{
          .local = session.localReg->span(size_t{0}, size),
          .remote = session.remoteReg->span(size_t{0}, size),
      };
      std::vector<TransferRequest> batchReqs(config.loopCount, singleReq);

      // Per-size barrier keeps rank 1 (passive in unidirectional mode)
      // in sync with rank 0, preventing early exit and TCP disconnect.
      auto barrierStatus = barrier(peers, bootstrap);
      if (!barrierStatus) {
        UNIFLOW_LOG_ERROR(
            "NVLinkBandwidthBenchmark: barrier failed: {}",
            barrierStatus.error().toString());
        return;
      }

      if (!isActiveRank) {
        continue;
      }

      for (int iter = 0; iter < totalIterations; ++iter) {
        auto start = std::chrono::steady_clock::now();

        // Issue all loopCount requests in a single batched transport call.
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
              "NVLinkBandwidthBenchmark: {} failed at size {}: {}",
              dir,
              size,
              opStatus.error().message());
          return;
        }

        auto end = std::chrono::steady_clock::now();

        if (iter >= config.warmupIterations) {
          double elapsedUs =
              std::chrono::duration<double, std::micro>(end - start).count();
          latenciesUs.push_back(elapsedUs / config.loopCount);
        }
      }

      if (!isActiveRank) {
        continue;
      }

      auto stats = Stats::compute(std::move(latenciesUs));
      double bandwidthGBs = (stats.avg > 0)
          ? (static_cast<double>(size) / (stats.avg * 1e-6)) / 1e9
          : 0;

      BenchmarkResult result{
          .benchmarkName = benchmarkName,
          .transport = "nvlink",
          .direction = dir,
          .messageSize = size,
          .iterations = config.iterations,
          .bandwidthGBs = bandwidthGBs,
          .latency = stats,
      };
      results.push_back(result);

      UNIFLOW_LOG_INFO(
          "NVLinkBandwidthBenchmark: {} size={} avg={:.1f}us bw={:.2f}GB/s",
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
    cudaStreamDestroy(benchStream);
  }

  return results;
}

} // namespace

std::vector<BenchmarkResult> NVLinkBandwidthBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("NVLinkBandwidthBenchmark: no peers, skipping");
    return {};
  }

  if (config.loopCount < 1) {
    UNIFLOW_LOG_ERROR(
        "NVLinkBandwidthBenchmark: loopCount must be >= 1, got {}",
        config.loopCount);
    return {};
  }

  auto session = setupTransport(config, peers, bootstrap);
  if (!session) {
    return {};
  }

  // In unidirectional mode only rank 0 issues transfers while rank 1 just
  // participates in barriers (matching nvbandwidth methodology).
  const bool isActiveRank = config.bidirectional || bootstrap.isRank0();

  UNIFLOW_LOG_INFO(
      "NVLinkBandwidthBenchmark: rank {} setup complete, sweeping sizes {}-{}"
      " (loopCount={}, {}directional, active={})",
      bootstrap.rank,
      config.minSize,
      config.maxSize,
      config.loopCount,
      config.bidirectional ? "bi" : "uni",
      isActiveRank);

  if (isActiveRank) {
    if (!prefaultAndVerify(*session, config.maxSize)) {
      return {};
    }
  }

  auto results = runBenchmarkLoop(
      config, peers, bootstrap, *session, name(), isActiveRank);

  // Ensure both ranks finish before shutting down.
  auto shutdownBarrier = barrier(peers, bootstrap);
  if (!shutdownBarrier) {
    UNIFLOW_LOG_WARN(
        "NVLinkBandwidthBenchmark: shutdown barrier failed: {}",
        shutdownBarrier.error().toString());
  }
  session->transport->shutdown();
  return results;
}

} // namespace uniflow::benchmark
