// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/RdmaBandwidthBenchmark.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <optional>
#include <utility>

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/SegmentHelper.h"
#include "comms/uniflow/benchmarks/Stats.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

namespace uniflow::benchmark {

namespace {

std::vector<std::string> discoverRdmaDevices(
    const std::shared_ptr<IbvApi>& ibvApi) {
  int numDevices = 0;
  auto devResult = ibvApi->getDeviceList(&numDevices);
  if (!devResult.hasValue() || numDevices == 0) {
    return {};
  }
  auto* deviceList = devResult.value();
  std::vector<std::string> names;
  for (int i = 0; i < numDevices; ++i) {
    auto nameResult = ibvApi->getDeviceName(deviceList[i]);
    if (nameResult.hasValue()) {
      names.emplace_back(nameResult.value());
    }
  }
  ibvApi->freeDeviceList(deviceList);
  return names;
}

struct BenchmarkBuffers {
  void* src{nullptr};
  void* dst{nullptr};
  bool useGpu{false}; // Controls deallocation path (cudaFree vs std::free).
  MemoryType memType{MemoryType::DRAM};
  int gpuDevice{0};

  BenchmarkBuffers() = default;
  ~BenchmarkBuffers() {
    release();
  }

  BenchmarkBuffers(BenchmarkBuffers&& o) noexcept
      : src(std::exchange(o.src, nullptr)),
        dst(std::exchange(o.dst, nullptr)),
        useGpu(o.useGpu),
        memType(o.memType),
        gpuDevice(o.gpuDevice) {}

  BenchmarkBuffers(const BenchmarkBuffers&) = delete;
  BenchmarkBuffers& operator=(const BenchmarkBuffers&) = delete;
  BenchmarkBuffers& operator=(BenchmarkBuffers&&) = delete;

 private:
  void release() noexcept {
    if (useGpu) {
      if (src) {
        cudaFree(src);
      }
      if (dst) {
        cudaFree(dst);
      }
    } else {
      std::free(src);
      std::free(dst);
    }
    src = dst = nullptr;
  }
};

std::optional<BenchmarkBuffers>
allocateBuffers(size_t maxSize, int cudaDevice, int rank) {
  BenchmarkBuffers bufs;
  bufs.useGpu = cudaDevice >= 0;
  bufs.memType = bufs.useGpu ? MemoryType::VRAM : MemoryType::DRAM;
  bufs.gpuDevice = bufs.useGpu ? cudaDevice : 0;

  if (bufs.useGpu) {
    auto cudaRet = cudaSetDevice(bufs.gpuDevice);
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaSetDevice({}) failed: {}",
          bufs.gpuDevice,
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
    cudaRet = cudaMalloc(&bufs.src, maxSize);
    if (cudaRet != cudaSuccess || bufs.src == nullptr) {
      UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: cudaMalloc(src) failed");
      return std::nullopt;
    }
    cudaRet = cudaMalloc(&bufs.dst, maxSize);
    if (cudaRet != cudaSuccess || bufs.dst == nullptr) {
      UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: cudaMalloc(dst) failed");
      return std::nullopt;
    }
    cudaRet = cudaMemset(bufs.src, 0xAB, maxSize);
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaMemset(src) failed: {}",
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
    cudaRet = cudaMemset(bufs.dst, 0, maxSize);
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaMemset(dst) failed: {}",
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
    cudaRet = cudaDeviceSynchronize();
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaDeviceSynchronize failed: {}",
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
    UNIFLOW_LOG_INFO(
        "RdmaBandwidthBenchmark: rank {} using GPU {} memory",
        rank,
        bufs.gpuDevice);
  } else {
    bufs.src = std::malloc(maxSize);
    bufs.dst = std::malloc(maxSize);
    if (bufs.src == nullptr || bufs.dst == nullptr) {
      UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: malloc failed");
      return std::nullopt;
    }
    std::memset(bufs.src, 0xAB, maxSize);
    std::memset(bufs.dst, 0, maxSize);
    UNIFLOW_LOG_INFO(
        "RdmaBandwidthBenchmark: rank {} using CPU (DRAM) memory", rank);
  }

  return bufs;
}

struct TransportSession {
  // Factory must be declared BEFORE transport so it is destroyed AFTER the
  // transport.  The factory owns the ibv_context and PD that the transport's
  // QPs and MRs depend on; destroying it first would invalidate those
  // resources (ibv_close_device closes the kernel fd, tearing down all
  // associated QPs/MRs/CQs).
  std::unique_ptr<RdmaTransportFactory> factory;
  std::unique_ptr<Transport> transport;
  RegisteredSegment localReg;
  RemoteRegisteredSegment remoteReg;
  // The local destination MR must stay alive so the remote side's rkey
  // remains valid for RDMA writes.  Without this, ibv_dereg_mr fires
  // when setupTransport returns and the remote gets R_Key violation.
  std::unique_ptr<RegistrationHandle> localDstReg;
};

std::optional<TransportSession> setupTransport(
    const std::string& device,
    const BenchmarkBuffers& bufs,
    size_t maxSize,
    ScopedEventBaseThread& evbThread,
    const std::shared_ptr<IbvApi>& ibvApi,
    PeerConnection& peer,
    const BootstrapConfig& bootstrap,
    size_t chunkSize) {
  RdmaTransportConfig rdmaConfig{};
  rdmaConfig.chunkSize = chunkSize;

  auto cudaDriverApi = std::make_shared<CudaDriverApi>();
  auto factory = std::make_unique<RdmaTransportFactory>(
      std::vector<std::string>{device},
      evbThread.getEventBase(),
      rdmaConfig,
      ibvApi,
      cudaDriverApi);

  auto localTopology = factory->getTopology();
  auto remoteTopologyResult =
      exchangeMetadata(*peer.ctrl, localTopology, bootstrap.isRank0());
  if (!remoteTopologyResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: topology exchange failed: {}",
        remoteTopologyResult.error().toString());
    return std::nullopt;
  }

  auto transportResult =
      factory->createTransport(std::move(remoteTopologyResult).value());
  if (!transportResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: createTransport failed: {}",
        transportResult.error().toString());
    return std::nullopt;
  }
  auto transport = std::move(transportResult).value();

  auto localInfo = transport->bind();
  auto remoteInfoResult =
      exchangeMetadata(*peer.ctrl, localInfo, bootstrap.isRank0());
  if (!remoteInfoResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: transport info exchange failed: {}",
        remoteInfoResult.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  auto connectStatus = transport->connect(std::move(remoteInfoResult).value());
  if (!connectStatus) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: connect failed: {}",
        connectStatus.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  Segment srcSeg(bufs.src, maxSize, bufs.memType, bufs.gpuDevice);
  Segment dstSeg(bufs.dst, maxSize, bufs.memType, bufs.gpuDevice);

  auto srcRegResult = factory->registerSegment(srcSeg);
  if (!srcRegResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: registerSegment(src) failed: {}",
        srcRegResult.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  auto dstRegResult = factory->registerSegment(dstSeg);
  if (!dstRegResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: registerSegment(dst) failed: {}",
        dstRegResult.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  UNIFLOW_LOG_INFO(
      "setupTransport: src={:#x} dst={:#x} maxSize={} memType={}",
      reinterpret_cast<uintptr_t>(bufs.src),
      reinterpret_cast<uintptr_t>(bufs.dst),
      maxSize,
      static_cast<int>(bufs.memType));
  auto dstPayload = dstRegResult.value()->serialize();
  uint64_t dstAddr = reinterpret_cast<uint64_t>(bufs.dst);
  std::vector<uint8_t> dstPayloadWithAddr(sizeof(dstAddr) + dstPayload.size());
  std::memcpy(dstPayloadWithAddr.data(), &dstAddr, sizeof(dstAddr));
  std::memcpy(
      dstPayloadWithAddr.data() + sizeof(dstAddr),
      dstPayload.data(),
      dstPayload.size());

  auto remotePayloadResult =
      exchangeMetadata(*peer.ctrl, dstPayloadWithAddr, bootstrap.isRank0());
  if (!remotePayloadResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: registration exchange failed: {}",
        remotePayloadResult.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  auto& remotePayload = remotePayloadResult.value();
  if (remotePayload.size() < sizeof(uint64_t)) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: remote payload too small: {}",
        remotePayload.size());
    transport->shutdown();
    return std::nullopt;
  }
  uint64_t remoteDstAddr = 0;
  std::memcpy(&remoteDstAddr, remotePayload.data(), sizeof(remoteDstAddr));
  UNIFLOW_LOG_INFO(
      "setupTransport: remoteDstAddr={:#x} payloadSize={}",
      remoteDstAddr,
      remotePayload.size());
  std::vector<uint8_t> remoteRegPayload(
      remotePayload.begin() + sizeof(remoteDstAddr), remotePayload.end());

  auto remoteHandleResult =
      factory->importSegment(maxSize, std::move(remoteRegPayload));
  if (!remoteHandleResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: importSegment failed: {}",
        remoteHandleResult.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  auto localReg =
      SegmentTest::makeRegistered(srcSeg, std::move(srcRegResult.value()));
  auto remoteReg = SegmentTest::makeRemote(
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(remoteDstAddr),
      maxSize,
      std::move(remoteHandleResult.value()));

  return TransportSession{
      std::move(factory),
      std::move(transport),
      std::move(localReg),
      std::move(remoteReg),
      std::move(dstRegResult.value()),
  };
}

/// Batched put/get: pass batchSize requests per put() call, measure the
/// latency of the whole batch, divide by batchSize for per-op latency.
std::vector<BenchmarkResult> runBenchmarkLoop(
    Transport& transport,
    RegisteredSegment& localReg,
    RemoteRegisteredSegment& remoteReg,
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    const std::string& benchmarkName) {
  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;
  const int batchSize = std::max(1, config.batchSize);

  const bool isActiveRank = config.bidirectional || bootstrap.isRank0();

  auto runDirection = [&](const std::string& dir) {
    for (auto size : sizes) {
      auto barrierStatus = barrier(peers, bootstrap);
      if (!barrierStatus) {
        UNIFLOW_LOG_ERROR(
            "RdmaBandwidthBenchmark: barrier failed: {}",
            barrierStatus.error().toString());
        return;
      }

      if (!isActiveRank) {
        continue;
      }

      std::vector<TransferRequest> batch(
          batchSize,
          TransferRequest{
              .local = localReg.span(size_t{0}, size),
              .remote = remoteReg.span(size_t{0}, size),
          });

      int numBatches =
          std::max(1, (config.iterations + batchSize - 1) / batchSize);
      // totalOps may exceed config.iterations when iterations is not evenly
      // divisible by batchSize (we round up to complete batches).
      int totalOps = numBatches * batchSize;

      auto submitBatch = [&]() -> Status {
        auto fut = (dir == "put") ? transport.put(batch, {})
                                  : transport.get(batch, {});
        return fut.get();
      };

      for (int iter = 0; iter < config.warmupIterations; ++iter) {
        auto status = submitBatch();
        if (status.hasError()) {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: warmup {} failed at size {}: {}",
              dir,
              size,
              status.error().message());
          return;
        }
      }

      std::vector<double> latenciesUs;
      latenciesUs.reserve(numBatches);

      auto overallStart = std::chrono::steady_clock::now();

      for (int b = 0; b < numBatches; ++b) {
        auto t0 = std::chrono::steady_clock::now();
        auto status = submitBatch();
        auto t1 = std::chrono::steady_clock::now();
        if (status.hasError()) {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: {} failed at size {}: {}",
              dir,
              size,
              status.error().message());
          return;
        }
        double batchUs =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
        latenciesUs.push_back(batchUs / batchSize);
      }

      auto overallEnd = std::chrono::steady_clock::now();

      double totalTimeSec =
          std::chrono::duration<double>(overallEnd - overallStart).count();
      double totalBytes =
          static_cast<double>(size) * static_cast<double>(totalOps);
      double bandwidthGBs =
          (totalTimeSec > 0) ? (totalBytes / totalTimeSec) / 1e9 : 0;
      double msgRateMops =
          (totalTimeSec > 0) ? (totalOps / totalTimeSec) / 1e6 : 0;

      auto stats = Stats::compute(std::move(latenciesUs));

      results.push_back({
          .benchmarkName = benchmarkName,
          .transport = "rdma",
          .direction = dir,
          .messageSize = size,
          .iterations = totalOps,
          .batchSize = batchSize,
          .chunkSize = config.chunkSize,
          .bandwidthGBs = bandwidthGBs,
          .latency = stats,
          .messageRateMops = msgRateMops,
      });

      fprintf(
          stderr,
          "[rank %d] %s size=%-10zu batch=%-3d iters=%-6d "
          "bw=%.2f GB/s  avg=%.1f us  %s\n",
          bootstrap.rank,
          dir.c_str(),
          size,
          batchSize,
          totalOps,
          bandwidthGBs,
          stats.avg,
          config.bidirectional ? "(bidirectional)" : "(unidirectional)");
    }
  };

  if (config.direction == "put" || config.direction == "both") {
    runDirection("put");
  }
  if (config.direction == "get" || config.direction == "both") {
    runDirection("get");
  }

  return results;
}

} // anonymous namespace

std::vector<BenchmarkResult> RdmaBandwidthBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("RdmaBandwidthBenchmark: no peers, skipping");
    return {};
  }

  auto ibvApi = std::make_shared<IbvApi>();
  auto initStatus = ibvApi->init();
  if (initStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: IbvApi init failed: {}",
        initStatus.error().message());
    return {};
  }

  std::vector<std::string> deviceNames = rdmaDevices_;
  if (deviceNames.empty()) {
    deviceNames = discoverRdmaDevices(ibvApi);
  }
  if (deviceNames.empty()) {
    UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: no RDMA devices found");
    return {};
  }

  std::string myDevice;
  if (static_cast<size_t>(bootstrap.localRank) < deviceNames.size()) {
    myDevice = deviceNames[bootstrap.localRank];
  } else {
    myDevice = deviceNames[0];
  }
  UNIFLOW_LOG_INFO(
      "RdmaBandwidthBenchmark: rank {} using RDMA device {}",
      bootstrap.rank,
      myDevice);

  auto bufs =
      allocateBuffers(config.maxSize, config.cudaDevice, bootstrap.rank);
  if (!bufs) {
    return {};
  }

  // evbThread must outlive the transport (transport posts async work to it).
  ScopedEventBaseThread evbThread("bench-evb");
  auto session = setupTransport(
      myDevice,
      *bufs,
      config.maxSize,
      evbThread,
      ibvApi,
      peers[0],
      bootstrap,
      config.chunkSize);
  if (!session) {
    return {};
  }

  auto results = runBenchmarkLoop(
      *session->transport,
      session->localReg,
      session->remoteReg,
      config,
      peers,
      bootstrap,
      name());

  auto finalBarrier = barrier(peers, bootstrap);
  if (!finalBarrier) {
    UNIFLOW_LOG_WARN(
        "RdmaBandwidthBenchmark: final barrier failed: {}",
        finalBarrier.error().toString());
  }
  session->transport->shutdown();
  return results;
}

} // namespace uniflow::benchmark
