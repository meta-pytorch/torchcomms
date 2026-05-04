// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/RdmaBandwidthBenchmark.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
#include "comms/uniflow/transport/Topology.h"
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
  // Per-NIC separate allocations to avoid PCIe DMA contention when
  // multiple NICs read from the same GPU memory region simultaneously.
  std::vector<void*> srcs;
  std::vector<void*> dsts;
  bool useGpu{false};
  MemoryType memType{MemoryType::DRAM};
  int gpuDevice{0};

  BenchmarkBuffers() = default;
  ~BenchmarkBuffers() {
    release();
  }

  BenchmarkBuffers(BenchmarkBuffers&& o) noexcept
      : srcs(std::move(o.srcs)),
        dsts(std::move(o.dsts)),
        useGpu(o.useGpu),
        memType(o.memType),
        gpuDevice(o.gpuDevice) {}

  BenchmarkBuffers(const BenchmarkBuffers&) = delete;
  BenchmarkBuffers& operator=(const BenchmarkBuffers&) = delete;
  BenchmarkBuffers& operator=(BenchmarkBuffers&&) = delete;

 private:
  void release() noexcept {
    auto freeOne = [this](void* p) {
      if (p) {
        if (useGpu) {
          cudaFree(p);
        } else {
          std::free(p);
        }
      }
    };
    for (auto* p : srcs) {
      freeOne(p);
    }
    for (auto* p : dsts) {
      freeOne(p);
    }
    srcs.clear();
    dsts.clear();
  }
};

std::optional<BenchmarkBuffers>
allocateBuffers(size_t maxSize, int cudaDevice, int rank, int numBuffers) {
  BenchmarkBuffers bufs;
  bufs.useGpu = cudaDevice >= 0;
  bufs.memType = bufs.useGpu ? MemoryType::VRAM : MemoryType::DRAM;
  bufs.gpuDevice = bufs.useGpu ? cudaDevice : 0;

  auto allocOne = [&](void** out, uint8_t fill) -> bool {
    if (bufs.useGpu) {
      auto ret = cudaMalloc(out, maxSize);
      if (ret != cudaSuccess || *out == nullptr) {
        UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: cudaMalloc failed");
        return false;
      }
      ret = cudaMemset(*out, fill, maxSize);
      if (ret != cudaSuccess) {
        UNIFLOW_LOG_ERROR(
            "RdmaBandwidthBenchmark: cudaMemset failed: {}",
            cudaGetErrorString(ret));
        cudaFree(*out);
        *out = nullptr;
        return false;
      }
    } else {
      *out = std::malloc(maxSize);
      if (*out == nullptr) {
        UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: malloc failed");
        return false;
      }
      std::memset(*out, fill, maxSize);
    }
    return true;
  };

  if (bufs.useGpu) {
    auto cudaRet = cudaSetDevice(bufs.gpuDevice);
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaSetDevice({}) failed: {}",
          bufs.gpuDevice,
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
  }

  for (int i = 0; i < numBuffers; ++i) {
    void* src = nullptr;
    if (!allocOne(&src, 0xAB)) {
      return std::nullopt;
    }
    bufs.srcs.push_back(src);
    void* dst = nullptr;
    if (!allocOne(&dst, 0x00)) {
      return std::nullopt;
    }
    bufs.dsts.push_back(dst);
  }

  if (bufs.useGpu) {
    auto cudaRet = cudaDeviceSynchronize();
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaDeviceSynchronize failed: {}",
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
  }

  UNIFLOW_LOG_INFO(
      "RdmaBandwidthBenchmark: rank {} allocated {} buffer pairs ({} memory)",
      rank,
      numBuffers,
      bufs.useGpu ? "GPU" : "CPU");
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
  std::vector<RegisteredSegment> localRegs;
  std::vector<RemoteRegisteredSegment> remoteRegs;
  std::vector<std::unique_ptr<RegistrationHandle>> localDstRegs;
};

// Wire format for registration payload exchange between ranks.
// Layout: [uint64_t dstAddr | registration payload bytes]
struct RegistrationExchange {
  static std::vector<uint8_t> serialize(
      uint64_t dstAddr,
      const std::vector<uint8_t>& regPayload) {
    std::vector<uint8_t> buf(sizeof(dstAddr) + regPayload.size());
    std::memcpy(buf.data(), &dstAddr, sizeof(dstAddr));
    std::memcpy(
        buf.data() + sizeof(dstAddr), regPayload.data(), regPayload.size());
    return buf;
  }

  static std::optional<std::pair<uint64_t, std::vector<uint8_t>>> deserialize(
      const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(uint64_t)) {
      return std::nullopt;
    }
    uint64_t addr = 0;
    std::memcpy(&addr, data.data(), sizeof(addr));
    return std::make_pair(
        addr, std::vector<uint8_t>(data.begin() + sizeof(addr), data.end()));
  }
};

// Register per-NIC buffer pairs and exchange registration payloads with the
// remote peer.
bool registerBuffers(
    RdmaTransportFactory& factory,
    const BenchmarkBuffers& bufs,
    size_t maxSize,
    controller::Conn& ctrl,
    const BootstrapConfig& bootstrap,
    TransportSession& session) {
  int numBufs = static_cast<int>(bufs.srcs.size());
  if (numBufs == 0) {
    return true;
  }

  // Validate both ranks selected the same number of NICs. The loop below
  // calls exchangeMetadata once per NIC — a mismatch would deadlock.
  int32_t localCount = numBufs;
  std::vector<uint8_t> countPayload(sizeof(localCount));
  std::memcpy(countPayload.data(), &localCount, sizeof(localCount));
  auto remoteCountResult =
      exchangeMetadata(ctrl, countPayload, bootstrap.isRank0());
  if (!remoteCountResult ||
      remoteCountResult.value().size() < sizeof(int32_t)) {
    UNIFLOW_LOG_ERROR("registerBuffers: NIC count exchange failed");
    return false;
  }
  int32_t remoteCount = 0;
  std::memcpy(
      &remoteCount, remoteCountResult.value().data(), sizeof(remoteCount));
  if (localCount != remoteCount) {
    UNIFLOW_LOG_ERROR(
        "registerBuffers: NIC count mismatch (local={}, remote={})",
        localCount,
        remoteCount);
    return false;
  }

  for (int b = 0; b < numBufs; ++b) {
    Segment srcSeg(bufs.srcs[b], maxSize, bufs.memType, bufs.gpuDevice);
    Segment dstSeg(bufs.dsts[b], maxSize, bufs.memType, bufs.gpuDevice);

    auto srcRegResult = factory.registerSegment(srcSeg);
    if (!srcRegResult) {
      UNIFLOW_LOG_ERROR(
          "registerSegment(src[{}]) failed: {}",
          b,
          srcRegResult.error().toString());
      return false;
    }

    auto dstRegResult = factory.registerSegment(dstSeg);
    if (!dstRegResult) {
      UNIFLOW_LOG_ERROR(
          "registerSegment(dst[{}]) failed: {}",
          b,
          dstRegResult.error().toString());
      return false;
    }

    auto localPayload = RegistrationExchange::serialize(
        reinterpret_cast<uint64_t>(bufs.dsts[b]),
        dstRegResult.value()->serialize());

    auto remotePayloadResult =
        exchangeMetadata(ctrl, localPayload, bootstrap.isRank0());
    if (!remotePayloadResult) {
      UNIFLOW_LOG_ERROR(
          "registration exchange[{}] failed: {}",
          b,
          remotePayloadResult.error().toString());
      return false;
    }

    auto parsed =
        RegistrationExchange::deserialize(remotePayloadResult.value());
    if (!parsed) {
      UNIFLOW_LOG_ERROR(
          "remote payload[{}] too small: {}",
          b,
          remotePayloadResult.value().size());
      return false;
    }
    auto& [remoteDstAddr, remoteRegPayload] = *parsed;

    auto remoteHandleResult =
        factory.importSegment(maxSize, std::move(remoteRegPayload));
    if (!remoteHandleResult) {
      UNIFLOW_LOG_ERROR(
          "importSegment[{}] failed: {}",
          b,
          remoteHandleResult.error().toString());
      return false;
    }

    session.localRegs.push_back(
        SegmentTest::makeRegistered(srcSeg, std::move(srcRegResult.value())));
    session.remoteRegs.push_back(
        SegmentTest::makeRemote(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<void*>(remoteDstAddr),
            maxSize,
            std::move(remoteHandleResult.value())));
    session.localDstRegs.push_back(std::move(dstRegResult.value()));

    UNIFLOW_LOG_INFO(
        "registerBuffers: buf[{}] src={:#x} dst={:#x} remoteDst={:#x}",
        b,
        reinterpret_cast<uintptr_t>(bufs.srcs[b]),
        reinterpret_cast<uintptr_t>(bufs.dsts[b]),
        remoteDstAddr);
  }

  return true;
}

std::optional<TransportSession> setupTransport(
    const std::vector<std::string>& devices,
    const BenchmarkBuffers& bufs,
    size_t maxSize,
    ScopedEventBaseThread& evbThread,
    const std::shared_ptr<IbvApi>& ibvApi,
    PeerConnection& peer,
    const BootstrapConfig& bootstrap,
    size_t chunkSize) {
  RdmaTransportConfig rdmaConfig{};
  rdmaConfig.chunkSize = chunkSize;
  rdmaConfig.numQps = static_cast<uint32_t>(devices.size());

  auto cudaDriverApi = std::make_shared<CudaDriverApi>();
  auto factory = std::make_unique<RdmaTransportFactory>(
      devices, evbThread.getEventBase(), rdmaConfig, ibvApi, cudaDriverApi);

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

  TransportSession session;
  if (!registerBuffers(
          *factory, bufs, maxSize, *peer.ctrl, bootstrap, session)) {
    transport->shutdown();
    return std::nullopt;
  }

  session.factory = std::move(factory);
  session.transport = std::move(transport);
  return session;
}

/// Run batched put/get with pipelined submission (txDepth in-flight batches).
std::vector<BenchmarkResult> runBenchmarkLoop(
    Transport& transport,
    std::vector<RegisteredSegment>& localRegs,
    std::vector<RemoteRegisteredSegment>& remoteRegs,
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    const std::string& benchmarkName) {
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;
  const int batchSize = std::max(1, config.batchSize);
  const int txDepth = std::max(1, config.txDepth);
  const int numBufs = static_cast<int>(localRegs.size());

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

      // Map each request to its NIC's buffer. Must match spray()'s
      // contiguous chunk-to-QP assignment in the transport layer.
      auto nicForRequest = [&](int requestIdx) {
        return requestIdx * numBufs / batchSize;
      };
      std::vector<TransferRequest> batch;
      batch.reserve(batchSize);
      for (int i = 0; i < batchSize; ++i) {
        int bufIdx = nicForRequest(i);
        batch.push_back(
            TransferRequest{
                .local = localRegs[bufIdx].span(size_t{0}, size),
                .remote = remoteRegs[bufIdx].span(size_t{0}, size),
            });
      }

      int numBatches =
          std::max(1, (config.iterations + batchSize - 1) / batchSize);
      int totalOps = numBatches * batchSize;

      auto submitBatchAsync = [&]() -> std::future<Status> {
        return (dir == "put") ? transport.put(batch, {})
                              : transport.get(batch, {});
      };

      for (int iter = 0; iter < config.warmupIterations; ++iter) {
        auto status = submitBatchAsync().get();
        if (status.hasError()) {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: warmup {} failed at size {}: {}",
              dir,
              size,
              status.error().message());
          return;
        }
      }

      // Sliding window: keep up to txDepth batches in-flight.
      // txDepth=1 degenerates to synchronous behavior.
      std::deque<std::pair<std::future<Status>, TimePoint>> inflight;
      std::vector<double> latenciesUs;
      latenciesUs.reserve(numBatches);

      // Complete the oldest in-flight batch: get result, record latency.
      // Returns false on error after draining all remaining futures.
      auto completeOne = [&]() -> bool {
        auto& [fut, submitTime] = inflight.front();
        auto status = fut.get();
        auto completeTime = Clock::now();
        if (status.hasError()) {
          inflight.pop_front();
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: {} failed at size {}: {}",
              dir,
              size,
              status.error().message());
          for (auto& [f, _] : inflight) {
            f.wait();
          }
          return false;
        }
        double batchUs =
            std::chrono::duration<double, std::micro>(completeTime - submitTime)
                .count();
        latenciesUs.push_back(batchUs / batchSize);
        inflight.pop_front();
        return true;
      };

      auto overallStart = Clock::now();

      for (int b = 0; b < numBatches; ++b) {
        if (static_cast<int>(inflight.size()) >= txDepth) {
          if (!completeOne()) {
            return;
          }
        }
        inflight.emplace_back(submitBatchAsync(), Clock::now());
      }

      while (!inflight.empty()) {
        if (!completeOne()) {
          return;
        }
      }

      auto overallEnd = Clock::now();

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
          .txDepth = txDepth,
          .chunkSize = config.chunkSize,
          .bandwidthGBs = bandwidthGBs,
          .latency = stats,
          .messageRateMops = msgRateMops,
      });

      UNIFLOW_LOG_WARN(
          "[rank {}] {} size={:<10} batch={:<3} txdepth={:<3} iters={:<6} "
          "bw={:.2f} GB/s  avg={:.1f} us  {}",
          bootstrap.rank,
          dir,
          size,
          batchSize,
          txDepth,
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

  std::vector<std::string> myDevices;
  if (!rdmaDevices_.empty()) {
    myDevices = rdmaDevices_;
  } else {
    auto& topo = Topology::get();
    if (topo.available()) {
      myDevices = (config.cudaDevice < 0)
          ? topo.selectCpuNics()
          : topo.selectGpuNics(config.cudaDevice);
    }
    if (myDevices.empty()) {
      myDevices = deviceNames;
    }
    if (config.numNics > 0 &&
        config.numNics < static_cast<int>(myDevices.size())) {
      myDevices.resize(config.numNics);
    }
  }
  {
    std::string devList;
    for (const auto& d : myDevices) {
      if (!devList.empty()) {
        devList += ", ";
      }
      devList += d;
    }
    UNIFLOW_LOG_WARN(
        "RdmaBandwidthBenchmark: rank {} RDMA device(s): {}",
        bootstrap.rank,
        devList);
  }

  int numNics = static_cast<int>(myDevices.size());
  auto bufs = allocateBuffers(
      config.maxSize, config.cudaDevice, bootstrap.rank, numNics);
  if (!bufs) {
    return {};
  }

  assert(
      bufs->srcs.size() == myDevices.size() &&
      "Buffer count must equal NIC/QP count");

  ScopedEventBaseThread evbThread("bench-evb");
  auto session = setupTransport(
      myDevices,
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
      session->localRegs,
      session->remoteRegs,
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
