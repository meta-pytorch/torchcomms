// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/SendRecvBandwidthBenchmark.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <optional>
#include <utility>

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/uniflow/Segment.h"
#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/benchmarks/Stats.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
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

struct Buffers {
  void* src{nullptr};
  void* dst{nullptr};
  bool useGpu{false};
  MemoryType memType{MemoryType::DRAM};
  int gpuDevice{0};

  Buffers() = default;
  ~Buffers() {
    release();
  }

  Buffers(Buffers&& o) noexcept
      : src(std::exchange(o.src, nullptr)),
        dst(std::exchange(o.dst, nullptr)),
        useGpu(o.useGpu),
        memType(o.memType),
        gpuDevice(o.gpuDevice) {}

  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  Buffers& operator=(Buffers&&) = delete;

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
    freeOne(src);
    freeOne(dst);
    src = nullptr;
    dst = nullptr;
  }
};

std::optional<Buffers>
allocateBuffers(size_t maxSize, int cudaDevice, int rank) {
  Buffers bufs;
  bufs.useGpu = cudaDevice >= 0;
  bufs.memType = bufs.useGpu ? MemoryType::VRAM : MemoryType::DRAM;
  bufs.gpuDevice = bufs.useGpu ? cudaDevice : 0;

  auto allocOne = [&](void** out, uint8_t fill) -> bool {
    if (bufs.useGpu) {
      auto ret = cudaMalloc(out, maxSize);
      if (ret != cudaSuccess || *out == nullptr) {
        UNIFLOW_LOG_ERROR("SendRecvBandwidthBenchmark: cudaMalloc failed");
        return false;
      }
      ret = cudaMemset(*out, fill, maxSize);
      if (ret != cudaSuccess) {
        UNIFLOW_LOG_ERROR(
            "SendRecvBandwidthBenchmark: cudaMemset failed: {}",
            cudaGetErrorString(ret));
        cudaFree(*out);
        *out = nullptr;
        return false;
      }
    } else {
      *out = std::malloc(maxSize);
      if (*out == nullptr) {
        UNIFLOW_LOG_ERROR("SendRecvBandwidthBenchmark: malloc failed");
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
          "SendRecvBandwidthBenchmark: cudaSetDevice({}) failed: {}",
          bufs.gpuDevice,
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
  }

  if (!allocOne(&bufs.src, 0xAB) || !allocOne(&bufs.dst, 0x00)) {
    return std::nullopt;
  }

  if (bufs.useGpu) {
    auto cudaRet = cudaDeviceSynchronize();
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "SendRecvBandwidthBenchmark: cudaDeviceSynchronize failed: {}",
          cudaGetErrorString(cudaRet));
      return std::nullopt;
    }
  }

  UNIFLOW_LOG_INFO(
      "SendRecvBandwidthBenchmark: rank {} allocated buffers ({} memory)",
      rank,
      bufs.useGpu ? "GPU" : "CPU");
  return bufs;
}

// --- Multi-transport session: one factory, N peer transports ---

struct PeerTransport {
  std::unique_ptr<Transport> transport;
  cudaStream_t stream{nullptr};
  RequestOptions opts;
};

struct MultiTransportSession {
  // EventBase thread must outlive factory which must outlive transports.
  std::unique_ptr<ScopedEventBaseThread> evbThread;
  std::unique_ptr<RdmaTransportFactory> factory;
  std::vector<PeerTransport> peerTransports;

  MultiTransportSession() = default;
  MultiTransportSession(MultiTransportSession&&) = default;
  MultiTransportSession(const MultiTransportSession&) = delete;
  MultiTransportSession& operator=(const MultiTransportSession&) = delete;
  MultiTransportSession& operator=(MultiTransportSession&&) = delete;

  ~MultiTransportSession() {
    for (auto& pt : peerTransports) {
      if (pt.transport) {
        pt.transport->shutdown();
      }
      if (pt.stream) {
        cudaStreamDestroy(pt.stream);
      }
    }
  }
};

std::optional<std::unique_ptr<Transport>> connectOneTransport(
    RdmaTransportFactory& factory,
    controller::Conn& ctrl,
    bool isRank0,
    int peerRank) {
  auto localTopology = factory.getTopology();
  auto remoteTopoResult = exchangeMetadata(ctrl, localTopology, isRank0);
  if (!remoteTopoResult) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: topology exchange with peer {} "
        "failed: {}",
        peerRank,
        remoteTopoResult.error().toString());
    return std::nullopt;
  }

  auto transportResult =
      factory.createTransport(std::move(remoteTopoResult).value());
  if (!transportResult) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: createTransport for peer {} "
        "failed: {}",
        peerRank,
        transportResult.error().toString());
    return std::nullopt;
  }
  auto transport = std::move(transportResult).value();

  auto localInfo = transport->bind();
  auto remoteInfoResult = exchangeMetadata(ctrl, localInfo, isRank0);
  if (!remoteInfoResult) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: info exchange with peer {} "
        "failed: {}",
        peerRank,
        remoteInfoResult.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  auto connectStatus = transport->connect(std::move(remoteInfoResult).value());
  if (!connectStatus) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: connect to peer {} failed: {}",
        peerRank,
        connectStatus.error().toString());
    transport->shutdown();
    return std::nullopt;
  }

  return transport;
}

std::optional<MultiTransportSession> setupMultiTransport(
    const std::vector<std::string>& devices,
    const std::shared_ptr<IbvApi>& ibvApi,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    const BenchmarkConfig& config,
    bool useGpu) {
  RdmaTransportConfig rdmaConfig{};
  rdmaConfig.chunkSize = config.chunkSize;
  rdmaConfig.numQps = static_cast<uint32_t>(devices.size());
  rdmaConfig.pipelineDepth = static_cast<uint16_t>(config.pipelineDepth);
  const auto numPeers = std::max(size_t{1}, peers.size());
  const auto pipelineDepth =
      static_cast<size_t>(std::max(1, config.pipelineDepth));
  rdmaConfig.slabPoolConfig.slabSize =
      config.slabSize > 0 ? config.slabSize : config.chunkSize;
  rdmaConfig.slabPoolConfig.slabNum =
      config.slabNum > 0 ? config.slabNum : pipelineDepth * numPeers;
  auto cudaDriverApi = std::make_shared<CudaDriverApi>();

  MultiTransportSession session;
  session.evbThread = std::make_unique<ScopedEventBaseThread>("bench-evb");
  session.factory = std::make_unique<RdmaTransportFactory>(
      devices,
      session.evbThread->getEventBase(),
      rdmaConfig,
      ibvApi,
      cudaDriverApi);

  session.peerTransports.reserve(peers.size());

  for (size_t i = 0; i < peers.size(); ++i) {
    auto& peer = peers[i];
    auto t = connectOneTransport(
        *session.factory, *peer.ctrl, bootstrap.isRank0(), peer.peerRank);
    if (!t) {
      return std::nullopt;
    }

    PeerTransport pt;
    pt.transport = std::move(*t);
    if (useGpu) {
      auto ret = cudaStreamCreate(&pt.stream);
      if (ret != cudaSuccess) {
        UNIFLOW_LOG_ERROR(
            "SendRecvBandwidthBenchmark: cudaStreamCreate failed for peer {}: {}",
            peer.peerRank,
            cudaGetErrorString(ret));
        pt.transport->shutdown();
        return std::nullopt;
      }
      pt.opts.stream = static_cast<void*>(pt.stream);
    }

    UNIFLOW_LOG_INFO(
        "SendRecvBandwidthBenchmark: rank {} connected to peer {}",
        bootstrap.rank,
        peer.peerRank);
    session.peerTransports.push_back(std::move(pt));
  }

  return session;
}

// --- Benchmark loop: topology-driven concurrent send/recv ---

std::vector<BenchmarkResult> runBenchmarkLoop(
    MultiTransportSession& session,
    const Buffers& bufs,
    size_t maxSize,
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    const std::string& benchmarkName) {
  using Clock = std::chrono::steady_clock;

  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;
  const int txDepth = std::max(1, config.txDepth);
  const int numPeers = static_cast<int>(session.peerTransports.size());
  const auto& topology = config.topology;

  if (numPeers == 0) {
    UNIFLOW_LOG_ERROR("SendRecvBandwidthBenchmark: numPeers == 0");
    return {};
  }

  bool doSend = (topology == "fanout" && bootstrap.isRank0()) ||
      (topology == "fanin" && !bootstrap.isRank0());

  Segment srcSeg(bufs.src, maxSize, bufs.memType, bufs.gpuDevice);
  Segment dstSeg(bufs.dst, maxSize, bufs.memType, bufs.gpuDevice);

  struct InflightOp {
    std::future<Status> fut;
  };
  struct PeerState {
    std::deque<InflightOp> inflight;
  };

  for (auto size : sizes) {
    auto barrierStatus = barrier(peers, bootstrap);
    if (!barrierStatus) {
      UNIFLOW_LOG_ERROR(
          "SendRecvBandwidthBenchmark: barrier failed: {}",
          barrierStatus.error().toString());
      break;
    }

    auto srcSpan = srcSeg.span(size_t{0}, size);
    auto dstSpan = dstSeg.span(size_t{0}, size);

    // Warmup — launch ops on all peers, wait for all.
    bool warmupFailed = false;
    for (int w = 0; w < config.warmupIterations && !warmupFailed; ++w) {
      std::vector<std::future<Status>> futs;
      for (int p = 0; p < numPeers; ++p) {
        auto& pt = session.peerTransports[p];
        if (doSend) {
          futs.push_back(pt.transport->send(srcSpan, pt.opts));
        } else {
          futs.push_back(pt.transport->recv(dstSpan, pt.opts));
        }
      }
      for (auto& f : futs) {
        auto st = f.get();
        if (st.hasError()) {
          UNIFLOW_LOG_ERROR(
              "SendRecvBandwidthBenchmark: warmup failed at size {}: {}",
              size,
              st.error().message());
          warmupFailed = true;
        }
      }
    }
    if (warmupFailed) {
      break;
    }

    // Per-peer pipelining state.
    std::vector<PeerState> peerStates(numPeers);
    bool hadError = false;

    auto drainAll = [&]() {
      for (auto& ps : peerStates) {
        for (auto& op : ps.inflight) {
          op.fut.wait();
        }
        ps.inflight.clear();
      }
    };

    auto completePeer = [&](int p) -> bool {
      auto& ps = peerStates[p];
      auto st = ps.inflight.front().fut.get();
      if (st.hasError()) {
        UNIFLOW_LOG_ERROR(
            "SendRecvBandwidthBenchmark: op for peer {} failed at "
            "size {}: {}",
            p,
            size,
            st.error().message());
        ps.inflight.pop_front();
        drainAll();
        return false;
      }
      ps.inflight.pop_front();
      return true;
    };

    auto overallStart = Clock::now();

    for (int iter = 0; iter < config.iterations && !hadError; ++iter) {
      for (int p = 0; p < numPeers && !hadError; ++p) {
        auto& ps = peerStates[p];
        if (static_cast<int>(ps.inflight.size()) >= txDepth) {
          if (!completePeer(p)) {
            hadError = true;
            break;
          }
        }
        auto& pt = session.peerTransports[p];
        InflightOp op;
        if (doSend) {
          op.fut = pt.transport->send(srcSpan, pt.opts);
        } else {
          op.fut = pt.transport->recv(dstSpan, pt.opts);
        }
        ps.inflight.push_back(std::move(op));
      }
    }

    for (int p = 0; p < numPeers && !hadError; ++p) {
      while (!peerStates[p].inflight.empty()) {
        if (!completePeer(p)) {
          hadError = true;
        }
      }
    }

    if (hadError) {
      break;
    }

    auto overallEnd = Clock::now();

    double totalTimeSec =
        std::chrono::duration<double>(overallEnd - overallStart).count();
    double totalBytes = static_cast<double>(size) *
        static_cast<double>(config.iterations) * numPeers;
    double bandwidthGBs =
        (totalTimeSec > 0) ? (totalBytes / totalTimeSec) / 1e9 : 0;
    double msgRateMops = (totalTimeSec > 0)
        ? (static_cast<double>(config.iterations) * numPeers / totalTimeSec) /
            1e6
        : 0;
    double avgLatencyUs =
        (totalTimeSec > 0) ? (totalTimeSec * 1e6) / config.iterations : 0;

    results.push_back({
        .benchmarkName = benchmarkName,
        .transport = "rdma",
        .direction = topology,
        .messageSize = size,
        .iterations = config.iterations,
        .batchSize = 1,
        .txDepth = txDepth,
        .chunkSize = config.chunkSize,
        .bandwidthGBs = bandwidthGBs,
        .latency =
            {.min = avgLatencyUs,
             .max = avgLatencyUs,
             .avg = avgLatencyUs,
             .p50 = avgLatencyUs,
             .p99 = avgLatencyUs},
        .messageRateMops = msgRateMops,
        .numPeers = numPeers,
    });

    UNIFLOW_LOG_WARN(
        "[rank {}] {} peers={} size={:<10} txdepth={:<3} iters={:<6} "
        "aggBw={:.2f} GB/s  perLink={:.2f} GB/s  avg={:.1f} us",
        bootstrap.rank,
        topology,
        numPeers,
        size,
        txDepth,
        config.iterations,
        bandwidthGBs,
        numPeers > 0 ? bandwidthGBs / numPeers : 0.0,
        avgLatencyUs);
  }

  return results;
}

} // anonymous namespace

std::vector<BenchmarkResult> SendRecvBandwidthBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("SendRecvBandwidthBenchmark: no peers, skipping");
    return {};
  }

  if (config.topology != "fanout") {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: unsupported topology '{}' "
        "(only 'fanout' is currently implemented)",
        config.topology);
    return {};
  }

  auto ibvApi = std::make_shared<IbvApi>();
  auto initStatus = ibvApi->init();
  if (initStatus.hasError()) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: IbvApi init failed: {}",
        initStatus.error().message());
    return {};
  }

  std::vector<std::string> deviceNames = rdmaDevices_;
  if (deviceNames.empty()) {
    deviceNames = discoverRdmaDevices(ibvApi);
  }
  if (deviceNames.empty()) {
    UNIFLOW_LOG_ERROR("SendRecvBandwidthBenchmark: no RDMA devices found");
    return {};
  }

  std::vector<std::string> myDevices;
  const char* nicSelectionPolicy = "unknown";
  if (!rdmaDevices_.empty()) {
    myDevices = rdmaDevices_;
    nicSelectionPolicy = "explicit (--rdma-devices)";
  } else {
    auto& topo = sharedTopology();
    if (topo.available()) {
      if (config.cudaDevice < 0) {
        myDevices = topo.selectCpuNics();
        nicSelectionPolicy = "topology CPU-local";
      } else {
        myDevices = topo.selectGpuNics(config.cudaDevice);
        nicSelectionPolicy = "topology GPU-local";
      }
    }
    if (myDevices.empty()) {
      myDevices = deviceNames;
      nicSelectionPolicy = "fallback (all devices)";
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
        "SendRecvBandwidthBenchmark: rank {} using {} RDMA device(s): {} "
        "[selection={}] with {} peer(s), topology={}",
        bootstrap.rank,
        myDevices.size(),
        devList,
        nicSelectionPolicy,
        peers.size(),
        config.topology);
  }

  bool useGpu = config.cudaDevice >= 0;
  auto bufs =
      allocateBuffers(config.maxSize, config.cudaDevice, bootstrap.rank);
  if (!bufs) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: buffer allocation failed on rank {}",
        bootstrap.rank);
    return {};
  }

  auto session =
      setupMultiTransport(myDevices, ibvApi, peers, bootstrap, config, useGpu);
  if (!session) {
    UNIFLOW_LOG_ERROR(
        "SendRecvBandwidthBenchmark: transport setup failed on rank {}",
        bootstrap.rank);
    return {};
  }

  UNIFLOW_LOG_INFO(
      "SendRecvBandwidthBenchmark: rank {} set up {} transport(s)",
      bootstrap.rank,
      session->peerTransports.size());

  auto results = runBenchmarkLoop(
      *session, *bufs, config.maxSize, config, peers, bootstrap, name());

  auto finalBarrier = barrier(peers, bootstrap);
  if (!finalBarrier) {
    UNIFLOW_LOG_WARN(
        "SendRecvBandwidthBenchmark: final barrier failed: {}",
        finalBarrier.error().toString());
  }
  return results;
}

} // namespace uniflow::benchmark
