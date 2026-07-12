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
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/ibverbs/IbvApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/logging/Logger.h"
#include "comms/uniflow/transport/Topology.h"
#include "comms/uniflow/transport/rdma/RdmaResources.h"
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
  /*
   * Per-NIC separate allocations to avoid PCIe DMA contention when multiple
   * NICs read from the same GPU memory region simultaneously. srcs/dsts are the
   * (aligned) registration pointers; rawAllocs holds the original allocation
   * bases to free (an aligned buffer is an offset into a larger raw allocation,
   * so free the raw base, not the aligned pointer).
   */
  std::vector<void*> srcs;
  std::vector<void*> dsts;
  std::vector<void*> rawAllocs;
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
        rawAllocs(std::move(o.rawAllocs)),
        useGpu(o.useGpu),
        memType(o.memType),
        gpuDevice(o.gpuDevice) {}

  BenchmarkBuffers(const BenchmarkBuffers&) = delete;
  BenchmarkBuffers& operator=(const BenchmarkBuffers&) = delete;
  BenchmarkBuffers& operator=(BenchmarkBuffers&&) = delete;

 private:
  void release() noexcept {
    for (auto* p : rawAllocs) {
      if (p) {
        if (useGpu) {
          // hipFree is [[nodiscard]] on HIP (cudaFree is not); we are in a
          // best-effort noexcept release path, so discard the status.
          (void)cudaFree(p);
        } else {
          std::free(p);
        }
      }
    }
    srcs.clear();
    dsts.clear();
    rawAllocs.clear();
  }
};

std::optional<BenchmarkBuffers>
allocateBuffers(size_t maxSize, int cudaDevice, int rank, int numBuffers) {
  BenchmarkBuffers bufs;
  bufs.useGpu = cudaDevice >= 0;
  bufs.memType = bufs.useGpu ? MemoryType::VRAM : MemoryType::DRAM;
  bufs.gpuDevice = bufs.useGpu ? cudaDevice : 0;

  /*
   * mlx5 Data Direct on GB300 requires the registered buffer's *allocation
   * base* to be aligned to the CUDA VMM granularity (2 MiB) so the exported
   * dmabuf offset is 0; a non-zero offset makes
   * mlx5dv_reg_dmabuf_mr(DATA_DIRECT) return EOPNOTSUPP. cudaMalloc returns a
   * granularity-aligned base for allocations rounded up to the granularity (the
   * driver large-allocation path), so round the request up and use the base
   * directly -- aligning a pointer inside an over-allocation instead would
   * leave a non-zero offset relative to the true allocation base. See
   * D110262217.
   */
  constexpr size_t kVmmAlign = size_t{2} << 20; // 2 MiB VMM granularity
  const size_t allocBytes = ((maxSize + kVmmAlign - 1) / kVmmAlign) * kVmmAlign;

  /*
   * Contract: on success *out holds the buffer and true is returned; on any
   * failure *out is left nullptr. Raw allocations are tracked in bufs.rawAllocs
   * for cleanup regardless, so *out is assigned only once the buffer is fully
   * initialized.
   */
  auto allocOne = [&](void** out, uint8_t fill) -> bool {
    *out = nullptr;
    if (bufs.useGpu) {
      void* raw = nullptr;
      auto ret = cudaMalloc(&raw, allocBytes);
      if (ret != cudaSuccess || raw == nullptr) {
        UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: cudaMalloc failed");
        return false;
      }
      bufs.rawAllocs.push_back(raw);
      ret = cudaMemset(raw, fill, maxSize);
      if (ret != cudaSuccess) {
        UNIFLOW_LOG_ERROR(
            "RdmaBandwidthBenchmark: cudaMemset failed: {}",
            cudaGetErrorString(ret));
        return false;
      }
      *out = raw;
    } else {
      void* raw = std::malloc(maxSize);
      if (raw == nullptr) {
        UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: malloc failed");
        return false;
      }
      bufs.rawAllocs.push_back(raw);
      std::memset(raw, fill, maxSize);
      *out = raw;
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
    size_t chunkSize,
    bool dataDirect) {
  RdmaTransportConfig rdmaConfig{};
  rdmaConfig.chunkSize = chunkSize;
  rdmaConfig.numQps = static_cast<uint32_t>(devices.size());
  rdmaConfig.dataDirect = dataDirect;

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

/*
 * A single GPU's transport plus its registered local/remote buffer sets.
 * One RunUnit is driven by one worker thread during the concurrent run.
 */
struct RunUnit {
  int cudaDevice{-1};
  std::string label;
  Transport* transport{nullptr};
  std::vector<RegisteredSegment>* localRegs{nullptr};
  std::vector<RemoteRegisteredSegment>* remoteRegs{nullptr};
};

// Outcome of transferring one message size on one RunUnit.
struct TransferResult {
  bool ok{false};
  double bandwidthGBs{0};
  double messageRateMops{0};
  int totalOps{0};
  std::vector<double> latenciesUs;
};

/// Transfer `size` bytes in direction `dir` ("put"/"get") on a single unit:
/// warmup, then a pipelined timed loop keeping up to txDepth batches in flight.
/// Self-contained (no shared mutable state) so it can run on its own thread,
/// enabling concurrent multi-GPU aggregate measurement.
TransferResult runTransfer(
    RunUnit& unit,
    size_t size,
    const std::string& dir,
    const BenchmarkConfig& config) {
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  /*
   * The transport's RDMA path is host-initiated, but pin the worker to the
   * owning GPU so any CUDA-context-sensitive calls resolve to the right device.
   */
  if (unit.cudaDevice >= 0) {
    auto cudaRet = cudaSetDevice(unit.cudaDevice);
    if (cudaRet != cudaSuccess) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: cudaSetDevice({}) failed: {}",
          unit.cudaDevice,
          cudaGetErrorString(cudaRet));
      return {};
    }
  }

  const int batchSize = std::max(1, config.batchSize);
  const int txDepth = std::max(1, config.txDepth);
  const int numBufs = static_cast<int>(unit.localRegs->size());
  auto& localRegs = *unit.localRegs;
  auto& remoteRegs = *unit.remoteRegs;
  auto& transport = *unit.transport;

  TransferResult out;

  // Each request indexes localRegs/remoteRegs by NIC; both must be non-empty
  // and equal-sized (they are one-per-NIC from setup). Guard so the indexing
  // below can never touch an empty vector.
  if (numBufs <= 0 || localRegs.size() != remoteRegs.size()) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: {} has no or mismatched registered segments",
        unit.label);
    return out;
  }

  /*
   * Map each request to its NIC's buffer. Must match spray()'s contiguous
   * chunk-to-QP assignment in the transport layer.
   */
  auto nicForRequest = [&](int requestIdx) {
    return requestIdx * numBufs / batchSize;
  };
  std::vector<TransferRequest> batch;
  batch.reserve(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    int bufIdx = nicForRequest(i);
    batch.push_back(
        TransferRequest{
            .local = localRegs.at(bufIdx).span(size_t{0}, size),
            .remote = remoteRegs.at(bufIdx).span(size_t{0}, size),
        });
  }

  int numBatches = std::max(1, (config.iterations + batchSize - 1) / batchSize);
  int totalOps = numBatches * batchSize;

  auto submitBatchAsync = [&]() -> std::future<Status> {
    return (dir == "put") ? transport.put(batch, {}) : transport.get(batch, {});
  };

  for (int iter = 0; iter < config.warmupIterations; ++iter) {
    auto status = submitBatchAsync().get();
    if (status.hasError()) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: warmup {} failed on {} at size {}: {}",
          dir,
          unit.label,
          size,
          status.error().message());
      return out;
    }
  }

  /*
   * Sliding window: keep up to txDepth batches in-flight.
   * txDepth=1 degenerates to synchronous behavior.
   */
  std::deque<std::pair<std::future<Status>, TimePoint>> inflight;
  std::vector<double> latenciesUs;
  latenciesUs.reserve(numBatches);

  /*
   * Complete the oldest in-flight batch: get result, record latency.
   * Returns false on error after draining all remaining futures.
   */
  auto completeOne = [&]() -> bool {
    auto& [fut, submitTime] = inflight.front();
    auto status = fut.get();
    auto completeTime = Clock::now();
    if (status.hasError()) {
      inflight.pop_front();
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: {} failed on {} at size {}: {}",
          dir,
          unit.label,
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
        return out;
      }
    }
    inflight.emplace_back(submitBatchAsync(), Clock::now());
  }

  while (!inflight.empty()) {
    if (!completeOne()) {
      return out;
    }
  }

  auto overallEnd = Clock::now();

  double totalTimeSec =
      std::chrono::duration<double>(overallEnd - overallStart).count();
  double totalBytes = static_cast<double>(size) * static_cast<double>(totalOps);

  out.ok = true;
  out.bandwidthGBs = (totalTimeSec > 0) ? (totalBytes / totalTimeSec) / 1e9 : 0;
  out.messageRateMops =
      (totalTimeSec > 0) ? (totalOps / totalTimeSec) / 1e6 : 0;
  out.totalOps = totalOps;
  out.latenciesUs = std::move(latenciesUs);
  return out;
}

/// Run the message-size sweep across all GPU units concurrently and report
/// aggregate bandwidth (sum across units) per size. With a single unit this is
/// identical to a plain single-GPU run.
std::vector<BenchmarkResult> runBenchmarkLoop(
    std::vector<RunUnit>& units,
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap,
    const std::string& benchmarkName) {
  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;
  const int batchSize = std::max(1, config.batchSize);
  const int txDepth = std::max(1, config.txDepth);
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

      /*
       * Launch one worker per GPU so all units transfer this size between the
       * same pair of barriers; aggregate bandwidth is their sum.
       */
      std::vector<std::future<TransferResult>> futures;
      futures.reserve(units.size());
      for (auto& unit : units) {
        futures.push_back(
            std::async(std::launch::async, [&unit, size, dir, &config]() {
              return runTransfer(unit, size, dir, config);
            }));
      }

      bool allOk = true;
      double aggBandwidthGBs = 0;
      double aggMsgRateMops = 0;
      int aggOps = 0;
      std::vector<double> pooledLatencies;
      std::string perUnitBw;
      /*
       * runTransfer reports failures through TransferResult::ok rather than
       * exceptions, so get() below does not throw on a failed transfer. Even if
       * get() did throw (e.g. std::async failing to start a thread), unwinding
       * destroys `futures`, and a std::async(launch::async) future joins its
       * task in its destructor -- so every worker completes before the units it
       * captured by reference go out of scope. No dangling reference results.
       */
      for (size_t u = 0; u < units.size(); ++u) {
        auto r = futures[u].get();
        if (!r.ok) {
          allOk = false;
          continue;
        }
        aggBandwidthGBs += r.bandwidthGBs;
        aggMsgRateMops += r.messageRateMops;
        aggOps += r.totalOps;
        pooledLatencies.insert(
            pooledLatencies.end(), r.latenciesUs.begin(), r.latenciesUs.end());
        if (!perUnitBw.empty()) {
          perUnitBw += " ";
        }
        char bwBuf[32];
        std::snprintf(bwBuf, sizeof(bwBuf), "%.2f", r.bandwidthGBs);
        perUnitBw += units[u].label + "=" + bwBuf;
      }
      if (!allOk) {
        return;
      }

      auto stats = Stats::compute(std::move(pooledLatencies));

      results.push_back({
          .benchmarkName = benchmarkName,
          .transport = "rdma",
          .direction = dir,
          .messageSize = size,
          .iterations = aggOps,
          .batchSize = batchSize,
          .txDepth = txDepth,
          .chunkSize = config.chunkSize,
          .bandwidthGBs = aggBandwidthGBs,
          .latency = stats,
          .messageRateMops = aggMsgRateMops,
      });

      UNIFLOW_LOG_WARN(
          "[rank {}] {} size={:<10} gpus={} batch={:<3} txdepth={:<3} "
          "iters={:<6} bw={:.2f} GB/s [{}] avg={:.1f} us {}",
          bootstrap.rank,
          dir,
          size,
          units.size(),
          batchSize,
          txDepth,
          aggOps,
          aggBandwidthGBs,
          perUnitBw,
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

  std::vector<std::string> allDevices = rdmaDevices_;
  if (allDevices.empty()) {
    allDevices = discoverRdmaDevices(ibvApi);
  }
  if (allDevices.empty()) {
    UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: no RDMA devices found");
    return {};
  }

  /*
   * Resolve the GPU device list: an explicit multi-GPU list, else the single
   * --cuda-device (which may be -1 for CPU memory).
   */
  std::vector<int> gpuDevices = config.cudaDevices;
  if (gpuDevices.empty()) {
    gpuDevices.push_back(config.cudaDevice);
  }
  const bool multiGpu = gpuDevices.size() > 1;

  /*
   * Data Direct is a GPU-memory (VRAM) feature. Reject it up front if any
   * selected device is CPU (< 0): dataDirect is forwarded uniformly to every
   * unit's transport, so a mixed list would carry a meaningless dataDirect=true
   * into the CPU unit and fail later at MR registration with a less specific
   * error. Requiring an all-GPU device list keeps the failure mode clear.
   */
  if (config.dataDirect &&
      std::any_of(
          gpuDevices.begin(), gpuDevices.end(), [](int d) { return d < 0; })) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: --data-direct requires all selected devices to "
        "be GPUs (--cuda-device/--cuda-devices >= 0); it does not apply to CPU "
        "memory");
    return {};
  }

  /*
   * An explicit per-GPU NIC map (--gpu-nics) takes precedence over topology: it
   * gives each GPU its own NICs and avoids adjacent GPUs being assigned the
   * same NICs. If provided, it must have exactly one group per GPU.
   */
  const auto& nicGroups = config.gpuNicGroups;
  if (!nicGroups.empty() && nicGroups.size() != gpuDevices.size()) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: --gpu-nics has {} group(s) but there are {} "
        "GPU(s)",
        nicGroups.size(),
        gpuDevices.size());
    return {};
  }
  if (multiGpu && nicGroups.empty() && !rdmaDevices_.empty()) {
    UNIFLOW_LOG_WARN(
        "RdmaBandwidthBenchmark: --rdma-devices is not used in multi-GPU mode "
        "(it would double-book NICs across GPUs); topology selects each GPU's "
        "NICs -- pass --gpu-nics for an explicit per-GPU map");
  }

  /*
   * Select the NICs that serve the g-th GPU (device id `dev`). Precedence:
   * explicit --gpu-nics map, then single-GPU --rdma-devices override, then
   * topology selection. In Data Direct mode the candidate set is further
   * restricted to NICs whose data-direct PCIe domain matches the GPU's.
   */
  auto selectNicsForDevice = [&](size_t g,
                                 int dev) -> std::vector<std::string> {
    std::vector<std::string> sel;
    if (!nicGroups.empty()) {
      sel = nicGroups[g];
    } else if (!multiGpu && !rdmaDevices_.empty()) {
      sel = rdmaDevices_;
    } else {
      auto& topo = sharedTopology();
      if (topo.available()) {
        sel = (dev < 0) ? topo.selectCpuNics() : topo.selectGpuNics(dev);
      }
      if (sel.empty()) {
        // Falling back to the full device list assigns the same NICs to every
        // GPU. That is fine for a single GPU, but in multi-GPU mode it would
        // double-book NICs across GPUs and inflate the aggregate, so require an
        // explicit --gpu-nics map instead of guessing.
        if (multiGpu) {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: no per-GPU NICs for gpu {} (topology "
              "unavailable); pass --gpu-nics to give each GPU a distinct NIC set",
              dev);
          return {};
        }
        sel = allDevices;
      }
    }
    /*
     * Data Direct registers GPU memory over a dedicated NIC<->GPU PCIe path, so
     * it only succeeds on NICs whose data-direct interface shares the GPU's
     * PCIe domain. Physical/topology NIC adjacency does not imply the same
     * domain, so filter the candidates to the domain-matched subset instead of
     * relying on --gpu-nics.
     */
    if (config.dataDirect && dev >= 0) {
      char busId[32] = {};
      if (cudaDeviceGetPCIBusId(busId, sizeof(busId), dev) == cudaSuccess) {
        auto ddNics = selectDataDirectNicsForGpu(*ibvApi, sel, busId);
        if (!ddNics.empty()) {
          sel = std::move(ddNics);
        } else {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: no Data-Direct-domain-matched NICs for "
              "gpu {} ({}) among {} candidate(s); aborting (pass --gpu-nics to "
              "override)",
              dev,
              busId,
              sel.size());
          return {};
        }
      } else {
        /*
         * Without the GPU's bus id the domain filter cannot run. Continuing
         * with the unfiltered candidates would defer the failure to Data
         * Direct MR registration (EOPNOTSUPP on out-of-domain NICs) with a
         * less specific error, so abort here to match the no-match branch.
         */
        UNIFLOW_LOG_ERROR(
            "RdmaBandwidthBenchmark: cudaDeviceGetPCIBusId failed for gpu {}; "
            "cannot filter NICs by Data-Direct domain (pass --gpu-nics to "
            "override)",
            dev);
        return {};
      }
    }
    if (config.numNics > 0 && config.numNics < static_cast<int>(sel.size())) {
      sel.resize(config.numNics);
    }
    return sel;
  };

  /*
   * Declared so transports/MRs (sessions) are destroyed before the GPU buffers
   * they reference, which are destroyed before the EventBase threads the
   * factories hold. reserve() keeps element addresses stable across setup.
   */
  std::vector<std::unique_ptr<ScopedEventBaseThread>> evbThreads;
  std::vector<BenchmarkBuffers> buffers;
  std::vector<TransportSession> sessions;
  evbThreads.reserve(gpuDevices.size());
  buffers.reserve(gpuDevices.size());
  sessions.reserve(gpuDevices.size());

  for (size_t g = 0; g < gpuDevices.size(); ++g) {
    int dev = gpuDevices[g];
    auto nics = selectNicsForDevice(g, dev);
    if (nics.empty()) {
      UNIFLOW_LOG_ERROR(
          "RdmaBandwidthBenchmark: no NICs selected for gpu {}", dev);
      return {};
    }
    {
      std::string devList;
      for (const auto& d : nics) {
        if (!devList.empty()) {
          devList += ", ";
        }
        devList += d;
      }
      UNIFLOW_LOG_WARN(
          "RdmaBandwidthBenchmark: rank {} gpu {} RDMA device(s): {}",
          bootstrap.rank,
          dev,
          devList);
    }

    auto bufs = allocateBuffers(
        config.maxSize, dev, bootstrap.rank, static_cast<int>(nics.size()));
    if (!bufs) {
      return {};
    }
    assert(
        bufs->srcs.size() == nics.size() &&
        "Buffer count must equal NIC/QP count");
    buffers.push_back(std::move(*bufs));

    evbThreads.push_back(
        std::make_unique<ScopedEventBaseThread>(
            "bench-evb-gpu" + std::to_string(dev)));

    auto session = setupTransport(
        nics,
        buffers[g],
        config.maxSize,
        *evbThreads[g],
        ibvApi,
        peers[0],
        bootstrap,
        config.chunkSize,
        config.dataDirect);
    if (!session) {
      return {};
    }
    sessions.push_back(std::move(*session));
  }

  std::vector<RunUnit> units;
  units.reserve(sessions.size());
  /*
   * sessions is built 1:1 from gpuDevices (any failure returns early), so the
   * sizes match; bound the loop by both and read each element through at() so
   * the indexing is checked at the access site.
   */
  for (size_t g = 0; g < sessions.size() && g < gpuDevices.size(); ++g) {
    const int dev = gpuDevices.at(g);
    auto& session = sessions.at(g);
    units.push_back(
        RunUnit{
            .cudaDevice = dev,
            .label =
                (dev < 0) ? std::string("cpu") : "gpu" + std::to_string(dev),
            .transport = session.transport.get(),
            .localRegs = &session.localRegs,
            .remoteRegs = &session.remoteRegs,
        });
  }

  auto results = runBenchmarkLoop(units, config, peers, bootstrap, name());

  auto finalBarrier = barrier(peers, bootstrap);
  if (!finalBarrier) {
    UNIFLOW_LOG_WARN(
        "RdmaBandwidthBenchmark: final barrier failed: {}",
        finalBarrier.error().toString());
  }
  for (auto& session : sessions) {
    session.transport->shutdown();
  }
  return results;
}

} // namespace uniflow::benchmark
