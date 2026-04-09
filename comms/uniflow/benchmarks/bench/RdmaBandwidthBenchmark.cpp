// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/RdmaBandwidthBenchmark.h"

#include <chrono>
#include <cstring>

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

namespace {

/// Discover all RDMA device names via the IbvApi device list.
std::vector<std::string> discoverRdmaDevices(
    const std::shared_ptr<uniflow::IbvApi>& ibvApi) {
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

} // namespace

namespace uniflow::benchmark {

std::vector<BenchmarkResult> RdmaBandwidthBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  if (peers.empty()) {
    UNIFLOW_LOG_WARN("RdmaBandwidthBenchmark: no peers, skipping");
    return {};
  }

  // 1. Init ibverbs and discover/validate devices.
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

  // Use one device per rank for single-host benchmarking.
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

  // 2. Set CUDA device and allocate GPU buffers.
  auto cudaRet = cudaSetDevice(bootstrap.localRank);
  if (cudaRet != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: cudaSetDevice({}) failed: {}",
        bootstrap.localRank,
        cudaGetErrorString(cudaRet));
    return {};
  }
  void* srcBuf = nullptr;
  void* dstBuf = nullptr;
  cudaRet = cudaMalloc(&srcBuf, config.maxSize);
  if (cudaRet != cudaSuccess || srcBuf == nullptr) {
    UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: cudaMalloc(src) failed");
    return {};
  }
  cudaRet = cudaMalloc(&dstBuf, config.maxSize);
  if (cudaRet != cudaSuccess || dstBuf == nullptr) {
    cudaFree(srcBuf);
    UNIFLOW_LOG_ERROR("RdmaBandwidthBenchmark: cudaMalloc(dst) failed");
    return {};
  }

  // Fill source buffer with a pattern.
  cudaRet = cudaMemset(srcBuf, 0xAB, config.maxSize);
  if (cudaRet != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: cudaMemset(src) failed: {}",
        cudaGetErrorString(cudaRet));
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }
  cudaRet = cudaMemset(dstBuf, 0, config.maxSize);
  if (cudaRet != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: cudaMemset(dst) failed: {}",
        cudaGetErrorString(cudaRet));
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }
  cudaRet = cudaDeviceSynchronize();
  if (cudaRet != cudaSuccess) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: cudaDeviceSynchronize failed: {}",
        cudaGetErrorString(cudaRet));
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  // 3. Create RDMA factory, exchange topology, create transport, connect.
  ScopedEventBaseThread evbThread("bench-evb");
  auto cudaDriverApi = std::make_shared<CudaDriverApi>();
  RdmaTransportFactory factory(
      {myDevice},
      evbThread.getEventBase(),
      RdmaTransportConfig{},
      ibvApi,
      cudaDriverApi);

  auto localTopology = factory.getTopology();
  auto remoteTopologyResult =
      exchangeMetadata(*peers[0].ctrl, localTopology, bootstrap.isRank0());
  if (!remoteTopologyResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: topology exchange failed: {}",
        remoteTopologyResult.error().toString());
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  auto transportResult =
      factory.createTransport(std::move(remoteTopologyResult).value());
  if (!transportResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: createTransport failed: {}",
        transportResult.error().toString());
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }
  auto transport = std::move(transportResult).value();

  auto localInfo = transport->bind();
  auto remoteInfoResult =
      exchangeMetadata(*peers[0].ctrl, localInfo, bootstrap.isRank0());
  if (!remoteInfoResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: transport info exchange failed: {}",
        remoteInfoResult.error().toString());
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  auto connectStatus = transport->connect(std::move(remoteInfoResult).value());
  if (!connectStatus) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: connect failed: {}",
        connectStatus.error().toString());
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  // 4. Register segments and exchange registration payloads.
  Segment srcSeg(srcBuf, config.maxSize, MemoryType::VRAM, bootstrap.localRank);
  Segment dstSeg(dstBuf, config.maxSize, MemoryType::VRAM, bootstrap.localRank);

  auto srcRegResult = factory.registerSegment(srcSeg);
  if (!srcRegResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: registerSegment(src) failed: {}",
        srcRegResult.error().toString());
    transport->shutdown();
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  auto dstRegResult = factory.registerSegment(dstSeg);
  if (!dstRegResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: registerSegment(dst) failed: {}",
        dstRegResult.error().toString());
    transport->shutdown();
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  // Exchange destination registration payload so each side can write to
  // the peer's destination buffer. Prepend the dst buffer virtual address
  // so the remote side knows the correct RDMA target address (each process
  // has a different virtual address for its cudaMalloc'd buffer).
  auto dstPayload = dstRegResult.value()->serialize();
  uint64_t dstAddr = reinterpret_cast<uint64_t>(dstBuf);
  std::vector<uint8_t> dstPayloadWithAddr(sizeof(dstAddr) + dstPayload.size());
  std::memcpy(dstPayloadWithAddr.data(), &dstAddr, sizeof(dstAddr));
  std::memcpy(
      dstPayloadWithAddr.data() + sizeof(dstAddr),
      dstPayload.data(),
      dstPayload.size());

  auto remotePayloadResult =
      exchangeMetadata(*peers[0].ctrl, dstPayloadWithAddr, bootstrap.isRank0());
  if (!remotePayloadResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: registration exchange failed: {}",
        remotePayloadResult.error().toString());
    transport->shutdown();
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  // Extract the remote dst buffer address and registration payload.
  auto& remotePayload = remotePayloadResult.value();
  if (remotePayload.size() < sizeof(uint64_t)) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: remote payload too small: {}",
        remotePayload.size());
    transport->shutdown();
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }
  uint64_t remoteDstAddr = 0;
  std::memcpy(&remoteDstAddr, remotePayload.data(), sizeof(remoteDstAddr));
  std::vector<uint8_t> remoteRegPayload(
      remotePayload.begin() + sizeof(remoteDstAddr), remotePayload.end());

  auto remoteHandleResult =
      factory.importSegment(config.maxSize, std::move(remoteRegPayload));
  if (!remoteHandleResult) {
    UNIFLOW_LOG_ERROR(
        "RdmaBandwidthBenchmark: importSegment failed: {}",
        remoteHandleResult.error().toString());
    transport->shutdown();
    cudaFree(srcBuf);
    cudaFree(dstBuf);
    return {};
  }

  auto localReg =
      SegmentTest::makeRegistered(srcSeg, std::move(srcRegResult.value()));
  auto remoteReg = SegmentTest::makeRemote(
      reinterpret_cast<void*>(remoteDstAddr),
      config.maxSize,
      std::move(remoteHandleResult.value()));

  // 5. Benchmark loop — sweep message sizes.
  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;

  auto runDirection = [&](const std::string& dir) {
    for (auto size : sizes) {
      const int totalIterations = config.warmupIterations + config.iterations;
      std::vector<double> latenciesUs;
      latenciesUs.reserve(config.iterations);

      for (int iter = 0; iter < totalIterations; ++iter) {
        auto barrierStatus = barrier(peers, bootstrap);
        if (!barrierStatus) {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: barrier failed: {}",
              barrierStatus.error().toString());
          return;
        }

        TransferRequest req{
            .local = localReg.span(size_t{0}, size),
            .remote = remoteReg.span(size_t{0}, size),
        };

        auto start = std::chrono::steady_clock::now();

        Status opStatus;
        if (dir == "put") {
          opStatus = transport->put({&req, 1}, {}).get();
        } else {
          opStatus = transport->get({&req, 1}, {}).get();
        }

        auto end = std::chrono::steady_clock::now();

        if (opStatus.hasError()) {
          UNIFLOW_LOG_ERROR(
              "RdmaBandwidthBenchmark: {} failed at size {}: {}",
              dir,
              size,
              opStatus.error().message());
          return;
        }

        if (iter >= config.warmupIterations) {
          double elapsedUs =
              std::chrono::duration<double, std::micro>(end - start).count();
          latenciesUs.push_back(elapsedUs);
        }
      }

      auto stats = Stats::compute(std::move(latenciesUs));
      double bandwidthGBs = (stats.avg > 0)
          ? (static_cast<double>(size) / (stats.avg * 1e-6)) / 1e9
          : 0;

      BenchmarkResult result;
      result.benchmarkName = name();
      result.transport = "rdma";
      result.direction = dir;
      result.messageSize = size;
      result.iterations = config.iterations;
      result.bandwidthGBs = bandwidthGBs;
      result.latency = stats;
      results.push_back(result);

      UNIFLOW_LOG_INFO(
          "RdmaBandwidthBenchmark: {} size={} avg={:.1f}us bw={:.2f}GB/s",
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

  // Ensure both ranks finish before shutting down.
  barrier(peers, bootstrap);

  // 6. Cleanup.
  transport->shutdown();
  cudaFree(srcBuf);
  cudaFree(dstBuf);
  return results;
}

} // namespace uniflow::benchmark
