// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * DC vs RC RDMA Single-Pair Benchmark
 *
 * This benchmark compares Dynamic Connection (DC) vs Reliable Connection (RC)
 * RDMA performance for a single sender-receiver pair.
 *
 * Experiments:
 * 1. Single-Pair Steady-State: DC vs RC latency/bandwidth for various sizes
 * 2. QP Initialization Overhead: Time to create and transition QPs
 *
 * Methodology:
 * - Warm-up phase before each benchmark
 * - 1000 runs per message size
 * - Message sizes: 1B to 2GB (32 sizes, powers of 2)
 */

#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/benchmarks/IbverbxDcBenchUtils.h"
#include "comms/ctran/ibverbx/tests/dc_utils.h"
#include "comms/testinfra/BenchUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

DEFINE_bool(raw_only, true, "Print only RAW CSV results, suppress folly table");
DEFINE_int32(batch_size, 1, "Number of RDMA writes to batch before polling");

using namespace ibverbx;

namespace {

// Global flags to track if RDMA and DC are available
bool g_rdmaAvailable = false;
bool g_rdmaChecked = false;
bool g_dcAvailable = false;
bool g_dcChecked = false;
// All DC-capable device indices (need at least 2 for single-pair)
std::vector<int> g_dcCapableDevices;

// Check if RDMA devices are available
bool checkRdmaAvailable() {
  if (g_rdmaChecked) {
    return g_rdmaAvailable;
  }
  g_rdmaChecked = true;

  ncclCvarInit();
  if (!ibvInit()) {
    XLOG(WARNING) << "Failed to initialize ibverbs";
    return false;
  }

  auto devices = IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  if (!devices || devices->empty()) {
    XLOG(WARNING) << "No RDMA devices found";
    return false;
  }

  // Check if we have at least 2 devices (sender and receiver)
  if (devices->size() < 2) {
    XLOG(WARNING) << "Need at least 2 RDMA devices, found " << devices->size();
    return false;
  }

  g_rdmaAvailable = true;
  return true;
}

// Check if DC transport is supported on the hardware
bool checkDcAvailable() {
  if (g_dcChecked) {
    return g_dcAvailable;
  }
  g_dcChecked = true;

  if (!checkRdmaAvailable()) {
    XLOG(WARNING) << "DC check: RDMA not available";
    return false;
  }

  // Get device list to scan for DC-capable devices
  ncclCvarInit();
  auto devices = IbvDevice::ibvGetDeviceList(NCCL_IB_HCA, NCCL_IB_HCA_PREFIX);
  if (!devices || devices->empty()) {
    XLOG(WARNING) << "DC check: No RDMA devices found";
    return false;
  }

  // Scan through devices to find ones that support DC
  std::vector<int> dcCapableDevices;
  for (size_t i = 0; i < devices->size(); ++i) {
    DcEndPoint testEndpoint;
    auto initResult = testEndpoint.init(static_cast<int>(i));
    if (!initResult) {
      XLOGF(
          WARNING,
          "Device {} ({}): init failed: {}",
          i,
          devices->at(i).device()->name,
          initResult.error().errStr);
      continue;
    }

    auto dcResult = testEndpoint.initDc();
    if (!dcResult) {
      XLOGF(
          WARNING,
          "Device {} ({}): DC not supported: {}",
          i,
          devices->at(i).device()->name,
          dcResult.error().errStr);
      continue;
    }

    XLOGF(
        DBG, "Device {} ({}): DC supported!", i, devices->at(i).device()->name);
    dcCapableDevices.push_back(static_cast<int>(i));
  }

  if (dcCapableDevices.size() < 2) {
    XLOGF(
        WARNING,
        "Need at least 2 DC-capable devices, found {} (DC benchmarks will be skipped)",
        dcCapableDevices.size());
    return false;
  }

  g_dcCapableDevices = std::move(dcCapableDevices);
  g_dcAvailable = true;

  XLOGF(
      DBG,
      "DC transport is available - using devices {} and {} for single-pair benchmarks",
      g_dcCapableDevices[0],
      g_dcCapableDevices[1]);
  return true;
}

// Benchmark setup state shared across iterations
struct DcBenchmarkState {
  std::unique_ptr<DcEndPoint> sender;
  std::unique_ptr<DcEndPoint> receiver;
  std::unique_ptr<IbvAh> ah;
  std::unique_ptr<IbvMr> senderMr;
  std::unique_ptr<IbvMr> receiverMr;
  DcBusinessCard senderCard;
  DcBusinessCard receiverCard;
  std::vector<uint8_t> senderBuf;
  std::vector<uint8_t> receiverBuf;
  ibverbx::ibv_sge sge{};
  bool initialized{false};

  bool init(size_t bufferSize) {
    if (initialized && senderBuf.size() == bufferSize) {
      return true;
    }

    // Reset resources in correct order (AH, MRs before endpoints/PDs)
    ah.reset();
    senderMr.reset();
    receiverMr.reset();
    sender.reset();
    receiver.reset();

    // Allocate buffers
    senderBuf.resize(bufferSize);
    receiverBuf.resize(bufferSize);
    std::memset(senderBuf.data(), 0xAA, bufferSize);
    std::memset(receiverBuf.data(), 0x00, bufferSize);

    // Create sender endpoint (use discovered DC-capable device)
    sender = std::make_unique<DcEndPoint>();
    auto senderInitResult = sender->init(g_dcCapableDevices[0]);
    if (!senderInitResult) {
      XLOGF(ERR, "Sender init failed: {}", senderInitResult.error().errStr);
      return false;
    }
    auto senderDcResult = sender->initDc();
    if (!senderDcResult) {
      XLOGF(ERR, "Sender DC init failed: {}", senderDcResult.error().errStr);
      return false;
    }

    // Create receiver endpoint (use discovered DC-capable device)
    receiver = std::make_unique<DcEndPoint>();
    auto receiverInitResult = receiver->init(g_dcCapableDevices[1]);
    if (!receiverInitResult) {
      XLOGF(ERR, "Receiver init failed: {}", receiverInitResult.error().errStr);
      return false;
    }
    auto receiverDcResult = receiver->initDc();
    if (!receiverDcResult) {
      XLOGF(
          ERR, "Receiver DC init failed: {}", receiverDcResult.error().errStr);
      return false;
    }

    // Register memory
    auto senderMrResult =
        sender->registerMr(senderBuf.data(), senderBuf.size());
    if (!senderMrResult) {
      XLOGF(ERR, "Sender MR failed: {}", senderMrResult.error().errStr);
      return false;
    }
    senderMr = std::make_unique<IbvMr>(std::move(*senderMrResult));

    auto receiverMrResult =
        receiver->registerMr(receiverBuf.data(), receiverBuf.size());
    if (!receiverMrResult) {
      XLOGF(ERR, "Receiver MR failed: {}", receiverMrResult.error().errStr);
      return false;
    }
    receiverMr = std::make_unique<IbvMr>(std::move(*receiverMrResult));

    // Create business cards
    senderCard =
        sender->createBusinessCard(senderBuf.data(), senderMr->mr()->lkey);
    receiverCard = receiver->createBusinessCard(
        receiverBuf.data(), receiverMr->mr()->rkey);

    // Create address handle from sender to receiver
    auto ahResult = sender->createAh(receiverCard);
    if (!ahResult) {
      XLOGF(ERR, "AH creation failed: {}", ahResult.error().errStr);
      return false;
    }
    ah = std::make_unique<IbvAh>(std::move(*ahResult));

    // Setup SGE for sends
    sge.addr = reinterpret_cast<uint64_t>(senderBuf.data());
    sge.length = static_cast<uint32_t>(bufferSize);
    sge.lkey = senderMr->mr()->lkey;

    initialized = true;
    return true;
  }
};

struct RcBenchmarkState {
  std::unique_ptr<RcEndPoint> sender;
  std::unique_ptr<RcEndPoint> receiver;
  std::unique_ptr<IbvMr> senderMr;
  std::unique_ptr<IbvMr> receiverMr;
  RcBusinessCard senderCard;
  RcBusinessCard receiverCard;
  std::vector<uint8_t> senderBuf;
  std::vector<uint8_t> receiverBuf;
  ibverbx::ibv_sge sge{};
  bool initialized{false};

  bool init(size_t bufferSize) {
    if (initialized && senderBuf.size() == bufferSize) {
      return true;
    }

    // Reset resources in correct order (MRs before endpoints/PDs)
    senderMr.reset();
    receiverMr.reset();
    sender.reset();
    receiver.reset();

    // Allocate buffers
    senderBuf.resize(bufferSize);
    receiverBuf.resize(bufferSize);
    std::memset(senderBuf.data(), 0xAA, bufferSize);
    std::memset(receiverBuf.data(), 0x00, bufferSize);

    // Create sender endpoint (same device as DC sender, or device 0 if DC
    // unavailable)
    int senderDevIdx =
        g_dcCapableDevices.size() >= 2 ? g_dcCapableDevices[0] : 0;
    int receiverDevIdx =
        g_dcCapableDevices.size() >= 2 ? g_dcCapableDevices[1] : 1;

    sender = std::make_unique<RcEndPoint>();
    auto senderInitResult = sender->init(senderDevIdx);
    if (!senderInitResult) {
      XLOGF(ERR, "Sender init failed: {}", senderInitResult.error().errStr);
      return false;
    }
    auto senderRcResult = sender->initRc();
    if (!senderRcResult) {
      XLOGF(ERR, "Sender RC init failed: {}", senderRcResult.error().errStr);
      return false;
    }

    // Create receiver endpoint (same device as DC receiver, or device 1 if DC
    // unavailable)
    receiver = std::make_unique<RcEndPoint>();
    auto receiverInitResult = receiver->init(receiverDevIdx);
    if (!receiverInitResult) {
      XLOGF(ERR, "Receiver init failed: {}", receiverInitResult.error().errStr);
      return false;
    }
    auto receiverRcResult = receiver->initRc();
    if (!receiverRcResult) {
      XLOGF(
          ERR, "Receiver RC init failed: {}", receiverRcResult.error().errStr);
      return false;
    }

    // Register memory
    auto senderMrResult =
        sender->registerMr(senderBuf.data(), senderBuf.size());
    if (!senderMrResult) {
      XLOGF(ERR, "Sender MR failed: {}", senderMrResult.error().errStr);
      return false;
    }
    senderMr = std::make_unique<IbvMr>(std::move(*senderMrResult));

    auto receiverMrResult =
        receiver->registerMr(receiverBuf.data(), receiverBuf.size());
    if (!receiverMrResult) {
      XLOGF(ERR, "Receiver MR failed: {}", receiverMrResult.error().errStr);
      return false;
    }
    receiverMr = std::make_unique<IbvMr>(std::move(*receiverMrResult));

    // Create business cards
    senderCard =
        sender->createBusinessCard(senderBuf.data(), senderMr->mr()->lkey);
    receiverCard = receiver->createBusinessCard(
        receiverBuf.data(), receiverMr->mr()->rkey);

    // Connect QPs (bidirectional)
    auto senderConnectResult = sender->connect(receiverCard);
    if (!senderConnectResult) {
      XLOGF(
          ERR, "Sender connect failed: {}", senderConnectResult.error().errStr);
      return false;
    }

    auto receiverConnectResult = receiver->connect(senderCard);
    if (!receiverConnectResult) {
      XLOGF(
          ERR,
          "Receiver connect failed: {}",
          receiverConnectResult.error().errStr);
      return false;
    }

    // Setup SGE for sends
    sge.addr = reinterpret_cast<uint64_t>(senderBuf.data());
    sge.length = static_cast<uint32_t>(bufferSize);
    sge.lkey = senderMr->mr()->lkey;

    initialized = true;
    return true;
  }
};
thread_local std::unique_ptr<DcBenchmarkState> g_dcState;
thread_local std::unique_ptr<RcBenchmarkState> g_rcState;

struct RawResult {
  std::string name;
  size_t bufferBytes;
  double latencyUs;
  double postUs;
  double pollUs;
  double bwGbps;
};
std::vector<RawResult> g_rawResults;
std::mutex g_rawResultsMutex;

void recordRawResult(
    const char* name,
    size_t bufferBytes,
    double latencyUs,
    double postUs,
    double pollUs,
    double bwGbps) {
  std::lock_guard<std::mutex> lock(g_rawResultsMutex);
  // Overwrite if same name+size already exists (folly runs multiple times)
  for (auto& r : g_rawResults) {
    if (r.name == name && r.bufferBytes == bufferBytes) {
      r = {name, bufferBytes, latencyUs, postUs, pollUs, bwGbps};
      return;
    }
  }
  g_rawResults.push_back(
      {name, bufferBytes, latencyUs, postUs, pollUs, bwGbps});
}

void printAllRawResults() {
  fprintf(
      stderr,
      "RAW,benchmark,buffer_bytes,latency_us,post_us,poll_us,bw_gbps\n");
  for (const auto& r : g_rawResults) {
    fprintf(
        stderr,
        "RAW,%s,%zu,%.3f,%.3f,%.3f,%.6f\n",
        r.name.c_str(),
        r.bufferBytes,
        r.latencyUs,
        r.postUs,
        r.pollUs,
        r.bwGbps);
  }
}

} // namespace

//------------------------------------------------------------------------------
// DC RDMA Write Benchmark
//------------------------------------------------------------------------------

static void
dcRdmaWrite(uint32_t iters, size_t bufferSize, folly::UserCounters& counters) {
  if (!checkDcAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Initialize or reuse state
  if (!g_dcState) {
    g_dcState = std::make_unique<DcBenchmarkState>();
  }
  if (!g_dcState->init(bufferSize)) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Update SGE length for this buffer size
  g_dcState->sge.length = static_cast<uint32_t>(bufferSize);

  // Warm-up iterations
  constexpr int kWarmupIters = 10;
  for (int i = 0; i < kWarmupIters; ++i) {
    int ret = g_dcState->sender->postRdmaWrite(
        *g_dcState->ah, g_dcState->receiverCard, g_dcState->sge, i);
    if (ret != 0) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    // Poll sender CQ for send completion (no receiver poll with plain write)
    if (!g_dcState->sender->pollCqBlocking(1)) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
  }

  // Timed iterations — separate post and poll loops.
  // Batch to avoid overflowing the send queue (max 1024 WRs).
  constexpr uint32_t kBatchSize = 1;
  const uint32_t batchSize = FLAGS_batch_size > 0
      ? static_cast<uint32_t>(FLAGS_batch_size)
      : kBatchSize;
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds totalPostNs{0};
  std::chrono::nanoseconds totalPollNs{0};

  uint32_t remaining = iters;
  while (remaining > 0) {
    uint32_t batch = std::min(remaining, batchSize);

    // Post loop: submit all writes in a tight batch
    auto postStart = std::chrono::high_resolution_clock::now();
    for (uint32_t j = 0; j < batch; ++j) {
      int ret = g_dcState->sender->postRdmaWrite(
          *g_dcState->ah, g_dcState->receiverCard, g_dcState->sge, j);
      if (ret != 0) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
    }
    auto postEnd = std::chrono::high_resolution_clock::now();
    totalPostNs += (postEnd - postStart);

    // Poll loop: drain all completions
    auto pollStart = std::chrono::high_resolution_clock::now();
    for (uint32_t j = 0; j < batch; ++j) {
      if (!g_dcState->sender->pollCqBusySpin(1)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
    }
    auto pollEnd = std::chrono::high_resolution_clock::now();
    totalPollNs += (pollEnd - pollStart);

    remaining -= batch;
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
      1000.0;

  double avgLatencyUs = elapsedUs / iters;
  double avgPostUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(totalPostNs)
          .count() /
      1000.0 / iters;
  double avgPollUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(totalPollNs)
          .count() /
      1000.0 / iters;
  double bandwidthGBps = (bufferSize * iters / 1e9) / (elapsedUs / 1e6);

  counters["latency_us"] =
      folly::UserMetric(avgLatencyUs, folly::UserMetric::Type::METRIC);
  counters["post_us"] =
      folly::UserMetric(avgPostUs, folly::UserMetric::Type::METRIC);
  counters["poll_us"] =
      folly::UserMetric(avgPollUs, folly::UserMetric::Type::METRIC);
  counters["bw_gbps"] =
      folly::UserMetric(bandwidthGBps, folly::UserMetric::Type::METRIC);
  counters["buffer_bytes"] =
      folly::UserMetric(bufferSize, folly::UserMetric::Type::METRIC);
  recordRawResult(
      "dcRdmaWrite",
      bufferSize,
      avgLatencyUs,
      avgPostUs,
      avgPollUs,
      bandwidthGBps);
}

//------------------------------------------------------------------------------
// RC RDMA Write Benchmark
//------------------------------------------------------------------------------

static void
rcRdmaWrite(uint32_t iters, size_t bufferSize, folly::UserCounters& counters) {
  // Use checkDcAvailable to ensure g_dcCapableDevices is populated,
  // so RC uses the same physical devices as DC for fair comparison.
  // Falls back to checkRdmaAvailable if DC is not supported.
  if (!checkDcAvailable() && !checkRdmaAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Initialize or reuse state
  if (!g_rcState) {
    g_rcState = std::make_unique<RcBenchmarkState>();
  }
  if (!g_rcState->init(bufferSize)) {
    counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  // Update SGE length for this buffer size
  g_rcState->sge.length = static_cast<uint32_t>(bufferSize);

  // Warm-up iterations
  constexpr int kWarmupIters = 10;
  for (int i = 0; i < kWarmupIters; ++i) {
    int ret = g_rcState->sender->postRdmaWrite(
        g_rcState->receiverCard, g_rcState->sge, i);
    if (ret != 0) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    if (!g_rcState->sender->pollCqBlocking(1)) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
  }

  // Timed iterations — separate post and poll loops.
  // Batch to avoid overflowing the send queue (max 1024 WRs).
  constexpr uint32_t kBatchSize = 1;
  const uint32_t batchSize = FLAGS_batch_size > 0
      ? static_cast<uint32_t>(FLAGS_batch_size)
      : kBatchSize;
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds totalPostNs{0};
  std::chrono::nanoseconds totalPollNs{0};

  uint32_t remaining = iters;
  while (remaining > 0) {
    uint32_t batch = std::min(remaining, batchSize);

    // Post loop: submit all writes in a tight batch
    auto postStart = std::chrono::high_resolution_clock::now();
    for (uint32_t j = 0; j < batch; ++j) {
      int ret = g_rcState->sender->postRdmaWrite(
          g_rcState->receiverCard, g_rcState->sge, j);
      if (ret != 0) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
    }
    auto postEnd = std::chrono::high_resolution_clock::now();
    totalPostNs += (postEnd - postStart);

    // Poll loop: drain all completions
    auto pollStart = std::chrono::high_resolution_clock::now();
    for (uint32_t j = 0; j < batch; ++j) {
      if (!g_rcState->sender->pollCqBusySpin(1)) {
        counters["error"] =
            folly::UserMetric(1, folly::UserMetric::Type::METRIC);
        return;
      }
    }
    auto pollEnd = std::chrono::high_resolution_clock::now();
    totalPollNs += (pollEnd - pollStart);

    remaining -= batch;
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count() /
      1000.0;

  double avgLatencyUs = elapsedUs / iters;
  double avgPostUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(totalPostNs)
          .count() /
      1000.0 / iters;
  double avgPollUs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(totalPollNs)
          .count() /
      1000.0 / iters;
  double bandwidthGBps = (bufferSize * iters / 1e9) / (elapsedUs / 1e6);

  counters["latency_us"] =
      folly::UserMetric(avgLatencyUs, folly::UserMetric::Type::METRIC);
  counters["post_us"] =
      folly::UserMetric(avgPostUs, folly::UserMetric::Type::METRIC);
  counters["poll_us"] =
      folly::UserMetric(avgPollUs, folly::UserMetric::Type::METRIC);
  counters["bw_gbps"] =
      folly::UserMetric(bandwidthGBps, folly::UserMetric::Type::METRIC);
  counters["buffer_bytes"] =
      folly::UserMetric(bufferSize, folly::UserMetric::Type::METRIC);
  recordRawResult(
      "rcRdmaWrite",
      bufferSize,
      avgLatencyUs,
      avgPostUs,
      avgPollUs,
      bandwidthGBps);
}

//------------------------------------------------------------------------------
// QP Initialization Overhead Benchmarks
//------------------------------------------------------------------------------

static void dcQpCreation(uint32_t iters, folly::UserCounters& counters) {
  if (!checkDcAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  double totalTimeUs = 0;

  for (uint32_t i = 0; i < iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    DcEndPoint endpoint;
    auto initResult = endpoint.init(g_dcCapableDevices[0]);
    if (!initResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    auto dcResult = endpoint.initDc();
    if (!dcResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    totalTimeUs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() /
        1000.0;
  }

  double avgTimeUs = totalTimeUs / iters;
  counters["qp_init_us"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
}

static void rcQpCreation(uint32_t iters, folly::UserCounters& counters) {
  if (!checkRdmaAvailable()) {
    counters["skipped"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
    return;
  }

  double totalTimeUs = 0;

  for (uint32_t i = 0; i < iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    RcEndPoint endpoint;
    auto initResult = endpoint.init(0);
    if (!initResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }
    auto rcResult = endpoint.initRc();
    if (!rcResult) {
      counters["error"] = folly::UserMetric(1, folly::UserMetric::Type::METRIC);
      return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    totalTimeUs +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() /
        1000.0;
  }

  double avgTimeUs = totalTimeUs / iters;
  counters["qp_init_us"] =
      folly::UserMetric(avgTimeUs, folly::UserMetric::Type::METRIC);
}

//------------------------------------------------------------------------------
// Benchmark Registration
//------------------------------------------------------------------------------

// Helper macro for registering RDMA write benchmarks with buffer size
#define REGISTER_RDMA_BENCH(func, sizeName, sizeBytes) \
  BENCHMARK_MULTI_PARAM_COUNTERS(func, sizeName, sizeBytes)

// DC RDMA Write benchmarks - powers of 2 from 1B to 2GB
REGISTER_RDMA_BENCH(dcRdmaWrite, 1B, 1);
REGISTER_RDMA_BENCH(dcRdmaWrite, 2B, 2);
REGISTER_RDMA_BENCH(dcRdmaWrite, 4B, 4);
REGISTER_RDMA_BENCH(dcRdmaWrite, 8B, 8);
REGISTER_RDMA_BENCH(dcRdmaWrite, 16B, 16);
REGISTER_RDMA_BENCH(dcRdmaWrite, 32B, 32);
REGISTER_RDMA_BENCH(dcRdmaWrite, 64B, 64);
REGISTER_RDMA_BENCH(dcRdmaWrite, 128B, 128);
REGISTER_RDMA_BENCH(dcRdmaWrite, 256B, 256);
REGISTER_RDMA_BENCH(dcRdmaWrite, 512B, 512);
REGISTER_RDMA_BENCH(dcRdmaWrite, 1KB, 1024);
REGISTER_RDMA_BENCH(dcRdmaWrite, 2KB, 2048);
REGISTER_RDMA_BENCH(dcRdmaWrite, 4KB, 4096);
REGISTER_RDMA_BENCH(dcRdmaWrite, 8KB, 8192);
REGISTER_RDMA_BENCH(dcRdmaWrite, 16KB, 16384);
REGISTER_RDMA_BENCH(dcRdmaWrite, 32KB, 32768);
REGISTER_RDMA_BENCH(dcRdmaWrite, 64KB, 65536);
REGISTER_RDMA_BENCH(dcRdmaWrite, 128KB, 131072);
REGISTER_RDMA_BENCH(dcRdmaWrite, 256KB, 262144);
REGISTER_RDMA_BENCH(dcRdmaWrite, 512KB, 524288);
REGISTER_RDMA_BENCH(dcRdmaWrite, 1MB, 1048576);
REGISTER_RDMA_BENCH(dcRdmaWrite, 2MB, 2097152);
REGISTER_RDMA_BENCH(dcRdmaWrite, 4MB, 4194304);
REGISTER_RDMA_BENCH(dcRdmaWrite, 8MB, 8388608);
REGISTER_RDMA_BENCH(dcRdmaWrite, 16MB, 16777216);
REGISTER_RDMA_BENCH(dcRdmaWrite, 32MB, 33554432);
REGISTER_RDMA_BENCH(dcRdmaWrite, 64MB, 67108864);
REGISTER_RDMA_BENCH(dcRdmaWrite, 128MB, 134217728);
REGISTER_RDMA_BENCH(dcRdmaWrite, 256MB, 268435456);
REGISTER_RDMA_BENCH(dcRdmaWrite, 512MB, 536870912);
REGISTER_RDMA_BENCH(dcRdmaWrite, 1GB, 1073741824);
REGISTER_RDMA_BENCH(dcRdmaWrite, 2GB, 2147483648ULL);

BENCHMARK_DRAW_LINE();

// RC RDMA Write benchmarks - same sizes
REGISTER_RDMA_BENCH(rcRdmaWrite, 1B, 1);
REGISTER_RDMA_BENCH(rcRdmaWrite, 2B, 2);
REGISTER_RDMA_BENCH(rcRdmaWrite, 4B, 4);
REGISTER_RDMA_BENCH(rcRdmaWrite, 8B, 8);
REGISTER_RDMA_BENCH(rcRdmaWrite, 16B, 16);
REGISTER_RDMA_BENCH(rcRdmaWrite, 32B, 32);
REGISTER_RDMA_BENCH(rcRdmaWrite, 64B, 64);
REGISTER_RDMA_BENCH(rcRdmaWrite, 128B, 128);
REGISTER_RDMA_BENCH(rcRdmaWrite, 256B, 256);
REGISTER_RDMA_BENCH(rcRdmaWrite, 512B, 512);
REGISTER_RDMA_BENCH(rcRdmaWrite, 1KB, 1024);
REGISTER_RDMA_BENCH(rcRdmaWrite, 2KB, 2048);
REGISTER_RDMA_BENCH(rcRdmaWrite, 4KB, 4096);
REGISTER_RDMA_BENCH(rcRdmaWrite, 8KB, 8192);
REGISTER_RDMA_BENCH(rcRdmaWrite, 16KB, 16384);
REGISTER_RDMA_BENCH(rcRdmaWrite, 32KB, 32768);
REGISTER_RDMA_BENCH(rcRdmaWrite, 64KB, 65536);
REGISTER_RDMA_BENCH(rcRdmaWrite, 128KB, 131072);
REGISTER_RDMA_BENCH(rcRdmaWrite, 256KB, 262144);
REGISTER_RDMA_BENCH(rcRdmaWrite, 512KB, 524288);
REGISTER_RDMA_BENCH(rcRdmaWrite, 1MB, 1048576);
REGISTER_RDMA_BENCH(rcRdmaWrite, 2MB, 2097152);
REGISTER_RDMA_BENCH(rcRdmaWrite, 4MB, 4194304);
REGISTER_RDMA_BENCH(rcRdmaWrite, 8MB, 8388608);
REGISTER_RDMA_BENCH(rcRdmaWrite, 16MB, 16777216);
REGISTER_RDMA_BENCH(rcRdmaWrite, 32MB, 33554432);
REGISTER_RDMA_BENCH(rcRdmaWrite, 64MB, 67108864);
REGISTER_RDMA_BENCH(rcRdmaWrite, 128MB, 134217728);
REGISTER_RDMA_BENCH(rcRdmaWrite, 256MB, 268435456);
REGISTER_RDMA_BENCH(rcRdmaWrite, 512MB, 536870912);
REGISTER_RDMA_BENCH(rcRdmaWrite, 1GB, 1073741824);
REGISTER_RDMA_BENCH(rcRdmaWrite, 2GB, 2147483648ULL);

BENCHMARK_DRAW_LINE();

// QP Creation benchmarks
BENCHMARK_MULTI_PARAM_COUNTERS(dcQpCreation, dc_qp_init);
BENCHMARK_MULTI_PARAM_COUNTERS(rcQpCreation, rc_qp_init);

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  XLOG(INFO) << "DC vs RC RDMA Single-Pair Benchmark";
  XLOG(INFO) << "===================================";

  if (!checkRdmaAvailable()) {
    XLOG(ERR) << "RDMA not available - benchmarks will be skipped";
  } else {
    XLOG(INFO) << "RDMA devices available - running benchmarks";
  }

  // Suppress folly benchmark table when --raw_only is set
  int savedStdout = -1;
  if (FLAGS_raw_only) {
    savedStdout = dup(STDOUT_FILENO);
    int devNull = open("/dev/null", O_WRONLY);
    dup2(devNull, STDOUT_FILENO);
    close(devNull);
  }

  folly::runBenchmarks();

  if (FLAGS_raw_only && savedStdout >= 0) {
    dup2(savedStdout, STDOUT_FILENO);
    close(savedStdout);
  }

  printAllRawResults();

  return 0;
}
