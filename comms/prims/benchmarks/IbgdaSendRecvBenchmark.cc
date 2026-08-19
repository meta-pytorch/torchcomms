// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <mpi.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "comms/prims/benchmarks/IbgdaSendRecv.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/testinfra/ITestBootstrap.h"
#include "comms/testinfra/TcpStoreBootstrap.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::ITestBootstrap;
using meta::comms::MpiBootstrap;
using meta::comms::TcpStoreBootstrap;

DEFINE_bool(
    ibgda_sendrecv_enable_registered,
    false,
    "Enable the registered-source send comparison");
DEFINE_int32(
    ibgda_sendrecv_num_blocks,
    2,
    "Number of sender and receiver blocks");
DEFINE_int64(
    ibgda_sendrecv_per_channel_bytes,
    4 * 1024 * 1024,
    "Transport staging bytes per channel");
DEFINE_int32(ibgda_sendrecv_pipeline_depth, 2, "Transport slots per channel");
DEFINE_int32(
    ibgda_sendrecv_qps_per_connection,
    1,
    "QPs per channel, direction, and NIC");
DEFINE_int32(ibgda_sendrecv_warmup_iters, 5, "Warmup iterations");
DEFINE_uint32(
    ibgda_warp_proxy_queue_depth,
    comms::prims::benchmark::kDefaultIbgdaWarpProxyQueueDepth,
    "Maximum outstanding commands per IB warp proxy queue");

namespace comms::prims::benchmark {
namespace {

constexpr int kWorldSize = 2;
constexpr const char* kDefaultBenchmarkIters = "20";

// Sub-1MB transfers finish below folly's ~100us timing floor at the default
// iteration count, so folly drops their counters (printing
// 0.00fs/Infinity/NaN). Override just those sizes with a high, deterministic
// count: both ranks derive it identically from nbytes, so the paired send/recv
// stays in lockstep. Larger messages already clear the floor and keep folly's
// count.
constexpr std::size_t kSmallMessageThreshold = 1ULL << 20; // 1 MiB
constexpr uint32_t kSmallMessageIters = 2048;

enum class SendRecvApi {
  Blocking,
  Progress,
  RegisteredProgress,
  WarpProxy,
};

enum class SendRecvDirection {
  Bidirectional,
  Unidirectional,
};

// CopyOp policy driven through the transport. Only the fixed-size `Memcpy`
// path is wired here. Kept as an enum so benchmark names stay stable across
// the removal of the ANS variant and a second policy can be re-added without
// renaming every benchmark.
enum class SendRecvCopyOp {
  Memcpy,
};

// Wire protocol: Simple (put data + explicit DATA_READY signal) vs LL
// (low-latency data + inline flag, 2x wire). LL is wired for the blocking path
// only (the resumable Progress API has no LL wire/payload geometry yet).
enum class SendRecvProto {
  Simple,
  LL,
};

bool isTcpEnvironment() {
  return std::getenv("MASTER_ADDR") != nullptr &&
      std::getenv("MASTER_PORT") != nullptr && std::getenv("RANK") != nullptr &&
      std::getenv("WORLD_SIZE") != nullptr;
}

// Read a KEY=VALUE from /etc/nccl.conf (NCCL's global config file). Returns the
// trimmed value, or "" if the file is unreadable or the key is absent. Mirrors
// NCCL's own fallback: an env var takes precedence, otherwise /etc/nccl.conf is
// consulted. Minimal parser: skips blank/`#` lines, trims surrounding
// whitespace around the key and value.
std::string readNcclConf(const std::string& key) {
  std::ifstream conf("/etc/nccl.conf");
  if (!conf.is_open()) {
    return std::string();
  }
  std::string line;
  while (std::getline(conf, line)) {
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    const auto eq = line.find('=', first);
    if (eq == std::string::npos) {
      continue;
    }
    const auto keyEnd = line.find_last_not_of(" \t", eq - 1);
    if (keyEnd == std::string::npos ||
        line.substr(first, keyEnd - first + 1) != key) {
      continue;
    }
    const auto valFirst = line.find_first_not_of(" \t", eq + 1);
    if (valFirst == std::string::npos) {
      return std::string();
    }
    const auto valLast = line.find_last_not_of(" \t\r\n");
    return line.substr(valFirst, valLast - valFirst + 1);
  }
  return std::string();
}

// NIC selection for the benchmark transport. Returns an NCCL_IB_HCA-style
// filter string (e.g. "mlx5_0") for the transport, or "" for PCIe-topology
// auto-discovery. Resolution order matches NCCL: the NCCL_IB_HCA env var wins;
// if it is unset, fall back to /etc/nccl.conf. Needed to pin ranks onto
// rail-aligned NICs (auto-discovery otherwise picks NICs that cannot reach each
// other over RoCE).
std::string benchIbHca() {
  if (const char* hca = std::getenv("NCCL_IB_HCA")) {
    return std::string(hca);
  }
  return readNcclConf("NCCL_IB_HCA");
}

class DistributedBenchmarkEnvironment {
 public:
  DistributedBenchmarkEnvironment() {
    if (isTcpEnvironment()) {
      return;
    }
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
      MPI_Init(nullptr, nullptr);
      ownsMpi_ = true;
    }
  }

  ~DistributedBenchmarkEnvironment() {
    if (!ownsMpi_) {
      return;
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
    }
  }

  DistributedBenchmarkEnvironment(const DistributedBenchmarkEnvironment&) =
      delete;
  DistributedBenchmarkEnvironment& operator=(
      const DistributedBenchmarkEnvironment&) = delete;
  DistributedBenchmarkEnvironment(DistributedBenchmarkEnvironment&&) = delete;
  DistributedBenchmarkEnvironment& operator=(
      DistributedBenchmarkEnvironment&&) = delete;

 private:
  bool ownsMpi_{false};
};

std::shared_ptr<ITestBootstrap> makeBootstrap() {
  if (isTcpEnvironment()) {
    return std::make_shared<TcpStoreBootstrap>();
  }
  return std::make_shared<MpiBootstrap>();
}

void setDefaultBenchmarkFlags() {
  folly::gflags::SetCommandLineOptionWithMode(
      "bm_min_iters",
      kDefaultBenchmarkIters,
      folly::gflags::SET_FLAG_IF_DEFAULT);
}

struct BenchmarkSize {
  const char* name;
  std::size_t nbytes;
};

// ---------------------------------------------------------------------------
// Correctness pass (separate from the timed benchmarks; see
// runCorrectnessSweep)
// ---------------------------------------------------------------------------

constexpr uint32_t kDefaultCorrectnessIters = 32;
constexpr int kCorrectnessPoison = 0xEE;

// Bytes checked past the payload to catch a decoder that writes a padded
// length. LL's quantum is kData = 4, so 3 stray bytes is the worst case; 16
// covers any plausible future packet geometry.
constexpr std::size_t kCorrectnessGuardBytes = 16;

// Correctness verifies on the host, so each iteration costs an H2D of the
// pattern plus a D2H of the result, and holds two host buffers of `nbytes`.
// Simple's sweep runs to 4 GB, which would mean 8 GB of host memory and
// hundreds of GB over PCIe -- so the correctness pass caps the size instead of
// carrying a separate list. Sizes above the cap are skipped and reported (never
// silently dropped). Override with IBGDA_BENCH_CORRECTNESS_MAX_BYTES.
constexpr std::size_t kDefaultCorrectnessMaxBytes = 1ULL << 20; // 1 MiB

// SplitMix64 finalizer: cheap full-avalanche mixing.
inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Payload seed for one (iteration, size) case. Both ranks derive it from
// values they already agree on, so the pattern never has to be exchanged.
// Mixing in `nbytes` keeps two different sizes from sharing a prefix, so a
// stale buffer left by an earlier size cannot match either.
inline uint64_t correctnessSeed(uint32_t iter, std::size_t nbytes) {
  return mix64((static_cast<uint64_t>(iter) << 32) ^ nbytes);
}

// Fill `buf` with `nbytes` of pseudo-random payload for `seed`.
//
// Deliberately NOT a linear ramp: with a constant stride k, a whole-buffer
// shift of `s` bytes reproduces the expected values whenever k*s is a multiple
// of 256, so shifted-copy bugs at those offsets pass unnoticed. Mixing each
// 8 B word independently removes any stride, so no shift aliases.
inline void
fillPattern(std::vector<char>& buf, std::size_t nbytes, uint64_t seed) {
  buf.resize(nbytes);
  std::size_t i = 0;
  for (; i + sizeof(uint64_t) <= nbytes; i += sizeof(uint64_t)) {
    const uint64_t word = mix64(seed ^ i);
    std::memcpy(buf.data() + i, &word, sizeof(word));
  }
  if (i < nbytes) {
    const uint64_t word = mix64(seed ^ i);
    std::memcpy(buf.data() + i, &word, nbytes - i);
  }
}

constexpr std::array<BenchmarkSize, 33> kBenchmarkSizes{{
    {"1B", 1ULL},
    {"2B", 2ULL},
    {"4B", 4ULL},
    {"8B", 8ULL},
    {"16B", 16ULL},
    {"32B", 32ULL},
    {"64B", 64ULL},
    {"128B", 128ULL},
    {"256B", 256ULL},
    {"512B", 512ULL},
    {"1KB", 1ULL << 10},
    {"2KB", 2ULL << 10},
    {"4KB", 4ULL << 10},
    {"8KB", 8ULL << 10},
    {"16KB", 16ULL << 10},
    {"32KB", 32ULL << 10},
    {"64KB", 64ULL << 10},
    {"128KB", 128ULL << 10},
    {"256KB", 256ULL << 10},
    {"512KB", 512ULL << 10},
    {"1MB", 1ULL << 20},
    {"2MB", 2ULL << 20},
    {"4MB", 4ULL << 20},
    {"8MB", 8ULL << 20},
    {"16MB", 16ULL << 20},
    {"32MB", 32ULL << 20},
    {"64MB", 64ULL << 20},
    {"128MB", 128ULL << 20},
    {"256MB", 256ULL << 20},
    {"512MB", 512ULL << 20},
    {"1GB", 1ULL << 30},
    {"2GB", 2ULL << 30},
    {"4GB", 4ULL << 30},
}};

// LL sweep. Capped at 1 MiB: LL puts 2x payload on the wire (8 B packet per
// 4 B of data), so past ~1 MiB it is bandwidth-bound and strictly worse than
// Simple -- the interesting region is small-message latency. 1..64 B is
// enumerated byte-by-byte so every partial-final-packet remainder
// (nbytes % kData, kData = 4) and every packet count up to 16 is covered;
// past 64 B the sweep doubles, with a few deliberately non-4B-aligned sizes
// interleaved to keep exercising the partial final packet at scale.
constexpr std::array<BenchmarkSize, 86> kLlBenchmarkSizes{{
    {"1B", 1ULL},
    {"2B", 2ULL},
    {"3B", 3ULL},
    {"4B", 4ULL},
    {"5B", 5ULL},
    {"6B", 6ULL},
    {"7B", 7ULL},
    {"8B", 8ULL},
    {"9B", 9ULL},
    {"10B", 10ULL},
    {"11B", 11ULL},
    {"12B", 12ULL},
    {"13B", 13ULL},
    {"14B", 14ULL},
    {"15B", 15ULL},
    {"16B", 16ULL},
    {"17B", 17ULL},
    {"18B", 18ULL},
    {"19B", 19ULL},
    {"20B", 20ULL},
    {"21B", 21ULL},
    {"22B", 22ULL},
    {"23B", 23ULL},
    {"24B", 24ULL},
    {"25B", 25ULL},
    {"26B", 26ULL},
    {"27B", 27ULL},
    {"28B", 28ULL},
    {"29B", 29ULL},
    {"30B", 30ULL},
    {"31B", 31ULL},
    {"32B", 32ULL},
    {"33B", 33ULL},
    {"34B", 34ULL},
    {"35B", 35ULL},
    {"36B", 36ULL},
    {"37B", 37ULL},
    {"38B", 38ULL},
    {"39B", 39ULL},
    {"40B", 40ULL},
    {"41B", 41ULL},
    {"42B", 42ULL},
    {"43B", 43ULL},
    {"44B", 44ULL},
    {"45B", 45ULL},
    {"46B", 46ULL},
    {"47B", 47ULL},
    {"48B", 48ULL},
    {"49B", 49ULL},
    {"50B", 50ULL},
    {"51B", 51ULL},
    {"52B", 52ULL},
    {"53B", 53ULL},
    {"54B", 54ULL},
    {"55B", 55ULL},
    {"56B", 56ULL},
    {"57B", 57ULL},
    {"58B", 58ULL},
    {"59B", 59ULL},
    {"60B", 60ULL},
    {"61B", 61ULL},
    {"62B", 62ULL},
    {"63B", 63ULL},
    {"64B", 64ULL},
    // Doubling from 64 B, with non-4B-aligned sizes interleaved in order.
    {"127B", 127ULL},
    {"128B", 128ULL},
    {"256B", 256ULL},
    {"333B", 333ULL},
    {"512B", 512ULL},
    {"1023B", 1023ULL},
    {"1KB", 1ULL << 10},
    {"2KB", 2ULL << 10},
    {"3001B", 3001ULL},
    {"4KB", 4ULL << 10},
    {"8KB", 8ULL << 10},
    {"12345B", 12345ULL},
    {"16KB", 16ULL << 10},
    {"32KB", 32ULL << 10},
    {"64KB", 64ULL << 10},
    {"65537B", 65537ULL},
    {"128KB", 128ULL << 10},
    {"256KB", 256ULL << 10},
    {"300007B", 300007ULL},
    {"512KB", 512ULL << 10},
    {"999983B", 999983ULL},
    {"1MB", 1ULL << 20},
}};

constexpr std::size_t kMaxBenchmarkBytes = 4ULL << 30;

const char* apiName(SendRecvApi api) {
  switch (api) {
    case SendRecvApi::Blocking:
      return "blocking";
    case SendRecvApi::Progress:
      return "progress";
    case SendRecvApi::RegisteredProgress:
      return "registered_progress";
    case SendRecvApi::WarpProxy:
      return "warp_proxy";
  }
  return "unknown";
}

const char* directionName(SendRecvDirection direction) {
  switch (direction) {
    case SendRecvDirection::Bidirectional:
      return "bidirectional";
    case SendRecvDirection::Unidirectional:
      return "unidirectional";
  }
  return "unknown";
}

const char* copyOpName(SendRecvCopyOp copyOp) {
  switch (copyOp) {
    case SendRecvCopyOp::Memcpy:
      return "memcpy";
  }
  return "unknown";
}

const char* protoName(SendRecvProto proto) {
  switch (proto) {
    case SendRecvProto::Simple:
      return "simple";
    case SendRecvProto::LL:
      return "ll";
  }
  return "unknown";
}

std::string benchmarkName(
    SendRecvApi api,
    SendRecvDirection direction,
    SendRecvCopyOp copyOp,
    SendRecvProto proto,
    const char* sizeName,
    int repeat = -1) {
  std::string name = "ibgdaSendRecv(";
  name += protoName(proto);
  name += "_";
  name += apiName(api);
  name += "_";
  name += directionName(direction);
  name += "_";
  name += copyOpName(copyOp);
  name += "_";
  name += sizeName;
  if (repeat >= 0) {
    name += "_rep";
    name += std::to_string(repeat);
  }
  name += ")";
  return name;
}

class IbgdaSendRecvBenchmarkContext {
 public:
  IbgdaSendRecvBenchmarkContext(
      std::shared_ptr<ITestBootstrap> bootstrap,
      std::size_t maxBytes)
      : bootstrap_(std::move(bootstrap)),
        maxBytes_(maxBytes),
        perChannelSize_(
            static_cast<std::size_t>(FLAGS_ibgda_sendrecv_per_channel_bytes)),
        numBlocks_(FLAGS_ibgda_sendrecv_num_blocks),
        pipelineDepth_(FLAGS_ibgda_sendrecv_pipeline_depth),
        warmupIters_(FLAGS_ibgda_sendrecv_warmup_iters),
        registeredEnabled_(FLAGS_ibgda_sendrecv_enable_registered) {
    CHECK(bootstrap_ != nullptr);
    CHECK_GT(maxBytes_, 0);
    CHECK_GT(FLAGS_ibgda_sendrecv_per_channel_bytes, 0);
    CHECK_GT(FLAGS_ibgda_sendrecv_qps_per_connection, 0);
    CHECK_GT(perChannelSize_, 0);
    CHECK_GT(numBlocks_, 0);
    CHECK_GT(pipelineDepth_, 0);
    CHECK_GT(warmupIters_, 0);
    CHECK_EQ(perChannelSize_ % static_cast<std::size_t>(pipelineDepth_), 0);
    if (registeredEnabled_) {
      CHECK_EQ(numBlocks_, 1)
          << "registered-source comparison requires one block";
    }
    globalRank_ = bootstrap_->getGlobalRank();
    worldSize_ = bootstrap_->getWorldSize();
    localRank_ = bootstrap_->getLocalRank();

    CHECK_EQ(worldSize_, kWorldSize)
        << "IBGDA send/recv benchmark requires exactly two ranks";
    int deviceCount = 0;
    CHECK_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    CHECK_GT(deviceCount, localRank_)
        << "Not enough visible CUDA devices for local rank";
    CHECK_EQ(cudaSetDevice(localRank_), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);

    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank_,
        .perChannelSize = perChannelSize_,
        .max_num_channels = numBlocks_,
        .pipelineDepth = pipelineDepth_,
        .qpsPerConnection = FLAGS_ibgda_sendrecv_qps_per_connection,
    };
    transportConfig.ibHca = benchIbHca();
    transport_ = std::make_unique<MultipeerIbgdaTransport>(
        globalRank_, worldSize_, bootstrap_, transportConfig);
    transport_->exchange();
    CHECK_GT(transport_->numNics(), 0);

    sendBuf_ = std::make_unique<DeviceBuffer>(maxBytes_);
    recvBuf_ = std::make_unique<DeviceBuffer>(maxBytes_);
    CHECK_EQ(cudaMemset(sendBuf_->get(), 0xAA, maxBytes_), cudaSuccess);
    CHECK_EQ(cudaMemset(recvBuf_->get(), 0, maxBytes_), cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    if (registeredEnabled_ && globalRank_ == 0) {
      registeredSendBuf_ =
          transport_->registerBuffer(sendBuf_->get(), maxBytes_, true);
    }

    deviceTransport_ = transport_->getP2pTransportDevice(1 - globalRank_);
  }

  ~IbgdaSendRecvBenchmarkContext() {
    CHECK_EQ(cudaSetDevice(localRank_), cudaSuccess);
    if (stream_ != nullptr) {
      CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    if (bootstrap_) {
      bootstrap_->barrierAll();
    }
    if (transport_ && registeredSendBuf_.ptr != nullptr) {
      transport_->deregisterBuffer(sendBuf_->get());
      registeredSendBuf_ = {};
    }
    if (stream_ != nullptr) {
      CHECK_EQ(cudaStreamDestroy(stream_), cudaSuccess);
      stream_ = nullptr;
    }
    recvBuf_.reset();
    sendBuf_.reset();
    transport_.reset();
    bootstrap_.reset();
  }

  IbgdaSendRecvBenchmarkContext(const IbgdaSendRecvBenchmarkContext&) = delete;
  IbgdaSendRecvBenchmarkContext& operator=(
      const IbgdaSendRecvBenchmarkContext&) = delete;
  IbgdaSendRecvBenchmarkContext(IbgdaSendRecvBenchmarkContext&&) = delete;
  IbgdaSendRecvBenchmarkContext& operator=(IbgdaSendRecvBenchmarkContext&&) =
      delete;

  int numBlocks() const {
    return numBlocks_;
  }

  std::size_t pipelineChunkBytes() const {
    return perChannelSize_ / static_cast<std::size_t>(pipelineDepth_);
  }

  std::size_t perChannelSize() const {
    return perChannelSize_;
  }

  int pipelineDepth() const {
    return pipelineDepth_;
  }

  int qpsPerConnection() const {
    return FLAGS_ibgda_sendrecv_qps_per_connection;
  }

  bool registeredEnabled() const {
    return registeredEnabled_;
  }

  void warmup(
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction,
      SendRecvCopyOp copyOp,
      SendRecvProto proto) {
    CHECK_LE(nbytes, maxBytes_);
    bootstrap_->barrierAll();
    for (int i = 0; i < warmupIters_; ++i) {
      launchOperation(nbytes, api, direction, copyOp, proto);
      CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    bootstrap_->barrierAll();
  }

  float runLocalElapsed(
      uint32_t iters,
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction,
      SendRecvCopyOp copyOp,
      SendRecvProto proto) {
    CHECK_LE(nbytes, maxBytes_);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_EQ(cudaEventCreate(&start), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&stop), cudaSuccess);

    CHECK_EQ(cudaEventRecord(start, stream_), cudaSuccess);
    for (uint32_t i = 0; i < iters; ++i) {
      launchOperation(nbytes, api, direction, copyOp, proto);
    }
    CHECK_EQ(cudaEventRecord(stop, stream_), cudaSuccess);
    CHECK_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsedMs = 0.0f;
    CHECK_EQ(cudaEventElapsedTime(&elapsedMs, start, stop), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(start), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(stop), cudaSuccess);

    std::array<float, kWorldSize> rankElapsed{};
    rankElapsed[globalRank_] = elapsedMs;
    CHECK_EQ(
        bootstrap_
            ->allGather(
                rankElapsed.data(), sizeof(float), globalRank_, worldSize_)
            .get(),
        0);
    return *std::max_element(rankElapsed.begin(), rankElapsed.end());
  }

  // One correctness iteration: fill -> transfer -> verify. Untimed.
  //
  // The payload pattern is re-derived from `iter` every call, so a receiver
  // that accepts a previous ring pass's bytes (the stale-flag failure LL's
  // inline readiness is most exposed to) cannot match. recvBuf is poisoned
  // first for the same reason: a recv that writes nothing must not pass on
  // leftovers. Returns false on mismatch and reports the first bad offset.
  bool runVerifiedIteration(
      std::size_t nbytes,
      uint32_t iter,
      SendRecvDirection direction,
      SendRecvProto proto,
      std::size_t& badOffset) {
    // The guard bytes are poisoned and read back past the payload, so the recv
    // buffer must hold them too -- `maxBytes_` alone would let a large
    // IBGDA_BENCH_CORRECTNESS_MAX_BYTES run off the end of the allocation.
    CHECK_LE(nbytes + kCorrectnessGuardBytes, maxBytes_);
    const bool bidir = direction == SendRecvDirection::Bidirectional;
    const bool sends = bidir || globalRank_ == 0;
    const bool receives = bidir || globalRank_ == 1;

    // Both ranks build the same pattern from the same seed: the sender puts it
    // on the wire, the receiver compares against it. No exchange needed.
    fillPattern(hostPattern_, nbytes, correctnessSeed(iter, nbytes));

    if (sends) {
      CHECK_EQ(
          cudaMemcpyAsync(
              sendBuf_->get(),
              hostPattern_.data(),
              nbytes,
              cudaMemcpyHostToDevice,
              stream_),
          cudaSuccess);
    }
    if (receives) {
      // Poison the payload AND a guard region past it. A protocol that decodes
      // a padded length (LL rounds chunks up to kData) would write past nbytes;
      // without the guard those bytes land in allocated-but-unchecked memory
      // and the comparison below, which only covers nbytes, never sees them.
      CHECK_EQ(
          cudaMemsetAsync(
              recvBuf_->get(),
              kCorrectnessPoison,
              nbytes + kCorrectnessGuardBytes,
              stream_),
          cudaSuccess);
    }
    CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // Keep both ranks on the same iteration so the pattern they encode and the
    // pattern the peer expects always agree.
    bootstrap_->barrierAll();
    launchOperation(
        nbytes,
        SendRecvApi::Blocking,
        direction,
        SendRecvCopyOp::Memcpy,
        proto);
    CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    if (!receives) {
      return true;
    }
    hostRecv_.resize(nbytes + kCorrectnessGuardBytes);
    CHECK_EQ(
        cudaMemcpy(
            hostRecv_.data(),
            recvBuf_->get(),
            nbytes + kCorrectnessGuardBytes,
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    // Guard must still be poison: anything else is a write past the payload.
    for (std::size_t g = 0; g < kCorrectnessGuardBytes; ++g) {
      if (hostRecv_[nbytes + g] != static_cast<char>(kCorrectnessPoison)) {
        badOffset = nbytes + g;
        return false;
      }
    }
    if (std::memcmp(hostRecv_.data(), hostPattern_.data(), nbytes) == 0) {
      return true;
    }
    for (std::size_t i = 0; i < nbytes; ++i) {
      if (hostRecv_[i] != hostPattern_[i]) {
        badOffset = i;
        break;
      }
    }
    return false;
  }

  int globalRank() const {
    return globalRank_;
  }

  int numNics() const {
    return transport_->numNics();
  }

  int numLanes() const {
    return numNics() * qpsPerConnection();
  }

 private:
  void launchOperation(
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction,
      SendRecvCopyOp copyOp,
      SendRecvProto proto) {
    auto* sendBuf = static_cast<char*>(sendBuf_->get());
    auto* recvBuf = static_cast<char*>(recvBuf_->get());
    // LL is wired for the blocking path (both directions) and the
    // unidirectional progress path; registerBenchmarks never pairs LL with
    // bidirectional Progress, so that combination is Simple-only below.
    const bool useLL = (proto == SendRecvProto::LL);
    (void)copyOp;

    if (direction == SendRecvDirection::Bidirectional) {
      if (api == SendRecvApi::Blocking) {
        if (useLL) {
          launch_ibgda_send_recv_ll(
              deviceTransport_, sendBuf, recvBuf, nbytes, numBlocks_, stream_);
        } else {
          launch_ibgda_send_recv(
              deviceTransport_, sendBuf, recvBuf, nbytes, numBlocks_, stream_);
        }
      } else {
        CHECK(api == SendRecvApi::Progress);
        launch_ibgda_progress_send_recv(
            deviceTransport_, sendBuf, recvBuf, nbytes, numBlocks_, stream_);
      }
      return;
    }

    if (globalRank_ == 0) {
      if (api == SendRecvApi::Blocking) {
        if (useLL) {
          launch_ibgda_send_ll(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        } else {
          launch_ibgda_send(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        }
      } else if (api == SendRecvApi::RegisteredProgress) {
        CHECK(!useLL);
        CHECK(registeredEnabled_);
        launch_ibgda_registered_progress_send(
            deviceTransport_, registeredSendBuf_, nbytes, numBlocks_, stream_);
      } else if (api == SendRecvApi::Progress) {
        if (useLL) {
          launch_ibgda_progress_send_ll(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        } else if (registeredEnabled_) {
          launch_ibgda_progress_send_complete(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        } else {
          launch_ibgda_progress_send(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        }
      } else if (api == SendRecvApi::WarpProxy) {
        CHECK(!useLL);
        launch_ibgda_warp_proxy_send(
            deviceTransport_,
            sendBuf,
            nbytes,
            numBlocks_,
            stream_,
            /*maxSignalBytes=*/0,
            Timeout(),
            FLAGS_ibgda_warp_proxy_queue_depth);
      } else {
        LOG(FATAL) << "unsupported send/recv API";
      }
      return;
    }

    if (api == SendRecvApi::Blocking) {
      if (useLL) {
        launch_ibgda_recv_ll(
            deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
      } else {
        launch_ibgda_recv(
            deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
      }
    } else if (api == SendRecvApi::Progress) {
      if (useLL) {
        launch_ibgda_progress_recv_ll(
            deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
      } else {
        launch_ibgda_progress_recv(
            deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
      }
    } else if (api == SendRecvApi::RegisteredProgress) {
      CHECK(!useLL);
      CHECK(registeredEnabled_);
      launch_ibgda_progress_recv(
          deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
    } else if (api == SendRecvApi::WarpProxy) {
      CHECK(!useLL);
      launch_ibgda_warp_proxy_recv(
          deviceTransport_,
          recvBuf,
          nbytes,
          numBlocks_,
          stream_,
          /*maxSignalBytes=*/0,
          Timeout(),
          FLAGS_ibgda_warp_proxy_queue_depth);
    } else {
      LOG(FATAL) << "unsupported send/recv API";
    }
  }

  std::shared_ptr<ITestBootstrap> bootstrap_;
  std::unique_ptr<MultipeerIbgdaTransport> transport_;
  std::unique_ptr<DeviceBuffer> sendBuf_;
  std::unique_ptr<DeviceBuffer> recvBuf_;
  IbgdaLocalBuffer registeredSendBuf_{};
  // Correctness-pass staging (unused by the timed benchmarks).
  std::vector<char> hostPattern_;
  std::vector<char> hostRecv_;
  P2pIbgdaTransportDevice* deviceTransport_{nullptr};
  std::size_t maxBytes_{0};
  std::size_t perChannelSize_{0};
  cudaStream_t stream_{};
  int numBlocks_{0};
  int pipelineDepth_{0};
  int warmupIters_{0};
  int globalRank_{0};
  int worldSize_{0};
  int localRank_{0};
  bool registeredEnabled_{false};
};

static unsigned int ibgdaSendRecv(
    IbgdaSendRecvBenchmarkContext& context,
    uint32_t follyIters,
    std::size_t nbytes,
    SendRecvApi api,
    SendRecvDirection direction,
    SendRecvCopyOp copyOp,
    SendRecvProto proto,
    folly::UserCounters& counters) {
  // Small messages would run too briefly at folly's count and be dropped as
  // NaN; use a deterministic high count for them (identical on both ranks, so
  // the paired transfer stays in lockstep). Larger messages keep folly's count.
  const uint32_t iters =
      nbytes < kSmallMessageThreshold ? kSmallMessageIters : follyIters;
  CHECK_GT(iters, 0);

  BENCHMARK_SUSPEND {
    context.warmup(nbytes, api, direction, copyOp, proto);
  }

  const float elapsedMs =
      context.runLocalElapsed(iters, nbytes, api, direction, copyOp, proto);
  folly::doNotOptimizeAway(elapsedMs);

  BENCHMARK_SUSPEND {
    const double totalBytes =
        (direction == SendRecvDirection::Bidirectional ? 2.0 : 1.0) *
        static_cast<double>(nbytes) * iters;
    const double elapsedSec = static_cast<double>(elapsedMs) / 1000.0;
    counters["latency_us"] = folly::UserMetric(
        static_cast<double>(elapsedMs) * 1000.0 / iters,
        folly::UserMetric::Type::METRIC);
    counters["bandwidth_GBps"] = folly::UserMetric(
        (totalBytes / 1e9) / elapsedSec, folly::UserMetric::Type::METRIC);
    counters["message_size"] = folly::UserMetric(
        static_cast<double>(nbytes), folly::UserMetric::Type::METRIC);
    counters["num_blocks"] = folly::UserMetric(
        static_cast<double>(context.numBlocks()),
        folly::UserMetric::Type::METRIC);
    counters["pipeline_chunk_bytes"] = folly::UserMetric(
        static_cast<double>(context.pipelineChunkBytes()),
        folly::UserMetric::Type::METRIC);
    counters["pipeline_depth"] = folly::UserMetric(
        static_cast<double>(context.pipelineDepth()),
        folly::UserMetric::Type::METRIC);
    counters["total_staging_bytes"] = folly::UserMetric(
        static_cast<double>(context.perChannelSize()) * context.numBlocks(),
        folly::UserMetric::Type::METRIC);
    counters["per_channel_window_bytes"] = folly::UserMetric(
        static_cast<double>(context.perChannelSize()),
        folly::UserMetric::Type::METRIC);
    counters["slot_bytes"] = folly::UserMetric(
        static_cast<double>(context.pipelineChunkBytes()),
        folly::UserMetric::Type::METRIC);
    counters["warp_proxy_queue_depth"] = folly::UserMetric(
        static_cast<double>(FLAGS_ibgda_warp_proxy_queue_depth),
        folly::UserMetric::Type::METRIC);
    counters["num_channels"] = folly::UserMetric(
        static_cast<double>(context.numBlocks()),
        folly::UserMetric::Type::METRIC);
    counters["qps_per_connection"] = folly::UserMetric(
        static_cast<double>(context.qpsPerConnection()),
        folly::UserMetric::Type::METRIC);
    counters["num_nics"] = folly::UserMetric(
        static_cast<double>(context.numNics()),
        folly::UserMetric::Type::METRIC);
    counters["num_lanes"] = folly::UserMetric(
        static_cast<double>(context.numLanes()),
        folly::UserMetric::Type::METRIC);
  }
  return iters;
}

void registerBenchmark(
    IbgdaSendRecvBenchmarkContext& context,
    const BenchmarkSize& size,
    SendRecvApi api,
    SendRecvDirection direction,
    SendRecvCopyOp copyOp,
    SendRecvProto proto,
    int repeat = -1) {
  folly::addBenchmark(
      __FILE__,
      benchmarkName(api, direction, copyOp, proto, size.name, repeat),
      [&context, nbytes = size.nbytes, api, direction, copyOp, proto](
          folly::UserCounters& counters, unsigned int iters) -> unsigned int {
        return ibgdaSendRecv(
            context, iters, nbytes, api, direction, copyOp, proto, counters);
      });
}

// A single run uses exactly one protocol, selected via IBGDA_BENCH_PROTO
// (default: simple). Run the binary once per protocol to compare.
// Mixed mode: alternate Simple and LL on the SAME group_id within one run.
// This is the only configuration that exercises the protocol banks against each
// other -- both land on the same hardware QP (channelId % channelsPerBank) but
// must keep separate per-channel state. A bank mix-up shows up here as a
// payload mismatch; running the protocols in separate processes cannot catch
// it.
bool correctnessMixed() {
  if (const char* p = std::getenv("IBGDA_BENCH_PROTO")) {
    return std::string(p) == "mixed";
  }
  return false;
}

SendRecvProto selectProto() {
  SendRecvProto proto = SendRecvProto::Simple;
  if (const char* p = std::getenv("IBGDA_BENCH_PROTO")) {
    const std::string protoStr(p);
    if (protoStr == "ll") {
      proto = SendRecvProto::LL;
    } else if (protoStr == "simple" || protoStr == "mixed") {
      proto = SendRecvProto::Simple;
    } else {
      LOG(WARNING) << "Unknown IBGDA_BENCH_PROTO='" << protoStr
                   << "', defaulting to simple";
    }
  }
  return proto;
}

void registerBenchmarks(IbgdaSendRecvBenchmarkContext& context) {
  const SendRecvProto proto = selectProto();

  // LL uses its own sweep: denser below 64 B and capped at 1 MiB.
  const bool isLl = proto == SendRecvProto::LL;
  const BenchmarkSize* const sizes =
      isLl ? kLlBenchmarkSizes.data() : kBenchmarkSizes.data();
  const std::size_t numSizes =
      isLl ? kLlBenchmarkSizes.size() : kBenchmarkSizes.size();

  for (std::size_t i = 0; i < numSizes; ++i) {
    const auto& size = sizes[i];
    // Blocking + fixed-size memcpy is supported by every protocol.
    registerBenchmark(
        context,
        size,
        SendRecvApi::Blocking,
        SendRecvDirection::Bidirectional,
        SendRecvCopyOp::Memcpy,
        proto);
    registerBenchmark(
        context,
        size,
        SendRecvApi::Blocking,
        SendRecvDirection::Unidirectional,
        SendRecvCopyOp::Memcpy,
        proto);
    // Progress: the unidirectional path is wired for every protocol
    // (Simple + LL); the bidirectional progress kernel is Simple-only.
    if (proto == SendRecvProto::Simple) {
      registerBenchmark(
          context,
          size,
          SendRecvApi::Progress,
          SendRecvDirection::Bidirectional,
          SendRecvCopyOp::Memcpy,
          proto);
    }
    registerBenchmark(
        context,
        size,
        SendRecvApi::Progress,
        SendRecvDirection::Unidirectional,
        SendRecvCopyOp::Memcpy,
        proto);
#ifndef __HIP_PLATFORM_AMD__
    if (proto == SendRecvProto::Simple) {
      registerBenchmark(
          context,
          size,
          SendRecvApi::WarpProxy,
          SendRecvDirection::Unidirectional,
          SendRecvCopyOp::Memcpy,
          proto);
    }
#endif
    if (context.registeredEnabled() && proto == SendRecvProto::Simple) {
      registerBenchmark(
          context,
          size,
          SendRecvApi::RegisteredProgress,
          SendRecvDirection::Unidirectional,
          SendRecvCopyOp::Memcpy,
          proto);
    }

    if (context.registeredEnabled() &&
        (size.nbytes == (64ULL << 20) || size.nbytes == (256ULL << 20) ||
         size.nbytes == (1ULL << 30) || size.nbytes == (2ULL << 30))) {
      for (int repeat = 0; repeat < 5; ++repeat) {
        registerBenchmark(
            context,
            size,
            SendRecvApi::Progress,
            SendRecvDirection::Unidirectional,
            SendRecvCopyOp::Memcpy,
            proto,
            repeat);
        registerBenchmark(
            context,
            size,
            SendRecvApi::RegisteredProgress,
            SendRecvDirection::Unidirectional,
            SendRecvCopyOp::Memcpy,
            proto,
            repeat);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Correctness sweep
//
// A separate, untimed pass over the same size list. Unlike the folly
// benchmarks it verifies EVERY iteration and prints its own table -- no
// latency/bandwidth columns, because nothing here is timed and the
// per-iteration host round-trip would dominate if it were.
// Enabled with IBGDA_BENCH_CORRECTNESS=1; iteration count overridable with
// IBGDA_BENCH_CORRECTNESS_ITERS.
// ---------------------------------------------------------------------------

struct CorrectnessResult {
  const char* name;
  std::size_t nbytes;
  SendRecvDirection direction;
  uint32_t iters;
  uint32_t failures;
  uint32_t firstBadIter;
  std::size_t firstBadOffset;
};

uint32_t correctnessIters() {
  if (const char* s = std::getenv("IBGDA_BENCH_CORRECTNESS_ITERS")) {
    const int parsed = std::atoi(s);
    if (parsed > 0) {
      return static_cast<uint32_t>(parsed);
    }
    LOG(WARNING) << "Ignoring invalid IBGDA_BENCH_CORRECTNESS_ITERS='" << s
                 << "'";
  }
  return kDefaultCorrectnessIters;
}

std::size_t correctnessMaxBytes() {
  if (const char* s = std::getenv("IBGDA_BENCH_CORRECTNESS_MAX_BYTES")) {
    const long long parsed = std::atoll(s);
    if (parsed > 0) {
      return static_cast<std::size_t>(parsed);
    }
    LOG(WARNING) << "Ignoring invalid IBGDA_BENCH_CORRECTNESS_MAX_BYTES='" << s
                 << "'";
  }
  return kDefaultCorrectnessMaxBytes;
}

void printCorrectnessTable(
    const std::vector<CorrectnessResult>& results,
    SendRecvProto proto,
    uint32_t iters,
    int globalRank) {
  std::ostringstream ss;
  ss << "\n=== ibgdaSendRecv correctness (proto="
     << (correctnessMixed() ? "mixed(simple+ll)" : protoName(proto))
     << ", iters=" << iters << "/size, rank=" << globalRank << ") ===\n";
  ss << std::left << std::setw(12) << "size" << std::right << std::setw(12)
     << "bytes" << std::right << std::setw(16) << "direction" << std::right
     << std::setw(10) << "verified" << std::right << std::setw(10) << "failures"
     << std::right << std::setw(22) << "first bad (iter@off)"
     << "\n";
  ss << std::string(82, '-') << "\n";

  uint32_t totalFailures = 0;
  for (const auto& r : results) {
    totalFailures += r.failures;
    ss << std::left << std::setw(12) << r.name << std::right << std::setw(12)
       << r.nbytes << std::right << std::setw(16) << directionName(r.direction)
       << std::right << std::setw(10) << r.iters << std::right << std::setw(10)
       << r.failures << std::right << std::setw(22);
    if (r.failures == 0) {
      ss << "-";
    } else {
      ss
          << (std::to_string(r.firstBadIter) + "@" +
              std::to_string(r.firstBadOffset));
    }
    ss << "\n";
  }
  ss << std::string(82, '-') << "\n";
  ss << (totalFailures == 0 ? "RESULT: PASS" : "RESULT: FAIL") << " ("
     << totalFailures << " failing iterations across " << results.size()
     << " cases)\n";
  std::cout << ss.str() << std::flush;
}

// Returns true if every verified iteration matched.
bool runCorrectnessSweep(IbgdaSendRecvBenchmarkContext& context) {
  const bool mixed = correctnessMixed();
  const SendRecvProto proto = selectProto();
  const uint32_t iters = correctnessIters();
  // Mixed uses the LL size list: every size in it is valid for both protocols.
  // The list stops at 1 MiB because that is where LL has long since become
  // bandwidth-bound (see kLlBenchmarkSizes), not because the protocol imposes
  // any size limit -- LlxPacket computes its geometry in size_t with no cap.
  const bool useLlSizes = mixed || proto == SendRecvProto::LL;
  const BenchmarkSize* const sizes =
      useLlSizes ? kLlBenchmarkSizes.data() : kBenchmarkSizes.data();
  const std::size_t numSizes =
      useLlSizes ? kLlBenchmarkSizes.size() : kBenchmarkSizes.size();

  constexpr std::array<SendRecvDirection, 2> kDirections{
      SendRecvDirection::Bidirectional, SendRecvDirection::Unidirectional};

  std::vector<CorrectnessResult> results;
  results.reserve(numSizes * kDirections.size());

  const std::size_t maxBytes = correctnessMaxBytes();
  std::size_t skipped = 0;
  std::size_t largestSkipped = 0;

  for (std::size_t i = 0; i < numSizes; ++i) {
    const auto& size = sizes[i];
    if (size.nbytes > maxBytes) {
      ++skipped;
      largestSkipped = std::max(largestSkipped, size.nbytes);
      continue;
    }
    for (const auto direction : kDirections) {
      CorrectnessResult r{
          .name = size.name,
          .nbytes = size.nbytes,
          .direction = direction,
          .iters = iters,
          .failures = 0,
          .firstBadIter = 0,
          .firstBadOffset = 0};
      for (uint32_t it = 0; it < iters; ++it) {
        std::size_t badOffset = 0;
        // Mixed: alternate protocols on the SAME group_id so consecutive
        // iterations hit different banks over one shared QP.
        const SendRecvProto iterProto = mixed
            ? ((it % 2 == 0) ? SendRecvProto::Simple : SendRecvProto::LL)
            : proto;
        if (!context.runVerifiedIteration(
                size.nbytes, it, direction, iterProto, badOffset)) {
          if (r.failures == 0) {
            r.firstBadIter = it;
            r.firstBadOffset = badOffset;
          }
          ++r.failures;
        }
      }
      results.push_back(r);
    }
  }

  if (skipped != 0) {
    LOG(WARNING) << "Correctness: skipped " << skipped << " size(s) above "
                 << maxBytes << " B (largest " << largestSkipped
                 << " B); raise IBGDA_BENCH_CORRECTNESS_MAX_BYTES to include "
                 << "them (needs 2x that in host memory per rank)";
  }
  printCorrectnessTable(results, proto, iters, context.globalRank());
  for (const auto& r : results) {
    if (r.failures != 0) {
      return false;
    }
  }
  return true;
}

} // namespace
} // namespace comms::prims::benchmark

int main(int argc, char** argv) {
  if (const char* localRank = std::getenv("LOCAL_RANK")) {
    cudaError_t ret = cudaSetDevice(std::atoi(localRank));
    CHECK_EQ(ret, cudaSuccess) << cudaGetErrorString(ret);
  }
  folly::Init init(&argc, &argv);
  comms::prims::benchmark::setDefaultBenchmarkFlags();
  comms::prims::benchmark::DistributedBenchmarkEnvironment environment;
  auto bootstrap = comms::prims::benchmark::makeBootstrap();
  const int globalRank = bootstrap->getGlobalRank();
  comms::prims::benchmark::IbgdaSendRecvBenchmarkContext context(
      std::move(bootstrap), comms::prims::benchmark::kMaxBenchmarkBytes);

  // Correctness mode is a distinct run: it verifies every iteration and prints
  // its own table, so it deliberately does not also run the timed benchmarks
  // (the verified path is not representative of steady-state timing). Both
  // ranks participate; each prints the results for the transfers it received.
  if (const char* c = std::getenv("IBGDA_BENCH_CORRECTNESS");
      c != nullptr && std::string(c) != "0") {
    return comms::prims::benchmark::runCorrectnessSweep(context) ? 0 : 1;
  }

  comms::prims::benchmark::registerBenchmarks(context);
  // Both ranks must run every benchmark in lockstep (paired send/recv), but
  // only rank 0 prints the results table. Rank 1 runs and discards its output.
  if (globalRank == 0) {
    folly::runBenchmarks();
  } else {
    // NOLINTNEXTLINE(facebook-hte-DetailCall)
    (void)folly::detail::runBenchmarksWithResults();
  }
  return 0;
}
