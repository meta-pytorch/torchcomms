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
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "comms/prims/benchmarks/IbgdaSendRecv.h"
#include "comms/prims/benchmarks/IbgdaSendRecvAns.h"
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

namespace comms::prims::benchmark {
namespace {

constexpr int kWorldSize = 2;
constexpr const char* kDefaultBenchmarkIters = "20";
constexpr const char* kDefaultBenchmarkMaxIters = "21";

enum class SendRecvApi {
  Blocking,
  Progress,
  RegisteredProgress,
};

enum class SendRecvDirection {
  Bidirectional,
  Unidirectional,
};

// CopyOp policy driven through the transport. `Memcpy` is the fixed-size path;
// `Ans` drives the variable-size (compressed) `AnsCompress` CopyOp, exercising
// the compressed send/recv branch added in D111967119.
enum class SendRecvCopyOp {
  Memcpy,
  Ans,
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
  folly::gflags::SetCommandLineOptionWithMode(
      "bm_max_iters",
      kDefaultBenchmarkMaxIters,
      folly::gflags::SET_FLAG_IF_DEFAULT);
  folly::gflags::SetCommandLineOptionWithMode(
      "bm_max_trials", "1", folly::gflags::SET_FLAG_IF_DEFAULT);
}

struct BenchmarkSize {
  const char* name;
  std::size_t nbytes;
};

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

constexpr std::size_t kMaxBenchmarkBytes = 4ULL << 30;

// ANS (variable-size) benchmarks run only over this size window. Below 1MB the
// per-chunk compress/decompress cost dominates (see AnsCompress
// kActivationThreshold), and the window is capped so the compressed sweep
// stays bounded in wall-clock time.
constexpr std::size_t kAnsMinBytes = 1ULL << 20;
constexpr std::size_t kAnsMaxBytes = 256ULL << 20;

const char* apiName(SendRecvApi api) {
  switch (api) {
    case SendRecvApi::Blocking:
      return "blocking";
    case SendRecvApi::Progress:
      return "progress";
    case SendRecvApi::RegisteredProgress:
      return "registered_progress";
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
    case SendRecvCopyOp::Ans:
      return "ans";
  }
  return "unknown";
}

std::string benchmarkName(
    SendRecvApi api,
    SendRecvDirection direction,
    SendRecvCopyOp copyOp,
    const char* sizeName,
    int repeat = -1) {
  std::string name = "ibgdaSendRecv(";
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

  bool registeredEnabled() const {
    return registeredEnabled_;
  }

  void warmup(
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction,
      SendRecvCopyOp copyOp) {
    CHECK_LE(nbytes, maxBytes_);
    bootstrap_->barrierAll();
    for (int i = 0; i < warmupIters_; ++i) {
      launchOperation(nbytes, api, direction, copyOp);
      CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    bootstrap_->barrierAll();
  }

  float runLocalElapsed(
      uint32_t iters,
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction,
      SendRecvCopyOp copyOp) {
    CHECK_LE(nbytes, maxBytes_);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_EQ(cudaEventCreate(&start), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&stop), cudaSuccess);

    CHECK_EQ(cudaEventRecord(start, stream_), cudaSuccess);
    for (uint32_t i = 0; i < iters; ++i) {
      launchOperation(nbytes, api, direction, copyOp);
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

 private:
  void launchOperation(
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction,
      SendRecvCopyOp copyOp) {
    auto* sendBuf = static_cast<char*>(sendBuf_->get());
    auto* recvBuf = static_cast<char*>(recvBuf_->get());

    if (copyOp == SendRecvCopyOp::Ans) {
      // ANS is a variable-size CopyOp: only the blocking, unidirectional path
      // is wired here (the resumable progress API static_asserts against
      // variable-size CopyOps). rank 0 compresses+sends, rank 1 recvs+decomp.
      if (globalRank_ == 0) {
        launch_ibgda_send_ans(
            deviceTransport_, sendBuf, nbytes, kNumBlocks, stream_);
      } else {
        launch_ibgda_recv_ans(
            deviceTransport_, recvBuf, nbytes, kNumBlocks, stream_);
      }
      return;
    }

    if (direction == SendRecvDirection::Bidirectional) {
      if (api == SendRecvApi::Blocking) {
        launch_ibgda_send_recv(
            deviceTransport_, sendBuf, recvBuf, nbytes, numBlocks_, stream_);
      } else {
        launch_ibgda_progress_send_recv(
            deviceTransport_, sendBuf, recvBuf, nbytes, numBlocks_, stream_);
      }
      return;
    }

    if (globalRank_ == 0) {
      if (api == SendRecvApi::Blocking) {
        launch_ibgda_send(
            deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
      } else if (api == SendRecvApi::RegisteredProgress) {
        CHECK(registeredEnabled_);
        launch_ibgda_registered_progress_send(
            deviceTransport_, registeredSendBuf_, nbytes, numBlocks_, stream_);
      } else {
        if (registeredEnabled_) {
          launch_ibgda_progress_send_complete(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        } else {
          launch_ibgda_progress_send(
              deviceTransport_, sendBuf, nbytes, numBlocks_, stream_);
        }
      }
      return;
    }

    if (api == SendRecvApi::Blocking) {
      launch_ibgda_recv(deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
    } else {
      launch_ibgda_progress_recv(
          deviceTransport_, recvBuf, nbytes, numBlocks_, stream_);
    }
  }

  std::shared_ptr<ITestBootstrap> bootstrap_;
  std::unique_ptr<MultipeerIbgdaTransport> transport_;
  std::unique_ptr<DeviceBuffer> sendBuf_;
  std::unique_ptr<DeviceBuffer> recvBuf_;
  IbgdaLocalBuffer registeredSendBuf_{};
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
    uint32_t iters,
    std::size_t nbytes,
    SendRecvApi api,
    SendRecvDirection direction,
    SendRecvCopyOp copyOp,
    folly::UserCounters& counters) {
  CHECK_GT(iters, 0);

  BENCHMARK_SUSPEND {
    context.warmup(nbytes, api, direction, copyOp);
  }

  const float elapsedMs =
      context.runLocalElapsed(iters, nbytes, api, direction, copyOp);
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
  }
  return iters;
}

void registerBenchmark(
    IbgdaSendRecvBenchmarkContext& context,
    const BenchmarkSize& size,
    SendRecvApi api,
    SendRecvDirection direction,
    SendRecvCopyOp copyOp,
    int repeat = -1) {
  folly::addBenchmark(
      __FILE__,
      benchmarkName(api, direction, copyOp, size.name, repeat),
      [&context, nbytes = size.nbytes, api, direction, copyOp](
          folly::UserCounters& counters, unsigned int iters) -> unsigned int {
        return ibgdaSendRecv(
            context, iters, nbytes, api, direction, copyOp, counters);
      });
}

void registerBenchmarks(IbgdaSendRecvBenchmarkContext& context) {
  for (const auto& size : kBenchmarkSizes) {
    registerBenchmark(
        context,
        size,
        SendRecvApi::Blocking,
        SendRecvDirection::Bidirectional,
        SendRecvCopyOp::Memcpy);
    registerBenchmark(
        context,
        size,
        SendRecvApi::Progress,
        SendRecvDirection::Bidirectional,
        SendRecvCopyOp::Memcpy);
    registerBenchmark(
        context,
        size,
        SendRecvApi::Blocking,
        SendRecvDirection::Unidirectional,
        SendRecvCopyOp::Memcpy);
    registerBenchmark(
        context,
        size,
        SendRecvApi::Progress,
        SendRecvDirection::Unidirectional,
        SendRecvCopyOp::Memcpy);
    if (context.registeredEnabled()) {
      registerBenchmark(
          context,
          size,
          SendRecvApi::RegisteredProgress,
          SendRecvDirection::Unidirectional,
          SendRecvCopyOp::Memcpy);
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
            repeat);
        registerBenchmark(
            context,
            size,
            SendRecvApi::RegisteredProgress,
            SendRecvDirection::Unidirectional,
            SendRecvCopyOp::Memcpy,
            repeat);
      }
    }

    // ANS (variable-size CopyOp): blocking + unidirectional only, over a
    // bounded size window. This is the transport benchmark's coverage of the
    // compressed send/recv path from D111967119.
    if (size.nbytes >= kAnsMinBytes && size.nbytes <= kAnsMaxBytes) {
      registerBenchmark(
          context,
          size,
          SendRecvApi::Blocking,
          SendRecvDirection::Unidirectional,
          SendRecvCopyOp::Ans);
    }
  }
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
  comms::prims::benchmark::IbgdaSendRecvBenchmarkContext context(
      std::move(bootstrap), comms::prims::benchmark::kMaxBenchmarkBytes);
  comms::prims::benchmark::registerBenchmarks(context);
  folly::runBenchmarks();
  return 0;
}
