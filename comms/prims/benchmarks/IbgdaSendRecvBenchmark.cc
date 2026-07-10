// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <mpi.h>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <folly/portability/GFlags.h>
#include <glog/logging.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

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

namespace comms::prims::benchmark {
namespace {

constexpr int kWorldSize = 2;
constexpr int kNumBlocks = 2;
constexpr std::size_t kSlotSize = 8 * 1024 * 1024;
constexpr int kPipelineDepth = 2;
constexpr int kWarmupIters = 5;
constexpr const char* kDefaultBenchmarkIters = "20";
constexpr const char* kDefaultBenchmarkMaxIters = "21";

enum class SendRecvApi {
  Blocking,
  Progress,
};

enum class SendRecvDirection {
  Bidirectional,
  Unidirectional,
};

bool isTcpEnvironment() {
  return std::getenv("MASTER_ADDR") != nullptr &&
      std::getenv("MASTER_PORT") != nullptr && std::getenv("RANK") != nullptr &&
      std::getenv("WORLD_SIZE") != nullptr;
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

const char* apiName(SendRecvApi api) {
  switch (api) {
    case SendRecvApi::Blocking:
      return "blocking";
    case SendRecvApi::Progress:
      return "progress";
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

std::string benchmarkName(
    SendRecvApi api,
    SendRecvDirection direction,
    const char* sizeName) {
  std::string name = "ibgdaSendRecv(";
  name += apiName(api);
  name += "_";
  name += directionName(direction);
  name += "_";
  name += sizeName;
  name += ")";
  return name;
}

class IbgdaSendRecvBenchmarkContext {
 public:
  IbgdaSendRecvBenchmarkContext(
      std::shared_ptr<ITestBootstrap> bootstrap,
      std::size_t maxBytes)
      : bootstrap_(std::move(bootstrap)), maxBytes_(maxBytes) {
    CHECK(bootstrap_ != nullptr);
    CHECK_GT(maxBytes_, 0);
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
        .dataBufferSize = kSlotSize,
        .sendRecv =
            MultipeerIbgdaTransportConfig::SendRecvConfig{
                .maxGroups = kNumBlocks,
                .pipelineDepth = kPipelineDepth,
            },
    };
    transport_ = std::make_unique<MultipeerIbgdaTransport>(
        globalRank_, worldSize_, bootstrap_, transportConfig);
    transport_->exchange();

    sendBuf_ = std::make_unique<DeviceBuffer>(maxBytes_);
    recvBuf_ = std::make_unique<DeviceBuffer>(maxBytes_);
    CHECK_EQ(cudaMemset(sendBuf_->get(), 0xAA, maxBytes_), cudaSuccess);
    CHECK_EQ(cudaMemset(recvBuf_->get(), 0, maxBytes_), cudaSuccess);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

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

  void
  warmup(std::size_t nbytes, SendRecvApi api, SendRecvDirection direction) {
    CHECK_LE(nbytes, maxBytes_);
    bootstrap_->barrierAll();
    for (int i = 0; i < kWarmupIters; ++i) {
      launchOperation(nbytes, api, direction);
      CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
  }

  float runLocalElapsed(
      uint32_t iters,
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction) {
    CHECK_LE(nbytes, maxBytes_);

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_EQ(cudaEventCreate(&start), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&stop), cudaSuccess);

    CHECK_EQ(cudaEventRecord(start, stream_), cudaSuccess);
    for (uint32_t i = 0; i < iters; ++i) {
      launchOperation(nbytes, api, direction);
    }
    CHECK_EQ(cudaEventRecord(stop, stream_), cudaSuccess);
    CHECK_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float elapsedMs = 0.0f;
    CHECK_EQ(cudaEventElapsedTime(&elapsedMs, start, stop), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(start), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(stop), cudaSuccess);

    return elapsedMs;
  }

 private:
  void launchOperation(
      std::size_t nbytes,
      SendRecvApi api,
      SendRecvDirection direction) {
    auto* sendBuf = static_cast<char*>(sendBuf_->get());
    auto* recvBuf = static_cast<char*>(recvBuf_->get());

    if (direction == SendRecvDirection::Bidirectional) {
      if (api == SendRecvApi::Blocking) {
        launch_ibgda_send_recv(
            deviceTransport_, sendBuf, recvBuf, nbytes, kNumBlocks, stream_);
      } else {
        launch_ibgda_progress_send_recv(
            deviceTransport_, sendBuf, recvBuf, nbytes, kNumBlocks, stream_);
      }
      return;
    }

    if (globalRank_ == 0) {
      if (api == SendRecvApi::Blocking) {
        launch_ibgda_send(
            deviceTransport_, sendBuf, nbytes, kNumBlocks, stream_);
      } else {
        launch_ibgda_progress_send(
            deviceTransport_, sendBuf, nbytes, kNumBlocks, stream_);
      }
      return;
    }

    if (api == SendRecvApi::Blocking) {
      launch_ibgda_recv(deviceTransport_, recvBuf, nbytes, kNumBlocks, stream_);
    } else {
      launch_ibgda_progress_recv(
          deviceTransport_, recvBuf, nbytes, kNumBlocks, stream_);
    }
  }

  std::shared_ptr<ITestBootstrap> bootstrap_;
  std::unique_ptr<MultipeerIbgdaTransport> transport_;
  std::unique_ptr<DeviceBuffer> sendBuf_;
  std::unique_ptr<DeviceBuffer> recvBuf_;
  P2pIbgdaTransportDevice* deviceTransport_{nullptr};
  std::size_t maxBytes_{0};
  cudaStream_t stream_{};
  int globalRank_{0};
  int worldSize_{0};
  int localRank_{0};
};

static unsigned int ibgdaSendRecv(
    IbgdaSendRecvBenchmarkContext& context,
    uint32_t iters,
    std::size_t nbytes,
    SendRecvApi api,
    SendRecvDirection direction,
    folly::UserCounters& counters) {
  CHECK_GT(iters, 0);

  BENCHMARK_SUSPEND {
    context.warmup(nbytes, api, direction);
  }

  const float elapsedMs =
      context.runLocalElapsed(iters, nbytes, api, direction);
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
  }
  return iters;
}

void registerBenchmark(
    IbgdaSendRecvBenchmarkContext& context,
    const BenchmarkSize& size,
    SendRecvApi api,
    SendRecvDirection direction) {
  folly::addBenchmark(
      __FILE__,
      benchmarkName(api, direction, size.name),
      [&context, nbytes = size.nbytes, api, direction](
          folly::UserCounters& counters, unsigned int iters) -> unsigned int {
        return ibgdaSendRecv(context, iters, nbytes, api, direction, counters);
      });
}

void registerBenchmarks(IbgdaSendRecvBenchmarkContext& context) {
  for (const auto& size : kBenchmarkSizes) {
    registerBenchmark(
        context, size, SendRecvApi::Blocking, SendRecvDirection::Bidirectional);
    registerBenchmark(
        context, size, SendRecvApi::Progress, SendRecvDirection::Bidirectional);
    registerBenchmark(
        context,
        size,
        SendRecvApi::Blocking,
        SendRecvDirection::Unidirectional);
    registerBenchmark(
        context,
        size,
        SendRecvApi::Progress,
        SendRecvDirection::Unidirectional);
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
