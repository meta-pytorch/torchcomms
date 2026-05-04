#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD port of comms/pipes/benchmarks/IbgdaBenchmark.cc
// Same benchmark tests but uses MultipeerIbgdaTransportAmd + HIP APIs.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <hip/hip_runtime.h>

#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#include "HipDeviceBuffer.h"
#include "IbgdaBenchmarkAmdKernels.h"
#include "MultipeerIbgdaTransportAmd.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"

using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace pipes_gda::benchmark {

constexpr int kIbgdaBatchIters = 1000;

#define HIP_CHECK_VOID(call)        \
  do {                              \
    hipError_t err = call;          \
    if (err != hipSuccess) {        \
      XLOGF(                        \
          ERR,                      \
          "HIP error at {}:{}: {}", \
          __FILE__,                 \
          __LINE__,                 \
          hipGetErrorString(err));  \
      return;                       \
    }                               \
  } while (0)

struct IbgdaBenchmarkConfig {
  std::size_t nBytes = 0;
  std::string name;
};

struct IbgdaBenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  float bandwidth{}; // GB/s
  float latency{}; // microseconds
};

using ::pipes_gda::HipDeviceBuffer;

class IbgdaBenchmarkAmdFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    HIP_CHECK_VOID(hipSetDevice(localRank));
    HIP_CHECK_VOID(hipStreamCreate(&stream_));

    // AMD GPU clock: wall_clock64() runs at 100 MHz on MI200/MI300
    clockRateGHz_ = 0.1f; // 100 MHz = 0.1 GHz
  }

  void TearDown() override {
    HIP_CHECK_VOID(hipStreamDestroy(stream_));
    MpiBaseTestFixture::TearDown();
  }

  float cyclesToUs(unsigned long long cycles) const {
    return cycles / (clockRateGHz_ * 1000.0f);
  }

  std::string formatSize(std::size_t bytes) {
    std::stringstream ss;
    if (bytes >= 1024 * 1024 * 1024) {
      ss << std::fixed << std::setprecision(0)
         << (bytes / (1024.0 * 1024.0 * 1024.0)) << "GB";
    } else if (bytes >= 1024 * 1024) {
      ss << std::fixed << std::setprecision(0) << (bytes / (1024.0 * 1024.0))
         << "MB";
    } else if (bytes >= 1024) {
      ss << std::fixed << std::setprecision(0) << (bytes / 1024.0) << "KB";
    } else {
      ss << bytes << "B";
    }
    return ss.str();
  }

  void printResultsTable(
      const std::string& title,
      const std::vector<IbgdaBenchmarkResult>& results) {
    if (globalRank != 0)
      return;

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "                    " << title << "\n";
    ss << "                    (Raw latency, no kernel launch overhead)\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(20) << "Test Name" << std::right
       << std::setw(12) << "Msg Size" << std::right << std::setw(14)
       << "BW (GB/s)" << std::right << std::setw(14) << "Latency (us)\n";
    ss << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(20) << r.testName << std::right
         << std::setw(12) << formatSize(r.messageSize) << std::right
         << std::setw(14) << std::fixed << std::setprecision(2) << r.bandwidth
         << std::right << std::setw(14) << std::fixed << std::setprecision(2)
         << r.latency << "\n";
    }
    ss << "================================================================================\n";
    ss << "AMD GPU wall_clock64() @ " << clockRateGHz_ << " GHz, "
       << kIbgdaBatchIters << " iterations\n";
    ss << "================================================================================\n\n";

    XLOG(INFO) << ss.str();
  }

  std::vector<IbgdaBenchmarkConfig> getFullConfigs() {
    return {
        {.nBytes = 8, .name = "8B"},
        {.nBytes = 64, .name = "64B"},
        {.nBytes = 256, .name = "256B"},
        {.nBytes = 1024, .name = "1KB"},
        {.nBytes = 4 * 1024, .name = "4KB"},
        {.nBytes = 8 * 1024, .name = "8KB"},
        {.nBytes = 16 * 1024, .name = "16KB"},
        {.nBytes = 32 * 1024, .name = "32KB"},
        {.nBytes = 64 * 1024, .name = "64KB"},
        {.nBytes = 128 * 1024, .name = "128KB"},
        {.nBytes = 256 * 1024, .name = "256KB"},
        {.nBytes = 512 * 1024, .name = "512KB"},
        {.nBytes = 1024 * 1024, .name = "1MB"},
        {.nBytes = 2 * 1024 * 1024, .name = "2MB"},
        {.nBytes = 4 * 1024 * 1024, .name = "4MB"},
        {.nBytes = 8 * 1024 * 1024, .name = "8MB"},
        {.nBytes = 16 * 1024 * 1024, .name = "16MB"},
        {.nBytes = 32 * 1024 * 1024, .name = "32MB"},
        {.nBytes = 64 * 1024 * 1024, .name = "64MB"},
        {.nBytes = 128 * 1024 * 1024, .name = "128MB"},
    };
  }

  hipStream_t stream_{};
  float clockRateGHz_{0.0f};
};

TEST_F(IbgdaBenchmarkAmdFixture, PutWaitLocal) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  auto configs = getFullConfigs();

  std::size_t maxBufferSize = 0;
  for (const auto& config : configs)
    maxBufferSize = std::max(maxBufferSize, config.nBytes);

  std::vector<IbgdaBenchmarkResult> results;

  try {
    MultipeerIbgdaTransportAmdConfig transportConfig{
        .hipDevice = localRank,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransportAmd transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    HipDeviceBuffer dataBuffer(maxBufferSize);
    auto localDataBuf =
        transport.registerBuffer(dataBuffer.get(), maxBufferSize);
    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    unsigned long long* d_totalCycles;
    HIP_CHECK_VOID(hipMalloc(&d_totalCycles, sizeof(unsigned long long)));

    for (const auto& config : configs) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (globalRank == 0) {
        launchIbgdaPutWaitLocalBatch(
            deviceTransportPtr,
            localDataBuf,
            remoteDataBuf,
            config.nBytes,
            kIbgdaBatchIters,
            d_totalCycles,
            stream_);
        HIP_CHECK_VOID(hipStreamSynchronize(stream_));

        unsigned long long totalCycles;
        HIP_CHECK_VOID(hipMemcpy(
            &totalCycles,
            d_totalCycles,
            sizeof(unsigned long long),
            hipMemcpyDeviceToHost));

        IbgdaBenchmarkResult result;
        result.testName = config.name;
        result.messageSize = config.nBytes;
        result.latency = cyclesToUs(totalCycles) / kIbgdaBatchIters;
        result.bandwidth = (config.nBytes / 1e9f) / (result.latency / 1e6f);
        results.push_back(result);
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    HIP_CHECK_VOID(hipFree(d_totalCycles));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "AMD IBGDA transport not available: " << e.what();
  }

  printResultsTable("AMD IBGDA Put+WaitLocal (RDMA Write)", results);
}

TEST_F(IbgdaBenchmarkAmdFixture, PutSignalWaitLocal) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  auto configs = getFullConfigs();
  constexpr int kSignalId = 0;

  std::size_t maxBufferSize = 0;
  for (const auto& config : configs)
    maxBufferSize = std::max(maxBufferSize, config.nBytes);

  std::vector<IbgdaBenchmarkResult> results;

  try {
    MultipeerIbgdaTransportAmdConfig transportConfig{
        .hipDevice = localRank,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransportAmd transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    HipDeviceBuffer dataBuffer(maxBufferSize);
    auto localDataBuf =
        transport.registerBuffer(dataBuffer.get(), maxBufferSize);
    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    HipDeviceBuffer signalBuffer(sizeof(uint64_t));
    HIP_CHECK_VOID(hipMemset(signalBuffer.get(), 0, sizeof(uint64_t)));
    auto localSignalBuf =
        transport.registerBuffer(signalBuffer.get(), sizeof(uint64_t));
    auto remoteSignalBufs = transport.exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    unsigned long long* d_totalCycles;
    HIP_CHECK_VOID(hipMalloc(&d_totalCycles, sizeof(unsigned long long)));

    for (const auto& config : configs) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (globalRank == 0) {
        launchIbgdaPutSignalWaitLocalBatch(
            deviceTransportPtr,
            localDataBuf,
            remoteDataBuf,
            config.nBytes,
            remoteSignalBuf,
            kSignalId,
            kIbgdaBatchIters,
            d_totalCycles,
            stream_);
        HIP_CHECK_VOID(hipStreamSynchronize(stream_));

        unsigned long long totalCycles;
        HIP_CHECK_VOID(hipMemcpy(
            &totalCycles,
            d_totalCycles,
            sizeof(unsigned long long),
            hipMemcpyDeviceToHost));

        IbgdaBenchmarkResult result;
        result.testName = config.name;
        result.messageSize = config.nBytes;
        result.latency = cyclesToUs(totalCycles) / kIbgdaBatchIters;
        result.bandwidth = (config.nBytes / 1e9f) / (result.latency / 1e6f);
        results.push_back(result);
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    HIP_CHECK_VOID(hipFree(d_totalCycles));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "AMD IBGDA transport not available: " << e.what();
  }

  printResultsTable(
      "AMD IBGDA Put+Signal+WaitLocal (RDMA Write + Atomic)", results);
}

TEST_F(IbgdaBenchmarkAmdFixture, SignalOnly) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalId = 0;

  try {
    MultipeerIbgdaTransportAmdConfig transportConfig{
        .hipDevice = localRank,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransportAmd transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);

    HipDeviceBuffer signalBuffer(sizeof(uint64_t));
    HIP_CHECK_VOID(hipMemset(signalBuffer.get(), 0, sizeof(uint64_t)));
    auto localSignalBuf =
        transport.registerBuffer(signalBuffer.get(), sizeof(uint64_t));
    auto remoteSignalBufs = transport.exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    unsigned long long* d_totalCycles;
    HIP_CHECK_VOID(hipMalloc(&d_totalCycles, sizeof(unsigned long long)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      launchIbgdaSignalOnlyBatch(
          deviceTransportPtr,
          remoteSignalBuf,
          kSignalId,
          kIbgdaBatchIters,
          d_totalCycles,
          stream_);
      HIP_CHECK_VOID(hipStreamSynchronize(stream_));

      unsigned long long totalCycles;
      HIP_CHECK_VOID(hipMemcpy(
          &totalCycles,
          d_totalCycles,
          sizeof(unsigned long long),
          hipMemcpyDeviceToHost));

      float latencyUs = cyclesToUs(totalCycles) / kIbgdaBatchIters;

      XLOGF(
          INFO,
          "\n=== AMD IBGDA Signal-Only Latency ===\n"
          "Average latency: {:.2f} us\n"
          "Batch iterations: {}\n"
          "=====================================\n",
          latencyUs,
          kIbgdaBatchIters);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    HIP_CHECK_VOID(hipFree(d_totalCycles));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "AMD IBGDA transport not available: " << e.what();
  }
}

} // namespace pipes_gda::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
#endif
