// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>

#include "comms/ctran/algos/tests/CtranAlgoDevTestUtils.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevPerfUTBase.h"
#include "comms/testinfra/TestsDistUtils.h"

using CtranDistDequantizedReduceParams = std::tuple<
    unsigned int /*nGroups*/,
    size_t /*beginCount*/,
    size_t /*endCount*/,
    commRedOp_t /*redOp*/,
    int /*warmup*/,
    int /*iters*/>;

class CtranKernDequantizedReducePerf : public CtranDistAlgoDevPerfTestBase {
 public:
  void SetUp() override {
    CtranDistAlgoDevPerfTestBase::SetUp();
  }

  void TearDown() override {
    CtranDistAlgoDevPerfTestBase::TearDown();
  }

  template <typename T, typename RedT>
  void benchmark(
      std::string_view kernName,
      const std::function<void(const void*, void*, size_t, cudaStream_t)>&
          kernelLaunchWrapper,
      CtranDistDequantizedReduceParams param) {
    const auto& [nGroups, beginCount, endCount, op, warmup, iters] = param;

    const int localRank = comm_->ctranComm_->statex_->localRank();
    const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

    if (comm_->ctranComm_->statex_->rank() == 0) {
      std::cout << std::string(100, '-') << std::endl;
    }

    for (size_t count = beginCount; count <= endCount; count *= 2) {
      size_t totalCount = count * nLocalRanks;
      size_t bytesPerRank = totalCount * sizeof(T);
      size_t totalBytes = totalCount * sizeof(T);
      initIpcBufs<T>(totalCount);
      // Use ipcBuf as source of reduce, and localBuf as destination
      assignVal<T>(ipcBuf_, totalCount, localRank, true);
      assignVal<T>(localBuf_, totalCount, rand());
      // Ensure data has been stored before IPC access
      intraNodeBarrier(comm_);

      float timeMs = 0.0;
      for (int i = 0; i < warmup + iters; ++i) {
        CUDACHECK_TEST(cudaEventRecord(start, stream));
        kernelLaunchWrapper(ipcBuf_, localBuf_, count, stream);
        CUDACHECK_TEST(cudaEventRecord(end, stream));

        CUDACHECK_TEST(cudaStreamSynchronize(stream));
        if (i >= warmup) {
          float iterTimeMs = 0.0;
          CUDACHECK_TEST(cudaEventElapsedTime(&iterTimeMs, start, end));
          timeMs += iterTimeMs;
        }
      }

      if (comm_->ctranComm_->statex_->rank() == 0) {
        auto timeUsPerIter = (timeMs * 1000) / iters;
        std::cout << "[" << kernName << "-" << typeid(T).name() << "] Rank-"
                  << comm_->ctranComm_->statex_->rank() << ", nRanks "
                  << comm_->ctranComm_->statex_->nRanks() << ", redOp "
                  << commOpToString(op) << ", nGroups " << nGroups << ", count "
                  << std::setw(9) << count << ", nbytesPerRank " << std::setw(9)
                  << bytesPerRank << " => " << std::fixed
                  << std::setprecision(2) << std::setw(9) << timeUsPerIter
                  << " us per kernel, " << std::fixed << std::setprecision(2)
                  << (totalBytes / (timeUsPerIter)) << " MB/s" << std::endl;
      }
      // ensure everyone is done before freeing IPC buffer
      intraNodeBarrier(comm_);
      freeIpcBufs();
    }
    if (comm_->ctranComm_->statex_->rank() == 0) {
      std::cout << std::string(100, '-') << std::endl;
    }
  }

  // Clean up buffers
  template <typename T>
  void cleanupBuffers(T* sendbuff_d, T* recvbuff_d) {
    CUDACHECK_TEST(cudaFree(sendbuff_d));
    CUDACHECK_TEST(cudaFree(recvbuff_d));
  }
};

class CtranKernDequantizedReducePerfTestParam
    : public CtranKernDequantizedReducePerf,
      public ::testing::WithParamInterface<CtranDistDequantizedReduceParams> {};

TEST_P(
    CtranKernDequantizedReducePerfTestParam,
    ctranKernDequantizedReducePerf) {
  const auto& [nGroups, beginCount, endCount, op, warmup, iters] = GetParam();

  // For this test, we'll use int as T and int as RedT
  using T = int;
  using RedT = int;

  void* fn = getDequantizedAllToAllReduceKernelFn<T, RedT>(op);
  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {1024, 1, 1};

  benchmark<T, RedT>(
      "DequantizedReduce",
      [&](const void* sendbuff,
          void* recvbuff,
          size_t count,
          cudaStream_t stream) {
        auto devState = comm_->ctranComm_->ctran_->algo->getDevState();
        void* args[] = {&sendbuff, &recvbuff, &count, &devState};
        CUDA_LAUNCH_KERNEL_WITH_DYNAMIC_SHM_TEST(
            fn, grid, blocks, args, sizeof(CtranAlgoDeviceState), stream);
      },
      GetParam());
}

std::string getTestName(
    const ::testing::TestParamInfo<CtranDistDequantizedReduceParams>& info) {
  const auto& [nGroups, beginCount, endCount, op, warmup, iters] = info.param;
  std::ostringstream oss;
  oss << "nGroups_" << nGroups << "_beginCount_" << beginCount << "_endCount_"
      << endCount << "_op_" << commOpToString(op) << "_warmup_" << warmup
      << "_iters_" << iters;
  return oss.str();
}

INSTANTIATE_TEST_SUITE_P(
    CtranKernDequantizedReducePerf,
    CtranKernDequantizedReducePerfTestParam,
    ::testing::Values(
        std::make_tuple(
            8,
            1 << 20, // 1MB
            1 << 26, // 32MB
            commSum,
            10,
            100)),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
