// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>

#include "CtranDistAlgoDevUTBase.h"
#include "CtranDistAlgoDevUTKernels.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevPerfUTBase.h"

using CtranDistAlgoDevCopyPerfTestParams = std::tuple<
    unsigned int /*nGroups*/,
    size_t /*beginSize*/,
    size_t /*endSize*/,
    int /*warmup*/,
    int /*iters*/>;

class CtranDistAlgoDevCopyPerfTest : public CtranDistAlgoDevPerfTestBase {
 public:
  void SetUp() override {
    CtranDistAlgoDevPerfTestBase::SetUp();
  }

  void TearDown() override {
    CtranDistAlgoDevPerfTestBase::TearDown();
  }

  template <typename T>
  void P2Pbenchmark(
      std::string_view kernName,
      const std::function<void(const void*, void*, size_t, int, cudaStream_t)>&
          kernelLaunchWrapper,
      CtranDistAlgoDevCopyPerfTestParams param) {
    const auto& [nGroups, beginSize, endSize, warmup, iters] = param;
    const int localRank = comm_->ctranComm_->statex_->localRank();
    const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

    std::vector<std::unique_ptr<CtranMapperRequest>> requests;
    int rightRank = (localRank + 1) % nLocalRanks;

    if (comm_->ctranComm_->statex_->rank() == 0) {
      std::cout << std::string(100, '-') << std::endl;
    }

    size_t beginCount = beginSize / sizeof(T);
    size_t endCount = endSize / sizeof(T);

    for (size_t count = beginCount; count <= endCount; count *= 2) {
      size_t totalCount = count;
      size_t totalBytes = totalCount * sizeof(T);
      initIpcBufs<T>(totalCount);
      // Use tmpbuf as source of reduce, and localBuf as destination
      assignVal<T>(localBuf_, totalCount, localRank, true);
      // Ensure data has been stored before IPC access
      intraNodeBarrier(comm_);

      // warmup
      kernelLaunchWrapper(
          localBuf_,
          ipcRemMem_.at(rightRank)->getBase(),
          count,
          warmup,
          stream);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      // benchmark
      CUDACHECK_TEST(cudaEventRecord(start, stream));
      kernelLaunchWrapper(
          localBuf_, ipcRemMem_.at(rightRank)->getBase(), count, iters, stream);
      CUDACHECK_TEST(cudaEventRecord(end, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));
      float timeMs = 0.0;
      CUDACHECK_TEST(cudaEventElapsedTime(&timeMs, start, end));

      if (comm_->ctranComm_->statex_->rank() == 0) {
        auto timeUsPerIter = (timeMs * 1000) / iters;
        auto effBw = totalBytes / (timeMs / iters / 1000) / (1 << 30); // GB/s
        std::cout << "---> Copy Perf: nRanks " << nLocalRanks << ", nSMs "
                  << nGroups << ", msg " << count * sizeof(int) / (1 << 20)
                  << " MB, " << "BW " << std::fixed << std::setprecision(1)
                  << effBw << " GB/s, " << "latency " << std::fixed
                  << std::setprecision(1) << timeUsPerIter << " us"
                  << std::endl;
      }
      // ensure everyone is done before freeing IPC buffer
      intraNodeBarrier(comm_);
      freeIpcBufs();
    }
    if (comm_->ctranComm_->statex_->rank() == 0) {
      std::cout << std::string(100, '-') << std::endl;
    }
  }

  template <typename T>
  void D2Dbenchmark(
      std::string_view kernName,
      const std::function<void(const void*, void*, size_t, int, cudaStream_t)>&
          kernelLaunchWrapper,
      CtranDistAlgoDevCopyPerfTestParams param) {
    const auto& [nGroups, beginSize, endSize, warmup, iters] = param;

    if (comm_->ctranComm_->statex_->rank() == 0) {
      std::cout << std::string(100, '-') << std::endl;
    }

    size_t beginCount = beginSize / sizeof(T);
    size_t endCount = endSize / sizeof(T);

    for (size_t count = beginCount; count <= endCount; count *= 2) {
      size_t totalCount = count;
      size_t totalBytes = totalCount * sizeof(T);
      void* sPtr = nullptr;
      void* dPtr = nullptr;
      NCCLCHECK_TEST(ncclMemAlloc(&sPtr, totalBytes));
      NCCLCHECK_TEST(ncclMemAlloc(&dPtr, totalBytes));

      // warmup
      kernelLaunchWrapper(sPtr, dPtr, count, warmup, stream);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      // benchmark
      CUDACHECK_TEST(cudaEventRecord(start, stream));
      kernelLaunchWrapper(sPtr, dPtr, count, iters, stream);
      CUDACHECK_TEST(cudaEventRecord(end, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));
      float timeMs = 0.0;
      CUDACHECK_TEST(cudaEventElapsedTime(&timeMs, start, end));

      if (comm_->ctranComm_->statex_->rank() == 0) {
        auto timeUsPerIter = (timeMs * 1000) / iters;
        auto effBw = totalBytes / (timeMs / iters / 1000) / (1 << 30); // GB/s
        std::cout << "---> Copy Perf: nSMs " << nGroups << ", msg "
                  << count * sizeof(int) / (1 << 20) << " MB, " << "BW "
                  << std::fixed << std::setprecision(1) << effBw << " GB/s, "
                  << "latency " << std::fixed << std::setprecision(1)
                  << timeUsPerIter << " us" << std::endl;
      }
      NCCLCHECK_TEST(ncclMemFree(sPtr));
      NCCLCHECK_TEST(ncclMemFree(dPtr));
    }
    if (comm_->ctranComm_->statex_->rank() == 0) {
      std::cout << std::string(100, '-') << std::endl;
    }
  }
};

class CtranDistAlgoDevCopyPerfTestParam
    : public CtranDistAlgoDevCopyPerfTest,
      public ::testing::WithParamInterface<CtranDistAlgoDevCopyPerfTestParams> {
};

TEST_P(CtranDistAlgoDevCopyPerfTestParam, KernCopyP2PTest) {
  const auto& [nGroups, beginCount, endCount, warmup, iters] = GetParam();

  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {512, 1, 1};
  using T = int;
  void* fn = getKernCopyFn<T>();
  P2Pbenchmark<T>(
      "KernCopy",
      [&](const void* sendbuff,
          void* recvbuff,
          size_t count,
          int iters,
          cudaStream_t stream) {
        auto devState = comm_->ctranComm_->ctran_->algo->getDevState();
        void* args[] = {&sendbuff, &recvbuff, &count, &iters, &devState};
        CUDACHECK_TEST(cudaLaunchKernel(fn, grid, blocks, args, 0, stream));
      },
      GetParam());
}

TEST_P(CtranDistAlgoDevCopyPerfTestParam, KernCopyD2DTest) {
  const auto& [nGroups, beginCount, endCount, warmup, iters] = GetParam();

  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {256, 1, 1};
  using T = int;
  void* fn = getKernCopyFn<T>();
  D2Dbenchmark<T>(
      "KernCopy",
      [&](const void* sendbuff,
          void* recvbuff,
          size_t count,
          int iters,
          cudaStream_t stream) {
        auto devState = comm_->ctranComm_->ctran_->algo->getDevState();
        void* args[] = {&sendbuff, &recvbuff, &count, &iters, &devState};
        CUDACHECK_TEST(cudaLaunchKernel(fn, grid, blocks, args, 0, stream));
      },
      GetParam());
}

TEST_P(CtranDistAlgoDevCopyPerfTestParam, NaiveCopyD2DTest) {
  const auto& [nGroups, beginCount, endCount, warmup, iters] = GetParam();

  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {256, 1, 1};
  using T = int;
  void* fn = getNaiveKernCopyFn<T>();
  D2Dbenchmark<T>(
      "KernCopy",
      [&](const void* sendbuff,
          void* recvbuff,
          size_t count,
          int iters,
          cudaStream_t stream) {
        auto devState = comm_->ctranComm_->ctran_->algo->getDevState();
        void* args[] = {&sendbuff, &recvbuff, &count, &iters, &devState};
        CUDACHECK_TEST(cudaLaunchKernel(fn, grid, blocks, args, 0, stream));
      },
      GetParam());
}

std::string getTestName(
    const ::testing::TestParamInfo<CtranDistAlgoDevCopyPerfTestParams>& info) {
  const auto& [nGroups, beginCount, endCount, warmup, iters] = info.param;
  std::ostringstream oss;
  oss << "nGroups_" << nGroups << "_beginCount_" << beginCount << "_endCount_"
      << endCount << "_warmup_" << warmup << "_iters_" << iters;
  return oss.str();
}

INSTANTIATE_TEST_SUITE_P(
    CtranDistAlgoDevCopyPerf,
    CtranDistAlgoDevCopyPerfTestParam,
    ::testing::Values(
        std::make_tuple(
            1, // nGroups
            1 << 20, // beginSize (1MB)
            1 << 25, // endSize (32MB)
            100, // warmup
            500), // iters
        std::make_tuple(
            2, // nGroups
            1 << 20, // beginSize (1MB)
            1 << 25, // endSize (32MB)
            100, // warmup
            500), // iters
        std::make_tuple(
            4, // nGroups
            1 << 20, // beginSize (1MB)
            1 << 25, // endSize (32MB)
            100, // warmup
            500), // iters
        std::make_tuple(
            8, // nGroups
            1 << 20, // beginSize (1MB)
            1 << 25, // endSize (32MB)
            100, // warmup
            500), // iters
        std::make_tuple(
            16, // nGroups
            1 << 20, // begin (1MB)
            1 << 25, // endSize (32MB)
            100, // warmup
            500), // iters
        std::make_tuple(
            32, // nGroups
            1 << 20, // begin (1MB)
            1 << 25, // endSize (32MB)
            100, // warmup
            500) // iters
        ),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
