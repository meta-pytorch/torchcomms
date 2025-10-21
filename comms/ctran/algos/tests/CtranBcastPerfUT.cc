// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>

#include "CtranDistAlgoDevUTBase.h"
#include "CtranDistAlgoDevUTKernels.h"

class CtranDistAlgoDevBcastPerfTestParamFixture
    : public CtranDistAlgoDevTest,
      public ::testing::WithParamInterface<std::tuple<
          BcastTestAlgo,
          ElemTestType,
          int, /* nSteps*/
          unsigned int, /* nGroups*/
          size_t, /* count */
          size_t>> /*last count*/
{};

static void checkElemListFree(KernelElem* elemList) {
  KernelElem* elem = elemList;
  while (elem) {
    EXPECT_TRUE(elem->isFree());
    elem = elem->next;
  }
}

static void checkElemListCompleteAndFree(KernelElem* elemList) {
  KernelElem* elem = elemList;
  while (elem) {
    EXPECT_TRUE(elem->isComplete());
    elem->free();
    elem = elem->next;
  }
}

TEST_P(CtranDistAlgoDevBcastPerfTestParamFixture, Bcast_Peft) {
  const auto& [testAlgo, testType, nSteps, nGroups, count, lastCount] =
      GetParam();

  ASSERT_TRUE(count >= lastCount) << "lastCount should be smaller than count";

  cudaEvent_t start, stop;
  CUDACHECK_TEST(cudaEventCreate(&start));
  CUDACHECK_TEST(cudaEventCreate(&stop));

  cudaStream_t stream = nullptr;
  const int localRank = comm_->ctranComm_->statex_->localRank();
  const int nLocalRanks = comm_->ctranComm_->statex_->nLocalRanks();

  CUDACHECK_TEST(cudaStreamCreate(&stream));

  KernelElem* bcastElem = nullptr;
  COMMCHECK_TEST(
      comm_->ctranComm_->ctran_->gpe->allocKernelElems(1, nGroups, &bcastElem));

  initIpcBufs<int>(count * nSteps * nLocalRanks);
  // Use localBuf as sendbuf, and ipcBuf as recvbuf in an allgather patter
  assignVal<int>(localBuf_, count * nSteps, localRank, true);
  assignVal<int>(ipcBuf_, count * nSteps * nLocalRanks, rand());
  // Ensure data has been stored before IPC access
  CUDACHECK_TEST(cudaDeviceSynchronize());
  intraNodeBarrier(comm_);

  // Submit bcast kernel
  // Always barrier + flush to ensure every rank has finished copy and every
  // rank can check the local received data on host side
  bcastElem->bcast.barrier = true;
  bcastElem->bcast.flushMem = true;
  bcastElem->bcast.nvectors = nLocalRanks;

  dim3 grid = {nGroups, 1, 1};
  dim3 blocks = {256, 1, 1};
  CUDACHECK_TEST(cudaEventRecord(start, stream));
  testKernBcastWrapper(
      testAlgo,
      testType,
      grid,
      blocks,
      stream,
      bcastElem,
      nSteps,
      nGroups,
      comm_->ctranComm_->ctran_->algo->getDevState());

  for (int i = 0; i < nSteps; i++) {
    // Repost the same bcast op nSteps times, similar to usage in collective
    // algorithm
    size_t srcStart = count * i * sizeof(int);
    size_t curCount = (i == nSteps - 1) ? lastCount : count;
    size_t dstStart = count * (nLocalRanks * i + localRank) * sizeof(int);

    bcastElem->bcast.count = curCount;
    bcastElem->bcast.src = reinterpret_cast<char*>(localBuf_) + srcStart;
    for (int peer = 0; peer < nLocalRanks; peer++) {
      char* dstBase = peer == localRank
          ? reinterpret_cast<char*>(ipcMem_->getBase())
          : reinterpret_cast<char*>(ipcRemMem_.at(peer)->getBase());
      bcastElem->bcast.dsts[peer] = dstBase + dstStart;
    }
    bcastElem->post();
    bcastElem->wait();
  }
  CUDACHECK_TEST(cudaEventRecord(stop, stream));

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  if (testType == kTestElemComplete) {
    checkElemListCompleteAndFree(bcastElem);
  } else {
    checkElemListFree(bcastElem);
  }

  cudaEventSynchronize(stop);
  if (localRank == 0) {
    float atime = 0.0f;
    CUDACHECK_TEST(cudaEventElapsedTime(&atime, start, stop));
    atime /= nSteps;
    const auto eff_bw = count * (1 + nLocalRanks) * sizeof(int) /
        (atime / 1000) / (1 << 30); // GB/s
    std::cout << "---> Bcast Perf: nRanks " << nLocalRanks << ", nSMs "
              << nGroups << ", msg " << count * sizeof(int) / (1 << 20)
              << " MB, " << "BW " << std::fixed << std::setprecision(1)
              << eff_bw << " GB/s, " << "latency " << std::fixed
              << std::setprecision(1) << atime * 1000 << " us" << std::endl;
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  freeIpcBufs();
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(
    CtranDistAlgoDevTest,
    CtranDistAlgoDevBcastPerfTestParamFixture,
    ::testing::Values(
        // [testAlgo, testType, nSteps, nGroups, count, lastCount] =
        // first perf numbers are not reliable due to wormup
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            64,
            32 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            1,
            1 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            1,
            4 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            1,
            16 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            1,
            32 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            2,
            1 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            2,
            4 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            2,
            16 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            2,
            32 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            4,
            1 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            4,
            4 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            4,
            16 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            4,
            32 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            8,
            1 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            8,
            4 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            8,
            16 * 1024 * 1024,
            0),
        std::make_tuple(
            kTestDefaultBcast,
            kTestElemComplete,
            5,
            8,
            32 * 1024 * 1024,
            0)),
    [&](const testing::TestParamInfo<
        CtranDistAlgoDevBcastPerfTestParamFixture::ParamType>& info) {
      auto algo = std::get<0>(info.param);
      std::string algoStr =
          (algo == kTestDefaultBcast) ? "defaultBcast_" : "multiPutBcast_";
      return algoStr + "testElemComplte_" +
          std::to_string(std::get<2>(info.param)) + "nSteps" +
          std::to_string(std::get<3>(info.param)) + "groups_" +
          std::to_string(std::get<4>(info.param)) + "int_" +
          std::to_string(std::get<5>(info.param)) + "last";
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
