// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class AllGatherTest : public NcclxBaseTest {
 public:
  AllGatherTest() = default;
  void SetUp() override {
    NcclxBaseTest::SetUp();
    this->comm = createNcclComm(
        this->globalRank,
        this->numRanks,
        this->localRank,
        false,
        nullptr,
        server.get());

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECK_TEST(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

  void runTest(
      enum NCCL_ALLGATHER_ALGO algo,
      bool inplace,
      bool registFlag,
      bool useCudaGraph,
      MemAllocType memType,
      size_t count) {
    auto envGuard = EnvRAII(NCCL_ALLGATHER_ALGO, algo);

    if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
        ncclIsCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    if (algo != NCCL_ALLGATHER_ALGO::orig) {
#ifdef TEST_ENABLE_CTRAN
      if (!ctranAllGatherAlgoSupport(comm->ctranComm_.get(), algo)) {
        GTEST_SKIP() << "Ctran algorithm is not supported, skip test";
      }

      // FIXME: this should be fixed in Ctran algo level!
      if (memType == kMemCudaMalloc &&
          comm->ctranComm_->statex_->nLocalRanks() > 1) {
        GTEST_SKIP()
            << "Ctran does not support cudaMalloc-ed buffer with nLocalRanks > 1, skip test";
      }
#else
      GTEST_SKIP() << "Ctran is not enabled, skip test";
#endif
    }

    // Create and register buffers. If inplace, we use the same buffer for send
    // and recv.
    int *sendBuf = nullptr, *recvBuf = nullptr;
    std::vector<TestMemSegment> recvBufSegs, sendBufSegs;
    std::vector<void*> segHandles;

    recvBuf = reinterpret_cast<int*>(
        testAllocBuf(count * numRanks * sizeof(int), memType, recvBufSegs));
    ASSERT_NE(recvBuf, nullptr);

    if (inplace) {
      sendBuf = recvBuf + count * globalRank;
    } else {
      sendBuf = reinterpret_cast<int*>(
          testAllocBuf(count * sizeof(int), memType, sendBufSegs));
      ASSERT_NE(sendBuf, nullptr);
    }

    assignChunkValue(recvBuf, count * numRanks, -1);
    assignChunkValue(sendBuf, count, globalRank + 1);

    if (registFlag) {
      if (!inplace) {
        for (auto& reg : sendBufSegs) {
          void* handle = nullptr;
          NCCLCHECK_TEST(ncclCommRegister(comm, reg.ptr, reg.size, &handle));
          segHandles.push_back(handle);
        }
      }
      for (auto& reg : recvBufSegs) {
        void* handle = nullptr;
        NCCLCHECK_TEST(ncclCommRegister(comm, reg.ptr, reg.size, &handle));
        segHandles.push_back(handle);
      }
    }

    auto fn = [&]() {
      // Run communication
      for (int i = 0; i < 5; i++) {
        auto res =
            ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream);
        ASSERT_EQ(res, ncclSuccess);
      }
    };

    if (useCudaGraph) {
      cudaGraph_t graph;

      // Capture the graph
      CUDACHECK_TEST(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      fn();
      CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));

      // Instantiate the graph
      cudaGraphExec_t graphExec;
      CUDACHECK_TEST(
          cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
      // Launch the graph
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));

      // Destroy the graph
      CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
      CUDACHECK_TEST(cudaGraphDestroy(graph));
    } else {
      fn();
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    // Check each received chunk
    for (int r = 0; r < numRanks; r++) {
      int expectedVal = r + 1;
      int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                         << " at " << recvBuf + r * count << " with " << errs
                         << " errors";
    }

    // Deregister and free buffers
    if (registFlag) {
      for (auto& handle : segHandles) {
        NCCLCHECK_TEST(ncclCommDeregister(comm, handle));
      }
    }

    if (!inplace) {
      testFreeBuf(sendBuf, count * sizeof(int), memType);
    }
    testFreeBuf(recvBuf, count * numRanks * sizeof(int), memType);
  }

 protected:
  ncclComm_t comm{};
  cudaStream_t stream{};
};

class AllgatherTestParam : public AllGatherTest,
                           public ::testing::WithParamInterface<std::tuple<
                               enum NCCL_ALLGATHER_ALGO,
                               bool,
                               bool,
                               MemAllocType,
                               size_t>> {};

TEST_P(AllgatherTestParam, Test) {
  const auto& [algo, inplace, registFlag, memType, count] = GetParam();
  runTest(algo, inplace, registFlag, false /*useCudaGraph*/, memType, count);
}

class AllgatherMemtypeGraphTestParam
    : public AllGatherTest,
      public ::testing::WithParamInterface<
          std::tuple<enum NCCL_ALLGATHER_ALGO, MemAllocType, bool>> {};

TEST_P(AllgatherMemtypeGraphTestParam, Test) {
  const auto& [algo, memType, useCudaGraph] = GetParam();
  auto registFlag = false;
  constexpr size_t count = 10e6 * 8;

  auto GRAPH_REGISTER_OVERRIDE = 1L;
  // TODO: we should throw proper error in the unsupported case in baseline
  // NCCL graph register
  if (algo == NCCL_ALLGATHER_ALGO::orig && useCudaGraph &&
      memType == kCuMemAllocDisjoint) {
    GRAPH_REGISTER_OVERRIDE = 0L;
  }

  auto envGuard = EnvRAII(NCCL_GRAPH_REGISTER, GRAPH_REGISTER_OVERRIDE);
  runTest(algo, false /* inplace */, registFlag, useCudaGraph, memType, count);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherTestInstance,
    AllgatherMemtypeGraphTestParam,
    ::testing::Combine(
        ::testing::Values(
            NCCL_ALLGATHER_ALGO::orig,
            NCCL_ALLGATHER_ALGO::ctran),
        ::testing::Values(
            kMemCudaMalloc,
            kMemNcclMemAlloc,
            kCuMemAllocDisjoint),
        ::testing::Values(true, false)),
    [&](const testing::TestParamInfo<AllgatherMemtypeGraphTestParam::ParamType>&
            info) {
      auto str = allGatherAlgoName(std::get<0>(info.param)) + "_" +
          testMemAllocTypeToStr(std::get<1>(info.param));
      if (std::get<2>(info.param)) {
        str = str + "_graph";
      }
      return str;
    });

INSTANTIATE_TEST_SUITE_P(
    AllGatherTestInstance,
    AllgatherTestParam,
    ::testing::Combine(
        ::testing::Values(
            NCCL_ALLGATHER_ALGO::orig,
            NCCL_ALLGATHER_ALGO::ctran,
            NCCL_ALLGATHER_ALGO::ctrd,
            NCCL_ALLGATHER_ALGO::ctring,
            NCCL_ALLGATHER_ALGO::ctdirect,
            NCCL_ALLGATHER_ALGO::ctbrucks),
        ::testing::Values(true, false), // inplace
        ::testing::Values(true, false), // registFlag
        ::testing::Values(
            kMemCudaMalloc,
            kMemNcclMemAlloc,
            kCuMemAllocDisjoint),
        ::testing::Values(1048576, 1)),

    // algo, inplace, registFlag, memType, count
    [&](const testing::TestParamInfo<AllgatherTestParam::ParamType>& info) {
      return allGatherAlgoName(std::get<0>(info.param)) + "_inplace" +
          std::to_string(std::get<1>(info.param)) + "_register" +
          std::to_string(std::get<2>(info.param)) + "_" +
          testMemAllocTypeToStr(std::get<3>(info.param)) + "_count" +
          std::to_string(std::get<4>(info.param));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
