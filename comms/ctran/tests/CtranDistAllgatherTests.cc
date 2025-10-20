// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"

class CtranAllgatherTest : public CtranDistBaseTest {
 public:
  CtranAllgatherTest() = default;
  char expectedVal;
  commDataType_t dt = commBfloat16;
  size_t sendBytes, recvBytes;
  // Allocated buffers
  void *sendbuf, *recvbuf, *pairbuf;
  // Buffers with offset used in collective
  void *sCommBuf, *rCommBuf, *pCommBuf;
  std::vector<void*> segHandles;
  std::vector<TestMemSegment> segments;
  ncclComm_t comm;

  void SetUp() override {
    CtranDistBaseTest::SetUp();
    comm = commWorld;
    segments.clear();
    segHandles.clear();
    if (!ctranAllGatherSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "ctranAllGatherSupport returns fails, skip test";
    }
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void memorySetUp(
      MemAllocType memType,
      size_t offset,
      size_t count,
      TestInPlaceType inplace,
      TestPairCollType pairColl) {
    // Check cumem after comm creation to make sure we have loaded cu symbols
    if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
        ncclIsCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    expectedVal = globalRank;

    size_t offsetBytes = offset * commTypeSize(dt);
    size_t sCommBytes = count * commTypeSize(dt);
    size_t rCommBytes = sCommBytes * numRanks;
    sendbuf = recvbuf = pairbuf = nullptr;
    sCommBuf = rCommBuf = pCommBuf = nullptr;
    sendBytes = offsetBytes + sCommBytes;
    recvBytes = offsetBytes + rCommBytes;

    recvbuf = prepareBuf(pageAligned(recvBytes), memType, segments);
    rCommBuf = reinterpret_cast<char*>(recvbuf) + offsetBytes;
    CUDACHECK_TEST(cudaMemset(rCommBuf, rand(), rCommBytes));

    if (inplace == kTestOutOfPlace) {
      sendbuf = prepareBuf(pageAligned(sendBytes), memType, segments);
      sCommBuf = reinterpret_cast<char*>(sendbuf) + offsetBytes;
      CUDACHECK_TEST(cudaMemset(sCommBuf, expectedVal, sCommBytes));
    } else {
      // correct data for in-place allgather
      sCommBuf = reinterpret_cast<char*>(rCommBuf) + globalRank * sCommBytes;
      CUDACHECK_TEST(cudaMemset(sCommBuf, expectedVal, sCommBytes));
    }

    if (pairColl != kTestPairNone) {
      pairbuf = prepareBuf(pageAligned(sendBytes), memType, segments);
      pCommBuf = reinterpret_cast<char*>(pairbuf) + offsetBytes;
      // set up reduce buffer
      std::vector<int> redExpVal(count, globalRank);
      CUDACHECK_TEST(cudaMemcpy(
          pCommBuf, redExpVal.data(), sCommBytes, cudaMemcpyDefault));
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void memoryCleanUp(
      MemAllocType memType,
      TestInPlaceType inplace,
      TestPairCollType pairColl) {
    if (inplace == kTestOutOfPlace) {
      releaseBuf(sendbuf, pageAligned(sendBytes), memType);
    }
    if (recvbuf) {
      releaseBuf(recvbuf, pageAligned(recvBytes), memType);
    }
    if (pairColl != kTestPairNone) {
      releaseBuf(pairbuf, pageAligned(sendBytes), memType);
    }
  }
};

class CtranAllgatherTestParam : public CtranAllgatherTest,
                                public ::testing::WithParamInterface<std::tuple<
                                    enum NCCL_ALLGATHER_ALGO,
                                    size_t,
                                    size_t,
                                    TestInPlaceType,
                                    MemAllocType,
                                    int,
                                    TestPairCollType>> {};

TEST_P(CtranAllgatherTestParam, AllgatherAlgo) {
  const auto& [algo, offset, count, inplace, memType, iter, pairColl] =
      GetParam();

  // CollTrace will help check whether the specified algo is used
  EnvRAII env(NCCL_ALLGATHER_ALGO, algo);
  // Ensure CollTrace won't drop any record
  EnvRAII envCollTrace(
      NCCL_COLLTRACE_RECORD_MAX, iter * (1 + (pairColl != kTestPairNone)));

  if (memType == kCuMemAllocDisjoint &&
      (!comm->dmaBufSupport || !NCCL_CTRAN_IB_DMABUF_ENABLE)) {
    GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
  }

  // NVL is now using bcast kernel, which will fail with cudaMalloc
  if (comm->ctranComm_->statex_->nLocalRanks() > 1 &&
      memType == kMemCudaMalloc) {
    GTEST_SKIP() << allGatherAlgoName(algo)
                 << " cannot support nLocalRanks > 1 with cudaMalloc"
                 << ", skip test";
  }

  const int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
  if ((algo == NCCL_ALLGATHER_ALGO::ctrd ||
       algo == NCCL_ALLGATHER_ALGO::ctring) &&
      nLocalRanks != 1) {
    GTEST_SKIP() << "Test with " << allGatherAlgoName(algo)
                 << " only supports nLocalRanks=1, but got " << nLocalRanks
                 << ", skip test";
  }

  memorySetUp(memType, offset, count, inplace, pairColl);

  for (auto& segment : segments) {
    void* hdl = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(comm, segment.ptr, segment.size, &hdl));
    segHandles.push_back(hdl);
  }

  // Used for collTrace check
  std::vector<std::string> expOpNames;
  std::vector<std::string> expAlgoNames;

  comm->ctranComm_->collTrace_->resetPastColls();

  for (int x = 0; x < iter; x++) {
    expOpNames.push_back("AllGather");
    expAlgoNames.push_back(allGatherAlgoName(algo));

    auto res = ctranAllGather(
        sCommBuf, rCommBuf, count, dt, comm->ctranComm_.get(), stream);
    EXPECT_EQ(res, commSuccess);

    if (pairColl == kTestPairAllReduce) {
      expOpNames.push_back("AllReduce");
      expAlgoNames.push_back(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctdirect));
      auto res = ctranAllReduceDirect(
          pCommBuf,
          pCommBuf,
          count,
          dt,
          commSum,
          comm->ctranComm_.get(),
          stream);
      EXPECT_EQ(res, commSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
  }

  size_t sCommBytes = count * commTypeSize(dt);
  for (int i = 0; i < numRanks; ++i) {
    std::vector<char> observedVals(sCommBytes, rand());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        reinterpret_cast<char*>(rCommBuf) + sCommBytes * i,
        sCommBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i))
        << " on rank " << globalRank << " received from peer " << i;
  }

  verifyBackendsUsed(
      comm->ctranComm_->ctran_.get(),
      comm->ctranComm_->statex_.get(),
      memType,
      // AllGatherDirect uses kernel bcast not NVL iput
      {CtranMapperBackend::NVL});
  verifyGpeLeak(comm->ctranComm_->ctran_.get());

  // CollTrace is updated by a separate thread, need wait for it to finish to
  // avoid flaky test
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pastColls.size(), expOpNames.size());
  int idx = 0;
  for (auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.opName, expOpNames.at(idx));
    EXPECT_EQ(coll.count, count);
    EXPECT_EQ(coll.dataType, dt);
    EXPECT_EQ(coll.algoName, expAlgoNames.at(idx));
    idx++;
  }

  for (auto& hdl : segHandles) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
  }

  memoryCleanUp(memType, inplace, pairColl);
}

TEST_F(CtranAllgatherTest, OutOfPlaceAllgatherRingDynamicRegist) {
  size_t count = 8192;
  MemAllocType memType = kMemCudaMalloc;

  const int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
  if (nLocalRanks != 1) {
    GTEST_SKIP() << "Test only supports nLocalRanks=1, but got " << nLocalRanks
                 << ", skip test";
  }

  memorySetUp(memType, 0, count, kTestOutOfPlace, kTestPairNone);

  auto res = ctranAllGatherRing(
      sendbuf, recvbuf, count, dt, comm->ctranComm_.get(), stream);
  EXPECT_EQ(res, commSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  for (int i = 0; i < numRanks; ++i) {
    std::vector<char> observedVals(sendBytes, rand());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        (char*)recvbuf + sendBytes * i,
        sendBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i));
  }

  verifyBackendsUsed(
      comm->ctranComm_->ctran_.get(),
      comm->ctranComm_->statex_.get(),
      memType,
      // AllGatherDirect uses kernel bcast not NVL iput
      {CtranMapperBackend::NVL});
  verifyGpeLeak(comm->ctranComm_->ctran_.get());

  memoryCleanUp(memType, kTestOutOfPlace, kTestPairNone);
}

// common function to get test name from test parameter
inline std::string getTestName(
    const testing::TestParamInfo<CtranAllgatherTestParam::ParamType>& info) {
  return allGatherAlgoName(std::get<0>(info.param)) + "_" +
      std::to_string(std::get<1>(info.param)) + "offset_" +
      std::to_string(std::get<2>(info.param)) + "_" + "elements_" +
      testInPlaceTypeToStr(std::get<3>(info.param)) + "_" +
      testMemAllocTypeToStr(std::get<4>(info.param)) + "_" +
      std::to_string(std::get<5>(info.param)) + "iters_" +
      testPairCollTypeToStr(std::get<6>(info.param));
}

INSTANTIATE_TEST_SUITE_P(
    CtranTestCudaMalloc,
    CtranAllgatherTestParam,
    ::testing::Combine(
        testing::Values(
            NCCL_ALLGATHER_ALGO::ctdirect,
            NCCL_ALLGATHER_ALGO::ctring,
            NCCL_ALLGATHER_ALGO::ctrd),
        testing::Values(0),
        testing::Values(8192, 1),
        testing::Values(kTestInPlace, kTestOutOfPlace),
        testing::Values(kMemCudaMalloc),
        testing::Values(1),
        testing::Values(kTestPairNone)),
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTestCumemInPlace,
    CtranAllgatherTestParam,
    ::testing::Combine(
        testing::Values(
            NCCL_ALLGATHER_ALGO::ctdirect,
            NCCL_ALLGATHER_ALGO::ctring,
            NCCL_ALLGATHER_ALGO::ctrd),
        testing::Values(0),
        testing::Values(8192, 1048576, 1048567, 1),
        testing::Values(kTestInPlace),
        testing::Values(kMemNcclMemAlloc),
        testing::Values(1),
        testing::Values(kTestPairNone)),
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTestCumemOutOfPlace,
    CtranAllgatherTestParam,
    ::testing::Combine(
        testing::Values(
            NCCL_ALLGATHER_ALGO::ctdirect,
            NCCL_ALLGATHER_ALGO::ctring,
            NCCL_ALLGATHER_ALGO::ctrd),
        testing::Values(256),
        testing::Values(1048567, 1),
        testing::Values(kTestOutOfPlace),
        testing::Values(kMemNcclMemAlloc),
        testing::Values(1),
        testing::Values(kTestPairNone)),
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTestCumemPair,
    CtranAllgatherTestParam,
    ::testing::Combine(
        testing::Values(
            NCCL_ALLGATHER_ALGO::ctdirect,
            NCCL_ALLGATHER_ALGO::ctring,
            NCCL_ALLGATHER_ALGO::ctrd),
        testing::Values(0),
        testing::Values(8192, 1),
        testing::Values(kTestOutOfPlace),
        testing::Values(kMemNcclMemAlloc),
        testing::Values(20),
        testing::Values(kTestPairAllReduce)),
    getTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTestDisjoint,
    CtranAllgatherTestParam,
    ::testing::Combine(
        testing::Values(NCCL_ALLGATHER_ALGO::ctdirect),
        testing::Values(0, 256),
        testing::Values(1048568, 1),
        testing::Values(kTestInPlace, kTestOutOfPlace),
        testing::Values(kCuMemAllocDisjoint),
        testing::Values(5),
        testing::Values(kTestPairNone)),
    getTestName);

class CtranSocketAllgatherTestParam
    : public CtranAllgatherTest,
      public ::testing::WithParamInterface<
          std::tuple<size_t, size_t, TestInPlaceType, int, TestPairCollType>> {
  void SetUp() override {
    EnvRAII env1(
        NCCL_CTRAN_BACKENDS,
        std::vector<enum NCCL_CTRAN_BACKENDS>{
            NCCL_CTRAN_BACKENDS::socket, NCCL_CTRAN_BACKENDS::nvl});
    CtranAllgatherTest::SetUp();

    if (enableNolocal || localSize < numRanks) {
      GTEST_SKIP()
          << "Ctran Socket + NVL backend require intra-node only environment. Skip test";
    }
  }
};

TEST_P(CtranSocketAllgatherTestParam, AllgatherAlgo) {
  const auto& [offset, count, inplace, iter, pairColl] = GetParam();
  enum NCCL_ALLGATHER_ALGO algo = NCCL_ALLGATHER_ALGO::ctdirect;

  // CollTrace will help check whether the specified algo is used
  EnvRAII env(NCCL_ALLGATHER_ALGO, algo);
  // Ensure CollTrace won't drop any record
  EnvRAII envCollTrace(
      NCCL_COLLTRACE_RECORD_MAX, iter * (1 + (pairColl != kTestPairNone)));

  memorySetUp(kMemNcclMemAlloc, offset, count, inplace, pairColl);

  for (auto& segment : segments) {
    void* hdl = nullptr;
    NCCLCHECK_TEST(ncclCommRegister(comm, segment.ptr, segment.size, &hdl));
    segHandles.push_back(hdl);
  }

  // Used for collTrace check
  std::vector<std::string> expOpNames;
  std::vector<std::string> expAlgoNames;

  comm->ctranComm_->collTrace_->resetPastColls();

  for (int x = 0; x < iter; x++) {
    expOpNames.emplace_back("AllGather");
    expAlgoNames.push_back(allGatherAlgoName(algo));

    auto res = ctranAllGather(
        sCommBuf, rCommBuf, count, dt, comm->ctranComm_.get(), stream);
    EXPECT_EQ(res, commSuccess);

    if (pairColl == kTestPairAllReduce) {
      expOpNames.emplace_back("AllReduce");
      expAlgoNames.push_back(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctdirect));
      auto resAllReduce = ctranAllReduceDirect(
          pCommBuf,
          pCommBuf,
          count,
          dt,
          commSum,
          comm->ctranComm_.get(),
          stream);
      EXPECT_EQ(resAllReduce, commSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
  }

  size_t sCommBytes = count * commTypeSize(dt);
  for (int i = 0; i < numRanks; ++i) {
    std::vector<char> observedVals(sCommBytes, rand());
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        reinterpret_cast<char*>(rCommBuf) + sCommBytes * i,
        sCommBytes,
        cudaMemcpyDefault));
    EXPECT_THAT(observedVals, testing::Each(i))
        << " on rank " << globalRank << " received from peer " << i;
  }

  verifyBackendsUsed(
      comm->ctranComm_->ctran_.get(),
      comm->ctranComm_->statex_.get(),
      kMemNcclMemAlloc,
      // AllGatherDirect uses kernel bcast not NVL iput
      {CtranMapperBackend::NVL});
  verifyGpeLeak(comm->ctranComm_->ctran_.get());

  // CollTrace is updated by a separate thread, need wait for it to finish to
  // avoid flaky test
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pastColls.size(), expOpNames.size());
  int idx = 0;
  for (auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.opName, expOpNames.at(idx));
    EXPECT_EQ(coll.count, count);
    EXPECT_EQ(coll.dataType, dt);
    EXPECT_EQ(coll.algoName, expAlgoNames.at(idx));
    idx++;
  }

  for (auto& hdl : segHandles) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
  }

  memoryCleanUp(kMemNcclMemAlloc, inplace, pairColl);
}

// Socket test name generator function
inline std::string getSocketTestName(
    const testing::TestParamInfo<CtranSocketAllgatherTestParam::ParamType>&
        info) {
  return std::to_string(std::get<0>(info.param)) + "offset_" +
      std::to_string(std::get<1>(info.param)) + "_" + "elements_" +
      testInPlaceTypeToStr(std::get<2>(info.param)) + "_" + "socket_" +
      std::to_string(std::get<3>(info.param)) + "iters_" +
      testPairCollTypeToStr(std::get<4>(info.param));
}

INSTANTIATE_TEST_SUITE_P(
    CtranTestCumemInPlace,
    CtranSocketAllgatherTestParam,
    ::testing::Combine(
        testing::Values(0),
        testing::Values(8192, 1048576, 1048567, 1),
        testing::Values(kTestInPlace),
        testing::Values(1),
        testing::Values(kTestPairNone)),
    getSocketTestName);

INSTANTIATE_TEST_SUITE_P(
    CtranTestCumemOutOfPlace,
    CtranSocketAllgatherTestParam,
    ::testing::Combine(
        testing::Values(256),
        testing::Values(1048567, 1),
        testing::Values(kTestOutOfPlace),
        testing::Values(1),
        testing::Values(kTestPairNone)),
    getSocketTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
