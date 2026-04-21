// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <algorithm>
#include <thread>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class CtranAllgatherTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  CtranAllgatherTest() = default;
  char expectedVal;
  commDataType_t dt = commBfloat16;
  size_t sendBytes, recvBytes;
  // Allocated buffers
  void *sendbuf, *recvbuf, *pairbuf;
  // Buffers with offset used in collective
  void *sCommBuf, *rCommBuf, *pCommBuf;
  std::vector<TestMemSegment> segments;

  void SetUp() override {
    setenv("NCCL_CTRAN_TRANSPORT_PROFILER", "1", 0);
    setenv("NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT", "1", 0);
    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
    segments.clear();
  }

  static void checkProfiler(ctran::Profiler* profiler) {
    if (!profiler) {
      return;
    }
    uint64_t algoTotal =
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_TOTAL);
    uint64_t algoCtrl =
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_CTRL);
    uint64_t algoData =
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_DATA);
    uint64_t bufReg =
        profiler->getEventDurationUs(ctran::ProfilerEvent::BUF_REG);
    uint64_t oneMinUs = 1000 * 1000 * 60;
    EXPECT_GE(algoTotal, 0);
    EXPECT_LE(algoTotal, oneMinUs);
    EXPECT_GE(algoCtrl, 0);
    EXPECT_LE(algoCtrl, oneMinUs);
    EXPECT_GE(algoData, 0);
    EXPECT_LE(algoData, oneMinUs);
    EXPECT_GE(bufReg, 0);
    EXPECT_LE(bufReg, oneMinUs);
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
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

  // Drain colltrace, parse pastColls, and sort by collId (submission order).
  folly::dynamic getPastColls() {
    EXPECT_NE(ctranComm->colltraceNew_, nullptr);
    auto dumpMap = ctran::waitForCollTraceDrain(ctranComm.get());
    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColls"], "[]");
    auto pastColls = folly::parseJson(dumpMap["CT_pastColls"]);
    std::sort(
        pastColls.begin(),
        pastColls.end(),
        [](const folly::dynamic& a, const folly::dynamic& b) {
          return a["collId"].asInt() < b["collId"].asInt();
        });
    return pastColls;
  }

 protected:
  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
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

  if (memType == kCuMemAllocDisjoint && !NCCL_CTRAN_IB_DMABUF_ENABLE) {
    GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
  }

  // NVL is now using bcast kernel, which will fail with cudaMalloc
  if (ctranComm->statex_->nLocalRanks() > 1 && memType == kMemCudaMalloc) {
    GTEST_SKIP() << allGatherAlgoName(algo)
                 << " cannot support nLocalRanks > 1 with cudaMalloc"
                 << ", skip test";
  }

  const int nLocalRanks = ctranComm->statex_->nLocalRanks();
  if (!ctranAllGatherSupport(ctranComm.get(), algo)) {
    GTEST_SKIP() << "Test with " << allGatherAlgoName(algo)
                 << " only supports nLocalRanks=1, but got " << nLocalRanks
                 << ", skip test";
  }

  memorySetUp(memType, offset, count, inplace, pairColl);

  for (auto& segment : segments) {
    COMMCHECK_TEST(ctran::globalRegisterWithPtr(segment.ptr, segment.size));
  }

  // Used for collTrace check
  std::vector<std::string> expOpNames;
  std::vector<std::string> expAlgoNames;

  ASSERT_TRUE(
      meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

  for (int x = 0; x < iter; x++) {
    expOpNames.push_back("AllGather");
    expAlgoNames.push_back(allGatherAlgoName(algo));

    if (!ctranAllGatherSupport(ctranComm.get(), algo)) {
      GTEST_SKIP() << "ctranAllGatherSupport returns fails, skip test";
    }
    auto res = ctranAllGather(
        sCommBuf, rCommBuf, count, dt, ctranComm.get(), testStream, algo);
    EXPECT_EQ(res, commSuccess);

    if (pairColl == kTestPairAllReduce) {
      expOpNames.push_back("AllReduce");
      expAlgoNames.push_back(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctdirect));
      auto res = ctranAllReduceDirect(
          pCommBuf, pCommBuf, count, dt, commSum, ctranComm.get(), testStream);
      EXPECT_EQ(res, commSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));

    // Verify profiler event durations are populated
    checkProfiler(ctranComm->ctran_->profiler.get());
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
      ctranComm->ctran_.get(),
      ctranComm->statex_.get(),
      memType,
      // AllGatherDirect uses kernel bcast not NVL iput
      {CtranMapperBackend::NVL});
  verifyGpeLeak(ctranComm->ctran_.get());

  CUDACHECK_TEST(cudaDeviceSynchronize());
  auto pastCollsJson = getPastColls();
  EXPECT_EQ(pastCollsJson.size(), expOpNames.size());
  int idx = 0;
  for (const auto& coll : pastCollsJson) {
    EXPECT_EQ(coll["opName"].asString(), expOpNames.at(idx));
    EXPECT_EQ(coll["count"].asInt(), count);
    EXPECT_THAT(
        coll["algoName"].asString(), testing::HasSubstr(expAlgoNames.at(idx)));
    idx++;
  }

  for (auto& segment : segments) {
    COMMCHECK_TEST(ctran::globalDeregisterWithPtr(segment.ptr, segment.size));
  }

  memoryCleanUp(memType, inplace, pairColl);
}

TEST_F(CtranAllgatherTest, OutOfPlaceAllgatherRingDynamicRegist) {
  size_t count = 8192;
  MemAllocType memType = kMemCudaMalloc;

  const int nLocalRanks = ctranComm->statex_->nLocalRanks();
  if (nLocalRanks != 1) {
    GTEST_SKIP() << "Test only supports nLocalRanks=1, but got " << nLocalRanks
                 << ", skip test";
  }

  memorySetUp(memType, 0, count, kTestOutOfPlace, kTestPairNone);

  auto res = ctranAllGatherRing(
      sendbuf, recvbuf, count, dt, ctranComm.get(), testStream);
  EXPECT_EQ(res, commSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(testStream));

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
      ctranComm->ctran_.get(),
      ctranComm->statex_.get(),
      memType,
      // AllGatherDirect uses kernel bcast not NVL iput
      {CtranMapperBackend::NVL});
  verifyGpeLeak(ctranComm->ctran_.get());

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

  memorySetUp(kMemNcclMemAlloc, offset, count, inplace, pairColl);

  for (auto& segment : segments) {
    COMMCHECK_TEST(ctran::globalRegisterWithPtr(segment.ptr, segment.size));
  }

  // Used for collTrace check
  std::vector<std::string> expOpNames;
  std::vector<std::string> expAlgoNames;

  ASSERT_TRUE(
      meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

  for (int x = 0; x < iter; x++) {
    expOpNames.emplace_back("AllGather");
    expAlgoNames.push_back(allGatherAlgoName(algo));

    if (!ctranAllGatherSupport(ctranComm.get(), algo)) {
      GTEST_SKIP() << "ctranAllGatherSupport returns fails, skip test";
    }
    auto res = ctranAllGather(
        sCommBuf, rCommBuf, count, dt, ctranComm.get(), testStream, algo);

    EXPECT_EQ(res, commSuccess);

    if (pairColl == kTestPairAllReduce) {
      expOpNames.emplace_back("AllReduce");
      expAlgoNames.push_back(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctdirect));
      auto resAllReduce = ctranAllReduceDirect(
          pCommBuf, pCommBuf, count, dt, commSum, ctranComm.get(), testStream);
      EXPECT_EQ(resAllReduce, commSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));
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
      ctranComm->ctran_.get(),
      ctranComm->statex_.get(),
      kMemNcclMemAlloc,
      // AllGatherDirect uses kernel bcast not NVL iput
      {CtranMapperBackend::NVL});
  verifyGpeLeak(ctranComm->ctran_.get());

  CUDACHECK_TEST(cudaDeviceSynchronize());
  auto pastCollsJson = getPastColls();
  EXPECT_EQ(pastCollsJson.size(), expOpNames.size());
  int idx = 0;
  for (const auto& coll : pastCollsJson) {
    EXPECT_EQ(coll["opName"].asString(), expOpNames.at(idx));
    EXPECT_EQ(coll["count"].asInt(), count);
    EXPECT_THAT(
        coll["algoName"].asString(), testing::HasSubstr(expAlgoNames.at(idx)));
    idx++;
  }

  for (auto& segment : segments) {
    COMMCHECK_TEST(ctran::globalDeregisterWithPtr(segment.ptr, segment.size));
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
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
