// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "nccl.h"

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class CtranAllgatherPTest : public CtranDistBaseTest {
 public:
  CtranAllgatherPTest() = default;
  char expectedVal;
  commDataType_t dt = commBfloat16;
  size_t sendBytes, recvBytes;
  void *sendbuf, *recvbuf;
  void *sendHdl, *recvHdl;
  CtranComm* ctranComm{nullptr};

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistBaseTest::SetUp();

    ctranComm = commWorld->ctranComm_.get();
    if (!ctran::allGatherPSupport(ctranComm)) {
      GTEST_SKIP() << "allGatherPSupport returns false, skip test";
    }
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  char* prepareBuf(size_t bufSize, MemAllocType memType) {
    void* buf = nullptr;
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    } else {
      NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
    }
    return reinterpret_cast<char*>(buf);
  }

  void releaseBuf(char* buf, MemAllocType memType) {
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(buf));
    } else {
      ncclMemFree(buf);
    }
  }

  void
  memorySetUp(MemAllocType memType, size_t sendCount, size_t maxRecvCount) {
    // Check cumem after comm creation to make sure we have loaded cu symbols
    if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    expectedVal = globalRank;

    const size_t pageSize = getpagesize();
    sendbuf = recvbuf = nullptr;
    sendHdl = recvHdl = nullptr;
    sendBytes = sendCount * commTypeSize(dt);
    recvBytes = maxRecvCount * commTypeSize(dt);

    size_t bufSize;
    bufSize = ((sendBytes + pageSize - 1) / pageSize) * pageSize;
    sendbuf = prepareBuf(bufSize, memType);
    bufSize = ((recvBytes + pageSize - 1) / pageSize) * pageSize;
    recvbuf = prepareBuf(bufSize, memType);

    CUDACHECK_TEST(cudaMemset(sendbuf, expectedVal, sendBytes));
    CUDACHECK_TEST(cudaMemset(recvbuf, rand(), recvBytes));
    // correct data for in-place allgather
    CUDACHECK_TEST(cudaMemset(
        (char*)recvbuf + globalRank * sendBytes, expectedVal, sendBytes));

    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void memoryCleanUp(MemAllocType memType) {
    releaseBuf((char*)sendbuf, memType);
    releaseBuf((char*)recvbuf, memType);
  }

  void run(
      size_t maxSendCount,
      size_t count,
      TestInPlaceType inplace,
      MemAllocType memType,
      bool sendbufReg = true) {
    const auto maxRecvCount = maxSendCount * numRanks;
    memorySetUp(memType, count, maxRecvCount);
    if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
      GTEST_SKIP() << "No IB Backend found, skip test";
    }

    void* usedSendBuf = sendbuf;
    COMMCHECK_TEST(
        ctranComm->ctran_->commRegister(recvbuf, recvBytes, &recvHdl));
    if (inplace == kTestInPlace) {
      usedSendBuf = (char*)recvbuf + globalRank * sendBytes;
    } else if (sendbufReg) {
      COMMCHECK_TEST(
          ctranComm->ctran_->commRegister(sendbuf, sendBytes, &sendHdl));
    }
    meta::comms::Hints hints;
    CtranPersistentRequest* request;
    // Convert to int8_t for init to mimic FSDP use case
    const auto initMaxRecvCount =
        maxRecvCount * commTypeSize(dt) / commTypeSize(commInt8);
    const auto initDt = commInt8;
    COMMCHECK_TEST(
        ctran::allGatherPInit(
            recvbuf,
            initMaxRecvCount,
            hints,
            initDt,
            ctranComm,
            stream,
            request));

    constexpr int nIter = 5;
    for (int j = 0; j < nIter; j++) {
      // change the values in sendbuff in each iteration
      const int sendVal = j * 10 + globalRank;
      std::vector<char> sendVals(sendBytes, sendVal);
      ASSERT_EQ(
          cudaMemcpyAsync(
              usedSendBuf,
              sendVals.data(),
              sendBytes,
              cudaMemcpyDefault,
              stream),
          cudaSuccess);

      ASSERT_EQ(
          ctran::allGatherPExec(usedSendBuf, count, dt, request), commSuccess);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

      for (int i = 0; i < numRanks; ++i) {
        std::vector<char> observedVals(sendBytes, rand());
        ASSERT_EQ(
            cudaMemcpy(
                observedVals.data(),
                (char*)recvbuf + sendBytes * i,
                sendBytes,
                cudaMemcpyDefault),
            cudaSuccess);
        EXPECT_THAT(observedVals, testing::Each(i + j * 10))
            << "at rank " << globalRank << " in iteration " << j
            << " at chunk received from peer " << i;
      }
    }

    verifyGpeLeak(ctranComm->ctran_.get());

    ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
    delete request;

    if (inplace == kTestOutOfPlace && sendbufReg) {
      COMMCHECK_TEST(ctranComm->ctran_->commDeregister(sendHdl));
    }
    COMMCHECK_TEST(ctranComm->ctran_->commDeregister(recvHdl));

    memoryCleanUp(memType);
  }
};

class CtranAllgatherPTestParam
    : public CtranAllgatherPTest,
      public ::testing::WithParamInterface<std::tuple<
          size_t,
          size_t,
          TestInPlaceType,
          MemAllocType,
          enum NCCL_ALLGATHER_P_ALGO>> {};

TEST_P(CtranAllgatherPTestParam, Basic) {
  const auto& [maxSendCount, count, inplace, memType, algo] = GetParam();

  EnvRAII algoEnv(NCCL_ALLGATHER_P_ALGO, algo);

  run(maxSendCount, count, inplace, memType);
}

TEST_F(CtranAllgatherPTestParam, DynamicSendRegDirect) {
  const auto count = 8192;
  const auto maxSendCount = count * numRanks;
  const auto inplace = kTestOutOfPlace;
  const auto memType = kMemNcclMemAlloc;

  EnvRAII algoEnv(NCCL_ALLGATHER_P_ALGO, NCCL_ALLGATHER_P_ALGO::ctdirect);
  run(maxSendCount, count, inplace, memType, false /* sendbufReg */);
}

TEST_F(CtranAllgatherPTestParam, DynamicSendRegPipeline) {
  const auto count = 8192;
  const auto maxSendCount = count * numRanks;
  const auto inplace = kTestOutOfPlace;
  const auto memType = kMemNcclMemAlloc;

  EnvRAII algoEnv(NCCL_ALLGATHER_P_ALGO, NCCL_ALLGATHER_P_ALGO::ctpipeline);
  run(maxSendCount, count, inplace, memType, false /* sendbufReg */);
}

TEST_F(CtranAllgatherPTest, InvalidPreq) {
  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLTOALL_P, ctranComm, stream);
  void* sendBuf = reinterpret_cast<void*>(0x9000);
  ASSERT_EQ(
      ctran::allGatherPExec(sendBuf, 64, commInt32, request.get()),
      commInvalidArgument);
}

TEST_F(CtranAllgatherPTest, InvalidCount) {
  const auto count = 65536;
  const auto maxRecvCount = 8192 * numRanks;
  memorySetUp(kMemNcclMemAlloc, count, maxRecvCount);

  COMMCHECK_TEST(ctranComm->ctran_->commRegister(recvbuf, recvBytes, &recvHdl));
  COMMCHECK_TEST(ctranComm->ctran_->commRegister(sendbuf, sendBytes, &sendHdl));
  meta::comms::Hints hints;
  CtranPersistentRequest* request;

  const auto initDt = commInt8;
  const auto initMaxRecvCount =
      maxRecvCount * commTypeSize(dt) / commTypeSize(initDt);
  // Convert to int8_t for init to mimic FSDP use case
  COMMCHECK_TEST(
      ctran::allGatherPInit(
          recvbuf,
          initMaxRecvCount,
          hints,
          initDt,
          ctranComm,
          stream,
          request));

  // count * sizeof(dt) * numRanks must be less than maxRecvCount *
  // sizeof(initDt)
  ASSERT_EQ(
      ctran::allGatherPExec(sendbuf, count, dt, request), commInvalidArgument);

  // Check count < initMaxRecvCount / numRanks, but > maxRecvCount / numRanks,
  // to ensure it compares based on bytes
  const auto count1 = maxRecvCount / numRanks + 1;
  ASSERT_EQ(
      ctran::allGatherPExec(sendbuf, count1, dt, request), commInvalidArgument);

  // Release resources
  ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
  delete request;

  COMMCHECK_TEST(ctranComm->ctran_->commDeregister(sendHdl));
  COMMCHECK_TEST(ctranComm->ctran_->commDeregister(recvHdl));
  memoryCleanUp(kMemNcclMemAlloc);
}

inline std::string getTestName(
    const testing::TestParamInfo<CtranAllgatherPTestParam::ParamType>& info) {
  return std::to_string(std::get<0>(info.param)) + "maxSendCount_" +
      std::to_string(std::get<1>(info.param)) + "count_" +
      testInPlaceTypeToStr(std::get<2>(info.param)) + "_" +
      testMemAllocTypeToStr(std::get<3>(info.param)) + "_" +
      ctran::allgatherp::AlgoImpl::algoName(std::get<4>(info.param));
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllgatherPTestParam,
    ::testing::Combine(
        testing::Values(2097152UL), // maxRecvCount / nRanks
        testing::Values(8192, 1048576, 1048567),
        testing::Values(kTestInPlace, kTestOutOfPlace),
        testing::Values(kMemNcclMemAlloc),
        testing::Values(
            NCCL_ALLGATHER_P_ALGO::ctdirect,
            NCCL_ALLGATHER_P_ALGO::ctpipeline)),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
