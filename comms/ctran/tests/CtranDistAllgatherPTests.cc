// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdlib.h>

#if !defined(USE_ROCM)
// NCCL-specific includes only needed for CUDA builds
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#endif

#include <nccl.h>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGatherP/AlgoImpl.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
// Test sources uses ncclMemAlloc API from nccl.h/rccl.h, so adding this check
// macro here so to avoid including TestUtils.h which doesn't support
// cross-platform
#define NCCLCHECK_TEST(cmd)                  \
  do {                                       \
    ncclResult_t r = cmd;                    \
    if (r != ncclSuccess) {                  \
      printf(                                \
          "Failed, NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(r));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

class CtranAllgatherPTestEnv : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    CtranEnvironmentBase::SetUp();

    // set logging level to WARN
    setenv("NCCL_DEBUG", "WARN", 1);
  }
};

class CtranAllgatherPTest : public ctran::CtranDistTestFixture {
 public:
  CtranAllgatherPTest() = default;
  char expectedVal;
  commDataType_t dt = commBfloat16;
  size_t sendBytes, recvBytes;
  void *sendbuf, *recvbuf;
  void *sendHdl, *recvHdl;
  std::unique_ptr<CtranComm> ctranComm;
  cudaStream_t stream = 0;

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTestFixture::SetUp();

    CUDACHECK_TEST(cudaStreamCreate(&stream));
    ctranComm = makeCtranComm();
    EXPECT_TRUE(ctran::allGatherPSupport(ctranComm.get()))
        << "allGatherP algo is not supported!";
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    CtranDistTestFixture::TearDown();
  }

  // Check no GPE internal memory leak after finished collective kernel
  void verifyGpeLeak(ICtran* ctran) {
    ASSERT_EQ(ctran->gpe->numInUseKernelElems(), 0);
    ASSERT_EQ(ctran->gpe->numInUseKernelFlags(), 0);
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
    // Fix a bug of potential address overflow when recvBytes < sendBytes *
    // numRanks. cudaMemset
    // would fail in such cases. Nvdia may be fine if sendBytes * numRanks <
    // memory allocation granularity (e.g., 2MB). But AMD has strict memory
    // address OutofBoundary check.
    if (recvBytes >= sendBytes * numRanks) {
      CUDACHECK_TEST(cudaMemset(
          (char*)recvbuf + globalRank * sendBytes, expectedVal, sendBytes));
    }

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
            ctranComm.get(),
            stream,
            request));

    // Ensure async init completes before execution
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
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
  const std::string algoStr =
      (algo == NCCL_ALLGATHER_P_ALGO::ctdirect) ? "ctdirect" : "ctpipeline";
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", algoStr);

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  } else {
    run(maxSendCount, count, inplace, memType);
  }
}

TEST_F(CtranAllgatherPTestParam, DynamicSendRegDirect) {
  const auto count = 8192;
  const auto maxSendCount = count * numRanks;
  const auto inplace = kTestOutOfPlace;
  const auto memType = kMemNcclMemAlloc;

  // Use SysEnvRAII to set the OS environment variable (not just C++ global).
  // ncclMemAlloc() calls initEnv() which calls ncclCvarInit(), and
  // ncclCvarInit() reads from the OS environment. EnvRAII only sets the C++
  // global, which gets overwritten when ncclCvarInit() is triggered later in
  // ncclMemAlloc
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", "ctdirect");
  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  } else {
    run(maxSendCount, count, inplace, memType, false /* sendbufReg */);
  }
}

TEST_F(CtranAllgatherPTestParam, DynamicSendRegPipeline) {
  const auto count = 8192;
  const auto maxSendCount = count * numRanks;
  const auto inplace = kTestOutOfPlace;
  const auto memType = kMemNcclMemAlloc;

  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", "ctpipeline");
  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  } else {
    run(maxSendCount, count, inplace, memType, false /* sendbufReg */);
  }
}

TEST_F(CtranAllgatherPTest, InvalidPreq) {
  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLTOALL_P, ctranComm.get(), stream);
  void* sendBuf = reinterpret_cast<void*>(0x9000);
  ASSERT_EQ(
      ctran::allGatherPExec(sendBuf, 64, commInt32, request.get()),
      commInvalidArgument);
}

TEST_F(CtranAllgatherPTest, InvalidCount) {
  // Skip test if cuMem is not supported (e.g., on AMD)
  // Note GTEST_SKIP only works in the test body, not in SetUp/TearDown and
  // external functions see https://github.com/google/googletest/pull/1544

  MemAllocType memTypes[2] = {kMemNcclMemAlloc, kMemCudaMalloc};

  const auto count = 65536;
  const auto maxRecvCount = 8192 * numRanks;
  for (auto memType : memTypes) {
    if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
      XLOG(INFO)
          << "CuMem not supported, skipping InvalidCount test with memType = kMemNcclMemAlloc";
      ;
      continue;
    }

    memorySetUp(memType, count, maxRecvCount);

    COMMCHECK_TEST(
        ctranComm->ctran_->commRegister(recvbuf, recvBytes, &recvHdl));
    COMMCHECK_TEST(
        ctranComm->ctran_->commRegister(sendbuf, sendBytes, &sendHdl));
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
            ctranComm.get(),
            stream,
            request));

    // count * sizeof(dt) * numRanks must be less than maxRecvCount *
    // sizeof(initDt)
    ASSERT_EQ(
        ctran::allGatherPExec(sendbuf, count, dt, request),
        commInvalidArgument);

    // Check count < initMaxRecvCount / numRanks, but > maxRecvCount/numRanks;
    // to ensure it compares based on bytes
    const auto count1 = maxRecvCount / numRanks + 1;
    ASSERT_EQ(
        ctran::allGatherPExec(sendbuf, count1, dt, request),
        commInvalidArgument);

    // Release resources
    ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
    delete request;

    COMMCHECK_TEST(ctranComm->ctran_->commDeregister(sendHdl));
    COMMCHECK_TEST(ctranComm->ctran_->commDeregister(recvHdl));
    memoryCleanUp(memType);
  }
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
  ::testing::AddGlobalTestEnvironment(new CtranAllgatherPTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
