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
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/MathUtils.h"
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
    ctran::CtranEnvironmentBase::SetUp();

    // set logging level to WARN but allow override by manual run
    setenv("NCCL_DEBUG", "WARN", 0);
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
    setenv("NCCL_CTRAN_TRANSPORT_PROFILER", "1", 0);
    setenv("NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT", "1", 0);
    CtranDistTestFixture::SetUp();

    CUDACHECK_TEST(cudaStreamCreate(&stream));
    ctranComm = makeCtranComm();
    if (!ctran::allGatherPSupport(ctranComm.get())) {
      GTEST_SKIP() << "allGatherP algo is not supported!";
    }
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

  void
  cumemBufSetup(size_t sendCount, size_t recvCount, char** sBuf, char** rBuf) {
    expectedVal = globalRank;

    const size_t pageSize = getpagesize();
    auto sBytes = sendCount * commTypeSize(dt);
    auto rBytes = recvCount * commTypeSize(dt);

    size_t bufSize;
    bufSize = ((sBytes + pageSize - 1) / pageSize) * pageSize;
    *sBuf = prepareBuf(bufSize, kMemNcclMemAlloc);
    bufSize = ((rBytes + pageSize - 1) / pageSize) * pageSize;
    *rBuf = prepareBuf(bufSize, kMemNcclMemAlloc);

    CUDACHECK_TEST(cudaMemset(*sBuf, expectedVal, sBytes));
    CUDACHECK_TEST(cudaMemset(*rBuf, rand(), rBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  void cumemBufCleanUp(void* sBuf, void* rBuf) {
    releaseBuf((char*)sBuf, kMemNcclMemAlloc);
    releaseBuf((char*)rBuf, kMemNcclMemAlloc);
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
      CtranComm* testComm,
      bool sendbufReg = true) {
    const auto maxRecvCount = maxSendCount * numRanks;
    memorySetUp(memType, count, maxRecvCount);

    void* usedSendBuf = sendbuf;
    COMMCHECK_TEST(
        testComm->ctran_->commRegister(recvbuf, recvBytes, &recvHdl));
    if (inplace == kTestInPlace) {
      usedSendBuf = (char*)recvbuf + globalRank * sendBytes;
    } else if (sendbufReg) {
      COMMCHECK_TEST(
          testComm->ctran_->commRegister(sendbuf, sendBytes, &sendHdl));
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
            testComm,
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
        const std::vector<char> expectedVals(
            sendBytes, static_cast<char>(i + j * 10));
        EXPECT_EQ(observedVals, expectedVals)
            << "at rank " << globalRank << " in iteration " << j
            << " at chunk received from peer " << i;
      }
    }

    // Verify profiler event durations are populated
    checkProfiler(testComm->ctran_->profiler.get());

    verifyGpeLeak(testComm->ctran_.get());

    ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
    delete request;

    if (inplace == kTestOutOfPlace && sendbufReg) {
      COMMCHECK_TEST(testComm->ctran_->commDeregister(sendHdl));
    }
    COMMCHECK_TEST(testComm->ctran_->commDeregister(recvHdl));

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

static std::string algoToStr(enum NCCL_ALLGATHER_P_ALGO algo) {
  switch (algo) {
    case NCCL_ALLGATHER_P_ALGO::ctdirect:
      return "ctdirect";
    case NCCL_ALLGATHER_P_ALGO::ctpipeline:
      return "ctpipeline";
    case NCCL_ALLGATHER_P_ALGO::ctsrdpipeline:
      return "ctsrdpipeline";
    default:
      return "unknown";
  }
}

static bool requiresPowerOfTwoNodes(enum NCCL_ALLGATHER_P_ALGO algo) {
  return algo == NCCL_ALLGATHER_P_ALGO::ctsrdpipeline;
}

TEST_P(CtranAllgatherPTestParam, Basic) {
  const auto& [maxSendCount, count, inplace, memType, algo] = GetParam();
  const std::string algoStr = algoToStr(algo);
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", algoStr);
  const auto nNodes = ctranComm->statex_->nNodes();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  } else if (
      requiresPowerOfTwoNodes(algo) && nNodes > 1 &&
      !ctran::utils::isPowerOfTwo(nNodes)) {
    GTEST_SKIP() << algoStr << " requires nNodes to be a power of 2, skip test";
  } else {
    run(maxSendCount, count, inplace, memType, ctranComm.get());
  }
}

TEST_P(CtranAllgatherPTestParam, VnodeBasic) {
  const auto& [maxSendCount, count, inplace, memType, algo] = GetParam();
  const std::string algoStr = algoToStr(algo);
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", algoStr);
  EnvRAII vnodeEnv(
      NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::vnode);
  EnvRAII ppnEnv(NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS, 4);
  if (ctranComm->statex_->nLocalRanks() % 4 != 0) {
    GTEST_SKIP()
        << "Vnode test requires number of local ranks to be multiple of 4, skip test";
  }

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  } else {
    // Create a new communicator with virtual node + 4 local ranks setup
    auto testComm = makeCtranComm();
    ASSERT_EQ(testComm->statex_->nLocalRanks(), 4);
    const auto vnodeNNodes = numRanks / 4;
    ASSERT_EQ(testComm->statex_->nNodes(), vnodeNNodes);
    if (requiresPowerOfTwoNodes(algo) && vnodeNNodes > 1 &&
        !ctran::utils::isPowerOfTwo(vnodeNNodes)) {
      GTEST_SKIP() << algoStr
                   << " requires nNodes to be a power of 2, skip test";
    }
    run(maxSendCount, count, inplace, memType, testComm.get());
  }
}

// Exercises the log2(nNodes) striping with nLocalRanks=2, nNodes=numRanks/2,
// which forces ctsrdpipeline through 2+ recursive-doubling steps and thus the
// multi-step chunk-offset and step > 0 recvbuff-read path that the
// 1-step VnodeBasic never hits. Other algos also run through this config
// so the extra parameterization is cheap.
TEST_P(CtranAllgatherPTestParam, VnodeBasicMultiStep) {
  const auto& [maxSendCount, count, inplace, memType, algo] = GetParam();
  const std::string algoStr = algoToStr(algo);
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", algoStr);
  EnvRAII vnodeEnv(
      NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::vnode);
  EnvRAII ppnEnv(NCCL_COMM_STATE_DEBUG_TOPO_VNODE_NLOCALRANKS, 2);
  if (ctranComm->statex_->nLocalRanks() % 2 != 0 || numRanks / 2 < 4) {
    GTEST_SKIP()
        << "VnodeBasicMultiStep requires nLocalRanks divisible by 2 and nNodes >= 4";
  }

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  } else {
    auto testComm = makeCtranComm();
    ASSERT_EQ(testComm->statex_->nLocalRanks(), 2);
    const auto vnodeNNodes = numRanks / 2;
    ASSERT_EQ(testComm->statex_->nNodes(), vnodeNNodes);
    if (requiresPowerOfTwoNodes(algo) && vnodeNNodes > 1 &&
        !ctran::utils::isPowerOfTwo(vnodeNNodes)) {
      GTEST_SKIP() << algoStr
                   << " requires nNodes to be a power of 2, skip test";
    }
    run(maxSendCount, count, inplace, memType, testComm.get());
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
    run(maxSendCount,
        count,
        inplace,
        memType,
        ctranComm.get(),
        false /* sendbufReg */);
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
    run(maxSendCount,
        count,
        inplace,
        memType,
        ctranComm.get(),
        false /* sendbufReg */);
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

TEST_F(CtranAllgatherPTest, InternalRegisteredMemory) {
  // Test using CTRAN's internal registered temporary buffers
  // This validates that buffers allocated by CTRAN internally
  // (e.g., in CtranAlgo.cc) work correctly with AllGatherP operations

  EnvRAII algoEnv(NCCL_ALLGATHER_P_ALGO, NCCL_ALLGATHER_P_ALGO::ctdirect);

  // Access CTRAN's internal temporary buffers
  auto* ctranAlgo = ctranComm->ctran_->algo.get();

  // Use MIN_REG_SRC_TMPBUF and MIN_REG_DST_TMPBUF which are sized
  // CTRAN_MIN_REGISTRATION_SIZE (typically sufficient for small tests)
  auto [srcBuf, srcBufHdl] =
      ctranAlgo->getTmpBufInfo(CtranAlgo::TmpbufType::MIN_REG_SRC_TMPBUF);
  auto [dstBuf, dstBufHdl] =
      ctranAlgo->getTmpBufInfo(CtranAlgo::TmpbufType::MIN_REG_DST_TMPBUF);

  ASSERT_NE(srcBuf, nullptr) << "Internal src tmpbuf should be allocated";
  ASSERT_NE(dstBuf, nullptr) << "Internal dst tmpbuf should be allocated";
  ASSERT_NE(srcBufHdl, nullptr) << "Internal src tmpbuf should be registered";
  ASSERT_NE(dstBufHdl, nullptr) << "Internal dst tmpbuf should be registered";

  // Use a small count that fits within CTRAN_MIN_REGISTRATION_SIZE
  const size_t count = 64;
  const commDataType_t testDt = commInt8;
  const size_t sendBytes = count * commTypeSize(testDt);
  const size_t recvBytes = sendBytes * numRanks;

  // Initialize source buffer with rank-specific pattern
  const char expectedVal = static_cast<char>(globalRank);
  CUDACHECK_TEST(cudaMemset(srcBuf, expectedVal, sendBytes));
  CUDACHECK_TEST(cudaMemset(dstBuf, 0xAB, recvBytes));

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Initialize AllGatherP with internal recvbuf
  // Note: We don't need to call commRegister - buffers are already registered
  meta::comms::Hints hints;
  CtranPersistentRequest* request;

  COMMCHECK_TEST(
      ctran::allGatherPInit(
          dstBuf,
          count * numRanks,
          hints,
          testDt,
          ctranComm.get(),
          stream,
          request));

  // Execute AllGatherP using internal buffers
  constexpr int nIter = 3;
  for (int j = 0; j < nIter; j++) {
    // Update source buffer for each iteration
    const char sendVal = static_cast<char>(j * 10 + globalRank);
    std::vector<char> sendVals(sendBytes, sendVal);
    ASSERT_EQ(
        cudaMemcpyAsync(
            srcBuf, sendVals.data(), sendBytes, cudaMemcpyDefault, stream),
        cudaSuccess);

    ASSERT_EQ(
        ctran::allGatherPExec(srcBuf, count, testDt, request), commSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Verify results: each rank's data should be present in recvbuf
    for (int i = 0; i < numRanks; ++i) {
      std::vector<char> observedVals(sendBytes, 0xFF);
      ASSERT_EQ(
          cudaMemcpy(
              observedVals.data(),
              static_cast<char*>(dstBuf) + sendBytes * i,
              sendBytes,
              cudaMemcpyDefault),
          cudaSuccess);

      const std::vector<char> expectedVals(
          sendBytes, static_cast<char>(i + j * 10));
      EXPECT_EQ(observedVals, expectedVals)
          << "at rank " << globalRank << " in iteration " << j
          << " at chunk received from peer " << i;
    }
  }

  verifyGpeLeak(ctranComm->ctran_.get());

  ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
  delete request;
}

TEST_F(CtranAllgatherPTest, SharePersistentBuffer) {
  // Test use case where AGP uses the same persistent buffer for multiple
  // communicators
  const auto recvCount = 67108864;
  const auto sendCount = recvCount / numRanks;

  const auto kComms = 20;
  const auto kBufs = 3;
  std::vector<cudaStream_t> streams(kComms);
  std::vector<std::unique_ptr<CtranComm>> testComms(kComms);
  std::vector<CtranPersistentRequest*> requests(kComms * kBufs);
  std::vector<char*> sendBufs(kBufs);
  std::vector<char*> recvBufs(kBufs);
  std::vector<void*> sendHdls(kBufs);
  std::vector<void*> recvHdls(kBufs);

  for (int c = 0; c < kComms; c++) {
    CUDACHECK_TEST(cudaStreamCreate(&streams[c]));
    testComms[c] = makeCtranComm();
  }

  if (ncclIsCuMemSupported() == false) {
    XLOG(INFO)
        << "CuMem not supported, skipping SharePersistentBuffer test with memType = kMemNcclMemAlloc";
    return;
  }

  // Allocate persistent buffers and register with ALL communicators
  // before any allGatherPInit (export) calls
  for (int b = 0; b < kBufs; b++) {
    cumemBufSetup(sendCount, recvCount, &sendBufs.at(b), &recvBufs.at(b));
    for (int c = 0; c < kComms; c++) {
      COMMCHECK_TEST(
          testComms[c]->ctran_->commRegister(
              recvBufs.at(b), recvCount * commTypeSize(dt), &recvHdls.at(b)));
      COMMCHECK_TEST(
          testComms[c]->ctran_->commRegister(
              sendBufs.at(b), sendCount * commTypeSize(dt), &sendHdls.at(b)));
    }
  }
  // Convert to int8_t for init to mimic FSDP use case
  const auto initMaxRecvCount =
      recvCount * commTypeSize(dt) / commTypeSize(commInt8);
  const auto initDt = commInt8;

  for (int b = 0; b < kBufs; b++) {
    for (int c = 0; c < kComms; c++) {
      meta::comms::Hints hints;
      COMMCHECK_TEST(
          ctran::allGatherPInit(
              recvBufs.at(b),
              initMaxRecvCount,
              hints,
              initDt,
              testComms[c].get(),
              streams.at(c),
              requests.at(c * kBufs + b)));
    }
  }

  // Run allgather execute in parallel
  for (int b = 0; b < kBufs; b++) {
    for (int c = 0; c < kComms; c++) {
      ASSERT_EQ(
          ctran::allGatherPExec(
              sendBufs.at(b), sendCount, dt, requests[c * kBufs + b]),
          commSuccess);
    }
  }

  // synchronize all streams
  for (int c = 0; c < kComms; c++) {
    ASSERT_EQ(cudaStreamSynchronize(streams.at(c)), cudaSuccess);
  }

  // Release resources
  for (int r = 0; r < requests.size(); r++) {
    ASSERT_EQ(ctran::allGatherPDestroy(requests[r]), commSuccess);
    delete requests[r];
  }

  // Deregister buffers from all communicators
  for (int b = 0; b < kBufs; b++) {
    for (int c = 0; c < kComms; c++) {
      COMMCHECK_TEST(testComms[c]->ctran_->commDeregister(sendHdls.at(b)));
      COMMCHECK_TEST(testComms[c]->ctran_->commDeregister(recvHdls.at(b)));
    }
    cumemBufCleanUp(sendBufs.at(b), recvBufs.at(b));
  }

  // Destroy comms before streams
  testComms.clear();

  for (int c = 0; c < kComms; c++) {
    CUDACHECK_TEST(cudaStreamDestroy(streams[c]));
  }
}

TEST_F(CtranAllgatherPTest, DestroyOneShareRecvBuf) {
  // Two persistent AGP requests share the SAME recvbuff. Destroying one must
  // not release imports/registration still needed by the other: with the
  // per-request ScopedIpcRegHdl deferred release on destroy, an over-release or
  // stale key would make the surviving request's exec fail or produce wrong
  // data.
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", "ctpipeline");
  if (ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping DestroyOneShareRecvBuf test";
  }

  const auto recvCount = 2097152UL * numRanks;
  const auto sendCount = recvCount / numRanks;

  char* localSendBuf = nullptr;
  char* localRecvBuf = nullptr;
  cumemBufSetup(sendCount, recvCount, &localSendBuf, &localRecvBuf);

  const auto recvRegBytes = recvCount * commTypeSize(dt);
  const auto sendRegBytes = sendCount * commTypeSize(dt);
  void* localRecvHdl = nullptr;
  void* localSendHdl = nullptr;
  COMMCHECK_TEST(ctranComm->ctran_->commRegister(
      localRecvBuf, recvRegBytes, &localRecvHdl));
  COMMCHECK_TEST(ctranComm->ctran_->commRegister(
      localSendBuf, sendRegBytes, &localSendHdl));

  const auto initMaxRecvCount =
      recvCount * commTypeSize(dt) / commTypeSize(commInt8);
  const auto initDt = commInt8;

  // Two persistent requests over the SAME recvbuff on the same comm.
  meta::comms::Hints hints;
  CtranPersistentRequest* reqA = nullptr;
  CtranPersistentRequest* reqB = nullptr;
  COMMCHECK_TEST(
      ctran::allGatherPInit(
          localRecvBuf,
          initMaxRecvCount,
          hints,
          initDt,
          ctranComm.get(),
          stream,
          reqA));
  COMMCHECK_TEST(
      ctran::allGatherPInit(
          localRecvBuf,
          initMaxRecvCount,
          hints,
          initDt,
          ctranComm.get(),
          stream,
          reqB));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Runs one exec on the given request and verifies every peer chunk.
  auto runAndVerify = [&](CtranPersistentRequest* request, int iter) {
    const char sendVal = static_cast<char>(iter * 10 + globalRank);
    std::vector<char> sendVals(sendRegBytes, sendVal);
    ASSERT_EQ(
        cudaMemcpyAsync(
            localSendBuf,
            sendVals.data(),
            sendRegBytes,
            cudaMemcpyDefault,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        ctran::allGatherPExec(localSendBuf, sendCount, dt, request),
        commSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int i = 0; i < numRanks; ++i) {
      std::vector<char> observedVals(sendRegBytes, 0xFF);
      ASSERT_EQ(
          cudaMemcpy(
              observedVals.data(),
              localRecvBuf + sendRegBytes * i,
              sendRegBytes,
              cudaMemcpyDefault),
          cudaSuccess);
      const std::vector<char> expectedVals(
          sendRegBytes, static_cast<char>(i + iter * 10));
      EXPECT_EQ(observedVals, expectedVals)
          << "at rank " << globalRank << " in iteration " << iter
          << " at chunk received from peer " << i;
    }
  };

  // Both requests work before any destroy.
  runAndVerify(reqA, 0);
  runAndVerify(reqB, 1);

  // Destroy ONE; the shared recvbuff imports must remain valid for the other.
  ASSERT_EQ(ctran::allGatherPDestroy(reqA), commSuccess);
  delete reqA;

  // The surviving request must still execute correctly over the shared buffer.
  runAndVerify(reqB, 2);

  verifyGpeLeak(ctranComm->ctran_.get());

  ASSERT_EQ(ctran::allGatherPDestroy(reqB), commSuccess);
  delete reqB;

  COMMCHECK_TEST(ctranComm->ctran_->commDeregister(localSendHdl));
  COMMCHECK_TEST(ctranComm->ctran_->commDeregister(localRecvHdl));
  cumemBufCleanUp(localSendBuf, localRecvBuf);
}

// Validates the ctran-level persistent-request teardown-safety mechanism: a
// request's pooled pipeSync must be released before CtranGpe::terminate()'s
// pool-drain spin-wait, regardless of whether the CtranComm or the persistent
// request is destroyed first. Without the comm-drain-before-terminate fix, the
// "comm before preq" sub-case would hang forever in terminate().
TEST_F(CtranAllgatherPTest, CommDestroyBeforePreqDestroy) {
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", "ctpipeline");
  if (ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skipping this test";
  } else if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const auto recvCount = 2097152UL * numRanks;
  const auto sendCount = recvCount / numRanks;
  const auto sendRegBytes = sendCount * commTypeSize(dt);
  const auto recvRegBytes = recvCount * commTypeSize(dt);
  const auto initMaxRecvCount =
      recvCount * commTypeSize(dt) / commTypeSize(commInt8);
  const auto initDt = commInt8;

  // Creates a comm + persistent request over a freshly allocated recvbuf, runs
  // exactly one exec, and returns the pieces so the caller controls teardown
  // ordering. Buffers are freed by the caller after the request is destroyed.
  auto makeReqAndExec = [&](std::unique_ptr<CtranComm>& comm,
                            CtranPersistentRequest*& request,
                            char*& sendBuf,
                            char*& recvBuf,
                            void*& sendHdl,
                            void*& recvHdl) {
    comm = makeCtranComm();
    cumemBufSetup(sendCount, recvCount, &sendBuf, &recvBuf);

    // allGatherPInit resolves the recvbuf's registration via searchRegHandle,
    // so the recvbuf must be pre-registered on this comm; mirror the sibling
    // AGP tests by also registering the sendbuf used by exec.
    COMMCHECK_TEST(comm->ctran_->commRegister(recvBuf, recvRegBytes, &recvHdl));
    COMMCHECK_TEST(comm->ctran_->commRegister(sendBuf, sendRegBytes, &sendHdl));

    meta::comms::Hints hints;
    COMMCHECK_TEST(
        ctran::allGatherPInit(
            recvBuf,
            initMaxRecvCount,
            hints,
            initDt,
            comm.get(),
            stream,
            request));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<char> sendVals(sendRegBytes, static_cast<char>(globalRank));
    ASSERT_EQ(
        cudaMemcpyAsync(
            sendBuf, sendVals.data(), sendRegBytes, cudaMemcpyDefault, stream),
        cudaSuccess);
    ASSERT_EQ(
        ctran::allGatherPExec(sendBuf, sendCount, dt, request), commSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  };

  // Sub-case 1: normal ordering -- destroy the preq (comm alive), then the
  // comm. The comm drain then finds the token already spent (no-op).
  {
    std::unique_ptr<CtranComm> comm;
    CtranPersistentRequest* request = nullptr;
    char *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHdl = nullptr, *recvHdl = nullptr;
    makeReqAndExec(comm, request, sendBuf, recvBuf, sendHdl, recvHdl);

    ASSERT_EQ(ctran::allGatherPDestroy(request), commSuccess);
    delete request;
    COMMCHECK_TEST(comm->ctran_->commDeregister(sendHdl));
    COMMCHECK_TEST(comm->ctran_->commDeregister(recvHdl));
    comm.reset();
    cumemBufCleanUp(sendBuf, recvBuf);
  }

  // Sub-case 2: the hang case -- destroy the COMM before the preq. The comm
  // drain must release the pooled pipeSync before terminate() so this returns
  // instead of spinning forever. The user then only frees the request object
  // (the comm is gone; allGatherPDestroy is not valid post-comm-destroy).
  {
    std::unique_ptr<CtranComm> comm;
    CtranPersistentRequest* request = nullptr;
    char *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHdl = nullptr, *recvHdl = nullptr;
    makeReqAndExec(comm, request, sendBuf, recvBuf, sendHdl, recvHdl);

    // The comm is destroyed first; its drain releases the registrations, so no
    // explicit commDeregister is valid post-comm-destroy.
    comm.reset();
    delete request;
    cumemBufCleanUp(sendBuf, recvBuf);
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
            NCCL_ALLGATHER_P_ALGO::ctpipeline,
            NCCL_ALLGATHER_P_ALGO::ctsrdpipeline)),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranAllgatherPTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
