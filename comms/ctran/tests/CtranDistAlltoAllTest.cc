// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <cstdio>
#include <thread>

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class CtranAllToAllTest : public ctran::CtranDistTestFixture,
                          public CtranBaseTest {
 public:
  CtranAllToAllTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    oobBroadcast(&expectedVal, 1, 0);
  }

  void* createDataBuf(size_t nbytes, bool doRegister) {
    void* buf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf) {
      FB_CUDACHECKIGNORE(cudaMemset(buf, -1, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      if (doRegister) {
        COMMCHECK_TEST(ctran::globalRegisterWithPtr(buf, nbytes));
      }
    }
    return buf;
  }

  void releaseDataBuf(void* buf, size_t nbytes, bool doDeregister) {
    if (doDeregister) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(buf, nbytes));
    }
    CUDACHECK_TEST(cudaFree(buf));
  }

  bool checkTestPrerequisite(size_t count, commDataType_t dataType) {
    EXPECT_NE(nullptr, ctranComm.get());
    EXPECT_NE(nullptr, ctranComm->ctran_);
    if (!ctranAllToAllSupport(
            count, dataType, ctranComm.get(), NCCL_ALLTOALL_ALGO::ctran)) {
      if (globalRank == 0) {
        printf("Skip test because ctranAllToAllSupport returns false\n");
      }
      return false;
    }
    return true;
  }

  void SetUp() override {
    // Always run ctran alltoall no matter the message size
    setenv("NCCL_CTRAN_ALLTOALL_THRESHOLD", "0", 0);

    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
    CUDACHECK_TEST(cudaEventCreate(&start));
    CUDACHECK_TEST(cudaEventCreate(&stop));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaEventDestroy(start));
    CUDACHECK_TEST(cudaEventDestroy(stop));
    ctran::CtranDistTestFixture::TearDown();
  }

  template <commDataType_t DataType = commInt>
  void run(
      const size_t count,
      const size_t bufCount,
      bool registerFlag = true,
      // TODO: Move perf measurement to a separate benchmark file
      bool reportPerf = false) {
    using DT = typename CommTypeTraits<DataType>::T;
    size_t dataTypeSize = sizeof(DT);

    DT *sendBuf = nullptr, *recvBuf = nullptr;
    size_t bufNbytes = bufCount * dataTypeSize;

    assert(count * numRanks <= bufCount);

    if (!checkTestPrerequisite(count, DataType)) {
      GTEST_SKIP() << "Skip test because ctranAllToAllSupport returns false";
    }

    ASSERT_TRUE(
        meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

    generateDistRandomExpValue();

    // Allocate data buffer and register
    sendBuf = (DT*)createDataBuf(bufNbytes, registerFlag);
    recvBuf = (DT*)createDataBuf(bufNbytes, registerFlag);

    // Assign different value for each send chunk
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<DT>(
          sendBuf + i * count,
          count,
          DT(expectedVal + globalRank * 10 + i + 1));
    }

    // Run communication
    auto res = ctranAllToAll(
        sendBuf,
        recvBuf,
        count,
        DataType,
        ctranComm.get(),
        testStream,
        NCCL_ALLTOALL_ALGO::ctran);
    ASSERT_EQ(res, commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));

    // Check each received chunk
    for (int i = 0; i < numRanks; ++i) {
      int errs = checkChunkValue<DT>(
          recvBuf + i * count,
          count,
          DT(expectedVal + i * 10 + globalRank + 1));
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << i
                         << " at " << recvBuf + i * count << " with " << errs
                         << " errors";
    }
    // Check remaining chunks in receive buffer is not updated
    if (count * numRanks < bufCount) {
      int errs = checkChunkValue<DT>(
          recvBuf + count * numRanks, bufCount - count * numRanks, -1);
      EXPECT_EQ(errs, 0) << "rank " << globalRank
                         << " checked remaining chunk at "
                         << recvBuf + count * numRanks << " with " << errs
                         << " errors";
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());

    if (count > 0) {
      // Alltoall uses kernel staged copy not NVL iput
      std::vector<CtranMapperBackend> excludedBackends = {
          CtranMapperBackend::NVL};
      // If single node, uses only kernel staged copy
      if (ctranComm->statex_->nNodes() == 1) {
        excludedBackends.push_back(CtranMapperBackend::IB);
      }
      verifyBackendsUsed(
          ctranComm->ctran_.get(),
          ctranComm->statex_.get(),
          kMemCudaMalloc,
          excludedBackends);
    }
    verifyGpeLeak(ctranComm->ctran_.get());

    int totalColls = 1;
    if (reportPerf) {
      constexpr int iter = 500, warm = 100;
      float gpuTime_ = 0.0;

      totalColls += iter + warm;
      for (int x = 0; x < warm; x++) {
        COMMCHECK_TEST(ctranAllToAll(
            sendBuf,
            recvBuf,
            count,
            DataType,
            ctranComm.get(),
            testStream,
            NCCL_ALLTOALL_ALGO::ctran));
      }

      CUDACHECK_TEST(cudaEventRecord(start, testStream));
      for (int x = 0; x < iter; x++) {
        COMMCHECK_TEST(ctranAllToAll(
            sendBuf,
            recvBuf,
            count,
            DataType,
            ctranComm.get(),
            testStream,
            NCCL_ALLTOALL_ALGO::ctran));
      }
      CUDACHECK_TEST(cudaEventRecord(stop, testStream));
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));
      CUDACHECK_TEST(cudaEventElapsedTime(&gpuTime_, start, stop));
      gpuTime_ = gpuTime_ * 1000 / iter; // in us
      double bw = count * sizeof(DT) * (numRanks - 1) / gpuTime_ / 1000;

      std::cout
          << ::testing::UnitTest::GetInstance()->current_test_info()->name()
          << " with count " << count << " * int on rank " << globalRank
          << " took " << gpuTime_ << " us" << " BusBW " << bw << std::endl;
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    // Sleep for a while to make sure all the colls are finished
    std::this_thread::sleep_for(std::chrono::seconds(2));

    ASSERT_NE(ctranComm->colltraceNew_, nullptr);
    auto dumpMap = ctran::dumpCollTrace(ctranComm.get());

    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColls"], "[]");

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    if (count == 0) {
      totalColls = 0;
    }
    EXPECT_EQ(pastCollsJson.size(), totalColls);

    for (const auto& coll : pastCollsJson) {
      EXPECT_EQ(coll["opName"].asString(), "AllToAll");
      EXPECT_EQ(coll["count"].asInt(), count);
      EXPECT_THAT(
          coll["algoName"].asString(),
          testing::HasSubstr(allToAllAlgoName(NCCL_ALLTOALL_ALGO::ctran)));
    }
    verifyGpeLeak(ctranComm->ctran_.get());

    releaseDataBuf(sendBuf, bufNbytes, registerFlag);
    releaseDataBuf(recvBuf, bufNbytes, registerFlag);
  }

 protected:
  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
  int expectedVal{0};
  cudaEvent_t start;
  cudaEvent_t stop;
};

class CtranAllToAllTestParam
    : public CtranAllToAllTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(CtranAllToAllTestParam, AllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  // test 1 byte types
  run<commInt8>(8192, 8192 * ctranComm->statex_->nRanks());
  run<commInt8>(
      8192 + 41,
      (8192 + 41) * ctranComm->statex_->nRanks()); // non-power of 2

  // test 2 byte types
  run<commFloat16>(8192, 8192 * ctranComm->statex_->nRanks());
  run<commFloat16>(8192 + 103, (8192 + 103) * ctranComm->statex_->nRanks());

  // test 4 byte types
  run<commInt32>(8192, 8192 * ctranComm->statex_->nRanks());
  run<commInt32>((8192 + 60), (8192 + 60) * ctranComm->statex_->nRanks());

  // test 8 byte types
  run<commInt64>(8192, 8192 * ctranComm->statex_->nRanks());
  run<commInt64>((8192 + 192), (8192 + 192) * ctranComm->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, UnalignedAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(9991, 9991 * ctranComm->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, SmallAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  // Even for small data transfer size, need buffer size >= pagesize for IB
  // registration
  run(2, 8192 * ctranComm->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, LargeAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(1024 * 1024 * 128UL, 1024 * 1024 * 128UL * ctranComm->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, ZeroByteAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(0, 8192 * ctranComm->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, AllToAllDynamicRegister) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(8192, 8192 * ctranComm->statex_->nRanks(), false);
}

#ifdef TEST_CUDA_GRAPH_MODE
TEST_P(CtranAllToAllTestParam, CudaGraphAwareAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  EnvRAII env4(NCCL_CTRAN_ALLTOALL_CUDAGRAPH_AWARE_ENABLE, true);
  size_t count = 2, bufCount = 8192 * ctranComm->statex_->nRanks();
  commDataType_t DataType = commInt;
  using DT = int;
  size_t dataTypeSize = sizeof(DT);

  DT *sendBuf = nullptr, *recvBuf = nullptr;

  assert(count * numRanks <= bufCount);

  if (!checkTestPrerequisite(count, DataType)) {
    GTEST_SKIP() << "Skip test because ctranAllToAllSupport returns false";
  }

  generateDistRandomExpValue();

  // Allocate data buffer and register
  size_t bufNbytes = bufCount * dataTypeSize;
  sendBuf = (DT*)createDataBuf(bufNbytes, true);
  recvBuf = (DT*)createDataBuf(bufNbytes, true);

  // Assign different value for each send chunk
  for (int i = 0; i < numRanks; ++i) {
    assignChunkValue<DT>(
        sendBuf + i * count, count, DT(expectedVal + globalRank * 10 + i + 1));
  }
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  // FIXME: if using the stream created in SetUp(), got error "operation not
  // permitted when stream is capturing" when calling cudaStreamBeginCapture. So
  // use this local-managed cudagraph_stream as a workaround to test cuda graph
  // mode.
  cudaStream_t cudagraph_stream;
  CUDACHECK_TEST(cudaStreamCreate(&cudagraph_stream));
  // Capture cudagraph which will launch 1 AllToAll
  CUDACHECK_TEST(
      cudaStreamBeginCapture(cudagraph_stream, cudaStreamCaptureModeGlobal));
  // Run communication
  auto res = ctranAllToAll(
      sendBuf,
      recvBuf,
      count,
      DataType,
      ctranComm.get(),
      cudagraph_stream,
      NCCL_ALLTOALL_ALGO::ctran);
  ASSERT_EQ(res, commSuccess);
  CUDACHECK_TEST(cudaStreamEndCapture(cudagraph_stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  // Replay the graph for 1 time: because we skip sync in alltoallp, run
  // multiple alltoalls sharing the same recvbuff would cause data corruption;
  // need to allocate double buffers for testing multi-iters.
  constexpr int numIters = 1;
  for (int i = 0; i < numIters; i++) {
    CUDACHECK_TEST(cudaGraphLaunch(instance, cudagraph_stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(cudagraph_stream));
  // Check each received chunk
  for (int i = 0; i < numRanks; ++i) {
    int errs = checkChunkValue<DT>(
        recvBuf + i * count, count, DT(expectedVal + i * 10 + globalRank + 1));
    EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << i
                       << " at " << recvBuf + i * count << " with " << errs
                       << " errors";
  }
  // Check remaining chunks in receive buffer is not updated
  if (count * numRanks < bufCount) {
    int errs = checkChunkValue<DT>(
        recvBuf + count * numRanks, bufCount - count * numRanks, -1);
    EXPECT_EQ(errs, 0) << "rank " << globalRank
                       << " checked remaining chunk at "
                       << recvBuf + count * numRanks << " with " << errs
                       << " errors";
  }

  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaStreamDestroy(cudagraph_stream));
  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  verifyGpeLeak(ctranComm->ctran_.get());
  releaseDataBuf(sendBuf, bufNbytes, true);
  releaseDataBuf(recvBuf, bufNbytes, true);
}
#endif

// Tests for fast put configs
inline std::string getTestName(
    const testing::TestParamInfo<CtranAllToAllTestParam::ParamType>& info) {
  return "lowlatencyconfig_" + std::to_string(std::get<0>(info.param)) +
      "_enablefastput_" + std::to_string(std::get<1>(info.param));
}

INSTANTIATE_TEST_SUITE_P(
    CtranAllToAllTest,
    CtranAllToAllTestParam,
    ::testing::Combine(
        testing::Values(true, false),
        testing::Values(true, false)),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
