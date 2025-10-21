// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <nccl.h>
#include <stdlib.h>
#include <cstdio>
#include <new>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"

class CtranAllToAllTest : public CtranDistBaseTest {
 public:
  CtranAllToAllTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    MPI_Bcast(&expectedVal, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  void* createDataBuf(size_t nbytes, void** handle) {
    void* buf = nullptr;
    // Allocate data buffer, and assign different value for each send chunk
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf) {
      FB_CUDACHECKIGNORE(cudaMemset(buf, -1, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      if (handle) {
        NCCLCHECK_TEST(ncclCommRegister(comm, buf, nbytes, handle));
      }
    }
    return buf;
  }

  void releaseDataBuf(void* buf, void* handle) {
    if (handle) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, handle));
    }
    CUDACHECK_TEST(cudaFree(buf));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              int(observedVals[i]),
              int(val));
        }
        errs++;
      }
    }
    return errs;
  }

  bool checkTestPrerequisite(size_t count, commDataType_t dataType) {
    EXPECT_NE(nullptr, comm);
    EXPECT_NE(nullptr, comm->ctranComm_->ctran_);
    if (!ctranAllToAllSupport(count, dataType, comm->ctranComm_.get())) {
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

    CtranDistBaseTest::SetUp();
    comm = commWorld;
    CUDACHECK_TEST(cudaEventCreate(&start));
    CUDACHECK_TEST(cudaEventCreate(&stop));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaEventDestroy(start));
    CUDACHECK_TEST(cudaEventDestroy(stop));
    CtranDistBaseTest::TearDown();
  }

  template <commDataType_t DataType = commInt>
  void run(
      const size_t count,
      const size_t bufCount,
      bool registerFlag = true,
      bool reportPerf = false) {
    using DT = typename CommTypeTraits<DataType>::T;
    size_t dataTypeSize = sizeof(DT);

    DT *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendHdl = nullptr, *recvHdl = nullptr;

    assert(count * numRanks <= bufCount);

    if (!checkTestPrerequisite(count, DataType)) {
      GTEST_SKIP() << "Skip test because ctranAllToAllSupport returns false";
    }

    generateDistRandomExpValue();

    // Allocate data buffer and register
    sendBuf = (DT*)createDataBuf(
        bufCount * dataTypeSize, registerFlag ? &sendHdl : nullptr);
    recvBuf = (DT*)createDataBuf(
        bufCount * dataTypeSize, registerFlag ? &recvHdl : nullptr);

    // Assign different value for each send chunk
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<DT>(
          sendBuf + i * count,
          count,
          DT(expectedVal + globalRank * 10 + i + 1));
    }

    comm->ctranComm_->collTrace_->resetPastColls();

    // Run communication
    auto res = ctranAllToAll(
        sendBuf, recvBuf, count, DataType, comm->ctranComm_.get(), stream);
    ASSERT_EQ(res, commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

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

    if (count > 0) {
      // Alltoall uses kernel staged copy not NVL iput
      std::vector<CtranMapperBackend> excludedBackends = {
          CtranMapperBackend::NVL};
      // If single node, uses only kernel staged copy
      if (comm->ctranComm_->statex_->nNodes() == 1) {
        excludedBackends.push_back(CtranMapperBackend::IB);
      }
      verifyBackendsUsed(
          comm->ctranComm_->ctran_.get(),
          comm->ctranComm_->statex_.get(),
          kMemCudaMalloc,
          excludedBackends);
    }
    verifyGpeLeak(comm->ctranComm_->ctran_.get());

    int totalColls = 1;
    if (reportPerf) {
      constexpr int iter = 500, warm = 100;
      float gpuTime_ = 0.0;

      totalColls += iter + warm;
      for (int x = 0; x < warm; x++) {
        COMMCHECK_TEST(ctranAllToAll(
            sendBuf, recvBuf, count, DataType, comm->ctranComm_.get(), stream));
      }

      CUDACHECK_TEST(cudaEventRecord(start, stream));
      for (int x = 0; x < iter; x++) {
        COMMCHECK_TEST(ctranAllToAll(
            sendBuf, recvBuf, count, DataType, comm->ctranComm_.get(), stream));
      }
      CUDACHECK_TEST(cudaEventRecord(stop, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));
      CUDACHECK_TEST(cudaEventElapsedTime(&gpuTime_, start, stop));
      gpuTime_ = gpuTime_ * 1000 / iter; // in us
      double bw = count * sizeof(DT) * (numRanks - 1) / gpuTime_ / 1000;

      std::cout
          << ::testing::UnitTest::GetInstance()->current_test_info()->name()
          << " with count " << count << " * int on rank " << globalRank
          << " took " << gpuTime_ << " us" << " BusBW " << bw << std::endl;
    }

    // CollTrace is updated by a separate thread, need wait for it to finish to
    // avoid flaky test
    comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

    auto dump = comm->ctranComm_->collTrace_->dump();
    if (totalColls > NCCL_COLLTRACE_RECORD_MAX) {
      totalColls = NCCL_COLLTRACE_RECORD_MAX;
    }
    if (count == 0) {
      totalColls = 0;
    }
    EXPECT_EQ(dump.pastColls.size(), totalColls);

    for (auto& coll : dump.pastColls) {
      EXPECT_EQ(coll.opName, "AllToAll");
      EXPECT_EQ(coll.count, count);
      EXPECT_EQ(coll.dataType, DataType);
      EXPECT_EQ(coll.algoName, allToAllAlgoName(NCCL_ALLTOALL_ALGO::ctran));
    }
    verifyGpeLeak(comm->ctranComm_->ctran_.get());
    releaseDataBuf(sendBuf, registerFlag ? sendHdl : nullptr);
    releaseDataBuf(recvBuf, registerFlag ? recvHdl : nullptr);
  }

 protected:
  cudaStream_t stream{0};
  ncclComm_t comm{nullptr};
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
  run<commInt8>(8192, 8192 * comm->ctranComm_->statex_->nRanks());
  run<commInt8>(
      8192 + 41,
      (8192 + 41) * comm->ctranComm_->statex_->nRanks()); // non-power of 2

  // test 2 byte types
  run<commFloat16>(8192, 8192 * comm->ctranComm_->statex_->nRanks());
  run<commFloat16>(
      8192 + 103, (8192 + 103) * comm->ctranComm_->statex_->nRanks());

  // test 4 byte types
  run<commInt32>(8192, 8192 * comm->ctranComm_->statex_->nRanks());
  run<commInt32>(
      (8192 + 60), (8192 + 60) * comm->ctranComm_->statex_->nRanks());

  // test 8 byte types
  run<commInt64>(8192, 8192 * comm->ctranComm_->statex_->nRanks());
  run<commInt64>(
      (8192 + 192), (8192 + 192) * comm->ctranComm_->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, UnalignedAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(9991, 9991 * comm->ctranComm_->statex_->nRanks());
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
  run(2, 8192 * comm->ctranComm_->statex_->nRanks(), true, true);
}

TEST_P(CtranAllToAllTestParam, LargeAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(1024 * 1024 * 128UL,
      1024 * 1024 * 128UL * comm->ctranComm_->statex_->nRanks(),
      true,
      true);
}

TEST_P(CtranAllToAllTestParam, ZeroByteAllToAll) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(0, 8192 * comm->ctranComm_->statex_->nRanks());
}

TEST_P(CtranAllToAllTestParam, AllToAllDynamicRegister) {
  const auto& [enable_lowlatency_config, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  run(8192, 8192 * comm->ctranComm_->statex_->nRanks(), false);
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
  // FIXME: if enable colltrace, waitForWorkerFinishQueue() hangs when ppn > 1;
  // so disable for now.
  EnvRAII env5(NCCL_COLLTRACE, {});
  size_t count = 2, bufCount = 8192 * comm->ctranComm_->statex_->nRanks();
  commDataType_t DataType = commInt;
  using DT = int;
  size_t dataTypeSize = sizeof(DT);

  DT *sendBuf = nullptr, *recvBuf = nullptr;
  void *sendHdl = nullptr, *recvHdl = nullptr;

  assert(count * numRanks <= bufCount);

  if (!checkTestPrerequisite(count, DataType)) {
    GTEST_SKIP() << "Skip test because ctranAllToAllSupport returns false";
  }

  generateDistRandomExpValue();

  // Allocate data buffer and register
  sendBuf = (DT*)createDataBuf(bufCount * dataTypeSize, &sendHdl);
  recvBuf = (DT*)createDataBuf(bufCount * dataTypeSize, &recvHdl);

  // Assign different value for each send chunk
  for (int i = 0; i < numRanks; ++i) {
    assignChunkValue<DT>(
        sendBuf + i * count, count, DT(expectedVal + globalRank * 10 + i + 1));
  }
  // comm->ctranComm_->collTrace_->resetPastColls();
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
      comm->ctranComm_.get(),
      cudagraph_stream);
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

  // FIXME: uncomment after colltrace hang issue resolved when ppn > 1.
  // comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
  // auto statex = comm->ctranComm_->statex_.get();
  // auto dump = comm->ctranComm_->collTrace_->dump();

  // constexpr int numTimesRunInit = 1;
  // EXPECT_EQ(dump.pastColls.size(), numIters + numTimesRunInit);

  // // Skip the check for the AllToAllPInit (first 1) colls.
  // for (int i = numTimesRunInit; i < dump.pastColls.size(); i++) {
  //   auto& coll = dump.pastColls[i];
  //   if (statex->nNodes() == 1) {
  //     // If only cuda kernel is launched (no IB put), AlltoAllP is
  //     essentially
  //     // alltoall because it shares cuda kernel logic with AlltoAll.
  //     EXPECT_EQ(coll.opName, "AllToAll");
  //   } else {
  //     EXPECT_EQ(coll.opName, "AllToAllP");
  //   }
  //   EXPECT_THAT(coll.count, count);
  //   EXPECT_EQ(coll.dataType, DataType);
  //   EXPECT_EQ(
  //       coll.algoName,
  //       ctran::alltoallp::AlgoImpl::algoName(NCCL_ALLTOALL_ALGO::ctran));
  // }

  CUDACHECK_TEST(cudaStreamDestroy(cudagraph_stream));
  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  verifyGpeLeak(comm->ctranComm_->ctran_.get());
  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
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
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
