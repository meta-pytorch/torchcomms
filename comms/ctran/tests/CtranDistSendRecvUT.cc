// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CommGroupUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

#include <folly/json/json.h>

class CtranTestFixture : public NcclxBaseTest, public CtranBaseTest {
 public:
  std::vector<TestMemSegment> segments;
  std::vector<void*> segHandles;
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
    // -1 for not limiting the number of colls to trace
    setenv("NCCL_COLLTRACE_RECORD_MAX", "-1", 0);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_TRANSPORT_PROFILER", "1", 0);
    setenv("NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT", "1", 0);
    NcclxBaseTest::SetUp();
    srand(time(NULL));
    ctran::logGpuMemoryStats(globalRank);

    regCache = ctran::RegCache::getInstance();
    ctran::CHECK_VALID_REGCACHE(regCache);
  }

  void TearDown() override {
    ctran::logGpuMemoryStats(globalRank);
    NcclxBaseTest::TearDown();
  }

  static void checkProfiler(ctran::Profiler* profiler, uint64_t opCount) {
    // algo profiler currently only enabled for IB backend
    if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE ||
        NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctstaged) {
      return;
    }
    ASSERT_NE(profiler, nullptr);
    EXPECT_EQ(profiler->getOpCount(), opCount);
    uint64_t oneMinUs = 1000 * 1000 * 60;
    EXPECT_GE(
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_TOTAL), 0);
    EXPECT_LE(
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_TOTAL),
        oneMinUs);
    EXPECT_GE(profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_CTRL), 0);
    EXPECT_LE(
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_CTRL),
        oneMinUs);
    EXPECT_GE(profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_DATA), 0);
    EXPECT_LE(
        profiler->getEventDurationUs(ctran::ProfilerEvent::ALGO_DATA),
        oneMinUs);
    EXPECT_GE(profiler->getEventDurationUs(ctran::ProfilerEvent::BUF_REG), 0);
    EXPECT_LE(
        profiler->getEventDurationUs(ctran::ProfilerEvent::BUF_REG), oneMinUs);
  }

  /**
   * Run a send/recv test between ranks.
   *
   * @param offset Offset in bytes from the start of the buffer
   * @param count Number of elements to send/receive
   * @param numMaxQp Maximum number of Queue Pairs to use
   * @param nIter Number of iterations to run the test
   * @param memType Type of memory allocation to use
   * @param oneToOne If true, only test send/recv between rank 0 and the last
   * rank. If false, rank 0 sends to all other ranks.
   */
  void runTest(
      size_t offset,
      ssize_t count,
      int numMaxQp,
      int nIter,
      MemAllocType memType,
      bool oneToOne = false) {
    const commDataType_t dt = commInt;

    // Setup NCCL_CTRAN_IB_MAX_QPS before comm creation so that internal QP
    // containers can be initialized
    EnvRAII env(NCCL_CTRAN_IB_MAX_QPS, numMaxQp);
    NcclCommRAII comm(globalRank, numRanks, localRank);
    ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
    ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

    // Check cumem after comm creation to make sure we have loaded cu symbols
    if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
        ncclIsCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    if (memType == kCuMemAllocDisjoint &&
        (!comm->dmaBufSupport || !NCCL_CTRAN_IB_DMABUF_ENABLE)) {
      GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
    }

    for (int peer = 0; peer < comm->ctranComm_->statex_->nRanks(); peer++) {
      if (!ctranSendRecvSupport(peer, comm->ctranComm_.get())) {
        GTEST_SKIP()
            << "Skip test since ctran cannot support SendRecv with peer "
            << peer;
      }
    }

    // always allocate buffer in page size
    size_t bufSize = pageAligned((offset + count) * commTypeSize(dt));
    size_t sendSize = count * commTypeSize(dt);
    const int sendRank = 0;
    const int oneRecvRank = numRanks - 1;
    const bool isReceiver = (oneToOne && globalRank == oneRecvRank) ||
        (!oneToOne && globalRank != sendRank);
    void* base = prepareBuf(bufSize, memType, segments);
    cudaStream_t stream = 0;
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (auto& segment : segments) {
      void* hdl = nullptr;
      NCCLCHECK_TEST(ncclCommRegister(comm, segment.ptr, segment.size, &hdl));
      segHandles.push_back(hdl);
    }

    int* buf = reinterpret_cast<int*>(reinterpret_cast<char*>(base) + offset);

    for (int x = 0; x < nIter; x++) {
      // Indicating that we are in a group to populate same opCount for all ops
      // in the group
      commGroupDepth++;
      bool doSendRecv = false;

      auto opCount = comm->ctranComm_->ctran_->getOpCount();
      if (globalRank == sendRank) {
        printf(
            "Rank %d sendRank %d send to %d other ranks with offset %ld count %ld numMaxQP %d memType %s\n",
            comm->ctranComm_->statex_->rank(),
            sendRank,
            numRanks - 1, // exclude itself
            offset,
            count,
            numMaxQp,
            memType == kMemNcclMemAlloc ? "ncclMemAlloc" : "cudaMalloc");

        std::vector<int> sendVals(count);
        std::iota(std::begin(sendVals), std::end(sendVals), sendRank + x);
        CUDACHECK_TEST(
            cudaMemcpy(buf, sendVals.data(), sendSize, cudaMemcpyDefault));

        std::vector<int> recvRanks;
        if (oneToOne) {
          // Only send to the last rank
          recvRanks.push_back(oneRecvRank);
        } else {
          recvRanks.resize(numRanks);
          std::iota(recvRanks.begin(), recvRanks.end(), 0);
        }
        for (auto recvRank : recvRanks) {
          if (recvRank != globalRank) {
            EXPECT_EQ(
                ctranSend(
                    buf, count, dt, recvRank, comm->ctranComm_.get(), stream),
                commSuccess);
            doSendRecv = true;
          }
        }

        // Expect same opCount for all ops in the group
        EXPECT_EQ(comm->ctranComm_->ctran_->getOpCount(), opCount);
      } else {
        if (isReceiver) {
          CUDACHECK_TEST(cudaMemset(base, rand(), bufSize));
          EXPECT_EQ(
              ctranRecv(
                  buf, count, dt, sendRank, comm->ctranComm_.get(), stream),
              commSuccess);
          doSendRecv = true;
        }
      }

      // Indicating end of group
      commGroupDepth--;
      EXPECT_EQ(ctranGroupEndHook(), commSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      if (doSendRecv) {
        checkProfiler(comm->ctranComm_->ctran_->profiler.get(), opCount);
      }

      if (isReceiver) {
        EXPECT_EQ(
            checkChunkValue(buf, count, sendRank + x, 1, this->globalRank), 0);
      }
    }

    if (globalRank == sendRank &&
        (NCCL_SENDRECV_ALGO != NCCL_SENDRECV_ALGO::ctstaged)) {
      verifyBackendsUsed(
          comm->ctranComm_->ctran_.get(),
          comm->ctranComm_->statex_.get(),
          memType);
    }
    verifyGpeLeak(comm->ctranComm_->ctran_.get());

    // First deregister buffer to catch potential 'remote access error' caused
    // by incomplete ctranSend when ctranRecv has returned incorrectly.
    // Delaying it after check can lead to false positive since ctranSend may
    // eventually complete.
    for (auto& hdl : segHandles) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    // Sleep for a while to make sure all the colls are finished
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check the coll trace
    ASSERT_TRUE(comm->newCollTrace != nullptr);
    auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

    std::string expAlgoName;
    if (globalRank == sendRank) {
      // Sender issues numRanks - 1 sends; if more than 1, the current algoName
      // logic doesn't distingush whether send or recv.
      expAlgoName = numRanks > 2 ? "CtranSendRecv" : "CtranSend";
    } else {
      expAlgoName = "CtranRecv";
    }

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    for (const auto& coll : pastCollsJson) {
      // Ignore handle exchange
      if (coll["opName"].asString().find("HanldeExchange") ==
          std::string::npos) {
        continue;
      }
      EXPECT_EQ(
          coll["opName"].asString(), globalRank == sendRank ? "Send" : "Recv");
      // For pure send/recv, count should be set to the count of the send/recv
      if (globalRank == sendRank && numRanks - 1 > 1) {
        // If sendRank sends to multiple peers, count is 0
        EXPECT_EQ(coll["count"].asInt(), 0);
      } else {
        EXPECT_EQ(coll["count"].asInt(), count);
      }
      EXPECT_EQ(coll["algoName"].asString(), expAlgoName);
    }

    releaseBuf(base, bufSize, memType);
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }
};

class CtranTestParamFixture
    : public CtranTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<size_t, ssize_t, int, MemAllocType>> {};

TEST_P(CtranTestParamFixture, sendRecv) {
  const auto& [offset, count, numMaxQp, memType] = GetParam();

  regCache->init();

  runTest(offset, count, numMaxQp, 1 /* nIter */, memType);

  // Destroy regCache for later test with different NCCL_CTRAN_REGISTER config.
  COMMCHECK_TEST(regCache->destroy());
}

TEST_P(CtranTestParamFixture, sendRecvStagedCopyKernel) {
  const auto& [offset, count, numMaxQp, memType] = GetParam();
  EnvRAII env1(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctstaged);
  regCache->init();
  runTest(offset, count, numMaxQp, 1 /* nIter */, memType);

  // Destroy regCache for later test with different NCCL_CTRAN_REGISTER config.
  COMMCHECK_TEST(regCache->destroy());
}

TEST_F(CtranTestFixture, oneToOneSendRecv) {
  const size_t offset = 0;
  const ssize_t count = 4096;
  const int numMaxQp = 1;
  const MemAllocType memType = kMemCudaMalloc;

  regCache->init();

  runTest(offset, count, numMaxQp, 1 /* nIter */, memType, true);

  COMMCHECK_TEST(regCache->destroy());
}

class CtranAsyncRegTestParamFixture
    : public CtranTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<size_t, ssize_t, MemAllocType>> {};

TEST_P(CtranAsyncRegTestParamFixture, sendRecvWithAsyncReg) {
  constexpr int numMaxQp = 1;
  const auto& [offset, count, memType] = GetParam();
  EnvRAII env(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::async);

  // Reinitialize global cache to enable asyncReg thread (require
  // NCCL_CTRAN_REGISTER::async)
  regCache->init();

  // Run 3 iterations to test asyncReg's behavior at first iteration and
  // registration reuse in later iterations
  runTest(offset, count, numMaxQp, 3, memType);

  COMMCHECK_TEST(regCache->destroy());
}

class CtranSocketTestParamFixture
    : public CtranTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<size_t, ssize_t, MemAllocType>> {};

TEST_P(CtranSocketTestParamFixture, nvlSendRecv) {
  if (enableNolocal || localSize < numRanks) {
    GTEST_SKIP()
        << "Ctran Socket + NVL backend require intra-node only environment. Skip test";
  }
  const int numMaxQp = 1;
  const auto& [offset, count, memType] = GetParam();

  regCache->init();
  EnvRAII env1(
      NCCL_CTRAN_BACKENDS,
      std::vector<enum NCCL_CTRAN_BACKENDS>{
          NCCL_CTRAN_BACKENDS::socket, NCCL_CTRAN_BACKENDS::nvl});

  runTest(offset, count, numMaxQp, 1 /* nIter */, memType);

  // Destroy regCache for later test with different NCCL_CTRAN_REGISTER config.
  COMMCHECK_TEST(regCache->destroy());
}

class CtranSendRecvCopyEngineTestParamFixture
    : public CtranTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<size_t, ssize_t, MemAllocType>> {};

TEST_P(CtranSendRecvCopyEngineTestParamFixture, sendRecv) {
  const int numMaxQp = 1;
  const auto& [offset, count, memType] = GetParam();

  regCache->init();
  EnvRAII env1(NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE, true);
  EnvRAII env2(NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK, true);

  runTest(offset, count, numMaxQp, 1 /* nIter */, memType);

  // Destroy regCache for later test with different NCCL_CTRAN_REGISTER config.
  COMMCHECK_TEST(regCache->destroy());
}

// test various size and various num of max QP, intentionally make some sizes
// not aligned
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranTestParamFixture,
    ::testing::Values(
        // offset in bytes, count of float32, numMaxQp
        // test cudaMalloc based memory
        std::make_tuple(0, 4096, 1, kMemCudaMalloc),
        std::make_tuple(0, 4096, 4, kMemCudaMalloc),
        std::make_tuple(0, 65536, 1, kMemCudaMalloc),
        std::make_tuple(0, 65536, 4, kMemCudaMalloc),
        std::make_tuple(0, 65536, 8, kMemCudaMalloc),
        // test ncclMemAlloc based memory
        std::make_tuple(0, 4096, 1, kMemNcclMemAlloc),
        std::make_tuple(0, 4096, 4, kMemNcclMemAlloc),
        // unaligned addr and size
        std::make_tuple(5, 2097155, 1, kMemNcclMemAlloc),
        // unaligned size
        // TODO: hang here
        std::make_tuple(0, 2097155, 1, kMemNcclMemAlloc),
        // unaligned with multiple QPs
        std::make_tuple(0, 2097155, 8, kMemNcclMemAlloc),
        // large and unaligned
        std::make_tuple(5, 1073741819, 1, kMemNcclMemAlloc),
        // large with multiple QPs
        std::make_tuple(0, 2147483648, 4, kMemNcclMemAlloc),
        std::make_tuple(0, 2147483648, 16, kMemNcclMemAlloc),
        // test ncclMemAllocDisjoint memory
        std::make_tuple(0, 1UL << 21, 16, kCuMemAllocDisjoint)),
    [&](const testing::TestParamInfo<CtranTestParamFixture::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "offset_" +
          std::to_string(std::get<1>(info.param)) + "int_" +
          std::to_string(std::get<2>(info.param)) + "maxQp_" +
          testMemAllocTypeToStr(std::get<3>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    CtranAsyncRegTest,
    CtranAsyncRegTestParamFixture,
    ::testing::Values(
        // offset in bytes, count of float32
        // test cudaMalloc based memory
        std::make_tuple(0, 4096, kMemCudaMalloc),
        std::make_tuple(0, 65536, kMemCudaMalloc),
        // // test ncclMemAlloc based memory
        std::make_tuple(0, 4096, kMemNcclMemAlloc),
        // // unaligned addr and size
        std::make_tuple(5, 2097155, kMemNcclMemAlloc),
        // // unaligned size
        std::make_tuple(0, 2097155, kMemNcclMemAlloc),
        // large and unaligned
        std::make_tuple(5, 1073741819, kMemNcclMemAlloc),
        // // test ncclMemAllocDisjoint memory
        std::make_tuple(0, 1UL << 21, kCuMemAllocDisjoint)),
    [&](const testing::TestParamInfo<CtranAsyncRegTestParamFixture::ParamType>&
            info) {
      return std::to_string(std::get<0>(info.param)) + "offset_" +
          std::to_string(std::get<1>(info.param)) + "int_" +
          testMemAllocTypeToStr(std::get<2>(info.param));
    });

// test various size and various num of max QP, intentionally make some sizes
// not aligned
INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranSocketTestParamFixture,
    ::testing::Values(
        // // test ncclMemAlloc based memory
        std::make_tuple(0, 4096, kMemNcclMemAlloc),
        // // unaligned addr and size
        std::make_tuple(0, 2097155, kMemNcclMemAlloc),
        // // unaligned size
        std::make_tuple(5, 2097155, kMemNcclMemAlloc),
        // large and unaligned
        std::make_tuple(5, 1073741819, kMemNcclMemAlloc)),
    [&](const testing::TestParamInfo<CtranSocketTestParamFixture::ParamType>&
            info) {
      return std::to_string(std::get<0>(info.param)) + "offset_" +
          std::to_string(std::get<1>(info.param)) + "int_" +
          testMemAllocTypeToStr(std::get<2>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranSendRecvCopyEngineTestParamFixture,
    ::testing::Values(
        // // test ncclMemAlloc based memory
        std::make_tuple(0, 4096, kMemNcclMemAlloc),
        // // unaligned addr and size
        std::make_tuple(0, 2097155, kMemNcclMemAlloc),
        // // unaligned size
        std::make_tuple(5, 2097155, kMemNcclMemAlloc),
        // large and unaligned
        std::make_tuple(5, 1073741819, kMemNcclMemAlloc),
        std::make_tuple(0, 1UL << 21, kCuMemAllocDisjoint)),
    [&](const testing::TestParamInfo<
        CtranSendRecvCopyEngineTestParamFixture::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "offset_" +
          std::to_string(std::get<1>(info.param)) + "int_" +
          testMemAllocTypeToStr(std::get<2>(info.param));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
