// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/utils/CommGroupUtils.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class CtranTestFixture : public ctran::CtranDistTestFixture,
                         public CtranBaseTest {
 public:
  std::vector<TestMemSegment> segments;
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
    // -1 for not limiting the number of colls to trace
    setenv("NCCL_COLLTRACE_RECORD_MAX", "-1", 0);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_TRANSPORT_PROFILER", "1", 0);
    setenv("NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT", "1", 0);
    ctran::CtranDistTestFixture::SetUp();
    srand(time(NULL));
    ctran::logGpuMemoryStats(globalRank);

    regCache = ctran::RegCache::getInstance();
    ctran::CHECK_VALID_REGCACHE(regCache);
  }

  void TearDown() override {
    ctran::logGpuMemoryStats(globalRank);
    ctran::CtranDistTestFixture::TearDown();
  }

  static void checkProfiler(ctran::Profiler* profiler, uint64_t opCount) {
    // algo profiler currently only enabled for IB backend
    if (NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE ||
        NCCL_SENDRECV_ALGO == NCCL_SENDRECV_ALGO::ctp2p) {
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
   * @param numSegments Number of segments for kCuMemAllocDisjoint (default: 2)
   */
  void runTest(
      size_t offset,
      ssize_t count,
      int numMaxQp,
      int nIter,
      MemAllocType memType,
      bool oneToOne = false,
      size_t numSegments = 2,
      size_t numOpPairsPerPeer = 1,
      bool useGraph = false) {
    const commDataType_t dt = commInt;

    // Setup NCCL_CTRAN_IB_MAX_QPS before comm creation so that internal QP
    // containers can be initialized
    EnvRAII env(NCCL_CTRAN_IB_MAX_QPS, numMaxQp);
    auto ctranComm = makeCtranComm();
    ASSERT_NE(nullptr, ctranComm.get());
    ASSERT_NE(nullptr, ctranComm->ctran_.get());

    // Check cumem after comm creation to make sure we have loaded cu symbols
    if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
        ctran::utils::isCuMemSupported() == false) {
      GTEST_SKIP() << "CuMem not supported, skip test";
    }

    if (memType == kCuMemAllocDisjoint) {
      int cudaDev;
      CUDACHECK_TEST(cudaGetDevice(&cudaDev));
      auto ibSingleton = CtranIbSingleton::getInstance();
      if (!ibSingleton || !ibSingleton->getDevToDmaBufSupport(cudaDev) ||
          !NCCL_CTRAN_IB_DMABUF_ENABLE) {
        GTEST_SKIP() << "dmabuf is not supported, skip disjoint test";
      }
    }

    for (int peer = 0; peer < ctranComm->statex_->nRanks(); peer++) {
      if (!ctranSendRecvSupport(peer, ctranComm.get())) {
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
    void* base = prepareBuf(bufSize, memType, segments, numSegments);
    cudaStream_t stream = 0;
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (auto& segment : segments) {
      COMMCHECK_TEST(ctran::globalRegisterWithPtr(segment.ptr, segment.size));
    }

    int* buf = reinterpret_cast<int*>(reinterpret_cast<char*>(base) + offset);

    // graph mode: prepare data once before capture
    if (useGraph) {
      if (globalRank == sendRank) {
        std::vector<int> sendVals(count);
        std::iota(std::begin(sendVals), std::end(sendVals), sendRank);
        CUDACHECK_TEST(
            cudaMemcpy(buf, sendVals.data(), sendSize, cudaMemcpyDefault));
      } else {
        CUDACHECK_TEST(cudaMemset(base, 0, bufSize));
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());

      CUDACHECK_TEST(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    }

    for (int x = 0; x < nIter; x++) {
      // Indicating that we are in a group to populate same opCount for all ops
      // in the group
      commGroupDepth++;
      bool doSendRecv = false;

      auto opCount = ctranComm->ctran_->getOpCount();
      if (globalRank == sendRank) {
        printf(
            "Rank %d sendRank %d send to %d other ranks with offset %ld count %ld numMaxQP %d memType %s\n",
            ctranComm->statex_->rank(),
            sendRank,
            numRanks - 1, // exclude itself
            offset,
            count,
            numMaxQp,
            memType == kMemNcclMemAlloc ? "ncclMemAlloc" : "cudaMalloc");

        if (!useGraph) {
          std::vector<int> sendVals(count);
          std::iota(std::begin(sendVals), std::end(sendVals), sendRank + x);
          CUDACHECK_TEST(
              cudaMemcpy(buf, sendVals.data(), sendSize, cudaMemcpyDefault));
        }

        std::vector<int> recvRanks;
        if (oneToOne) {
          // Only send to the last rank
          recvRanks.push_back(oneRecvRank);
        } else {
          recvRanks.resize(numRanks);
          std::iota(recvRanks.begin(), recvRanks.end(), 0);
        }
        for (size_t p = 0; p < numOpPairsPerPeer; p++) {
          for (auto recvRank : recvRanks) {
            if (recvRank != globalRank) {
              EXPECT_EQ(
                  ctranSend(buf, count, dt, recvRank, ctranComm.get(), stream),
                  commSuccess);
              doSendRecv = true;
            }
          }
        }

        // Expect same opCount for all ops in the group
        EXPECT_EQ(ctranComm->ctran_->getOpCount(), opCount);
      } else {
        if (isReceiver) {
          if (!useGraph) {
            CUDACHECK_TEST(cudaMemset(base, rand(), bufSize));
          }
          for (size_t p = 0; p < numOpPairsPerPeer; p++) {
            EXPECT_EQ(
                ctranRecv(buf, count, dt, sendRank, ctranComm.get(), stream),
                commSuccess);
          }
          doSendRecv = true;
        }
      }

      // Indicating end of group
      commGroupDepth--;
      EXPECT_EQ(ctranGroupEndHook(NCCL_SENDRECV_ALGO), commSuccess);

      if (!useGraph) {
        CUDACHECK_TEST(cudaStreamSynchronize(stream));

        if (doSendRecv) {
          checkProfiler(ctranComm->ctran_->profiler.get(), opCount);
        }

        if (isReceiver) {
          EXPECT_EQ(
              checkChunkValue(buf, count, sendRank + x, 1, this->globalRank),
              0);
        }
      }
    }

    if (useGraph) {
      cudaGraph_t graph;
      CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
      ASSERT_NE(graph, nullptr);

      cudaGraphExec_t instance;
      CUDACHECK_TEST(
          cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

      CUDACHECK_TEST(cudaGraphLaunch(instance, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      if (isReceiver) {
        EXPECT_EQ(
            checkChunkValue(buf, count, sendRank, 1, this->globalRank), 0);
      }

      CUDACHECK_TEST(cudaGraphExecDestroy(instance));
      CUDACHECK_TEST(cudaGraphDestroy(graph));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Wait for GPE thread to finish processing the destroyed cmd
      while (ctranComm->ctran_->gpe->numInUseKernelFlags() > 0 ||
             ctranComm->ctran_->gpe->numInUseKernelElems() > 0) {
        std::this_thread::yield();
      }
    }

    if (!useGraph) {
      if (globalRank == sendRank &&
          (NCCL_SENDRECV_ALGO != NCCL_SENDRECV_ALGO::ctp2p)) {
        verifyBackendsUsed(
            ctranComm->ctran_.get(), ctranComm->statex_.get(), memType);
      }
    }

    verifyGpeLeak(ctranComm->ctran_.get());

    if (!useGraph) {
      // Brief wait for GPE thread to flush colltrace entries after stream sync
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Check the coll trace only for participating ranks
      bool participated = (globalRank == sendRank) || isReceiver;
      if (participated) {
        auto dumpMap = ctran::dumpCollTrace(ctranComm.get());
        ASSERT_FALSE(dumpMap.empty()) << "Colltrace should be initialized";

        int numSendPeers = oneToOne ? 1 : (numRanks - 1);
        size_t totalOps = (globalRank == sendRank)
            ? numSendPeers * numOpPairsPerPeer
            : numOpPairsPerPeer;
        std::string expAlgoName;
        if (totalOps > 1) {
          expAlgoName = "CtranSendRecv";
        } else if (globalRank == sendRank) {
          expAlgoName = "CtranSend";
        } else {
          expAlgoName = "CtranRecv";
        }

        auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
        ASSERT_FALSE(pastCollsJson.empty()) << "pastColls should not be empty";
        for (const auto& coll : pastCollsJson) {
          auto algoName = coll.getDefault("algoName", "").asString();
          auto opName = coll.getDefault("opName", "").asString();
          // Skip handle exchange entries
          if (algoName.find("HanldeExchange") != std::string::npos) {
            continue;
          }
          // algoName is always populated
          EXPECT_EQ(algoName, expAlgoName);
          // opName and count are only populated when GPE opGroup is non-empty
          // (i.e., the default algo). For ctp2p/CopyEngine notify
          // kernels, the opGroup is empty so opName/count are not set.
          if (!opName.empty() && coll.count("count")) {
            EXPECT_EQ(opName, globalRank == sendRank ? "Send" : "Recv");
            if (globalRank == sendRank && numSendPeers > 1) {
              // getGroupedP2PMetaData sums counts across all send ops
              EXPECT_EQ(
                  coll["count"].asInt(),
                  static_cast<int64_t>(count) * numSendPeers);
            } else {
              EXPECT_EQ(coll["count"].asInt(), count);
            }
          }
        }
      }
    }

    // First deregister buffer to catch potential 'remote access error' caused
    // by incomplete ctranSend when ctranRecv has returned incorrectly.
    // Delaying it after check can lead to false positive since ctranSend may
    // eventually complete.
    for (auto& segment : segments) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(segment.ptr, segment.size));
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());

    releaseBuf(base, bufSize, memType, numSegments);
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

TEST_P(CtranTestParamFixture, sendRecvP2pCopyKernel) {
  const auto& [offset, count, numMaxQp, memType] = GetParam();
  EnvRAII env1(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctp2p);
  regCache->init();
  runTest(offset, count, numMaxQp, 1 /* nIter */, memType);

  // Destroy regCache for later test with different NCCL_CTRAN_REGISTER config.
  COMMCHECK_TEST(regCache->destroy());
}

class CtranP2pUseListTestFixture
    : public CtranTestFixture,
      public ::testing::WithParamInterface<size_t /* numOpPairsPerPeer */> {};

TEST_P(CtranP2pUseListTestFixture, sendRecvP2pUseList) {
  const size_t numOpPairsPerPeer = GetParam();

  // Need enough total send ops to trigger useList (> kCtranMaxNvlSendRecvOps)
  if (numOpPairsPerPeer * (numRanks - 1) <=
      ctran::sendrecv::kCtranMaxNvlSendRecvOps) {
    GTEST_SKIP() << "Not enough ops to trigger useList with " << numRanks
                 << " ranks";
  }

  EnvRAII env1(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctp2p);
  regCache->init();
  runTest(
      0 /* offset */,
      4096 /* count */,
      1 /* numMaxQp */,
      1 /* nIter */,
      kMemNcclMemAlloc,
      false /* oneToOne */,
      2 /* numSegments */,
      numOpPairsPerPeer);
  COMMCHECK_TEST(regCache->destroy());
}

INSTANTIATE_TEST_SUITE_P(
    CtranP2pUseListTest,
    CtranP2pUseListTestFixture,
    ::testing::Values(
        /* pool path */ 1,
        /* ad-hoc alloc path */
        ctran::sendrecv::kMaxSendRecvOpsPerPoolBuf + 1),
    [](const testing::TestParamInfo<size_t>& info) {
      return "numOpPairsPerPeer_" + std::to_string(info.param);
    });

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

// Test case for NVL zero-copy path with 3+ segments to expose
// CTRAN_IPC_INLINE_SEGMENTS limitation. This test demonstrates the bug where
// Ctran NVL zero-copy path fails when memory is backed by 3+ physical memory
// allocations (expandable segments). The current implementation is limited to 2
// segments due to fixed-size CtranIpcDesc.segments array.
//
// Expected behavior with current code: FAIL with error:
// "CTRAN-IPC: tried to export CtranIpcMem backed by too many physical memory
// allocations."
//
// After fix: Test should PASS
TEST_F(CtranTestFixture, DISABLED_sendRecvCopyEngineMultiSegment) {
  // Use kCuMemAllocDisjoint with 3 segments to trigger the bug
  const MemAllocType memType = kCuMemAllocDisjoint;
  constexpr size_t numSegments = 3;
  const size_t offset = 0;
  // Use 6MB buffer = 3 x 2MB segments to ensure 3 physical allocations
  const ssize_t count = 6 * 1024 * 1024 / sizeof(int); // 6MB in int elements
  const int numMaxQp = 1;

  EnvRAII env1(NCCL_CTRAN_NVL_SENDRECV_COPY_ENGINE_ENABLE, true);

  if (ctran::utils::isCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  regCache->init();

  // This test currently exposes a bug - the runTest will fail with:
  // "CTRAN ERROR CTRAN-IPC: tried to export CtranIpcMem backed by too many
  // physical memory allocations."
  // The dmaBuf support check is done inside runTest after comm creation.
  runTest(
      offset,
      count,
      numMaxQp,
      1 /* nIter */,
      memType,
      false /* oneToOne */,
      numSegments);

  COMMCHECK_TEST(regCache->destroy());
}

#if not defined(__HIP_PLATFORM_AMD__) and not defined(__HIP_PLATFORM_HCC__)

class CtranP2pCudaGraphTestFixture
    : public CtranTestFixture,
      public ::testing::WithParamInterface<bool /* oneToOne */> {};

TEST_P(CtranP2pCudaGraphTestFixture, sendRecvP2p) {
  const bool oneToOne = GetParam();
  EnvRAII env1(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctp2p);
  regCache->init();
  runTest(
      0,
      4096,
      1 /* numMaxQp */,
      1 /* nIter */,
      kMemNcclMemAlloc,
      oneToOne,
      2,
      1,
      true /* useGraph */);
  COMMCHECK_TEST(regCache->destroy());
}

INSTANTIATE_TEST_SUITE_P(
    CtranP2pCudaGraphTest,
    CtranP2pCudaGraphTestFixture,
    ::testing::Values(
        /* useList path */ false,
        /* non-useList path */ true),
    [](const testing::TestParamInfo<bool>& info) {
      return "useList_" + std::to_string(!info.param);
    });

#endif

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
