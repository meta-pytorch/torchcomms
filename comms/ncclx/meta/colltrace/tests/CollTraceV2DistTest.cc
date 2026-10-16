// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comm.h" // @manual
#include "nccl.h" // @manual

#include "comms/ctran/Ctran.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/colltrace/v2/CollTraceV2.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

using meta::comms::colltracev2::CollTraceV2;
using meta::comms::colltracev2::Event;
using meta::comms::colltracev2::EventKind;

constexpr int kCount = 1024;
constexpr int kIters = 8;

struct CollSummary {
  int starts{0};
  int ends{0};
  std::chrono::system_clock::time_point firstStartTs;
  std::chrono::system_clock::time_point lastEndTs;
};

std::map<uint64_t, CollSummary> summarize(const std::vector<Event>& events) {
  std::map<uint64_t, CollSummary> out;
  for (const auto& e : events) {
    auto& s = out[e.collId];
    switch (e.kind) {
      case EventKind::kCollStartEvent:
        if (s.starts++ == 0) {
          s.firstStartTs = e.ts;
        }
        break;
      case EventKind::kCollEndEvent:
        ++s.ends;
        s.lastEndTs = e.ts;
        break;
    }
  }
  return out;
}

// Real NCCL collectives under NCCL_COLLTRACE_VERSION=2, read back through the
// colltrace v2 API keyed by the comm's commDesc.
class CollTraceV2DistTest : public NcclxBaseTestFixture {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"WORLD_SIZE", "4"},
        {"NCCL_COLLTRACE", "lifecycle"},
        {"NCCL_COLLTRACE_VERSION", "2"},
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET", "1"},
    });
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
    int dev = 0;
    int major = 0;
    CUDACHECK_TEST(cudaGetDevice(&dev));
    CUDACHECK_TEST(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    if (major < 9) {
      GTEST_SKIP() << "colltrace v2 needs sm_90+";
    }
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    if (sendBuf_ != nullptr) {
      CUDACHECK_TEST(cudaFree(sendBuf_));
    }
    if (recvBuf_ != nullptr) {
      CUDACHECK_TEST(cudaFree(recvBuf_));
    }
    NcclxBaseTestFixture::TearDown();
  }

  void allocBuffers(size_t sendCount, size_t recvCount) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf_, sendCount * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf_, recvCount * sizeof(int)));
  }

  static CollTraceV2& colltrace() {
    return CollTraceV2::instance();
  }

  static const std::string& desc(ncclComm_t comm) {
    return comm->logMetaData.commDesc;
  }

  // Connects transports, then discards everything traced so far.
  void warmup(
      ncclComm_t comm,
      const std::function<void(cudaStream_t)>& launch) {
    launch(stream_);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    colltrace().getCollEvents(desc(comm));
  }

  // Baseline launches one kernel per call. Grouped ctran send/recv launch at
  // group end, so their kernels are attributed to the last id of the group.
  void runEagerCheck(
      ncclComm_t comm,
      const std::function<void(cudaStream_t)>& launch,
      bool oneKernelPerCall = true) {
    warmup(comm, launch);
    const uint64_t before = colltrace().getLatestCollId(desc(comm));

    for (int i = 0; i < kIters; ++i) {
      launch(stream_);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    const uint64_t issued = colltrace().getLatestCollId(desc(comm)) - before;
    if (oneKernelPerCall) {
      EXPECT_EQ(issued, kIters);
    } else {
      EXPECT_GE(issued, kIters);
    }

    const auto summary = summarize(colltrace().getCollEvents(desc(comm)));
    ASSERT_LE(summary.size(), issued);
    uint64_t expectedId = before + 1;
    std::chrono::system_clock::time_point prevEnd;
    int totalStarts = 0;
    for (const auto& [collId, s] : summary) {
      if (oneKernelPerCall) {
        EXPECT_EQ(collId, expectedId++);
      } else {
        EXPECT_GT(collId, before);
        EXPECT_LE(collId, before + issued);
      }
      EXPECT_EQ(colltrace().getNumKernels(collId), s.starts)
          << "collId " << collId;
      EXPECT_EQ(s.starts, s.ends) << "collId " << collId;
      if (oneKernelPerCall) {
        EXPECT_EQ(s.starts, 1) << "collId " << collId;
      }
      totalStarts += s.starts;
      EXPECT_LE(s.firstStartTs, s.lastEndTs) << "collId " << collId;
      EXPECT_GE(s.firstStartTs, prevEnd)
          << "collId " << collId << " overlaps the previous collective";
      prevEnd = s.lastEndTs;
    }
    EXPECT_GE(totalStarts, kIters);
    if (oneKernelPerCall) {
      EXPECT_EQ(summary.size(), issued) << "every call traced one kernel";
    }
    EXPECT_TRUE(colltrace().getCollEvents(desc(comm)).empty())
        << "a second read must not replay events";
  }

  // The AllGatherP pipeline is the ctran algorithm that launches several
  // kernels per collective: PipeStart, nNodes-1 PipeSync, PipeEnd. It needs
  // nNodes > 1 and nLocalRanks > 1, so on one host the comm is split into two
  // virtual nodes of two ranks (vCliqueSize=2). The interior kernels carry no
  // GPE work; every kernel must still be traced, eager or captured.
  void checkPipelineAllGather(bool captureGraph) {
    constexpr int kVnodeRanks = 2;
    if (!ncclIsCuMemSupported()) {
      GTEST_SKIP() << "CuMem not supported";
    }
    if (numRanks % kVnodeRanks != 0 || numRanks / kVnodeRanks < 2) {
      GTEST_SKIP() << "needs at least two virtual nodes of " << kVnodeRanks;
    }
    const int nVnodes = numRanks / kVnodeRanks;
    const int kPipelineKernels = 1 + (nVnodes - 1) + 1;

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    hints.set("allgatherAlgo", "ctwin_pipeline");
    hints.set("win_register_symmetric", "1");
    hints.set("vCliqueSize", std::to_string(kVnodeRanks));
    config.hints = &hints;
    ncclx::test::NcclCommRAII comm{
        globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
    ASSERT_NE(comm->ctranComm_, nullptr);
    if (comm->ctranComm_->ctran_->mapper->ctranIbPtr() == nullptr) {
      GTEST_SKIP() << "no IB backend for the inter-vnode leg";
    }
    ASSERT_EQ(comm->ctranComm_->statex_->nNodes(), nVnodes);
    ASSERT_EQ(comm->ctranComm_->statex_->nLocalRanks(), kVnodeRanks);

    const size_t totalBytes = kCount * numRanks * sizeof(int);
    std::vector<TestMemSegment> winSegs;
    void* winBase = testAllocBuf(totalBytes, kMemNcclMemAlloc, winSegs);
    ASSERT_NE(winBase, nullptr);
    NCCLCHECK_TEST(ncclGlobalRegisterWithPtr(winBase, totalBytes));
    ncclWindow_t win = nullptr;
    ASSERT_EQ(
        ncclCommWindowRegister(
            comm, winBase, totalBytes, &win, NCCL_WIN_COLL_SYMMETRIC),
        ncclSuccess);
    auto winGuard = folly::makeGuard([&] {
      EXPECT_EQ(ncclCommWindowDeregister(comm, win), ncclSuccess);
      NCCLCHECK_TEST(ncclGlobalDeregisterWithPtr(winBase, totalBytes));
      testFreeBuf(winBase, totalBytes, kMemNcclMemAlloc);
    });
    // In place: this rank's chunk lives at its own offset in the window.
    int* recvBuf = static_cast<int*>(winBase);
    int* sendBuf = recvBuf + kCount * globalRank;
    if (!ctranAllGatherSupport(
            comm->ctranComm_.get(),
            NCCL_ALLGATHER_ALGO::ctwin_pipeline,
            stream_,
            recvBuf,
            totalBytes)) {
      GTEST_SKIP() << "ctwin_pipeline unsupported on this host";
    }

    auto launch = [&](cudaStream_t s) {
      NCCLCHECK_TEST(ncclAllGather(sendBuf, recvBuf, kCount, ncclInt, comm, s));
    };
    warmup(comm, launch);
    const uint64_t before = colltrace().getLatestCollId(desc(comm));

    if (!captureGraph) {
      for (int i = 0; i < kIters; ++i) {
        launch(stream_);
      }
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      EXPECT_EQ(colltrace().getLatestCollId(desc(comm)) - before, kIters);

      const auto summary = summarize(colltrace().getCollEvents(desc(comm)));
      ASSERT_EQ(summary.size(), kIters);
      uint64_t expectedId = before + 1;
      for (const auto& [collId, s] : summary) {
        EXPECT_EQ(collId, expectedId++);
        EXPECT_EQ(colltrace().getNumKernels(collId), kPipelineKernels)
            << "collId " << collId;
        EXPECT_EQ(s.starts, kPipelineKernels) << "collId " << collId;
        EXPECT_EQ(s.ends, kPipelineKernels) << "collId " << collId;
        EXPECT_LE(s.firstStartTs, s.lastEndTs) << "collId " << collId;
      }
      return;
    }

    constexpr int kReplays = 5;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    CUDACHECK_TEST(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed));
    launch(stream_);
    CUDACHECK_TEST(cudaStreamEndCapture(stream_, &graph));
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    const uint64_t captured = colltrace().getLatestCollId(desc(comm));
    EXPECT_EQ(captured, before + 1) << "one collId for the whole pipeline";
    EXPECT_EQ(colltrace().getNumKernels(captured), kPipelineKernels);

    for (int i = 0; i < kReplays; ++i) {
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream_));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    const auto summary = summarize(colltrace().getCollEvents(desc(comm)));
    ASSERT_EQ(summary.size(), 1u);
    const auto& [collId, s] = *summary.begin();
    EXPECT_EQ(collId, captured);
    EXPECT_EQ(s.starts, kPipelineKernels * kReplays);
    EXPECT_EQ(s.ends, kPipelineKernels * kReplays);
    EXPECT_LE(s.firstStartTs, s.lastEndTs);

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
  }

  cudaStream_t stream_{nullptr};
  int* sendBuf_{nullptr};
  int* recvBuf_{nullptr};
};

TEST_F(CollTraceV2DistTest, BaselineAllReduceOnePairPerCollective) {
  auto algoGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  EXPECT_EQ(comm->newCollTrace, nullptr) << "v1 must stay off under v2";
  allocBuffers(kCount, kCount);

  runEagerCheck(comm, [&](cudaStream_t s) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf_, recvBuf_, kCount, ncclInt, ncclSum, comm, s));
  });
}

TEST_F(CollTraceV2DistTest, BaselineAllGatherOnePairPerCollective) {
  auto algoGuard = EnvRAII(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::orig);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  allocBuffers(kCount, kCount * numRanks);

  runEagerCheck(comm, [&](cudaStream_t s) {
    NCCLCHECK_TEST(ncclAllGather(sendBuf_, recvBuf_, kCount, ncclInt, comm, s));
  });
}

TEST_F(CollTraceV2DistTest, BaselineGraphReplayReusesCapturedCollId) {
  constexpr int kReplays = 5;
  auto algoGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  allocBuffers(kCount, kCount);
  auto launch = [&](cudaStream_t s) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf_, recvBuf_, kCount, ncclInt, ncclSum, comm, s));
  };
  warmup(comm, launch);
  const uint64_t before = colltrace().getLatestCollId(desc(comm));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  CUDACHECK_TEST(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed));
  launch(stream_);
  CUDACHECK_TEST(cudaStreamEndCapture(stream_, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  const uint64_t captured = colltrace().getLatestCollId(desc(comm));
  EXPECT_EQ(captured, before + 1) << "capture assigns exactly one collId";

  for (int i = 0; i < kReplays; ++i) {
    CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream_));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));
  EXPECT_EQ(colltrace().getLatestCollId(desc(comm)), captured)
      << "replays do not enqueue";

  const auto summary = summarize(colltrace().getCollEvents(desc(comm)));
  ASSERT_EQ(summary.size(), 1u);
  const auto& [collId, s] = *summary.begin();
  EXPECT_EQ(collId, captured);
  EXPECT_EQ(colltrace().getNumKernels(collId), 1) << "armed once, at capture";
  EXPECT_EQ(s.starts, kReplays);
  EXPECT_EQ(s.ends, kReplays);

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
}

TEST_F(CollTraceV2DistTest, CtranSendRecvOnePairPerKernel) {
  auto algoGuard = EnvRAII(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctran);
  // noLocal makes ctran use IB/socket; NVLink is not up on every test host.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints;
  hints.set("noLocal", "1");
  config.hints = &hints;
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
  ASSERT_NE(comm->ctranComm_, nullptr);
  if (numRanks < 3) {
    GTEST_SKIP() << "needs >= 3 ranks for distinct send/recv peers";
  }
  const int sendPeer = (globalRank + 1) % numRanks;
  const int recvPeer = (globalRank + numRanks - 1) % numRanks;
  if (!ctranSendRecvSupport(sendPeer, comm->ctranComm_.get()) ||
      !ctranSendRecvSupport(recvPeer, comm->ctranComm_.get())) {
    GTEST_SKIP() << "ctran sendrecv unsupported on this host";
  }
  allocBuffers(kCount, kCount);
  const auto ctranOpsBefore = comm->ctranComm_->getCtranOpCount();

  runEagerCheck(
      comm,
      [&](cudaStream_t s) {
        NCCLCHECK_TEST(ncclGroupStart());
        NCCLCHECK_TEST(ncclSend(sendBuf_, kCount, ncclInt, sendPeer, comm, s));
        NCCLCHECK_TEST(ncclRecv(recvBuf_, kCount, ncclInt, recvPeer, comm, s));
        NCCLCHECK_TEST(ncclGroupEnd());
      },
      /*oneKernelPerCall=*/false);

  EXPECT_GE(comm->ctranComm_->getCtranOpCount() - ctranOpsBefore, kIters)
      << "send/recv did not go through ctran";
}

TEST_F(CollTraceV2DistTest, CtranAllGatherPipelineEagerTracesEveryKernel) {
  checkPipelineAllGather(/*captureGraph=*/false);
}

TEST_F(CollTraceV2DistTest, CtranAllGatherPipelineGraphTracesEveryKernel) {
  checkPipelineAllGather(/*captureGraph=*/true);
}

TEST_F(CollTraceV2DistTest, ZeroCountCollectiveGetsAnIdAndNoKernels) {
  auto algoGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  allocBuffers(kCount, kCount);
  auto launch = [&](cudaStream_t s) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf_, recvBuf_, kCount, ncclInt, ncclSum, comm, s));
  };
  warmup(comm, launch);
  const uint64_t before = colltrace().getLatestCollId(desc(comm));

  // NCCL discards a zero-count collective before it becomes a task.
  NCCLCHECK_TEST(
      ncclAllReduce(sendBuf_, recvBuf_, 0, ncclInt, ncclSum, comm, stream_));
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));

  const uint64_t collId = colltrace().getLatestCollId(desc(comm));
  EXPECT_EQ(collId, before + 1) << "every call gets an id, kernel or not";
  EXPECT_EQ(colltrace().getNumKernels(collId), 0);
  EXPECT_TRUE(colltrace().getCollEvents(desc(comm)).empty());
}

TEST_F(CollTraceV2DistTest, TwoCommsAreReportedSeparately) {
  auto algoGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);
  ncclx::test::NcclCommRAII commA{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ncclx::test::NcclCommRAII commB{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_NE(desc(commA), desc(commB));
  allocBuffers(kCount, kCount);

  auto launch = [&](ncclComm_t comm, cudaStream_t s) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf_, recvBuf_, kCount, ncclInt, ncclSum, comm, s));
  };
  launch(commA, stream_);
  launch(commB, stream_);
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));
  colltrace().getCollEvents(desc(commA));
  colltrace().getCollEvents(desc(commB));

  for (int i = 0; i < 3; ++i) {
    launch(commA, stream_);
  }
  launch(commB, stream_);
  CUDACHECK_TEST(cudaStreamSynchronize(stream_));

  const auto a = summarize(colltrace().getCollEvents(desc(commA)));
  const auto b = summarize(colltrace().getCollEvents(desc(commB)));
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(b.size(), 1u);
  for (const auto* summary : {&a, &b}) {
    for (const auto& [collId, s] : *summary) {
      EXPECT_EQ(colltrace().getNumKernels(collId), 1) << "collId " << collId;
      EXPECT_EQ(s.starts, 1) << "collId " << collId;
      EXPECT_EQ(s.ends, 1) << "collId " << collId;
    }
  }
}

TEST_F(CollTraceV2DistTest, V1LifecycleApiIsOffUnderV2) {
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  uint64_t id = 0;
  EXPECT_EQ(ncclx::colltrace::getCollTraceCommId(comm, id), ncclInvalidUsage);
  EXPECT_EQ(
      ncclx::colltrace::getLatestCollTraceCollectiveId(comm, id),
      ncclInvalidUsage);
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
