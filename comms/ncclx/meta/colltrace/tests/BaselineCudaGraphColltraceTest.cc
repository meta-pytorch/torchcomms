// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gtest/gtest.h>

#include "comm.h" // @manual
#include "nccl.h" // @manual

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

namespace {

// Baseline (orig NCCL, non-ctran) analog of CtranCudaGraphColltraceTest:
// capture a real collective into a CUDA graph, replay it, and assert baseline
// colltrace produced exactly one start/end pair per replay via the in-kernel
// emit path. The synthetic graph_colltrace_ut drives the ring directly, so it
// never catches a baseline kernel whose end event is not armed -- which would
// leave the collective stuck mid-flight and hang the drain. In-kernel emit is
// the only graph timestamp path and is gated on sm_90+; on older archs
// graph-captured collectives are not colltrace-timestamped.
class BaselineCudaGraphColltraceTest : public NcclxBaseTestFixture {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"WORLD_SIZE", "4"},
        {"NCCL_COLLTRACE", "trace"},
        {"NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "true"},
        // In-kernel device write is off by default until the cutover diff, so
        // enable it explicitly to exercise the in-kernel colltrace path.
        {"NCCLX_COLLTRACE_DEVICE_WRITE", "true"},
    });
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
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

  std::unordered_map<std::string, std::string> flushAndDump(ncclComm_t comm) {
    EXPECT_NE(comm->newCollTrace, nullptr);
    if (comm->newCollTrace == nullptr) {
      return {};
    }
    comm->newCollTrace->waitFlush(comm->newCollTrace->requestFlush());
    EXPECT_TRUE(meta::comms::ncclx::waitForCollTraceDrain(*comm->newCollTrace));
    return meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);
  }

  // Warms the collective eagerly (establishes transport connections and a
  // stable colltrace baseline), captures it once, and replays it `numReplays`
  // times under one CUDA graph on a single stream. Asserts colltrace fully
  // drains (nothing left mid-flight) and produces exactly one past record per
  // replay.
  void runGraphColltraceCheck(
      ncclComm_t comm,
      int numReplays,
      const std::string& expectedOpName,
      const std::function<void(cudaStream_t)>& launch) {
    // Eager warmup + baseline record count.
    launch(stream_);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    auto baseline = flushAndDump(comm);
    ASSERT_EQ(baseline["CT_currentColls"], "[]") << "warmup never drained";
    const int baseCount =
        static_cast<int>(folly::parseJson(baseline["CT_pastColls"]).size());

    // Capture the collective once.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    ASSERT_EQ(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed),
        cudaSuccess);
    launch(stream_);
    ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
        cudaSuccess);

    // Replay N times on one stream so completions are ordered.
    for (int i = 0; i < numReplays; ++i) {
      ASSERT_EQ(cudaGraphLaunch(graphExec, stream_), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    auto dump = flushAndDump(comm);
    EXPECT_EQ(dump["CT_currentColls"], "[]")
        << "a replayed collective never completed -- missing colltrace end "
           "event for "
        << expectedOpName;
    EXPECT_EQ(dump["CT_pendingColls"], "[]");

    auto past = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(static_cast<int>(past.size()) - baseCount, numReplays)
        << "expected exactly one colltrace record per graph replay for "
        << expectedOpName;

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
  }

  cudaStream_t stream_{};
  int* sendBuf_{nullptr};
  int* recvBuf_{nullptr};
};

constexpr int kNumReplays = 5;
constexpr int kCount = 1024;

TEST_F(BaselineCudaGraphColltraceTest, AllReduceOneRecordPerReplay) {
  if (numRanks < 2) {
    GTEST_SKIP() << "needs >= 2 ranks";
  }
  // Force the baseline (orig NCCL) AllReduce kernel path, not ctran.
  auto algoGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_TRUE(comm->newCollTrace != nullptr);

  CUDACHECK_TEST(cudaMalloc(&sendBuf_, kCount * sizeof(int)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf_, kCount * sizeof(int)));

  runGraphColltraceCheck(comm, kNumReplays, "AllReduce", [&](cudaStream_t s) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf_, recvBuf_, kCount, ncclInt, ncclSum, comm, s));
  });
}

TEST_F(BaselineCudaGraphColltraceTest, AllGatherOneRecordPerReplay) {
  if (numRanks < 2) {
    GTEST_SKIP() << "needs >= 2 ranks";
  }
  // Force the baseline (orig NCCL) AllGather kernel path, not ctran.
  auto algoGuard = EnvRAII(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::orig);
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_TRUE(comm->newCollTrace != nullptr);

  CUDACHECK_TEST(cudaMalloc(&sendBuf_, kCount * sizeof(int)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf_, kCount * numRanks * sizeof(int)));

  runGraphColltraceCheck(comm, kNumReplays, "AllGather", [&](cudaStream_t s) {
    NCCLCHECK_TEST(ncclAllGather(sendBuf_, recvBuf_, kCount, ncclInt, comm, s));
  });
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
