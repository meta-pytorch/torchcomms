// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <functional>
#include <string>

#include <folly/json.h>

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphTestBase.h"
#include "comms/utils/CudaRAII.h"
#include "comms/utils/colltrace/CollTrace.h"

namespace {

// Exercises the in-kernel colltrace emit path end-to-end: a real collective is
// captured into a CUDA graph and replayed, and we assert colltrace produced
// exactly one start/end pair per replay. This is the coverage the synthetic
// graph_colltrace_ut misses -- that unit test drives the ring directly, so it
// never catches an algo whose end event is not armed (which leaves the
// collective stuck mid-flight and hangs the drain).
class CtranCudaGraphColltraceTest : public CtranCudaGraphTestBase {
 protected:
  void SetUp() override {
    // Enable graph-mode colltrace tracing. In-kernel emit is the only graph
    // timestamp path and is gated on sm_90+ in the GPE; on older archs
    // graph-captured collectives are not colltrace-timestamped.
    CtranDistTestFixture::SetUp(
        ctran::CtranEnvs{
            {"NCCL_COLLTRACE", "trace"},
            {"NCCL_COLLTRACE_TRACE_CUDA_GRAPH", "true"},
        });
    if (!meta::comms::colltrace::graphColltraceSupported(
            "CtranCudaGraphColltraceTest")) {
      GTEST_SKIP() << "graph colltrace unsupported on this device/driver "
                      "(needs sm_90+ and a CUDA driver outside 13.0-13.2)";
    }
  }

  // Warms the collective eagerly (establishes transport connections and gives a
  // stable colltrace baseline), then captures it once and replays it
  // `numReplays` times under one CUDA graph on a single stream. Asserts that:
  //   - colltrace fully drains (nothing left in CT_currentColls) -- a missing
  //     end event would leave a collective mid-flight and time out the drain;
  //   - exactly one past record is produced per replay;
  //   - every new record names the expected collective.
  void runGraphColltraceCheck(
      CtranComm* comm,
      int numReplays,
      const std::string& expectedOpName,
      const std::function<void(cudaStream_t)>& launch) {
    meta::comms::CudaStream stream(cudaStreamNonBlocking);

    auto flushAndDump = [&]() {
      if (comm->colltraceNew_ != nullptr) {
        comm->colltraceNew_->waitFlush(comm->colltraceNew_->requestFlush());
      }
      return ctran::waitForCollTraceDrain(comm);
    };

    // Eager warmup + baseline.
    launch(stream.get());
    CUDACHECK_TEST(cudaStreamSynchronize(stream.get()));
    auto baseline = flushAndDump();
    ASSERT_EQ(baseline["CT_currentColls"], "[]") << "warmup collective never "
                                                    "drained";
    const int baseCount = folly::parseJson(baseline["CT_pastColls"]).size();

    // Capture the collective once.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    ASSERT_EQ(
        cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeRelaxed),
        cudaSuccess);
    launch(stream.get());
    ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
        cudaSuccess);

    // Replay N times on one stream so completions are ordered.
    for (int i = 0; i < numReplays; ++i) {
      ASSERT_EQ(cudaGraphLaunch(graphExec, stream.get()), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

    auto dump = flushAndDump();
    EXPECT_EQ(dump["CT_currentColls"], "[]")
        << "a replayed collective never completed -- missing colltrace end "
           "event for "
        << expectedOpName;
    EXPECT_EQ(dump["CT_pendingColls"], "[]");

    auto past = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(static_cast<int>(past.size()) - baseCount, numReplays)
        << "expected exactly one colltrace record per graph replay";
    for (int i = baseCount; i < static_cast<int>(past.size()); ++i) {
      EXPECT_EQ(past[i]["opName"].asString(), expectedOpName);
    }

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
    waitAndVerifyGpeClean(comm);
  }
};

constexpr int kNumReplays = 5;
constexpr size_t kCount = 1024;

TEST_F(CtranCudaGraphColltraceTest, AllGatherOneRecordPerReplay) {
  if (numRanks < 2) {
    GTEST_SKIP() << "needs >= 2 ranks";
  }
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  const auto algo = NCCL_ALLGATHER_ALGO::ctran;
  if (!ctranAllGatherSupport(comm.get(), algo)) {
    GTEST_SKIP() << "AllGather not supported on this topology";
  }

  ctran::TestDeviceBuffer send(kCount * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(kCount * numRanks * sizeof(int32_t));
  fillSendBuf(send.get(), kCount, globalRank);

  runGraphColltraceCheck(
      comm.get(), kNumReplays, "AllGather", [&](cudaStream_t s) {
        ASSERT_EQ(
            ctranAllGather(
                send.get(), recv.get(), kCount, commInt32, comm.get(), s, algo),
            commSuccess);
      });
}

TEST_F(CtranCudaGraphColltraceTest, AllReduceOneRecordPerReplay) {
  if (numRanks < 2) {
    GTEST_SKIP() << "needs >= 2 ranks";
  }
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  const auto algo = NCCL_ALLREDUCE_ALGO::ctran;
  if (!ctranAllReduceSupport(comm.get(), algo)) {
    GTEST_SKIP() << "AllReduce not supported on this topology";
  }

  ctran::TestDeviceBuffer send(kCount * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(kCount * sizeof(int32_t));
  fillSendBuf(send.get(), kCount, globalRank);

  runGraphColltraceCheck(
      comm.get(), kNumReplays, "AllReduce", [&](cudaStream_t s) {
        ASSERT_EQ(
            ctranAllReduce(
                send.get(),
                recv.get(),
                kCount,
                commInt32,
                commSum,
                comm.get(),
                s,
                algo),
            commSuccess);
      });
}

TEST_F(CtranCudaGraphColltraceTest, ReduceScatterOneRecordPerReplay) {
  if (numRanks < 2) {
    GTEST_SKIP() << "needs >= 2 ranks";
  }
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  const auto algo = NCCL_REDUCESCATTER_ALGO::ctran;
  if (!ctranReduceScatterSupport(comm.get(), algo)) {
    GTEST_SKIP() << "ReduceScatter not supported on this topology";
  }

  ctran::TestDeviceBuffer send(kCount * numRanks * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(kCount * sizeof(int32_t));
  fillSendBuf(send.get(), kCount * numRanks, globalRank);

  runGraphColltraceCheck(
      comm.get(), kNumReplays, "ReduceScatter", [&](cudaStream_t s) {
        ASSERT_EQ(
            ctranReduceScatter(
                send.get(),
                recv.get(),
                kCount,
                commInt32,
                commSum,
                comm.get(),
                s,
                algo),
            commSuccess);
      });
}

// AllGatherP `ctpipeline` is the only path that uses the multi-kernel colltrace
// grouping (PipeStart=kBegin, PipeSync=kInner, PipeEnd=kEnd); every other
// collective is a single kSolo kernel. This asserts the grouping collapses the
// whole pipeline to exactly ONE colltrace record per graph replay (not one per
// pipe stage, and not zero from a dropped end). ctgraph_pipeline requires an
// intra-node NVL domain (nLocalRanks > 1) and a registered recvbuf, and is only
// used during capture (eager warmup uses the direct ctran algo).
TEST_F(CtranCudaGraphColltraceTest, AllGatherPPipelineOneRecordPerReplay) {
  if (numRanks < 2) {
    GTEST_SKIP() << "needs >= 2 ranks";
  }
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);
  if (!ctran::allGatherPSupport(comm.get()) ||
      comm->statex_.get()->nLocalRanks() <= 1) {
    GTEST_SKIP() << "AllGatherP ctpipeline requires nLocalRanks > 1";
  }

  ctran::TestDeviceBuffer send(kCount * sizeof(int32_t));
  ctran::TestDeviceBuffer recv(kCount * numRanks * sizeof(int32_t));
  fillSendBuf(send.get(), kCount, globalRank);

  // The graph AGP path requires the recvbuf segment cached in the regcache;
  // keep it registered until after the graph is destroyed below.
  const size_t recvBytes = kCount * numRanks * sizeof(int32_t);
  ctran::RegCache::getInstance()->globalRegister(
      recv.get(), recvBytes, /*forceReg=*/true);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);
  auto launch = [&](cudaStream_t s, enum NCCL_ALLGATHER_ALGO algo) {
    ASSERT_EQ(
        ctranAllGather(
            send.get(), recv.get(), kCount, commInt32, comm.get(), s, algo),
        commSuccess);
  };
  // Count only "AllGatherP" records (ignoring the one-time "AllGatherP_Init"
  // and the direct-warmup "AllGather"), after draining colltrace.
  auto countAllGatherP = [&]() -> int {
    if (comm->colltraceNew_ != nullptr) {
      comm->colltraceNew_->waitFlush(comm->colltraceNew_->requestFlush());
    }
    auto d = ctran::waitForCollTraceDrain(comm.get());
    EXPECT_EQ(d["CT_currentColls"], "[]");
    int n = 0;
    for (const auto& rec : folly::parseJson(d["CT_pastColls"])) {
      if (rec["opName"].asString() == "AllGatherP") {
        ++n;
      }
    }
    return n;
  };

  // Eager warmup on the direct algo, then capture the pipeline algo once.
  launch(stream.get(), NCCL_ALLGATHER_ALGO::ctran);
  CUDACHECK_TEST(cudaStreamSynchronize(stream.get()));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  launch(stream.get(), NCCL_ALLGATHER_ALGO::ctgraph_pipeline);
  ASSERT_EQ(cudaStreamEndCapture(stream.get(), &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(
      cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
      cudaSuccess);

  const int before = countAllGatherP();
  for (int i = 0; i < kNumReplays; ++i) {
    ASSERT_EQ(cudaGraphLaunch(graphExec, stream.get()), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  const int after = countAllGatherP();

  EXPECT_EQ(after - before, kNumReplays)
      << "AllGatherP ctpipeline should emit exactly one colltrace record per "
         "replay (one logical collective via kBegin/kInner/kEnd grouping, not "
         "one per pipe stage): before="
      << before << " after=" << after;

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  waitAndVerifyGpeClean(comm.get());
  ctran::RegCache::getInstance()->globalDeregister(recv.get(), recvBytes);
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
