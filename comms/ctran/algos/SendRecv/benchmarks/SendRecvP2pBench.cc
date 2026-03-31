// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// NCCL-level SendRecv P2P benchmark.
//
// Exercises the full ctranSend/ctranRecv path with NCCL_SENDRECV_ALGO=ctp2p.
// Three configurations:
//   Eager (pool): ops fit in pinned mempool buffer, kernel reads via UVA.
//   Eager (direct alloc): ops exceed pool buffer, direct cudaHostAlloc.
//   Graph capture: pool skipped, direct cudaHostAlloc for graph-safe memory.

#include <comm.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <nccl.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/utils/CommGroupUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

DEFINE_int32(warmup_iters, 100, "Number of warmup iterations");
DEFINE_int32(bench_iters, 1000, "Number of benchmark iterations");

class SendRecvP2pBenchmark : public NcclxBaseTest {
 public:
  std::shared_ptr<ctran::RegCache> regCache{nullptr};

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
    regCache = ctran::RegCache::getInstance();
    ctran::CHECK_VALID_REGCACHE(regCache);
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  }
};

TEST_F(SendRecvP2pBenchmark, BenchmarkSuite) {
  const ssize_t count = 4096;
  const commDataType_t dt = commInt;
  const int sendRank = 0;

  EnvRAII env1(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctp2p);
  EnvRAII env2(NCCL_CTRAN_IB_MAX_QPS, 1);
  regCache->init();

  NcclCommRAII comm(globalRank, numRanks, localRank);
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  for (int peer = 0; peer < comm->ctranComm_->statex_->nRanks(); peer++) {
    if (!ctranSendRecvSupport(peer, comm->ctranComm_.get())) {
      GTEST_SKIP() << "ctran SendRecv not supported for peer " << peer;
    }
  }

  size_t bufSize = count * commTypeSize(dt);
  void* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
  void* hdl = nullptr;
  NCCLCHECK_TEST(ncclCommRegister(comm, buf, bufSize, &hdl));

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Run one send/recv iteration.
  // maxPeers: how many peers rank 0 sends to (0 = all peers).
  // numSendsPerPeer: how many sends per peer (>1 to overflow the pool buffer).
  auto runIter = [&](int numSendsPerPeer, int maxPeers = 0) {
    int peersToSend = maxPeers > 0 ? maxPeers : (numRanks - 1);
    commGroupDepth++;
    if (globalRank == sendRank) {
      int peersSent = 0;
      for (int peer = 0; peer < numRanks && peersSent < peersToSend; peer++) {
        if (peer != globalRank) {
          for (int s = 0; s < numSendsPerPeer; s++) {
            EXPECT_EQ(
                ctranSend(buf, count, dt, peer, comm->ctranComm_.get(), stream),
                commSuccess);
          }
          peersSent++;
        }
      }
    } else if (globalRank <= peersToSend) {
      // Only ranks within maxPeers participate in recv
      for (int s = 0; s < numSendsPerPeer; s++) {
        EXPECT_EQ(
            ctranRecv(buf, count, dt, sendRank, comm->ctranComm_.get(), stream),
            commSuccess);
      }
    }
    commGroupDepth--;
    EXPECT_EQ(ctranGroupEndHook(NCCL_SENDRECV_ALGO), commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
  };

  // Benchmark helper: warmup + timed loop, returns avg ms per iteration.
  auto bench =
      [&](int numSendsPerPeer, const char* name, int maxPeers = 0) -> double {
    for (int i = 0; i < FLAGS_warmup_iters; i++) {
      runIter(numSendsPerPeer, maxPeers);
    }
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < FLAGS_bench_iters; i++) {
      runIter(numSendsPerPeer, maxPeers);
    }
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double avgMs =
        std::chrono::duration<double, std::milli>(end - start).count() /
        FLAGS_bench_iters;
    if (globalRank == 0) {
      std::cout << std::left << std::setw(40) << name << std::right
                << std::fixed << std::setprecision(3) << std::setw(14) << avgMs
                << " ms" << std::endl;
    }
    return avgMs;
  };

  if (globalRank == 0) {
    std::cout
        << "\n================================================================"
        << "\nSendRecv P2P Benchmark (ctp2p algo)"
        << "\n  Ranks: " << numRanks << ", Count: " << count << " int32 ("
        << bufSize << " bytes)"
        << "\n  Warmup: " << FLAGS_warmup_iters
        << ", Bench: " << FLAGS_bench_iters
        << "\n================================================================"
        << std::endl;
  }

  // 0. Baseline: 1 send to 1 peer, uses static array path (no useList)
  bench(1, "Baseline (static array, 1 peer)", /*maxPeers=*/1);

  // 1. Eager, pool path: 1 send per peer fits in pool buffer
  bench(1, "Eager (pool, 1 send/peer)");

  // 2. Eager, direct alloc: overflow pool buffer by sending enough times
  //    per peer that total ops > kMaxSendRecvOpsPerPoolBuf.
  int sendsToOverflow = (((CTRAN_MAX_NVL_PEERS - 1) * 2) / (numRanks - 1)) + 1;
  std::string overflowName = "Eager (direct alloc, " +
      std::to_string(sendsToOverflow) + " sends/peer)";
  bench(sendsToOverflow, overflowName.c_str());

  // 3. Graph capture path: direct alloc, pool skipped during capture.
  {
    // Warmup outside capture
    for (int i = 0; i < FLAGS_warmup_iters; i++) {
      runIter(1);
    }
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Capture one iteration into a graph
    CUDACHECK_TEST(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    commGroupDepth++;
    if (globalRank == sendRank) {
      for (int peer = 0; peer < numRanks; peer++) {
        if (peer != globalRank) {
          EXPECT_EQ(
              ctranSend(buf, count, dt, peer, comm->ctranComm_.get(), stream),
              commSuccess);
        }
      }
    } else {
      EXPECT_EQ(
          ctranRecv(buf, count, dt, sendRank, comm->ctranComm_.get(), stream),
          commSuccess);
    }
    commGroupDepth--;
    EXPECT_EQ(ctranGroupEndHook(NCCL_SENDRECV_ALGO), commSuccess);

    cudaGraph_t graph;
    CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Timed graph replay using CUDA events for accurate GPU timing
    cudaEvent_t startEvt, endEvt;
    CUDACHECK_TEST(cudaEventCreate(&startEvt));
    CUDACHECK_TEST(cudaEventCreate(&endEvt));

    // Warmup graph replay
    for (int i = 0; i < FLAGS_warmup_iters; i++) {
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    CUDACHECK_TEST(cudaEventRecord(startEvt, stream));
    for (int i = 0; i < FLAGS_bench_iters; i++) {
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
    }
    CUDACHECK_TEST(cudaEventRecord(endEvt, stream));
    CUDACHECK_TEST(cudaEventSynchronize(endEvt));

    float elapsedMs = 0;
    CUDACHECK_TEST(cudaEventElapsedTime(&elapsedMs, startEvt, endEvt));
    double avgMs = static_cast<double>(elapsedMs) / FLAGS_bench_iters;
    if (globalRank == 0) {
      std::cout << std::left << std::setw(40) << "Graph capture (replay)"
                << std::right << std::fixed << std::setprecision(3)
                << std::setw(14) << avgMs << " ms" << std::endl;
    }

    CUDACHECK_TEST(cudaEventDestroy(startEvt));
    CUDACHECK_TEST(cudaEventDestroy(endEvt));

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
  }

  if (globalRank == 0) {
    std::cout
        << "================================================================"
        << std::endl;
  }

  // Cleanup
  NCCLCHECK_TEST(ncclCommDeregister(comm, hdl));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CUDACHECK_TEST(cudaFree(buf));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
  COMMCHECK_TEST(regCache->destroy());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
