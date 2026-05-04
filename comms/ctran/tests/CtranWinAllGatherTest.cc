// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran;

class CtranWinAllGatherTest : public ctran::CtranDistTestFixture {
 public:
  CtranWinAllGatherTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTestFixture::SetUp();
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
  }

  void verifyAllGather(
      void* recvBuf,
      size_t sendCount,
      size_t sendBytes,
      int nRanks,
      int myRank,
      int iter) {
    for (int r = 0; r < nRanks; r++) {
      std::vector<float> observed(sendCount);
      CUDACHECK_TEST(cudaMemcpy(
          observed.data(),
          static_cast<char*>(recvBuf) + r * sendBytes,
          sendBytes,
          cudaMemcpyDeviceToHost));

      const float expected = static_cast<float>(r * 100 + iter);
      for (size_t i = 0; i < sendCount; i++) {
        EXPECT_EQ(observed[i], expected)
            << "rank " << myRank << " iter " << iter << " chunk from rank " << r
            << " element " << i;
      }
    }
  }

  void run(size_t sendCount, const std::string& algoStr) {
    SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", algoStr);

    auto comm = makeCtranComm();
    ASSERT_NE(comm, nullptr);

    auto statex = comm->statex_.get();
    ASSERT_NE(statex, nullptr);

    const auto nRanks = statex->nRanks();
    const auto myRank = statex->rank();
    const commDataType_t dt = commFloat;
    const size_t sendBytes = sendCount * commTypeSize(dt);
    const size_t recvBytes = sendBytes * nRanks;

    // Check support before allocating resources
    if (!CtranWin::allGatherPSupported(comm.get())) {
      GTEST_SKIP() << "allGatherP not supported on this topology";
    }
    const auto nNodes = statex->nNodes();
    if (algoStr == "ctrdpipeline" && nNodes > 1 &&
        (nNodes & (nNodes - 1)) != 0) {
      GTEST_SKIP() << "ctrd requires nNodes to be a power of 2";
    }

    cudaStream_t stream;
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Allocate recv buffer and register it with a window
    void* winBase = nullptr;
    CUDACHECK_TEST(cudaMalloc(&winBase, recvBytes));
    CtranWin* win = nullptr;
    auto res = ctranWinRegister(winBase, recvBytes, comm.get(), &win);
    ASSERT_EQ(res, commSuccess);

    // Allocate separate send buffer
    void* sendbuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendbuf, sendBytes));

    // Initialize allgather state from the window
    CtranPersistentRequest* request = nullptr;
    ASSERT_EQ(
        ctran::allGatherWinInit(win, comm.get(), stream, request), commSuccess);
    ASSERT_NE(request, nullptr);

    // Sync stream to ensure any init work is complete
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    constexpr int nIter = 3;
    for (int iter = 0; iter < nIter; iter++) {
      // Fill sendbuf with rank+iter specific values
      const float sendVal = static_cast<float>(myRank * 100 + iter);
      std::vector<float> sendVals(sendCount, sendVal);
      CUDACHECK_TEST(cudaMemcpyAsync(
          sendbuf, sendVals.data(), sendBytes, cudaMemcpyHostToDevice, stream));

      // Clear recvbuf (window data buffer)
      CUDACHECK_TEST(cudaMemsetAsync(winBase, 0, recvBytes, stream));

      ASSERT_EQ(
          ctran::allGatherWinExec(sendbuf, sendCount, dt, request),
          commSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      verifyAllGather(winBase, sendCount, sendBytes, nRanks, myRank, iter);
    }

    // Verify no GPE resource leak
    ASSERT_EQ(comm->ctran_->gpe->numInUseKernelElems(), 0);
    ASSERT_EQ(comm->ctran_->gpe->numInUseKernelFlags(), 0);

    ASSERT_EQ(ctran::allGatherWinDestroy(request), commSuccess);
    delete request;

    CUDACHECK_TEST(cudaFree(sendbuf));
    ASSERT_EQ(ctranWinFree(win), commSuccess);
    CUDACHECK_TEST(cudaFree(winBase));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }
};

class CtranWinAllGatherTestParam
    : public CtranWinAllGatherTest,
      public ::testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(CtranWinAllGatherTestParam, Basic) {
  const auto& [sendCount, algoStr] = GetParam();
  run(sendCount, algoStr);
}

INSTANTIATE_TEST_SUITE_P(
    WinAllGather,
    CtranWinAllGatherTestParam,
    ::testing::Combine(
        ::testing::Values(1024, 8192, 65536),
        ::testing::Values("ctdirect", "ctpipeline", "ctrdpipeline")),
    [](const ::testing::TestParamInfo<CtranWinAllGatherTestParam::ParamType>&
           info) {
      return "count_" + std::to_string(std::get<0>(info.param)) + "_" +
          std::get<1>(info.param);
    });

// Test window allgather with disjoint (multi-segment) memory allocation,
// simulating expandable segments. Verifies that ctranWinRegister works with
// globalRegisterWithPtr for buffers backed by multiple physical segments.
TEST_F(CtranWinAllGatherTest, DisjointMemory) {
  SysEnvRAII algoEnv("NCCL_ALLGATHER_P_ALGO", "ctdirect");

  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  auto statex = comm->statex_.get();
  ASSERT_NE(statex, nullptr);

  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();
  const commDataType_t dt = commFloat;
  // sendCount must be large enough that recvBytes (sendCount * 4 * nRanks)
  // spans multiple 2MB-aligned cuMem segments to trigger the multi-segment
  // path in pinRange. With 8 ranks: 1M * 4 * 8 = 32MB > 2MB.
  const size_t sendCount = 1024 * 1024;
  const size_t sendBytes = sendCount * commTypeSize(dt);
  const size_t recvBytes = sendBytes * nRanks;

  if (!CtranWin::allGatherPSupported(comm.get())) {
    GTEST_SKIP() << "allGatherP not supported on this topology";
  }

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  // Allocate recv buffer as disjoint segments (simulates expandable segments)
  constexpr int kNumSegments = 4;
  size_t segSize = recvBytes / kNumSegments;
  std::vector<size_t> segSizes(kNumSegments, segSize);
  std::vector<TestMemSegment> segments;
  void* winBase = nullptr;
  ASSERT_EQ(commMemAllocDisjoint(&winBase, segSizes, segments), commSuccess);

  CtranWin* win = nullptr;
  auto res = ctranWinRegister(winBase, recvBytes, comm.get(), &win);
  ASSERT_EQ(res, commSuccess);

  void* sendbuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendbuf, sendBytes));

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ctran::allGatherWinInit(win, comm.get(), stream, request), commSuccess);
  ASSERT_NE(request, nullptr);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  constexpr int nIter = 3;
  for (int iter = 0; iter < nIter; iter++) {
    const float sendVal = static_cast<float>(myRank * 100 + iter);
    std::vector<float> sendVals(sendCount, sendVal);
    CUDACHECK_TEST(cudaMemcpyAsync(
        sendbuf, sendVals.data(), sendBytes, cudaMemcpyHostToDevice, stream));
    CUDACHECK_TEST(cudaMemsetAsync(winBase, 0, recvBytes, stream));

    ASSERT_EQ(
        ctran::allGatherWinExec(sendbuf, sendCount, dt, request), commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    verifyAllGather(winBase, sendCount, sendBytes, nRanks, myRank, iter);
  }

  ASSERT_EQ(comm->ctran_->gpe->numInUseKernelElems(), 0);
  ASSERT_EQ(comm->ctran_->gpe->numInUseKernelFlags(), 0);

  ASSERT_EQ(ctran::allGatherWinDestroy(request), commSuccess);
  delete request;

  CUDACHECK_TEST(cudaFree(sendbuf));
  ASSERT_EQ(ctranWinFree(win), commSuccess);
  ASSERT_EQ(commMemFreeDisjoint(winBase, segSizes), commSuccess);
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// Test back-to-back window collective init+exec across multiple communicators.
// Exercises the GPE-thread init path (D101075857) to verify no races between
// overlapping init and exec on the mapper epoch lock.
TEST_F(CtranWinAllGatherTest, BackToBackMultiComm) {
  const size_t sendCount = 8192;
  const commDataType_t dt = commFloat;
  const int kComms = 4;

  auto firstComm = makeCtranComm();
  ASSERT_NE(firstComm, nullptr);

  const auto nRanks = firstComm->statex_->nRanks();
  const size_t sendBytes = sendCount * commTypeSize(dt);
  const size_t recvBytes = sendBytes * nRanks;

  // Check allGatherP support with a temporary window
  {
    void* tmpBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&tmpBuf, recvBytes));
    CtranWin* tmpWin = nullptr;
    ASSERT_EQ(
        ctranWinRegister(tmpBuf, recvBytes, firstComm.get(), &tmpWin),
        commSuccess);
    if (!tmpWin->allGatherPSupported()) {
      ASSERT_EQ(ctranWinFree(tmpWin), commSuccess);
      CUDACHECK_TEST(cudaFree(tmpBuf));
      GTEST_SKIP() << "allGatherP not supported on this topology";
    }
    ASSERT_EQ(ctranWinFree(tmpWin), commSuccess);
    CUDACHECK_TEST(cudaFree(tmpBuf));
  }
  firstComm.reset();

  std::vector<cudaStream_t> streams(kComms);
  std::vector<std::unique_ptr<CtranComm>> comms(kComms);
  std::vector<void*> winBases(kComms);
  std::vector<CtranWin*> wins(kComms);
  std::vector<void*> sendbufs(kComms);
  std::vector<CtranPersistentRequest*> requests(kComms);

  for (int c = 0; c < kComms; c++) {
    CUDACHECK_TEST(cudaStreamCreate(&streams[c]));
    comms[c] = makeCtranComm();
    ASSERT_NE(comms[c], nullptr);

    CUDACHECK_TEST(cudaMalloc(&winBases[c], recvBytes));
    ASSERT_EQ(
        ctranWinRegister(winBases[c], recvBytes, comms[c].get(), &wins[c]),
        commSuccess);

    CUDACHECK_TEST(cudaMalloc(&sendbufs[c], sendBytes));
  }

  const auto myRank = comms[0]->statex_->rank();

  // Init all windows back-to-back without synchronizing between them.
  // No stream sync here: execs below rely on waitInit() to wait for each
  // init's GPE callback to populate remote info, which is the mechanism
  // D101075857 establishes for window-based allGatherP.
  for (int c = 0; c < kComms; c++) {
    requests[c] = nullptr;
    ASSERT_EQ(
        ctran::allGatherWinInit(
            wins[c], comms[c].get(), streams[c], requests[c]),
        commSuccess);
    ASSERT_NE(requests[c], nullptr);
  }

  // Execute all back-to-back and verify correctness
  constexpr int nIter = 3;
  for (int iter = 0; iter < nIter; iter++) {
    for (int c = 0; c < kComms; c++) {
      const float sendVal = static_cast<float>(myRank * 100 + iter + c);
      std::vector<float> sendVals(sendCount, sendVal);
      CUDACHECK_TEST(cudaMemcpyAsync(
          sendbufs[c],
          sendVals.data(),
          sendBytes,
          cudaMemcpyHostToDevice,
          streams[c]));
      CUDACHECK_TEST(cudaMemsetAsync(winBases[c], 0, recvBytes, streams[c]));

      ASSERT_EQ(
          ctran::allGatherWinExec(sendbufs[c], sendCount, dt, requests[c]),
          commSuccess);
    }

    for (int c = 0; c < kComms; c++) {
      CUDACHECK_TEST(cudaStreamSynchronize(streams[c]));
      for (int r = 0; r < nRanks; r++) {
        std::vector<float> observed(sendCount);
        CUDACHECK_TEST(cudaMemcpy(
            observed.data(),
            static_cast<char*>(winBases[c]) + r * sendBytes,
            sendBytes,
            cudaMemcpyDeviceToHost));

        const float expected = static_cast<float>(r * 100 + iter + c);
        for (size_t i = 0; i < sendCount; i++) {
          EXPECT_EQ(observed[i], expected)
              << "comm " << c << " rank " << myRank << " iter " << iter
              << " chunk from rank " << r << " element " << i;
        }
      }
    }
  }

  // Verify no GPE resource leak
  for (int c = 0; c < kComms; c++) {
    ASSERT_EQ(comms[c]->ctran_->gpe->numInUseKernelElems(), 0);
    ASSERT_EQ(comms[c]->ctran_->gpe->numInUseKernelFlags(), 0);
  }

  // Cleanup
  for (int c = 0; c < kComms; c++) {
    ASSERT_EQ(ctran::allGatherWinDestroy(requests[c]), commSuccess);
    delete requests[c];
    CUDACHECK_TEST(cudaFree(sendbufs[c]));
    ASSERT_EQ(ctranWinFree(wins[c]), commSuccess);
    CUDACHECK_TEST(cudaFree(winBases[c]));
  }

  comms.clear();

  for (int c = 0; c < kComms; c++) {
    CUDACHECK_TEST(cudaStreamDestroy(streams[c]));
  }
}

class CtranWinAllGatherTestEnv : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();
    setenv("NCCL_DEBUG", "WARN", 0);
  }
};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranWinAllGatherTestEnv());
  return RUN_ALL_TESTS();
}
