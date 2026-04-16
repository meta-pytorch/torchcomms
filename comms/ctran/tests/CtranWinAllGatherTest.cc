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
        ::testing::Values("ctdirect", "ctpipeline")),
    [](const ::testing::TestParamInfo<CtranWinAllGatherTestParam::ParamType>&
           info) {
      return "count_" + std::to_string(std::get<0>(info.param)) + "_" +
          std::get<1>(info.param);
    });

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
