// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranStandaloneUTUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::testing {

struct TestParam {
  std::string name;
  enum NCCL_ALLGATHER_ALGO algo;
};

class CtranAllGatherTest : public CtranStandaloneMultiRankBaseTest,
                           public ::testing::WithParamInterface<TestParam> {
 protected:
  static constexpr int kNRanks = 4;
  static_assert(kNRanks % 2 == 0);
  static constexpr commDataType_t kDataType = commFloat32;
  static constexpr size_t kTypeSize = sizeof(float);
  static constexpr size_t kBufferNElem = kBufferSize / kTypeSize;

  void SetUp() override {
    CtranStandaloneMultiRankBaseTest::SetUp();
  }

  void overrideEnvConfig(const TestParam& param) {
    NCCL_ALLGATHER_ALGO = param.algo;

    if (NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctring ||
        NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctrd ||
        NCCL_ALLGATHER_ALGO == NCCL_ALLGATHER_ALGO::ctbrucks) {
      NCCL_COMM_STATE_DEBUG_TOPO = NCCL_COMM_STATE_DEBUG_TOPO::nolocal;
      NCCL_IGNORE_TOPO_LOAD_FAILURE = true;
    }
  }

  void startWorkers(
      const TestParam& param,
      std::optional<std::vector<std::shared_ptr<::ctran::utils::Abort>>>
          aborts = std::nullopt) {
    overrideEnvConfig(param);
    CtranStandaloneMultiRankBaseTest::startWorkers(
        kNRanks,
        /*aborts=*/
        aborts.value_or(std::vector<std::shared_ptr<::ctran::utils::Abort>>{}));
  }

  void runTest(const TestParam& param) {
    for (int rank = 0; rank < kNRanks; ++rank) {
      run(rank, [this](PerRankState& state) {
        runAllGather(kBufferNElem / kNRanks, state);
      });
    }
  }

  void validateConfigs(size_t nElem) {
    ASSERT_TRUE(nElem <= kBufferNElem);
  }

  void initBufferValues(size_t nElem, PerRankState& state) {
    std::vector<float> hostSrc(nElem, 1.0f);
    std::vector<float> hostDst(nElem, 0.0f);

    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            state.srcBuffer,
            hostSrc.data(),
            nElem * kTypeSize,
            cudaMemcpyHostToDevice));

    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            state.dstBuffer,
            hostDst.data(),
            nElem * kTypeSize,
            cudaMemcpyHostToDevice));
  }

  void runAllGather(size_t sendCount, PerRankState& state) {
    validateConfigs(sendCount * kNRanks);

    CLOGF(INFO, "rank {} allGather with {} sendCount", state.rank, sendCount);

    initBufferValues(sendCount, state);

    void* srcHandle;
    void* dstHandle;
    ASSERT_EQ(
        commSuccess,
        state.ctranComm->ctran_->commRegister(
            state.srcBuffer, kBufferSize, &srcHandle));
    ASSERT_EQ(
        commSuccess,
        state.ctranComm->ctran_->commRegister(
            state.dstBuffer, kBufferSize, &dstHandle));
    SCOPE_EXIT {
      state.ctranComm->ctran_->commDeregister(dstHandle);
      state.ctranComm->ctran_->commDeregister(srcHandle);
    };

    CLOGF(INFO, "rank {} allGather completed registration", state.rank);

    if (!ctranAllGatherSupport(state.ctranComm.get(), NCCL_ALLGATHER_ALGO)) {
      GTEST_SKIP() << "ctranAllGatherSupport returns fails, skip test";
    }
    auto result = ctranAllGather(
        state.srcBuffer,
        state.dstBuffer,
        sendCount,
        kDataType,
        state.ctranComm.get(),
        state.stream,
        NCCL_ALLGATHER_ALGO);
    EXPECT_EQ(commSuccess, result);

    CLOGF(INFO, "rank {} allGather scheduled", state.rank);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));
    EXPECT_EQ(commSuccess, state.ctranComm->getAsyncResult());

    validateAllGatherData(sendCount, state);

    CLOGF(INFO, "rank {} allGather task completed", state.rank);
  }

  void validateAllGatherData(size_t sendCount, PerRankState& state) {
    std::vector<float> hostDst(sendCount * kNRanks);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            hostDst.data(),
            state.dstBuffer,
            sendCount * kNRanks * kTypeSize,
            cudaMemcpyDeviceToHost));

    for (int rank = 0; rank < kNRanks; ++rank) {
      for (size_t i = 0; i < sendCount; ++i) {
        size_t dstIdx = rank * sendCount + i;
        float expected = 1.0f;
        EXPECT_FLOAT_EQ(hostDst[dstIdx], expected)
            << "Mismatch at dst index " << dstIdx << " (from rank " << rank
            << ") on rank " << state.rank;
      }
    }
  }
};

TEST_P(CtranAllGatherTest, AbortDisabled) {
  auto param = GetParam();

  startWorkers(param);

  runTest(param);
}

TEST_P(CtranAllGatherTest, AbortEnabled) {
  auto param = GetParam();

  std::vector<std::shared_ptr<::ctran::utils::Abort>> aborts;
  aborts.reserve(kNRanks);
  for (int i = 0; i < kNRanks; ++i) {
    aborts.push_back(ctran::utils::createAbort(/*enabled=*/true));
  }
  startWorkers(param, aborts);

  runTest(param);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    CtranAllGatherTest,
    ::testing::Values(
        TestParam{"allgather_ctdirect", NCCL_ALLGATHER_ALGO::ctdirect},
        TestParam{"allgather_ctring", NCCL_ALLGATHER_ALGO::ctring},
        TestParam{"allgather_ctrd", NCCL_ALLGATHER_ALGO::ctrd},
        TestParam{"allgather_ctbrucks", NCCL_ALLGATHER_ALGO::ctbrucks}),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return info.param.name;
    });

} // namespace ctran::testing
