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
  enum NCCL_BROADCAST_ALGO algo;
};

class CtranBroadcastTest : public CtranStandaloneMultiRankBaseTest,
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
    NCCL_BROADCAST_ALGO = param.algo;
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
      run(rank,
          [this](PerRankState& state) { runBroadcast(kBufferNElem, state); });
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

  void runBroadcast(size_t nElem, PerRankState& state, int root = 0) {
    validateConfigs(nElem);

    CLOGF(INFO, "rank {} broadcast with {} elems", state.rank, nElem);

    initBufferValues(nElem, state);

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

    CLOGF(INFO, "rank {} broadcast completed registration", state.rank);

    auto result = ctranBroadcast(
        state.srcBuffer,
        state.dstBuffer,
        nElem,
        kDataType,
        root,
        state.ctranComm.get(),
        state.stream);
    EXPECT_EQ(commSuccess, result);

    CLOGF(INFO, "rank {} broadcast scheduled", state.rank);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(state.stream));
    EXPECT_EQ(commSuccess, state.ctranComm->getAsyncResult());

    validateBroadcastData(nElem, state, root);

    CLOGF(INFO, "rank {} broadcast task completed", state.rank);
  }

  void validateBroadcastData(size_t nElem, PerRankState& state, int root) {
    std::vector<float> hostDst(nElem);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            hostDst.data(),
            state.dstBuffer,
            nElem * kTypeSize,
            cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nElem; ++i) {
      float expected = 1.0f;
      EXPECT_FLOAT_EQ(hostDst[i], expected)
          << "Mismatch at index " << i << " on rank " << state.rank;
    }
  }
};

TEST_P(CtranBroadcastTest, AbortDisabled) {
  auto param = GetParam();

  startWorkers(param);

  runTest(param);
}

TEST_P(CtranBroadcastTest, AbortEnabled) {
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
    CtranBroadcastTest,
    ::testing::Values(
        TestParam{"broadcast_ctdirect", NCCL_BROADCAST_ALGO::ctdirect},
        TestParam{"broadcast_ctbtree", NCCL_BROADCAST_ALGO::ctbtree}),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return info.param.name;
    });

} // namespace ctran::testing
